"""Train the frozen-CLIP temporal fusion baseline on the fixed split."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.dataset import ChallengeDataset, build_dataloader, build_image_transform
from src.model import (
    CLIP_CONTEXT_LENGTH,
    DEFAULT_CLIP_MODEL,
    FusionConfig,
    TemporalOrderModel,
    clip_image_preprocess_config,
)
from src.training import evaluate_model, select_device, set_global_seed, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CLIP + temporal Transformer 24-way baseline."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split-dir", type=Path, default=Path("outputs/splits"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/models/clip_fusion_v1")
    )
    parser.add_argument("--model-name", default=DEFAULT_CLIP_MODEL)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".cache/huggingface/hub")
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--max-validation-steps", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def count_truncated_captions(tokenizer: object, sentences: list[str]) -> int:
    encoded = tokenizer(  # type: ignore[operator]
        sentences,
        add_special_tokens=True,
        truncation=False,
        verbose=False,
    )
    return sum(
        len(input_ids) > CLIP_CONTEXT_LENGTH for input_ids in encoded["input_ids"]
    )


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive.")
    set_global_seed(args.seed)
    device = select_device(args.device)
    data_dir = args.data_dir.resolve()
    split_dir = args.split_dir.resolve()
    output_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()

    print("Scope: TRAIN ONLY (Test data is not opened or analyzed).")
    print(f"device: {device}")
    print(f"pretrained model: {args.model_name}")
    transform = build_image_transform(clip_image_preprocess_config())
    train_dataset = ChallengeDataset.from_split_csv(
        data_dir=data_dir,
        split_csv=split_dir / "train_ids.csv",
        expected_split="train",
        transform=transform,
    )
    validation_dataset = ChallengeDataset.from_split_csv(
        data_dir=data_dir,
        split_csv=split_dir / "validation_ids.csv",
        expected_split="validation",
        transform=transform,
    )
    train_loader = build_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = build_dataloader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
        cache_dir=cache_dir,
    )
    train_sentences = [sample.sentence for sample in train_dataset.samples]
    validation_sentences = [sample.sentence for sample in validation_dataset.samples]
    train_truncated_count = count_truncated_captions(tokenizer, train_sentences)
    validation_truncated_count = count_truncated_captions(
        tokenizer, validation_sentences
    )
    print(
        f"Train captions over {CLIP_CONTEXT_LENGTH} tokens: "
        f"{train_truncated_count:,}/{len(train_sentences):,}"
    )
    print(
        f"Validation captions over {CLIP_CONTEXT_LENGTH} tokens: "
        f"{validation_truncated_count:,}/{len(validation_sentences):,}"
    )

    fusion_config = FusionConfig(freeze_clip=True)
    model = TemporalOrderModel.from_pretrained(
        args.model_name,
        config=fusion_config,
        local_files_only=args.local_files_only,
        cache_dir=cache_dir,
    ).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"total parameters: {total_parameters:,}")
    print(f"trainable parameters: {model.trainable_parameter_count():,}")

    history: list[dict[str, object]] = []
    best_exact_match = -1.0
    for epoch in range(1, args.epochs + 1):
        train_result = train_one_epoch(
            model,
            train_loader,
            tokenizer=tokenizer,
            optimizer=optimizer,
            device=device,
            max_steps=args.max_train_steps,
        )
        validation_result = evaluate_model(
            model,
            validation_loader,
            tokenizer=tokenizer,
            device=device,
            max_steps=args.max_validation_steps,
        )
        epoch_row: dict[str, object] = {
            "epoch": epoch,
            "train": train_result.to_dict(),
            "validation": validation_result.to_dict(),
        }
        history.append(epoch_row)
        print(
            f"epoch {epoch:02d} | "
            f"train loss={train_result.loss:.4f}, "
            f"exact={train_result.exact_match_accuracy:.4f} | "
            f"val loss={validation_result.loss:.4f}, "
            f"exact={validation_result.exact_match_accuracy:.4f}, "
            f"position={validation_result.position_accuracy:.4f}"
        )

        if not args.no_save and validation_result.exact_match_accuracy > best_exact_match:
            best_exact_match = validation_result.exact_match_accuracy
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_name": args.model_name,
                    "fusion_config": asdict(fusion_config),
                    "fusion_state_dict": model.fusion_state_dict(),
                    "epoch": epoch,
                    "validation": validation_result.to_dict(),
                },
                output_dir / "best_fusion_head.pt",
            )

    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)
        run_config = {
            "model_name": args.model_name,
            "fusion_config": asdict(fusion_config),
            "training": vars(args),
            "device": str(device),
            "train_count": len(train_dataset),
            "validation_count": len(validation_dataset),
            "train_truncated_caption_count": train_truncated_count,
            "validation_truncated_caption_count": validation_truncated_count,
        }
        # Convert Path values before JSON serialization.
        run_config["training"] = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
            json.dump(run_config, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"outputs: {output_dir}")


if __name__ == "__main__":
    main()
