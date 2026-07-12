"""Run guarded BF16 LoRA training on the fixed Train split.

This script never opens Test data.  Train-only frame permutation augmentation
is applied before building Qwen multi-image conversations.  Validation keeps
the original four-image order and is used only for loss monitoring here;
Exact Match is measured afterward with ``scripts.evaluate_qwen_baseline``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from packaging.version import Version
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.qwen_training import (
    QwenTrainingExample,
    build_augmented_only_examples,
    build_qwen_sft_messages,
    build_training_examples,
    choose_training_samples,
    load_labeled_split_samples,
)


MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BF16 LoRA training for the four-frame Qwen model."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--train-split-csv",
        type=Path,
        default=Path("outputs/splits/train_ids.csv"),
    )
    parser.add_argument(
        "--validation-split-csv",
        type=Path,
        default=Path("outputs/splits/validation_ids.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/qwen25_vl_7b_lora_smoke"),
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument(
        "--initial-adapter-path",
        type=Path,
        default=None,
        help="Optional existing LoRA adapter to continue training as one model.",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".cache/huggingface/hub")
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=500,
        help="Unique Train samples before augmentation; 0 uses all 7,628.",
    )
    parser.add_argument(
        "--validation-limit",
        type=int,
        default=100,
        help="Original-order Validation samples used for eval loss; 0 uses all.",
    )
    parser.add_argument(
        "--augmentation-copies",
        type=int,
        default=1,
        help="Non-identity shuffled copies per Train sample; originals are also kept.",
    )
    parser.add_argument(
        "--augmented-only",
        action="store_true",
        help=(
            "Use exactly one deterministic shuffled view per Train ID instead "
            "of keeping originals plus copies. Requires --augmentation-copies 1."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=16 * 28 * 28,
        help="Minimum pixels per image used by the Qwen processor.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=576 * 28 * 28,
        help="Maximum pixels per image used by the Qwen processor.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


class QwenSupervisedDataset(Dataset[dict[str, object]]):
    """Lazily build messages so source images are never copied or modified."""

    def __init__(self, examples: Sequence[QwenTrainingExample], image_dir: Path):
        if not examples:
            raise ValueError("QwenSupervisedDataset requires at least one example.")
        self.examples = tuple(examples)
        self.image_dir = image_dir

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        example = self.examples[index]
        return {
            "example_id": example.example_id,
            "messages": build_qwen_sft_messages(example, self.image_dir),
        }


@dataclass
class QwenCompletionOnlyCollator:
    """Process four images and mask every token before the assistant Answer."""

    processor: object

    def __call__(self, examples: Sequence[dict[str, object]]) -> dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("Cannot collate an empty batch.")
        messages_list: list[list[dict[str, object]]] = []
        for example in examples:
            messages = example["messages"]
            if not isinstance(messages, list) or len(messages) != 2:
                raise ValueError("Each SFT example must contain user and assistant turns.")
            messages_list.append(messages)

        full_texts = [
            self.processor.apply_chat_template(  # type: ignore[attr-defined]
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            for messages in messages_list
        ]
        prompt_texts = [
            self.processor.apply_chat_template(  # type: ignore[attr-defined]
                messages[:-1],
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in messages_list
        ]

        batch_images: list[object] = []
        for messages in messages_list:
            image_inputs, video_inputs = process_vision_info(messages)
            if video_inputs:
                raise ValueError("This challenge pipeline expects images, not videos.")
            if image_inputs is None or len(image_inputs) != 4:
                raise ValueError("Each challenge sample must resolve to exactly four images.")
            batch_images.append(image_inputs)

        batch = self.processor(  # type: ignore[operator]
            text=full_texts,
            images=batch_images,
            padding=True,
            return_tensors="pt",
        )
        prompt_batch = self.processor(  # type: ignore[operator]
            text=prompt_texts,
            images=batch_images,
            padding=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        for index in range(len(examples)):
            full_length = int(batch["attention_mask"][index].sum().item())
            prompt_length = int(prompt_batch["attention_mask"][index].sum().item())
            if prompt_length >= full_length:
                raise ValueError("Assistant Answer contains no supervised tokens.")
            full_prefix = batch["input_ids"][index, :prompt_length]
            prompt_ids = prompt_batch["input_ids"][index, :prompt_length]
            if not torch.equal(full_prefix, prompt_ids):
                raise ValueError(
                    "The prompt-only chat template is not a prefix of the SFT example."
                )
            labels[index, :prompt_length] = -100

        if int((labels != -100).sum().item()) == 0:
            raise ValueError("The batch has no assistant Answer tokens for loss.")
        batch["labels"] = labels
        return dict(batch)


def require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen2.5-VL-7B LoRA training requires a CUDA GPU.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This guarded configuration requires BF16-capable CUDA.")
    try:
        torchao_version = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        torchao_version = None
    if torchao_version is not None and Version(torchao_version) < Version("0.16.0"):
        raise RuntimeError(
            f"Incompatible optional torchao {torchao_version} is installed. "
            "This BF16 LoRA run does not use torchao; run "
            "`pip uninstall -y torchao` and retry."
        )

    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    if args.train_limit < 0 or args.validation_limit < 0:
        raise ValueError("Train and Validation limits must be 0 or positive.")
    if args.augmentation_copies < 0:
        raise ValueError("augmentation-copies cannot be negative.")
    if args.augmented_only and args.augmentation_copies != 1:
        raise ValueError("augmented-only requires augmentation-copies=1.")
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("max-steps must be -1 for epoch training or a positive integer.")
    for name in (
        "batch_size",
        "eval_batch_size",
        "gradient_accumulation_steps",
        "logging_steps",
        "eval_steps",
        "save_steps",
        "lora_r",
        "lora_alpha",
    ):
        require_positive(name, int(getattr(args, name)))
    if args.min_pixels <= 0 or args.max_pixels < args.min_pixels:
        raise ValueError("Pixel limits must be positive and max-pixels >= min-pixels.")

    set_seed(args.seed)
    data_dir = args.data_dir.resolve()
    train_split_csv = args.train_split_csv.resolve()
    validation_split_csv = args.validation_split_csv.resolve()
    output_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    initial_adapter_path = (
        args.initial_adapter_path.resolve() if args.initial_adapter_path else None
    )
    if initial_adapter_path is not None and not initial_adapter_path.is_dir():
        raise FileNotFoundError(
            f"Initial LoRA adapter directory not found: {initial_adapter_path}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    all_train_samples = load_labeled_split_samples(
        data_dir=data_dir,
        split_csv=train_split_csv,
        expected_split="train",
    )
    all_validation_samples = load_labeled_split_samples(
        data_dir=data_dir,
        split_csv=validation_split_csv,
        expected_split="validation",
    )
    selected_train_samples = choose_training_samples(
        all_train_samples,
        limit=None if args.train_limit == 0 else args.train_limit,
        seed=args.seed,
    )
    selected_validation_samples = choose_training_samples(
        all_validation_samples,
        limit=None if args.validation_limit == 0 else args.validation_limit,
        seed=args.seed,
    )
    if args.augmented_only:
        train_examples = build_augmented_only_examples(
            selected_train_samples,
            seed=args.seed,
        )
        augmentation_strategy = "one_deterministic_shuffled_view_per_source"
    else:
        train_examples = build_training_examples(
            selected_train_samples,
            augmentation_copies=args.augmentation_copies,
            seed=args.seed,
        )
        augmentation_strategy = "original_plus_deterministic_copies"
    validation_examples = build_training_examples(
        selected_validation_samples,
        augmentation_copies=0,
        seed=args.seed,
    )
    train_dataset = QwenSupervisedDataset(train_examples, data_dir / "train")
    validation_dataset = QwenSupervisedDataset(
        validation_examples, data_dir / "train"
    )

    run_config = {
        "scope": "train_split_only_no_test_access",
        "model_name": args.model_name,
        "initial_adapter_path": (
            str(initial_adapter_path) if initial_adapter_path is not None else None
        ),
        "seed": args.seed,
        "split_counts": {
            "available_train": len(all_train_samples),
            "selected_unique_train": len(selected_train_samples),
            "train_examples_after_augmentation": len(train_examples),
            "augmentation_copies_per_train_sample": args.augmentation_copies,
            "augmentation_strategy": augmentation_strategy,
            "available_validation": len(all_validation_samples),
            "selected_validation": len(selected_validation_samples),
            "validation_augmentation_copies": 0,
        },
        "optimization": {
            "max_steps": args.max_steps,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": (
                args.batch_size * args.gradient_accumulation_steps
            ),
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
        },
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": list(LORA_TARGET_MODULES),
        },
        "vision": {
            "min_pixels_per_image": args.min_pixels,
            "max_pixels_per_image": args.max_pixels,
            "images_per_sample": 4,
        },
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print("Scope: FIXED TRAIN SPLIT ONLY; Test data is not opened.")
    print(
        f"Train: {len(selected_train_samples):,}/{len(all_train_samples):,} unique "
        f"samples -> {len(train_examples):,} examples after augmentation"
    )
    print(
        f"Validation: {len(selected_validation_samples):,}/"
        f"{len(all_validation_samples):,} original-order samples"
    )
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Initial adapter: {initial_adapter_path or 'none (new LoRA)'}")
    print(f"Output: {output_dir}")

    processor = AutoProcessor.from_pretrained(
        args.model_name,
        cache_dir=cache_dir,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        local_files_only=args.local_files_only,
    )
    processor.tokenizer.padding_side = "right"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        device_map={"": torch.cuda.current_device()},
        attn_implementation="sdpa",
        cache_dir=cache_dir,
        local_files_only=args.local_files_only,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if initial_adapter_path is not None:
        model = PeftModel.from_pretrained(
            model,
            str(initial_adapter_path),
            is_trainable=True,
        )
    else:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=list(LORA_TARGET_MODULES),
        )
        model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    collator = QwenCompletionOnlyCollator(processor)
    audit_batch = collator([train_dataset[0]])
    supervised_tokens = int((audit_batch["labels"] != -100).sum().item())
    total_tokens = int(audit_batch["attention_mask"].sum().item())
    print(
        f"Collator audit passed: {supervised_tokens} assistant tokens supervised "
        f"out of {total_tokens} non-padding tokens."
    )
    del audit_batch

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        max_steps=args.max_steps,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
    )
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    final_adapter_dir = output_dir / "final_adapter"
    model.config.use_cache = True
    trainer.save_state()
    trainer.save_model(str(final_adapter_dir))
    processor.save_pretrained(final_adapter_dir)
    summary = {
        "scope": "train_split_only_no_test_access",
        "train_metrics": train_result.metrics,
        "validation_loss_metrics": eval_metrics,
        "final_adapter_dir": str(final_adapter_dir),
        "continued_from_adapter": (
            str(initial_adapter_path) if initial_adapter_path is not None else None
        ),
        "exact_match_evaluated": False,
        "next_action": "Run fixed Validation generation with the saved adapter.",
    }
    with (output_dir / "training_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print("LoRA training completed.")
    print(f"Final adapter: {final_adapter_dir}")
    print("Exact Match has not been measured by this training script.")


if __name__ == "__main__":
    main()
