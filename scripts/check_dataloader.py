"""Load one Train and Validation batch and print the model-input contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.dataset import (
    ChallengeDataset,
    ImagePreprocessConfig,
    build_dataloader,
    build_image_transform,
    decode_class,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the Train-only preprocessing and DataLoader shapes."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split-dir", type=Path, default=Path("outputs/splits"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def print_batch(name: str, batch: dict[str, object]) -> None:
    images = batch["images"]
    answers = batch["answers"]
    targets = batch["targets"]
    sample_ids = batch["sample_ids"]
    sentences = batch["sentences"]
    input_names = batch["input_names"]
    no_ordering = batch["no_ordering"]
    assert isinstance(images, torch.Tensor)
    assert isinstance(answers, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
    assert isinstance(sample_ids, list)
    assert isinstance(sentences, list)
    assert isinstance(input_names, list)
    assert isinstance(no_ordering, torch.Tensor)

    print(f"\n[{name} batch]")
    print(f"images shape: {tuple(images.shape)}")
    print(f"images dtype: {images.dtype}")
    print(f"answers shape: {tuple(answers.shape)}")
    print(f"targets shape: {tuple(targets.shape)}")
    print(f"no_ordering shape: {tuple(no_ordering.shape)}")
    print(f"finite pixels: {bool(torch.isfinite(images).all())}")
    print(f"normalized range: {images.min().item():.4f} to {images.max().item():.4f}")
    print(f"first Id: {sample_ids[0]}")
    print(f"first caption: {sentences[0]}")
    print(f"first input names: {input_names[0]}")
    print(f"first Answer: {answers[0].tolist()}")
    print(f"first class target: {targets[0].item()}")
    print(f"decoded target: {list(decode_class(int(targets[0].item())))}")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    split_dir = args.split_dir.resolve()
    transform = build_image_transform(
        ImagePreprocessConfig(
            image_size=args.image_size,
            resize_size=args.resize_size,
        )
    )

    print("Scope: TRAIN ONLY (Test data is not opened or analyzed).")
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
        seed=42,
        num_workers=args.num_workers,
    )
    validation_loader = build_dataloader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        seed=42,
        num_workers=args.num_workers,
    )

    print(f"train samples: {len(train_dataset):,}")
    print(f"validation samples: {len(validation_dataset):,}")
    print_batch("Train", next(iter(train_loader)))
    print_batch("Validation", next(iter(validation_loader)))


if __name__ == "__main__":
    main()
