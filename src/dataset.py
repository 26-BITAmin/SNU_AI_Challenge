"""Train/Validation dataset and batching for the four-frame ordering task.

The dataset keeps the CSV order of Input_1 through Input_4.  It never sorts
images by the answer before training because that would leak the target into
the model input.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Callable, Sequence, TypedDict

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2

from .data import DataValidationError, Sample, load_samples, parse_answer


Permutation = tuple[int, int, int, int]
ALL_PERMUTATIONS: tuple[Permutation, ...] = tuple(
    permutations((1, 2, 3, 4))
)  # type: ignore[assignment]
PERMUTATION_TO_CLASS = {
    permutation: class_index
    for class_index, permutation in enumerate(ALL_PERMUTATIONS)
}
SPLIT_COLUMNS = (
    "Id",
    "Answer",
    "split",
    "duplicate_component_id",
    "duplicate_component_size",
)


@dataclass(frozen=True)
class ImagePreprocessConfig:
    """Deterministic image preprocessing used by the first baseline."""

    image_size: int = 224
    resize_size: int = 256
    normalize: bool = True
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def validate(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive.")
        if self.resize_size < self.image_size:
            raise ValueError("resize_size must be at least image_size.")
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("mean and std must contain three RGB values.")
        if any(value <= 0 for value in self.std):
            raise ValueError("std values must be positive.")


@dataclass(frozen=True)
class SplitRecord:
    """One row in a generated Train/Validation ID file."""

    sample_id: str
    answer: Permutation
    split: str
    duplicate_component_id: int
    duplicate_component_size: int


class ChallengeItem(TypedDict):
    """One sample before DataLoader collation."""

    sample_id: str
    sentence: str
    input_names: tuple[str, str, str, str]
    images: Tensor
    answer: Tensor
    target: int
    no_ordering: bool


class ChallengeBatch(TypedDict):
    """One model-ready batch, except for model-specific text tokenization."""

    sample_ids: list[str]
    sentences: list[str]
    input_names: list[tuple[str, str, str, str]]
    images: Tensor
    answers: Tensor
    targets: Tensor
    no_ordering: Tensor


def encode_answer(answer: Sequence[int]) -> int:
    """Map an official answer permutation to a stable class index from 0 to 23."""

    permutation = tuple(answer)
    if permutation not in PERMUTATION_TO_CLASS:
        raise DataValidationError(
            f"Answer must be a permutation of 1..4, got {permutation!r}."
        )
    return PERMUTATION_TO_CLASS[permutation]  # type: ignore[index]


def decode_class(class_index: int) -> Permutation:
    """Map a 24-way classifier output back to the official answer format."""

    if not 0 <= class_index < len(ALL_PERMUTATIONS):
        raise ValueError(
            f"class_index must be between 0 and 23, got {class_index}."
        )
    return ALL_PERMUTATIONS[class_index]


def build_image_transform(
    config: ImagePreprocessConfig | None = None,
) -> Callable[[Image.Image], Tensor]:
    """Build a deterministic RGB transform without modifying source images."""

    config = config or ImagePreprocessConfig()
    config.validate()
    operations: list[Callable] = [
        v2.Resize(
            config.resize_size,
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        v2.CenterCrop(config.image_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
    if config.normalize:
        operations.append(v2.Normalize(mean=config.mean, std=config.std))
    return v2.Compose(operations)


def load_split_records(
    split_csv: Path, *, expected_split: str | None = None
) -> list[SplitRecord]:
    """Read and validate one generated split file."""

    if expected_split not in {None, "train", "validation"}:
        raise ValueError("expected_split must be 'train', 'validation', or None.")

    records: list[SplitRecord] = []
    seen_ids: set[str] = set()
    with split_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SPLIT_COLUMNS:
            raise DataValidationError(
                f"Unexpected columns in {split_csv}: {reader.fieldnames!r}; "
                f"expected {list(SPLIT_COLUMNS)!r}."
            )
        for row_number, row in enumerate(reader, start=2):
            sample_id = row["Id"]
            if sample_id in seen_ids:
                raise DataValidationError(
                    f"Duplicate Id {sample_id!r} in {split_csv}:{row_number}."
                )
            if row["split"] not in {"train", "validation"}:
                raise DataValidationError(
                    f"Invalid split {row['split']!r} in {split_csv}:{row_number}."
                )
            if expected_split is not None and row["split"] != expected_split:
                raise DataValidationError(
                    f"Expected split {expected_split!r} but found {row['split']!r} "
                    f"in {split_csv}:{row_number}."
                )
            try:
                component_id = int(row["duplicate_component_id"])
                component_size = int(row["duplicate_component_size"])
            except ValueError as exc:
                raise DataValidationError(
                    f"Invalid duplicate component values in {split_csv}:{row_number}."
                ) from exc
            if component_id <= 0 or component_size <= 0:
                raise DataValidationError(
                    f"Duplicate component values must be positive in "
                    f"{split_csv}:{row_number}."
                )

            seen_ids.add(sample_id)
            records.append(
                SplitRecord(
                    sample_id=sample_id,
                    answer=parse_answer(row["Answer"]),
                    split=row["split"],
                    duplicate_component_id=component_id,
                    duplicate_component_size=component_size,
                )
            )

    if not records:
        raise DataValidationError(f"No split records found in {split_csv}.")
    return records


class ChallengeDataset(Dataset[ChallengeItem]):
    """Load four images and one caption for each labeled Train sample."""

    def __init__(
        self,
        samples: Sequence[Sample],
        *,
        image_dir: Path,
        transform: Callable[[Image.Image], Tensor] | None = None,
    ) -> None:
        if not samples:
            raise ValueError("ChallengeDataset requires at least one sample.")
        self.samples = tuple(samples)
        self.image_dir = image_dir
        self.transform = transform or build_image_transform()
        for sample in self.samples:
            if sample.answer is None or sample.no_ordering is None:
                raise DataValidationError(
                    f"Dataset sample {sample.sample_id!r} must have Train labels."
                )

    @classmethod
    def from_split_csv(
        cls,
        *,
        data_dir: Path,
        split_csv: Path,
        expected_split: str,
        transform: Callable[[Image.Image], Tensor] | None = None,
    ) -> "ChallengeDataset":
        """Build a Train-only dataset from the fixed split IDs."""

        all_samples = load_samples(data_dir / "train.csv", labeled=True)
        sample_by_id = {sample.sample_id: sample for sample in all_samples}
        records = load_split_records(split_csv, expected_split=expected_split)
        selected_samples: list[Sample] = []
        for record in records:
            sample = sample_by_id.get(record.sample_id)
            if sample is None:
                raise DataValidationError(
                    f"Split Id {record.sample_id!r} is missing from train.csv."
                )
            if sample.answer != record.answer:
                raise DataValidationError(
                    f"Answer mismatch for {record.sample_id!r}: "
                    f"train.csv={sample.answer}, split={record.answer}."
                )
            selected_samples.append(sample)
        return cls(
            selected_samples,
            image_dir=data_dir / "train",
            transform=transform,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ChallengeItem:
        sample = self.samples[index]
        image_tensors: list[Tensor] = []
        for image_path in sample.image_paths(self.image_dir):
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")
                image_tensor = self.transform(rgb_image)
            if image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
                raise DataValidationError(
                    f"Transform must return [3, H, W] for {image_path}, "
                    f"got {tuple(image_tensor.shape)}."
                )
            image_tensors.append(image_tensor)

        if sample.answer is None or sample.no_ordering is None:
            raise DataValidationError(f"Sample {sample.sample_id!r} has no Train label.")
        return {
            "sample_id": sample.sample_id,
            "sentence": sample.sentence,
            "input_names": sample.input_names,
            "images": torch.stack(image_tensors),
            "answer": torch.tensor(sample.answer, dtype=torch.long),
            "target": encode_answer(sample.answer),
            "no_ordering": sample.no_ordering,
        }


def collate_challenge_items(items: Sequence[ChallengeItem]) -> ChallengeBatch:
    """Combine samples while keeping text and filenames easy to inspect."""

    if not items:
        raise ValueError("Cannot collate an empty batch.")
    return {
        "sample_ids": [item["sample_id"] for item in items],
        "sentences": [item["sentence"] for item in items],
        "input_names": [item["input_names"] for item in items],
        "images": torch.stack([item["images"] for item in items]),
        "answers": torch.stack([item["answer"] for item in items]),
        "targets": torch.tensor(
            [item["target"] for item in items], dtype=torch.long
        ),
        "no_ordering": torch.tensor(
            [item["no_ordering"] for item in items], dtype=torch.bool
        ),
    }


def build_dataloader(
    dataset: ChallengeDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Build a reproducible loader; model-specific tokenization comes later."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_challenge_items,
        generator=generator,
        persistent_workers=num_workers > 0,
    )
