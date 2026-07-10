"""Read-only data loading and integrity checks for the competition dataset.

The official target describes the original temporal position of each input
frame.  For example, an answer of ``[3, 1, 2, 4]`` means that Input_1 is the
third frame, Input_2 is the first frame, and so on.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


INPUT_COLUMNS = ("Input_1", "Input_2", "Input_3", "Input_4")
TRAIN_COLUMNS = ("Id", *INPUT_COLUMNS, "Sentence", "Answer", "No_ordering")
TEST_COLUMNS = ("Id", *INPUT_COLUMNS, "Sentence")
SUBMISSION_COLUMNS = ("Id", "Answer")
VALID_POSITIONS = (1, 2, 3, 4)
ID_PATTERN = re.compile(r"^[A-Za-z0-9]{6}$")
IMAGE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9]{6}_[a-z]{3}\.jpg$")
IGNORED_METADATA_FILES = {".DS_Store"}


class DataValidationError(ValueError):
    """Raised when the dataset does not match the expected competition format."""


@dataclass(frozen=True)
class Sample:
    """One competition sample and its four input frames."""

    sample_id: str
    input_names: tuple[str, str, str, str]
    sentence: str
    answer: tuple[int, int, int, int] | None = None
    no_ordering: bool | None = None

    def image_paths(self, split_dir: Path) -> tuple[Path, Path, Path, Path]:
        """Return image paths in the CSV input order."""

        sample_dir = split_dir / self.sample_id
        return tuple(sample_dir / name for name in self.input_names)  # type: ignore[return-value]

    def ordered_input_names(self) -> tuple[str, str, str, str]:
        """Return the four filenames in the original chronological order."""

        if self.answer is None:
            raise DataValidationError(
                f"Sample {self.sample_id!r} has no answer and cannot be reordered."
            )
        answer_and_names = zip(self.answer, self.input_names, strict=True)
        ordered = tuple(name for _, name in sorted(answer_and_names))
        return ordered  # type: ignore[return-value]


@dataclass(frozen=True)
class DatasetAudit:
    """Compact result of a split-wide structural validation."""

    split: str
    sample_count: int
    image_count: int
    label_permutation_count: int | None


def parse_answer(raw_answer: str) -> tuple[int, int, int, int]:
    """Parse and validate a CSV answer cell as a permutation of 1..4."""

    try:
        parsed = json.loads(raw_answer)
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"Invalid Answer JSON: {raw_answer!r}") from exc

    if not isinstance(parsed, list) or len(parsed) != 4:
        raise DataValidationError(
            f"Answer must be a list of four positions, got {raw_answer!r}."
        )
    if any(type(position) is not int for position in parsed):
        raise DataValidationError(
            f"Answer positions must be integers, got {raw_answer!r}."
        )
    if tuple(sorted(parsed)) != VALID_POSITIONS:
        raise DataValidationError(
            f"Answer must be a permutation of 1..4, got {raw_answer!r}."
        )

    return tuple(parsed)  # type: ignore[return-value]


def _parse_bool(raw_value: str, *, field_name: str, sample_id: str) -> bool:
    normalized = raw_value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise DataValidationError(
        f"{field_name} for sample {sample_id!r} must be True or False, "
        f"got {raw_value!r}."
    )


def _validate_row_identity(sample_id: str, input_names: Sequence[str]) -> None:
    if not ID_PATTERN.fullmatch(sample_id):
        raise DataValidationError(
            f"Id must be six case-sensitive ASCII letters/digits, got {sample_id!r}."
        )
    if len(set(input_names)) != 4:
        raise DataValidationError(f"Sample {sample_id!r} does not have four unique inputs.")

    for image_name in input_names:
        if not IMAGE_NAME_PATTERN.fullmatch(image_name):
            raise DataValidationError(
                f"Unexpected image filename for sample {sample_id!r}: {image_name!r}."
            )
        if not image_name.startswith(f"{sample_id}_"):
            raise DataValidationError(
                f"Image {image_name!r} does not belong to sample {sample_id!r}."
            )


def _sample_from_row(
    row: Mapping[str, str], *, labeled: bool, csv_path: Path, row_number: int
) -> Sample:
    sample_id = row["Id"]
    input_names = tuple(row[column] for column in INPUT_COLUMNS)
    _validate_row_identity(sample_id, input_names)

    sentence = row["Sentence"]
    if not isinstance(sentence, str) or sentence == "":
        raise DataValidationError(
            f"Empty Sentence at {csv_path}:{row_number} for sample {sample_id!r}."
        )

    if not labeled:
        return Sample(
            sample_id=sample_id,
            input_names=input_names,  # type: ignore[arg-type]
            sentence=sentence,
        )

    answer = parse_answer(row["Answer"])
    no_ordering = _parse_bool(
        row["No_ordering"], field_name="No_ordering", sample_id=sample_id
    )
    expected_no_ordering = answer == VALID_POSITIONS
    if no_ordering != expected_no_ordering:
        raise DataValidationError(
            f"No_ordering disagrees with Answer for sample {sample_id!r}: "
            f"Answer={answer}, No_ordering={no_ordering}."
        )

    return Sample(
        sample_id=sample_id,
        input_names=input_names,  # type: ignore[arg-type]
        sentence=sentence,
        answer=answer,
        no_ordering=no_ordering,
    )


def load_samples(csv_path: Path, *, labeled: bool) -> list[Sample]:
    """Load a train or test CSV without modifying any source data."""

    expected_columns = TRAIN_COLUMNS if labeled else TEST_COLUMNS
    samples: list[Sample] = []
    seen_ids: set[str] = set()

    # utf-8-sig accepts ordinary UTF-8 and strips the BOM present in train.csv.
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected_columns:
            raise DataValidationError(
                f"Unexpected columns in {csv_path}: {reader.fieldnames!r}; "
                f"expected {list(expected_columns)!r}."
            )

        for row_number, row in enumerate(reader, start=2):
            sample = _sample_from_row(
                row, labeled=labeled, csv_path=csv_path, row_number=row_number
            )
            if sample.sample_id in seen_ids:
                raise DataValidationError(
                    f"Duplicate Id {sample.sample_id!r} in {csv_path}."
                )
            seen_ids.add(sample.sample_id)
            samples.append(sample)

    if not samples:
        raise DataValidationError(f"No samples found in {csv_path}.")
    return samples


def _visible_files(directory: Path) -> set[str]:
    return {
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.name not in IGNORED_METADATA_FILES
    }


def audit_split(data_dir: Path, split: str) -> DatasetAudit:
    """Validate the CSV, ID folders, filenames, labels, and image links."""

    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}.")

    labeled = split == "train"
    csv_path = data_dir / f"{split}.csv"
    split_dir = data_dir / split
    if not csv_path.is_file():
        raise DataValidationError(f"Missing CSV: {csv_path}")
    if not split_dir.is_dir():
        raise DataValidationError(f"Missing image directory: {split_dir}")

    samples = load_samples(csv_path, labeled=labeled)
    csv_ids = {sample.sample_id for sample in samples}
    folder_ids = {
        path.name
        for path in split_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }

    missing_folders = sorted(csv_ids - folder_ids)
    extra_folders = sorted(folder_ids - csv_ids)
    if missing_folders or extra_folders:
        raise DataValidationError(
            f"{split} CSV/folder Id mismatch: missing={missing_folders[:5]!r}, "
            f"extra={extra_folders[:5]!r}."
        )

    image_count = 0
    for sample in samples:
        sample_dir = split_dir / sample.sample_id
        expected_names = set(sample.input_names)
        actual_names = _visible_files(sample_dir)
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            raise DataValidationError(
                f"File mismatch for {split}/{sample.sample_id}: "
                f"missing={missing!r}, extra={extra!r}."
            )
        image_count += len(actual_names)

    label_permutations: set[tuple[int, int, int, int]] | None = None
    if labeled:
        label_permutations = {
            sample.answer for sample in samples if sample.answer is not None
        }

    return DatasetAudit(
        split=split,
        sample_count=len(samples),
        image_count=image_count,
        label_permutation_count=(
            len(label_permutations) if label_permutations is not None else None
        ),
    )


def validate_submission_template(data_dir: Path) -> int:
    """Verify that sample_submission IDs exactly match test.csv in row order."""

    test_samples = load_samples(data_dir / "test.csv", labeled=False)
    template_path = data_dir / "sample_submission.csv"
    template_ids: list[str] = []

    with template_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SUBMISSION_COLUMNS:
            raise DataValidationError(
                f"Unexpected columns in {template_path}: {reader.fieldnames!r}; "
                f"expected {list(SUBMISSION_COLUMNS)!r}."
            )
        for row_number, row in enumerate(reader, start=2):
            if row["Answer"] not in {"", None}:
                raise DataValidationError(
                    f"Expected an empty Answer in template at row {row_number}."
                )
            template_ids.append(row["Id"])

    test_ids = [sample.sample_id for sample in test_samples]
    if template_ids != test_ids:
        raise DataValidationError(
            "sample_submission.csv Ids do not match test.csv in the same row order."
        )
    if len(set(template_ids)) != len(template_ids):
        raise DataValidationError("Duplicate Id found in sample_submission.csv.")

    return len(template_ids)
