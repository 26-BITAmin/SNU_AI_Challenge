"""Run read-only structural checks on the competition data."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data import audit_split, validate_submission_template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate SNU AI Challenge CSV, labels, folders, and image links."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing train.csv, test.csv, train/, and test/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()

    train_audit = audit_split(data_dir, "train")
    test_audit = audit_split(data_dir, "test")
    submission_rows = validate_submission_template(data_dir)

    print("Data validation passed.")
    print(
        f"train: {train_audit.sample_count:,} samples, "
        f"{train_audit.image_count:,} images, "
        f"{train_audit.label_permutation_count} valid label permutations"
    )
    print(
        f"test: {test_audit.sample_count:,} samples, "
        f"{test_audit.image_count:,} images (structure only)"
    )
    print(
        f"sample_submission: {submission_rows:,} rows, "
        "Ids aligned with test.csv"
    )


if __name__ == "__main__":
    main()
