"""Strict construction and validation of the competition submission file."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

from .data import DataValidationError, SUBMISSION_COLUMNS, parse_answer


def build_submission_rows(
    template_ids: Sequence[str],
    predictions: Mapping[str, Sequence[int]],
) -> list[dict[str, str]]:
    """Order validated predictions exactly like sample_submission.csv."""

    if not template_ids:
        raise ValueError("At least one template Id is required.")
    if len(set(template_ids)) != len(template_ids):
        raise DataValidationError("Template IDs must be unique.")
    expected = set(template_ids)
    actual = set(predictions)
    if expected != actual:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise DataValidationError(
            f"Prediction IDs do not match template: missing={missing[:5]!r}, "
            f"extra={extra[:5]!r}."
        )

    rows: list[dict[str, str]] = []
    for sample_id in template_ids:
        raw_answer = json.dumps(list(predictions[sample_id]))
        answer = parse_answer(raw_answer)
        rows.append(
            {
                "Id": sample_id,
                "Answer": json.dumps(list(answer)),
            }
        )
    return rows


def write_submission(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    """Write only the official Id and Answer columns."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUBMISSION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def validate_submission_file(path: Path, template_ids: Sequence[str]) -> int:
    """Re-open a generated file and verify schema, order, IDs, and permutations."""

    ids: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SUBMISSION_COLUMNS:
            raise DataValidationError(
                f"Unexpected submission columns: {reader.fieldnames!r}."
            )
        for row_number, row in enumerate(reader, start=2):
            ids.append(row["Id"])
            try:
                parse_answer(row["Answer"])
            except DataValidationError as exc:
                raise DataValidationError(
                    f"Invalid Answer at submission row {row_number}."
                ) from exc
    if ids != list(template_ids):
        raise DataValidationError(
            "Generated submission IDs do not match the template row order."
        )
    return len(ids)
