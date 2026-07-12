from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.data import DataValidationError
from src.submission import (
    build_submission_rows,
    validate_submission_file,
    write_submission,
)


class SubmissionTests(unittest.TestCase):
    def test_writes_template_order_and_valid_permutations(self) -> None:
        template_ids = ["ABC123", "XYZ789"]
        predictions = {
            "XYZ789": (4, 3, 2, 1),
            "ABC123": (1, 3, 2, 4),
        }
        rows = build_submission_rows(template_ids, predictions)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            write_submission(path, rows)
            self.assertEqual(validate_submission_file(path, template_ids), 2)
            with path.open(newline="", encoding="utf-8") as handle:
                saved = list(csv.DictReader(handle))
        self.assertEqual(
            saved,
            [
                {"Id": "ABC123", "Answer": "[1, 3, 2, 4]"},
                {"Id": "XYZ789", "Answer": "[4, 3, 2, 1]"},
            ],
        )

    def test_rejects_missing_prediction_id(self) -> None:
        with self.assertRaises(DataValidationError):
            build_submission_rows(
                ["ABC123", "XYZ789"],
                {"ABC123": (1, 2, 3, 4)},
            )

    def test_rejects_invalid_answer(self) -> None:
        with self.assertRaises(DataValidationError):
            build_submission_rows(
                ["ABC123"],
                {"ABC123": (1, 1, 3, 4)},
            )


if __name__ == "__main__":
    unittest.main()
