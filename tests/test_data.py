from __future__ import annotations

import unittest
from pathlib import Path

from src.data import (
    DataValidationError,
    Sample,
    audit_split,
    load_samples,
    parse_answer,
    validate_submission_template,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


class AnswerParsingTests(unittest.TestCase):
    def test_valid_answer(self) -> None:
        self.assertEqual(parse_answer("[3, 1, 2, 4]"), (3, 1, 2, 4))

    def test_rejects_duplicate_position(self) -> None:
        with self.assertRaises(DataValidationError):
            parse_answer("[1, 1, 3, 4]")

    def test_rejects_non_integer_position(self) -> None:
        with self.assertRaises(DataValidationError):
            parse_answer('[1, 2, 3, "4"]')

    def test_reconstructs_chronological_order(self) -> None:
        sample = Sample(
            sample_id="ABC123",
            input_names=("a.jpg", "d.jpg", "b.jpg", "c.jpg"),
            sentence="Example",
            answer=(1, 4, 2, 3),
            no_ordering=False,
        )
        self.assertEqual(
            sample.ordered_input_names(),
            ("a.jpg", "b.jpg", "c.jpg", "d.jpg"),
        )


@unittest.skipUnless(DATA_DIR.is_dir(), "competition data is not available")
class LocalDatasetIntegrationTests(unittest.TestCase):
    def test_known_train_sample_label_direction(self) -> None:
        samples = load_samples(DATA_DIR / "train.csv", labeled=True)
        sample = next(item for item in samples if item.sample_id == "u7w0lr")
        self.assertEqual(sample.answer, (3, 1, 2, 4))
        self.assertEqual(
            sample.ordered_input_names(),
            (
                "u7w0lr_shc.jpg",
                "u7w0lr_uyv.jpg",
                "u7w0lr_aot.jpg",
                "u7w0lr_vuj.jpg",
            ),
        )

    def test_full_dataset_structure(self) -> None:
        train = audit_split(DATA_DIR, "train")
        test = audit_split(DATA_DIR, "test")

        self.assertEqual(train.sample_count, 9_535)
        self.assertEqual(train.image_count, 38_140)
        self.assertEqual(train.label_permutation_count, 24)
        self.assertEqual(test.sample_count, 819)
        self.assertEqual(test.image_count, 3_276)
        self.assertEqual(validate_submission_template(DATA_DIR), 819)


if __name__ == "__main__":
    unittest.main()
