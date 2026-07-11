from __future__ import annotations

import csv
import json
import tempfile
import unittest
from itertools import permutations
from pathlib import Path

import torch
from PIL import Image

from src.data import DataValidationError
from src.dataset import (
    ALL_PERMUTATIONS,
    ChallengeDataset,
    ImagePreprocessConfig,
    build_dataloader,
    build_image_transform,
    decode_class,
    encode_answer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SPLIT_DIR = PROJECT_ROOT / "outputs" / "splits"


class PermutationCodecTests(unittest.TestCase):
    def test_all_24_permutations_round_trip(self) -> None:
        expected = tuple(permutations((1, 2, 3, 4)))
        self.assertEqual(ALL_PERMUTATIONS, expected)
        self.assertEqual(len(ALL_PERMUTATIONS), 24)
        for class_index, answer in enumerate(ALL_PERMUTATIONS):
            self.assertEqual(encode_answer(answer), class_index)
            self.assertEqual(decode_class(class_index), answer)

    def test_rejects_invalid_answer_and_class(self) -> None:
        with self.assertRaises(DataValidationError):
            encode_answer((1, 1, 3, 4))
        with self.assertRaises(ValueError):
            decode_class(24)


class DatasetPipelineTests(unittest.TestCase):
    def _write_fixture(self, root: Path) -> tuple[Path, Path]:
        data_dir = root / "data"
        image_dir = data_dir / "train" / "ABC123"
        image_dir.mkdir(parents=True)
        input_names = (
            "ABC123_aaa.jpg",
            "ABC123_bbb.jpg",
            "ABC123_ccc.jpg",
            "ABC123_ddd.jpg",
        )
        colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255))
        for filename, color in zip(input_names, colors, strict=True):
            Image.new("RGB", (12, 10), color).save(image_dir / filename)

        with (data_dir / "train.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "Id",
                    "Input_1",
                    "Input_2",
                    "Input_3",
                    "Input_4",
                    "Sentence",
                    "Answer",
                    "No_ordering",
                ]
            )
            writer.writerow(
                [
                    "ABC123",
                    *input_names,
                    "A synthetic fixture for testing.",
                    json.dumps([3, 1, 4, 2]),
                    "False",
                ]
            )

        split_csv = root / "train_ids.csv"
        with split_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "Id",
                    "Answer",
                    "split",
                    "duplicate_component_id",
                    "duplicate_component_size",
                ]
            )
            writer.writerow(
                ["ABC123", json.dumps([3, 1, 4, 2]), "train", 1, 1]
            )
        return data_dir, split_csv

    def test_preserves_input_order_and_builds_expected_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir, split_csv = self._write_fixture(Path(temporary_directory))
            transform = build_image_transform(
                ImagePreprocessConfig(image_size=8, resize_size=8, normalize=False)
            )
            dataset = ChallengeDataset.from_split_csv(
                data_dir=data_dir,
                split_csv=split_csv,
                expected_split="train",
                transform=transform,
            )
            item = dataset[0]
            self.assertEqual(tuple(item["images"].shape), (4, 3, 8, 8))
            expected_rgb = torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                ]
            )
            torch.testing.assert_close(
                item["images"][:, :, 0, 0],
                expected_rgb,
                atol=0.01,
                rtol=0.0,
            )
            self.assertEqual(item["answer"].tolist(), [3, 1, 4, 2])
            self.assertEqual(decode_class(item["target"]), (3, 1, 4, 2))

            loader = build_dataloader(
                dataset, batch_size=1, shuffle=False, num_workers=0
            )
            batch = next(iter(loader))
            self.assertEqual(tuple(batch["images"].shape), (1, 4, 3, 8, 8))
            self.assertEqual(tuple(batch["answers"].shape), (1, 4))
            self.assertEqual(tuple(batch["targets"].shape), (1,))
            self.assertEqual(batch["sample_ids"], ["ABC123"])

    def test_rejects_split_answer_disagreement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_dir, split_csv = self._write_fixture(root)
            with split_csv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
            rows[1][1] = json.dumps([1, 2, 3, 4])
            with split_csv.open("w", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerows(rows)
            with self.assertRaises(DataValidationError):
                ChallengeDataset.from_split_csv(
                    data_dir=data_dir,
                    split_csv=split_csv,
                    expected_split="train",
                )


@unittest.skipUnless(
    DATA_DIR.is_dir() and SPLIT_DIR.is_dir(), "competition data is not available"
)
class LocalDatasetPipelineIntegrationTests(unittest.TestCase):
    def test_real_train_and_validation_contract(self) -> None:
        transform = build_image_transform(
            ImagePreprocessConfig(image_size=32, resize_size=40)
        )
        train_dataset = ChallengeDataset.from_split_csv(
            data_dir=DATA_DIR,
            split_csv=SPLIT_DIR / "train_ids.csv",
            expected_split="train",
            transform=transform,
        )
        validation_dataset = ChallengeDataset.from_split_csv(
            data_dir=DATA_DIR,
            split_csv=SPLIT_DIR / "validation_ids.csv",
            expected_split="validation",
            transform=transform,
        )
        self.assertEqual(len(train_dataset), 7_628)
        self.assertEqual(len(validation_dataset), 1_907)
        sample = validation_dataset[0]
        self.assertEqual(tuple(sample["images"].shape), (4, 3, 32, 32))
        self.assertEqual(decode_class(sample["target"]), tuple(sample["answer"].tolist()))
        self.assertTrue(torch.isfinite(sample["images"]).all())


if __name__ == "__main__":
    unittest.main()
