from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data import Sample
from src.eda import (
    EDAConfig,
    analyze_caption,
    analyze_image,
    analyze_train,
    build_sample_quality_row,
    describe,
    hash_hamming_distance,
)


class CaptionMetricTests(unittest.TestCase):
    def test_detects_temporal_cues_and_newline(self) -> None:
        sample = Sample(
            sample_id="ABC123",
            input_names=("a.jpg", "b.jpg", "c.jpg", "d.jpg"),
            sentence="First, a person enters.\nThen the camera shifts to the door.",
            answer=(1, 2, 3, 4),
            no_ordering=True,
        )
        metrics = analyze_caption(sample)

        self.assertTrue(metrics["has_temporal_cue"])
        self.assertEqual(metrics["newline_count"], 1)
        self.assertIn("then", metrics["temporal_cues"])


class ImageMetricTests(unittest.TestCase):
    def test_dark_image_metrics_and_quality_flag(self) -> None:
        config = EDAConfig()
        sample = Sample(
            sample_id="ABC123",
            input_names=("a.jpg", "b.jpg", "c.jpg", "d.jpg"),
            sentence="First this happens, then that happens.",
            answer=(1, 2, 3, 4),
            no_ordering=True,
        )
        caption = analyze_caption(sample)

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            rows = []
            for index, filename in enumerate(sample.input_names, start=1):
                path = directory / filename
                Image.new("RGB", (320, 240), color=(0, 0, 0)).save(path, "JPEG")
                rows.append(
                    analyze_image(
                        path,
                        sample_id=sample.sample_id,
                        input_index=index,
                        relative_path=Path(filename),
                        config=config,
                    )
                )

        quality = build_sample_quality_row(sample, caption, rows, config)
        self.assertTrue(quality["dark_frame"])
        self.assertFalse(quality["decode_error"])
        self.assertFalse(quality["mixed_resolution"])
        self.assertTrue(quality["review_candidate"])

    def test_hash_distance_and_descriptive_statistics(self) -> None:
        self.assertEqual(
            hash_hamming_distance("0000000000000000", "000000000000000f"), 4
        )
        self.assertEqual(describe([1, 2, 3, 4])["median"], 2.5)

    def test_train_eda_does_not_require_test_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            sample_dir = data_dir / "train" / "ABC123"
            sample_dir.mkdir(parents=True)
            image_names = [
                "ABC123_aaa.jpg",
                "ABC123_bbb.jpg",
                "ABC123_ccc.jpg",
                "ABC123_ddd.jpg",
            ]
            for index, image_name in enumerate(image_names):
                Image.new("RGB", (320, 240), color=(index * 40, 80, 120)).save(
                    sample_dir / image_name, "JPEG"
                )

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
                        *image_names,
                        "First this happens, then that happens.",
                        "[1, 2, 3, 4]",
                        "True",
                    ]
                )

            result = analyze_train(data_dir, progress_every=0)

        self.assertEqual(result.summary["sample_count"], 1)
        self.assertEqual(result.summary["image_count"], 4)


if __name__ == "__main__":
    unittest.main()
