from __future__ import annotations

import unittest
from itertools import permutations
from pathlib import Path

from src.data import Sample
from src.qwen_baseline import (
    IDENTITY_ANSWER,
    answer_to_chronological_order,
    build_qwen_messages,
    choose_evaluation_samples,
    chronological_order_to_answer,
    parse_qwen_output,
    summarize_qwen_evaluation,
)


class QwenAnswerConversionTests(unittest.TestCase):
    def test_all_24_permutations_round_trip(self) -> None:
        for answer in permutations((1, 2, 3, 4)):
            chronological_order = answer_to_chronological_order(answer)
            self.assertEqual(
                chronological_order_to_answer(chronological_order), answer
            )

    def test_parses_official_baseline_output(self) -> None:
        parsed = parse_qwen_output("The answer is [4, 2, 1, 3]")
        self.assertTrue(parsed.success)
        self.assertEqual(parsed.chronological_order, (4, 2, 1, 3))
        self.assertEqual(parsed.answer, (3, 2, 4, 1))

    def test_parses_direct_answer_without_inversion(self) -> None:
        parsed = parse_qwen_output(
            "[3, 2, 4, 1]", output_mode="direct_answer"
        )
        self.assertTrue(parsed.success)
        self.assertEqual(parsed.answer, (3, 2, 4, 1))
        self.assertEqual(parsed.chronological_order, (4, 2, 1, 3))

    def test_invalid_output_exposes_identity_fallback(self) -> None:
        parsed = parse_qwen_output("I cannot determine the order.")
        self.assertFalse(parsed.success)
        self.assertEqual(parsed.answer, IDENTITY_ANSWER)
        self.assertEqual(parsed.error, "missing_brackets")


class QwenPromptAndSamplingTests(unittest.TestCase):
    def test_prompt_preserves_input_order_and_paths(self) -> None:
        sample = Sample(
            sample_id="ABC123",
            input_names=(
                "ABC123_aaa.jpg",
                "ABC123_bbb.jpg",
                "ABC123_ccc.jpg",
                "ABC123_ddd.jpg",
            ),
            sentence="First this happens, then that happens.",
            answer=(1, 2, 3, 4),
            no_ordering=True,
        )
        messages = build_qwen_messages(sample, Path("data/train"))
        content = messages[0]["content"]
        self.assertIsInstance(content, list)
        assert isinstance(content, list)
        image_entries = [entry for entry in content if entry["type"] == "image"]
        self.assertEqual(
            [entry["image"] for entry in image_entries],
            [
                "data/train/ABC123/ABC123_aaa.jpg",
                "data/train/ABC123/ABC123_bbb.jpg",
                "data/train/ABC123/ABC123_ccc.jpg",
                "data/train/ABC123/ABC123_ddd.jpg",
            ],
        )

    def test_v2_prompt_labels_images_first_and_asks_for_direct_answer(self) -> None:
        sample = Sample(
            sample_id="ABC123",
            input_names=(
                "ABC123_aaa.jpg",
                "ABC123_bbb.jpg",
                "ABC123_ccc.jpg",
                "ABC123_ddd.jpg",
            ),
            sentence="First this happens, then that happens.",
            answer=(3, 2, 4, 1),
            no_ordering=True,
        )
        messages = build_qwen_messages(
            sample,
            Path("data/train"),
            prompt_version="v2_direct_answer",
        )
        content = messages[0]["content"]
        self.assertIsInstance(content, list)
        assert isinstance(content, list)
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[0]["text"], "\nImage 1:\n")
        self.assertEqual(content[1]["type"], "image")
        final_prompt = str(content[-1]["text"])
        self.assertIn("must not be assumed to be chronological", final_prompt)
        self.assertIn("original chronological position of Image 1", final_prompt)
        self.assertNotIn("Example:", final_prompt)

    def test_random_subset_is_reproducible(self) -> None:
        samples = [
            Sample(
                sample_id=f"ID{index:04d}",
                input_names=("a", "b", "c", "d"),
                sentence="Example",
                answer=(1, 2, 3, 4),
                no_ordering=True,
            )
            for index in range(20)
        ]
        first = choose_evaluation_samples(samples, limit=5, seed=42)
        second = choose_evaluation_samples(samples, limit=5, seed=42)
        self.assertEqual(
            [sample.sample_id for sample in first],
            [sample.sample_id for sample in second],
        )


class QwenSummaryTests(unittest.TestCase):
    def test_summary_separates_model_and_identity_baseline(self) -> None:
        targets = [(1, 2, 3, 4), (4, 3, 2, 1)]
        predictions = [(1, 2, 3, 4), (4, 3, 2, 1)]
        summary = summarize_qwen_evaluation(
            targets=targets,
            predictions=predictions,
            parse_successes=[True, False],
            parse_errors=[None, "missing_brackets"],
            latencies_seconds=[1.0, 3.0],
        )
        metrics = summary["metrics"]
        identity_metrics = summary["identity_baseline_metrics"]
        parsing = summary["parsing"]
        latency = summary["latency_seconds"]
        assert isinstance(metrics, dict)
        assert isinstance(identity_metrics, dict)
        assert isinstance(parsing, dict)
        assert isinstance(latency, dict)
        self.assertEqual(metrics["exact_match_accuracy"], 1.0)
        self.assertEqual(identity_metrics["exact_match_accuracy"], 0.5)
        self.assertEqual(summary["exact_match_lift_over_identity"], 0.5)
        self.assertEqual(parsing["failure_count"], 1)
        self.assertEqual(latency["mean"], 2.0)


if __name__ == "__main__":
    unittest.main()
