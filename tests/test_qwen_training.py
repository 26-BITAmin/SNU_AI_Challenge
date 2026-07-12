from __future__ import annotations

import unittest
from itertools import permutations
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.train_qwen_lora import QwenCompletionOnlyCollator
from src.data import Sample
from src.qwen_training import (
    IDENTITY_INPUT_ORDER,
    build_augmented_only_examples,
    build_qwen_sft_messages,
    build_training_examples,
    deterministic_input_order,
    permute_answer,
    permute_sample,
)


def make_sample(answer: tuple[int, int, int, int] = (3, 1, 4, 2)) -> Sample:
    return Sample(
        sample_id="ABC123",
        input_names=(
            "ABC123_aaa.jpg",
            "ABC123_bbb.jpg",
            "ABC123_ccc.jpg",
            "ABC123_ddd.jpg",
        ),
        sentence="First one action happens, then another action follows.",
        answer=answer,
        no_ordering=answer == IDENTITY_INPUT_ORDER,
    )


class QwenPermutationAugmentationTests(unittest.TestCase):
    def test_known_permutation_reorders_images_and_answer_together(self) -> None:
        augmented = permute_sample(make_sample(), (3, 1, 4, 2))
        self.assertEqual(
            augmented.input_names,
            (
                "ABC123_ccc.jpg",
                "ABC123_aaa.jpg",
                "ABC123_ddd.jpg",
                "ABC123_bbb.jpg",
            ),
        )
        self.assertEqual(augmented.answer, (4, 3, 2, 1))
        self.assertFalse(augmented.no_ordering)

    def test_all_answer_and_input_permutations_preserve_frame_positions(self) -> None:
        for answer in permutations((1, 2, 3, 4)):
            for input_order in permutations((1, 2, 3, 4)):
                transformed = permute_answer(answer, input_order)
                expected = tuple(answer[old_index - 1] for old_index in input_order)
                self.assertEqual(transformed, expected)
                self.assertEqual(tuple(sorted(transformed)), (1, 2, 3, 4))

    def test_deterministic_variants_are_reproducible_and_non_identity(self) -> None:
        first = deterministic_input_order("ABC123", seed=42, variant_index=1)
        second = deterministic_input_order("ABC123", seed=42, variant_index=1)
        another_variant = deterministic_input_order(
            "ABC123", seed=42, variant_index=2
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, IDENTITY_INPUT_ORDER)
        self.assertNotEqual(first, another_variant)

    def test_original_is_kept_and_validation_can_remain_unaugmented(self) -> None:
        sample = make_sample()
        train_examples = build_training_examples(
            [sample], augmentation_copies=2, seed=42
        )
        validation_examples = build_training_examples(
            [sample], augmentation_copies=0, seed=42
        )
        self.assertEqual(len(train_examples), 3)
        self.assertEqual([row.variant_index for row in train_examples], [0, 1, 2])
        self.assertEqual(len(validation_examples), 1)
        self.assertIs(validation_examples[0].sample, sample)
        self.assertEqual(sample.input_names[0], "ABC123_aaa.jpg")

    def test_augmented_only_keeps_every_source_once_without_original_view(self) -> None:
        samples = [
            make_sample(),
            Sample(
                sample_id="XYZ789",
                input_names=(
                    "XYZ789_aaa.jpg",
                    "XYZ789_bbb.jpg",
                    "XYZ789_ccc.jpg",
                    "XYZ789_ddd.jpg",
                ),
                sentence="An action changes over time.",
                answer=(1, 2, 3, 4),
                no_ordering=True,
            ),
        ]
        examples = build_augmented_only_examples(samples, seed=42)
        self.assertEqual(len(examples), len(samples))
        self.assertEqual(
            [row.source_sample_id for row in examples],
            [sample.sample_id for sample in samples],
        )
        self.assertTrue(all(row.variant_index == 1 for row in examples))
        self.assertTrue(
            all(row.input_order != IDENTITY_INPUT_ORDER for row in examples)
        )


class QwenSFTMessageTests(unittest.TestCase):
    def test_messages_follow_augmented_order_and_supervise_direct_answer(self) -> None:
        sample = permute_sample(make_sample(), (3, 1, 4, 2))
        example = build_training_examples(
            [sample], augmentation_copies=0, seed=42
        )[0]
        messages = build_qwen_sft_messages(example, Path("data/train"))

        self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
        user_content = messages[0]["content"]
        self.assertIsInstance(user_content, list)
        assert isinstance(user_content, list)
        image_paths = [
            item["image"] for item in user_content if item["type"] == "image"
        ]
        self.assertEqual(
            image_paths,
            [
                "data/train/ABC123/ABC123_ccc.jpg",
                "data/train/ABC123/ABC123_aaa.jpg",
                "data/train/ABC123/ABC123_ddd.jpg",
                "data/train/ABC123/ABC123_bbb.jpg",
            ],
        )
        assistant_content = messages[1]["content"]
        self.assertEqual(assistant_content, [{"type": "text", "text": "[4, 3, 2, 1]"}])


class FakeProcessor:
    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        self.assert_template_arguments(tokenize, add_generation_prompt, messages)
        return "prompt" if len(messages) == 1 else "prompt answer"

    @staticmethod
    def assert_template_arguments(
        tokenize: bool,
        add_generation_prompt: bool,
        messages: list[dict[str, object]],
    ) -> None:
        if tokenize:
            raise AssertionError("The fake expects text rendering before processing.")
        if len(messages) == 1 and not add_generation_prompt:
            raise AssertionError("Prompt-only rendering must add the assistant header.")
        if len(messages) == 2 and add_generation_prompt:
            raise AssertionError("Full SFT rendering already contains the assistant turn.")

    def __call__(
        self,
        *,
        text: list[str],
        images: list[object],
        padding: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        if not padding or return_tensors != "pt" or len(images) != len(text):
            raise AssertionError("Unexpected fake processor arguments.")
        if text == ["prompt answer"]:
            return {
                "input_ids": torch.tensor([[10, 11, 20, 21]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1]]),
            }
        if text == ["prompt"]:
            return {
                "input_ids": torch.tensor([[10, 11]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }
        raise AssertionError(f"Unexpected text: {text!r}")


class QwenCompletionOnlyCollatorTests(unittest.TestCase):
    @patch(
        "scripts.train_qwen_lora.process_vision_info",
        return_value=([object(), object(), object(), object()], None),
    )
    def test_masks_prompt_and_supervises_only_assistant_answer(self, _) -> None:
        example = {
            "messages": [
                {"role": "user", "content": []},
                {"role": "assistant", "content": []},
            ]
        }
        batch = QwenCompletionOnlyCollator(FakeProcessor())([example])
        self.assertEqual(batch["labels"].tolist(), [[-100, -100, 20, 21]])


if __name__ == "__main__":
    unittest.main()
