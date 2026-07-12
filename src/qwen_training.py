"""Train-only data preparation for Qwen vision-language fine-tuning.

The official Answer stores the chronological position of each displayed input.
If the four displayed inputs are permuted for augmentation, the Answer must be
permuted in exactly the same way.  Validation samples are loaded through the
same validated split reader but are never augmented by the training script.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Sequence

from .data import DataValidationError, Sample, load_samples
from .dataset import Permutation, load_split_records
from .qwen_baseline import build_qwen_messages


IDENTITY_INPUT_ORDER: Permutation = (1, 2, 3, 4)
NON_IDENTITY_INPUT_ORDERS: tuple[Permutation, ...] = tuple(
    order
    for order in permutations(IDENTITY_INPUT_ORDER)
    if order != IDENTITY_INPUT_ORDER
)  # type: ignore[assignment]


@dataclass(frozen=True)
class QwenTrainingExample:
    """One original or deterministically augmented supervised example."""

    sample: Sample
    source_sample_id: str
    variant_index: int
    input_order: Permutation

    @property
    def example_id(self) -> str:
        return f"{self.source_sample_id}__v{self.variant_index:02d}"


def validate_input_order(input_order: Sequence[int]) -> Permutation:
    """Validate a one-based mapping from new inputs to original inputs."""

    normalized = tuple(input_order)
    if (
        len(normalized) != 4
        or any(type(value) is not int for value in normalized)
        or tuple(sorted(normalized)) != IDENTITY_INPUT_ORDER
    ):
        raise ValueError(
            "input_order must be a permutation of the original Input numbers 1..4."
        )
    return normalized  # type: ignore[return-value]


def permute_answer(
    answer: Sequence[int], input_order: Sequence[int]
) -> Permutation:
    """Re-express an official Answer after changing the displayed input order.

    ``input_order=(3, 1, 4, 2)`` means that the new Input_1 is the old Input_3,
    the new Input_2 is the old Input_1, and so on.  Since Answer values already
    describe each old input's time position, the new Answer is selected in that
    same order.
    """

    normalized_answer = tuple(answer)
    if (
        len(normalized_answer) != 4
        or any(type(value) is not int for value in normalized_answer)
        or tuple(sorted(normalized_answer)) != IDENTITY_INPUT_ORDER
    ):
        raise ValueError("answer must be a permutation of 1..4.")
    normalized_order = validate_input_order(input_order)
    return tuple(
        normalized_answer[original_input_number - 1]
        for original_input_number in normalized_order
    )  # type: ignore[return-value]


def permute_sample(sample: Sample, input_order: Sequence[int]) -> Sample:
    """Create an augmented Train sample without modifying the source sample."""

    if sample.answer is None or sample.no_ordering is None:
        raise DataValidationError(
            f"Sample {sample.sample_id!r} must have Train labels before augmentation."
        )
    normalized_order = validate_input_order(input_order)
    new_input_names = tuple(
        sample.input_names[original_input_number - 1]
        for original_input_number in normalized_order
    )
    new_answer = permute_answer(sample.answer, normalized_order)
    return Sample(
        sample_id=sample.sample_id,
        input_names=new_input_names,  # type: ignore[arg-type]
        sentence=sample.sentence,
        answer=new_answer,
        no_ordering=new_answer == IDENTITY_INPUT_ORDER,
    )


def deterministic_input_order(
    sample_id: str, *, seed: int, variant_index: int
) -> Permutation:
    """Return a reproducible non-identity order for one augmented variant."""

    if variant_index < 0:
        raise ValueError("variant_index cannot be negative.")
    if variant_index == 0:
        return IDENTITY_INPUT_ORDER
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    sample_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    orders = list(NON_IDENTITY_INPUT_ORDERS)
    random.Random(sample_seed).shuffle(orders)
    return orders[(variant_index - 1) % len(orders)]


def build_training_examples(
    samples: Sequence[Sample], *, augmentation_copies: int, seed: int
) -> list[QwenTrainingExample]:
    """Keep every original sample and append deterministic shuffled copies."""

    if not samples:
        raise ValueError("At least one labeled sample is required.")
    if augmentation_copies < 0:
        raise ValueError("augmentation_copies cannot be negative.")
    examples: list[QwenTrainingExample] = []
    for sample in samples:
        if sample.answer is None or sample.no_ordering is None:
            raise DataValidationError(
                f"Sample {sample.sample_id!r} must have Train labels."
            )
        for variant_index in range(augmentation_copies + 1):
            input_order = deterministic_input_order(
                sample.sample_id,
                seed=seed,
                variant_index=variant_index,
            )
            permuted = (
                sample
                if variant_index == 0
                else permute_sample(sample, input_order)
            )
            examples.append(
                QwenTrainingExample(
                    sample=permuted,
                    source_sample_id=sample.sample_id,
                    variant_index=variant_index,
                    input_order=input_order,
                )
            )
    return examples


def build_augmented_only_examples(
    samples: Sequence[Sample], *, seed: int, variant_index: int = 1
) -> list[QwenTrainingExample]:
    """Build exactly one deterministic shuffled view for every source sample.

    This keeps full source-ID coverage without doubling the number of examples,
    which is useful for a time-bounded one-epoch run.  ``variant_index`` must be
    positive because index zero represents the unmodified original view.
    """

    if not samples:
        raise ValueError("At least one labeled sample is required.")
    if variant_index <= 0:
        raise ValueError("variant_index must be positive for augmented-only data.")
    examples: list[QwenTrainingExample] = []
    for sample in samples:
        input_order = deterministic_input_order(
            sample.sample_id,
            seed=seed,
            variant_index=variant_index,
        )
        augmented = permute_sample(sample, input_order)
        examples.append(
            QwenTrainingExample(
                sample=augmented,
                source_sample_id=sample.sample_id,
                variant_index=variant_index,
                input_order=input_order,
            )
        )
    return examples


def build_qwen_sft_messages(
    example: QwenTrainingExample, image_dir: Path
) -> list[dict[str, object]]:
    """Build the Prompt-v2 user turn plus the supervised direct Answer turn."""

    if example.sample.answer is None:
        raise DataValidationError(
            f"Training example {example.example_id!r} has no Answer."
        )
    messages = build_qwen_messages(
        example.sample,
        image_dir,
        prompt_version="v2_direct_answer",
    )
    messages.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(list(example.sample.answer)),
                }
            ],
        }
    )
    return messages


def load_labeled_split_samples(
    *, data_dir: Path, split_csv: Path, expected_split: str
) -> list[Sample]:
    """Load a fixed labeled split and verify every ID and Answer."""

    all_samples = load_samples(data_dir / "train.csv", labeled=True)
    sample_by_id = {sample.sample_id: sample for sample in all_samples}
    records = load_split_records(split_csv, expected_split=expected_split)
    selected: list[Sample] = []
    for record in records:
        sample = sample_by_id.get(record.sample_id)
        if sample is None:
            raise DataValidationError(
                f"Split Id {record.sample_id!r} is missing from train.csv."
            )
        if sample.answer != record.answer:
            raise DataValidationError(
                f"Answer mismatch for split sample {record.sample_id!r}."
            )
        selected.append(sample)
    return selected


def choose_training_samples(
    samples: Sequence[Sample], *, limit: int | None, seed: int
) -> list[Sample]:
    """Choose a reproducible development subset; ``None`` keeps the full split."""

    if not samples:
        raise ValueError("At least one sample is required.")
    if limit is None:
        return list(samples)
    if limit <= 0:
        raise ValueError("limit must be positive or None.")
    if limit > len(samples):
        raise ValueError(f"limit={limit} exceeds {len(samples)} available samples.")
    indices = random.Random(seed).sample(range(len(samples)), limit)
    return [samples[index] for index in indices]
