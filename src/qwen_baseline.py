"""Utilities for evaluating the official zero-shot Qwen baseline on Train-only data."""

from __future__ import annotations

import ast
import csv
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Sequence

from .data import DataValidationError, Sample, load_samples
from .dataset import load_split_records
from .metrics import evaluate_predictions


QWEN_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
IDENTITY_ANSWER = (1, 2, 3, 4)
QWEN_PROMPT_VERSIONS = ("v1_chronological", "v2_direct_answer")


@dataclass(frozen=True)
class ParsedQwenOutput:
    """Parsed chronological image order and official submission answer."""

    answer: tuple[int, int, int, int]
    chronological_order: tuple[int, int, int, int] | None
    success: bool
    error: str | None


def chronological_order_to_answer(
    chronological_order: Sequence[int],
) -> tuple[int, int, int, int]:
    """Convert ordered Input numbers to each Input's original time position."""

    normalized = tuple(chronological_order)
    if len(normalized) != 4 or any(type(value) is not int for value in normalized):
        raise ValueError("Chronological order must contain four integer Input numbers.")
    if tuple(sorted(normalized)) != IDENTITY_ANSWER:
        raise ValueError("Chronological order must be a permutation of 1..4.")
    answer = [0, 0, 0, 0]
    for position, input_number in enumerate(normalized, start=1):
        answer[input_number - 1] = position
    return tuple(answer)  # type: ignore[return-value]


def answer_to_chronological_order(answer: Sequence[int]) -> tuple[int, int, int, int]:
    """Convert the official Answer into ordered Input numbers for prompting."""

    normalized = tuple(answer)
    if len(normalized) != 4 or any(type(value) is not int for value in normalized):
        raise ValueError("Answer must contain four integer positions.")
    if tuple(sorted(normalized)) != IDENTITY_ANSWER:
        raise ValueError("Answer must be a permutation of 1..4.")
    return tuple(sorted(IDENTITY_ANSWER, key=lambda index: normalized[index - 1]))


def parse_qwen_output(
    output_text: str, *, output_mode: str = "chronological"
) -> ParsedQwenOutput:
    """Parse one permutation as chronological order or direct official Answer."""

    if output_mode not in {"chronological", "direct_answer"}:
        raise ValueError("output_mode must be 'chronological' or 'direct_answer'.")

    start_index = output_text.find("[")
    end_index = output_text.rfind("]")
    if start_index == -1 or end_index == -1 or start_index >= end_index:
        return ParsedQwenOutput(
            answer=IDENTITY_ANSWER,
            chronological_order=None,
            success=False,
            error="missing_brackets",
        )
    candidate = output_text[start_index : end_index + 1]
    try:
        parsed = ast.literal_eval(candidate)
    except (SyntaxError, ValueError):
        return ParsedQwenOutput(
            answer=IDENTITY_ANSWER,
            chronological_order=None,
            success=False,
            error="invalid_python_list",
        )
    if (
        not isinstance(parsed, list)
        or len(parsed) != 4
        or any(type(value) is not int for value in parsed)
        or sorted(parsed) != [1, 2, 3, 4]
    ):
        return ParsedQwenOutput(
            answer=IDENTITY_ANSWER,
            chronological_order=None,
            success=False,
            error="not_a_permutation",
        )
    parsed_permutation = tuple(parsed)
    if output_mode == "direct_answer":
        answer = parsed_permutation
        chronological_order = answer_to_chronological_order(answer)
    else:
        chronological_order = parsed_permutation
        answer = chronological_order_to_answer(chronological_order)
    return ParsedQwenOutput(
        answer=answer,  # type: ignore[arg-type]
        chronological_order=chronological_order,  # type: ignore[arg-type]
        success=True,
        error=None,
    )


def build_qwen_messages(
    sample: Sample,
    image_dir: Path,
    *,
    prompt_version: str = "v1_chronological",
) -> list[dict[str, object]]:
    """Build a versioned four-image ordering prompt."""

    if prompt_version not in QWEN_PROMPT_VERSIONS:
        raise ValueError(
            f"prompt_version must be one of {QWEN_PROMPT_VERSIONS!r}."
        )

    content: list[dict[str, object]] = []
    if prompt_version == "v1_chronological":
        for index, image_path in enumerate(sample.image_paths(image_dir), start=1):
            content.append({"type": "image", "image": str(image_path)})
            content.append({"type": "text", "text": f"\nImage {index}\n"})
        user_text = (
            f'Thinking about the sentence: "{sample.sentence}"\n'
            "Look at the 4 images above labeled Image 1 to Image 4. "
            "Determine the correct chronological order of these images to match the sentence. "
            "Provide the answer ONLY as a Python list of integers. "
            "Example: [1, 2, 3, 4]"
        )
    else:
        for index, image_path in enumerate(sample.image_paths(image_dir), start=1):
            content.append({"type": "text", "text": f"\nImage {index}:\n"})
            content.append({"type": "image", "image": str(image_path)})
        user_text = (
            f'The sentence describing the temporal sequence is: "{sample.sentence}"\n'
            "The displayed image order is shuffled and must not be assumed to be chronological. "
            "Compare the action state, body pose, and object position across all four images "
            "with the sequence described by the sentence. "
            "Return the original chronological position of Image 1, Image 2, Image 3, "
            "and Image 4 in that exact input order. Position 1 is earliest and position 4 "
            "is latest. Use each integer from 1 through 4 exactly once. "
            "Output only the four-integer Python list with no explanation."
        )
    content.append({"type": "text", "text": user_text})
    return [{"role": "user", "content": content}]


def choose_evaluation_samples(
    samples: Sequence[Sample], *, limit: int | None, seed: int
) -> list[Sample]:
    """Choose a deterministic random subset without changing the full split."""

    if not samples:
        raise ValueError("At least one Validation sample is required.")
    if limit is None:
        return list(samples)
    if limit <= 0:
        raise ValueError("limit must be positive or None.")
    if limit > len(samples):
        raise ValueError(f"limit={limit} exceeds {len(samples)} available samples.")
    random_generator = random.Random(seed)
    selected_indices = random_generator.sample(range(len(samples)), limit)
    return [samples[index] for index in selected_indices]


def load_validation_samples(
    *, data_dir: Path, validation_split_csv: Path
) -> list[Sample]:
    """Load only the fixed Validation IDs from the labeled Train CSV."""

    all_train_samples = load_samples(data_dir / "train.csv", labeled=True)
    sample_by_id = {sample.sample_id: sample for sample in all_train_samples}
    records = load_split_records(
        validation_split_csv, expected_split="validation"
    )
    validation_samples: list[Sample] = []
    for record in records:
        sample = sample_by_id.get(record.sample_id)
        if sample is None:
            raise DataValidationError(
                f"Validation Id {record.sample_id!r} is missing from train.csv."
            )
        if sample.answer != record.answer:
            raise DataValidationError(
                f"Answer mismatch for Validation sample {record.sample_id!r}."
            )
        validation_samples.append(sample)
    return validation_samples


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence.")
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def summarize_qwen_evaluation(
    *,
    targets: Sequence[Sequence[int]],
    predictions: Sequence[Sequence[int]],
    parse_successes: Sequence[bool],
    parse_errors: Sequence[str | None],
    latencies_seconds: Sequence[float],
) -> dict[str, object]:
    """Summarize accuracy, fallback use, prediction collapse, and latency."""

    sample_count = len(targets)
    if not (
        sample_count
        == len(predictions)
        == len(parse_successes)
        == len(parse_errors)
        == len(latencies_seconds)
    ):
        raise ValueError("All evaluation fields must have the same length.")
    if sample_count == 0:
        raise ValueError("At least one evaluated sample is required.")

    model_metrics = evaluate_predictions(targets, predictions)
    identity_predictions = [IDENTITY_ANSWER] * sample_count
    identity_metrics = evaluate_predictions(targets, identity_predictions)
    prediction_counts = Counter(json.dumps(list(row)) for row in predictions)
    parse_error_counts = Counter(error for error in parse_errors if error is not None)
    most_common_answer, most_common_count = prediction_counts.most_common(1)[0]
    parse_success_count = sum(parse_successes)
    model_exact = float(model_metrics["exact_match_accuracy"])
    identity_exact = float(identity_metrics["exact_match_accuracy"])
    return {
        "sample_count": sample_count,
        "metrics": model_metrics,
        "identity_baseline_metrics": identity_metrics,
        "exact_match_lift_over_identity": round(model_exact - identity_exact, 8),
        "parsing": {
            "success_count": parse_success_count,
            "failure_count": sample_count - parse_success_count,
            "success_rate": round(parse_success_count / sample_count, 8),
            "error_counts": dict(sorted(parse_error_counts.items())),
        },
        "predictions": {
            "unique_answer_count": len(prediction_counts),
            "most_common_answer": most_common_answer,
            "most_common_count": most_common_count,
            "most_common_fraction": round(most_common_count / sample_count, 8),
            "answer_counts": dict(sorted(prediction_counts.items())),
        },
        "latency_seconds": {
            "total": round(sum(latencies_seconds), 6),
            "mean": round(mean(latencies_seconds), 6),
            "median": round(median(latencies_seconds), 6),
            "p95": round(_percentile(latencies_seconds, 0.95), 6),
        },
    }


def write_qwen_predictions(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Write auditable Validation predictions, including raw model text."""

    fieldnames = (
        "Id",
        "target_answer",
        "predicted_answer",
        "chronological_order",
        "parse_success",
        "parse_error",
        "latency_seconds",
        "raw_output",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
