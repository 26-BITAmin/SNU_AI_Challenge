"""Competition-aligned metrics for four-frame temporal ordering."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence


Permutation = tuple[int, int, int, int]
VALID_POSITIONS = (1, 2, 3, 4)


def validate_permutation(values: Sequence[int]) -> Permutation:
    """Validate and normalize one predicted or target permutation."""

    normalized = tuple(values)
    if len(normalized) != 4:
        raise ValueError(f"A permutation must have four positions: {values!r}")
    if any(type(position) is not int for position in normalized):
        raise ValueError(f"Permutation positions must be integers: {values!r}")
    if tuple(sorted(normalized)) != VALID_POSITIONS:
        raise ValueError(f"Expected a permutation of 1..4: {values!r}")
    return normalized  # type: ignore[return-value]


def _validated_pairs(
    targets: Iterable[Sequence[int]], predictions: Iterable[Sequence[int]]
) -> list[tuple[Permutation, Permutation]]:
    target_rows = [validate_permutation(row) for row in targets]
    prediction_rows = [validate_permutation(row) for row in predictions]
    if not target_rows:
        raise ValueError("At least one sample is required for evaluation.")
    if len(target_rows) != len(prediction_rows):
        raise ValueError(
            "targets and predictions must have the same number of samples."
        )
    return list(zip(target_rows, prediction_rows, strict=True))


def exact_match_accuracy(
    targets: Iterable[Sequence[int]], predictions: Iterable[Sequence[int]]
) -> float:
    """Official metric: all four predicted positions must match exactly."""

    pairs = _validated_pairs(targets, predictions)
    return sum(target == prediction for target, prediction in pairs) / len(pairs)


def position_accuracy(
    targets: Iterable[Sequence[int]], predictions: Iterable[Sequence[int]]
) -> float:
    """Diagnostic metric: fraction of individual frame positions that match."""

    pairs = _validated_pairs(targets, predictions)
    correct = sum(
        target_position == predicted_position
        for target, prediction in pairs
        for target_position, predicted_position in zip(
            target, prediction, strict=True
        )
    )
    return correct / (len(pairs) * 4)


def pairwise_ordering_accuracy(
    targets: Iterable[Sequence[int]], predictions: Iterable[Sequence[int]]
) -> float:
    """Diagnostic metric over the six pairwise before/after relationships."""

    pairs = _validated_pairs(targets, predictions)
    correct = 0
    total = 0
    for target, prediction in pairs:
        for first_index, second_index in combinations(range(4), 2):
            target_before = target[first_index] < target[second_index]
            predicted_before = prediction[first_index] < prediction[second_index]
            correct += target_before == predicted_before
            total += 1
    return correct / total


def mean_kendall_tau(
    targets: Iterable[Sequence[int]], predictions: Iterable[Sequence[int]]
) -> float:
    """Mean Kendall tau for error analysis; ranges from -1 to 1."""

    pairs = _validated_pairs(targets, predictions)
    sample_scores: list[float] = []
    for target, prediction in pairs:
        concordant = 0
        discordant = 0
        for first_index, second_index in combinations(range(4), 2):
            target_sign = target[first_index] < target[second_index]
            prediction_sign = prediction[first_index] < prediction[second_index]
            if target_sign == prediction_sign:
                concordant += 1
            else:
                discordant += 1
        sample_scores.append((concordant - discordant) / 6)
    return sum(sample_scores) / len(sample_scores)


def evaluate_predictions(
    targets: Iterable[Sequence[int]], predictions: Iterable[Sequence[int]]
) -> dict[str, float | int]:
    """Return the official score first, followed by diagnostic metrics."""

    pairs = _validated_pairs(targets, predictions)
    normalized_targets = [target for target, _ in pairs]
    normalized_predictions = [prediction for _, prediction in pairs]
    return {
        "sample_count": len(pairs),
        "exact_match_accuracy": exact_match_accuracy(
            normalized_targets, normalized_predictions
        ),
        "position_accuracy": position_accuracy(
            normalized_targets, normalized_predictions
        ),
        "pairwise_ordering_accuracy": pairwise_ordering_accuracy(
            normalized_targets, normalized_predictions
        ),
        "mean_kendall_tau": mean_kendall_tau(
            normalized_targets, normalized_predictions
        ),
    }
