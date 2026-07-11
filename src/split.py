"""Reproducible Train/Validation splitting with duplicate diagnostics."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .data import Sample, load_samples


Permutation = tuple[int, int, int, int]


@dataclass(frozen=True)
class SplitConfig:
    """All choices that define a reproducible validation protocol."""

    validation_fraction: float = 0.20
    seed: int = 42
    min_shared_informative_frames: int = 3
    search_trials: int = 500
    uniform_fraction_threshold: float = 0.98
    uniform_brightness_std_max: float = 1.0

    def validate(self) -> None:
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1.")
        if not 1 <= self.min_shared_informative_frames <= 4:
            raise ValueError("min_shared_informative_frames must be between 1 and 4.")
        if self.search_trials < 1:
            raise ValueError("search_trials must be positive.")
        if not 0.0 <= self.uniform_fraction_threshold <= 1.0:
            raise ValueError("uniform_fraction_threshold must be between 0 and 1.")
        if self.uniform_brightness_std_max < 0:
            raise ValueError("uniform_brightness_std_max cannot be negative.")


@dataclass(frozen=True)
class DuplicateIndex:
    """Exact-image hash membership and hashes excluded as non-informative."""

    sample_hashes: dict[str, set[str]]
    hash_samples: dict[str, set[str]]
    hash_metadata: dict[str, dict[str, str]]
    uniform_hashes: set[str]


@dataclass
class SplitResult:
    """In-memory split rows and auditable diagnostic tables."""

    summary: dict[str, object]
    train_rows: list[dict[str, object]]
    validation_rows: list[dict[str, object]]
    label_rows: list[dict[str, object]]
    cross_split_hash_rows: list[dict[str, object]]


class _DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            if first_root < second_root:
                self.parent[second_root] = first_root
            else:
                self.parent[first_root] = second_root


def load_duplicate_index(image_metrics_csv: Path, config: SplitConfig) -> DuplicateIndex:
    """Load Train image SHA-256 membership and uniform-image flags."""

    sample_hashes: dict[str, set[str]] = defaultdict(set)
    hash_samples: dict[str, set[str]] = defaultdict(set)
    hash_metadata: dict[str, dict[str, str]] = {}
    uniform_hashes: set[str] = set()

    with image_metrics_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "Id",
            "filename",
            "sha256",
            "decode_error",
            "dark_pixel_fraction",
            "bright_pixel_fraction",
            "brightness_std",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{image_metrics_csv} is missing columns: {sorted(missing)!r}"
            )

        for row in reader:
            if row["decode_error"].lower() == "true":
                raise ValueError(
                    "Split creation requires all Train images to decode successfully."
                )
            sample_id = row["Id"]
            sha256 = row["sha256"]
            sample_hashes[sample_id].add(sha256)
            hash_samples[sha256].add(sample_id)
            hash_metadata.setdefault(sha256, row)
            is_uniform = (
                float(row["dark_pixel_fraction"])
                >= config.uniform_fraction_threshold
                or float(row["bright_pixel_fraction"])
                >= config.uniform_fraction_threshold
                or float(row["brightness_std"])
                <= config.uniform_brightness_std_max
            )
            if is_uniform:
                uniform_hashes.add(sha256)

    return DuplicateIndex(
        sample_hashes=dict(sample_hashes),
        hash_samples=dict(hash_samples),
        hash_metadata=hash_metadata,
        uniform_hashes=uniform_hashes,
    )


def count_shared_hashes(
    hash_samples: Mapping[str, set[str]], *, excluded_hashes: set[str] | None = None
) -> Counter[tuple[str, str]]:
    """Count exact hashes shared by each pair of different samples."""

    excluded_hashes = excluded_hashes or set()
    pair_counts: Counter[tuple[str, str]] = Counter()
    for sha256, sample_ids in hash_samples.items():
        if sha256 in excluded_hashes or len(sample_ids) < 2:
            continue
        for first, second in combinations(sorted(sample_ids), 2):
            pair_counts[(first, second)] += 1
    return pair_counts


def build_duplicate_components(
    sample_ids: Iterable[str],
    pair_shared_hash_counts: Mapping[tuple[str, str], int],
    *,
    min_shared_frames: int,
) -> list[tuple[str, ...]]:
    """Keep samples sharing at least N informative frames in one component."""

    sample_ids = sorted(set(sample_ids))
    disjoint_set = _DisjointSet(sample_ids)
    for (first, second), shared_count in pair_shared_hash_counts.items():
        if shared_count >= min_shared_frames:
            disjoint_set.union(first, second)

    components: dict[str, list[str]] = defaultdict(list)
    for sample_id in sample_ids:
        components[disjoint_set.find(sample_id)].append(sample_id)
    return sorted(
        (tuple(sorted(members)) for members in components.values()),
        key=lambda members: (members[0], len(members)),
    )


def _split_score(
    validation_ids: set[str],
    labels: Mapping[str, Permutation],
    target_validation_count: int,
) -> tuple[float, float]:
    global_counts = Counter(labels.values())
    validation_counts = Counter(labels[sample_id] for sample_id in validation_ids)
    total_count = len(labels)
    validation_count = len(validation_ids)
    max_percentage_point_difference = max(
        abs(
            validation_counts[label] / validation_count
            - global_counts[label] / total_count
        )
        for label in global_counts
    )
    normalized_squared_error = sum(
        (
            (validation_counts[label] - global_counts[label] * validation_count / total_count)
            / max(1.0, global_counts[label] * validation_count / total_count)
        )
        ** 2
        for label in global_counts
    )
    size_penalty = abs(validation_count - target_validation_count) / total_count
    return (
        round(max_percentage_point_difference + size_penalty, 12),
        round(normalized_squared_error, 12),
    )


def assign_components_to_validation(
    components: Sequence[tuple[str, ...]],
    labels: Mapping[str, Permutation],
    *,
    validation_fraction: float,
    seed: int,
    search_trials: int,
) -> set[str]:
    """Search deterministic group assignments and keep the best label balance."""

    target_validation_count = round(len(labels) * validation_fraction)
    best_ids: set[str] | None = None
    best_score: tuple[float, float] | None = None

    for trial in range(search_trials):
        random_generator = random.Random(seed + trial)
        candidates = list(components)
        random_generator.shuffle(candidates)
        validation_ids: set[str] = set()
        for component in candidates:
            if len(validation_ids) + len(component) <= target_validation_count:
                validation_ids.update(component)
            if len(validation_ids) == target_validation_count:
                break

        score = _split_score(validation_ids, labels, target_validation_count)
        if best_score is None or score < best_score:
            best_score = score
            best_ids = validation_ids

    if best_ids is None or not best_ids:
        raise RuntimeError("Could not create a non-empty validation split.")
    return best_ids


def _component_index(
    components: Sequence[tuple[str, ...]]
) -> tuple[dict[str, int], dict[int, int]]:
    sample_to_component: dict[str, int] = {}
    component_sizes: dict[int, int] = {}
    for component_id, members in enumerate(components, start=1):
        component_sizes[component_id] = len(members)
        for sample_id in members:
            sample_to_component[sample_id] = component_id
    return sample_to_component, component_sizes


def _build_label_rows(
    samples: Sequence[Sample], validation_ids: set[str]
) -> list[dict[str, object]]:
    total_counts = Counter(sample.answer for sample in samples)
    validation_counts = Counter(
        sample.answer for sample in samples if sample.sample_id in validation_ids
    )
    train_counts = total_counts - validation_counts
    train_count = len(samples) - len(validation_ids)
    validation_count = len(validation_ids)

    rows: list[dict[str, object]] = []
    for answer in sorted(total_counts):
        if answer is None:
            raise ValueError("Train sample without Answer.")
        total_percentage = total_counts[answer] / len(samples) * 100
        train_percentage = train_counts[answer] / train_count * 100
        validation_percentage = validation_counts[answer] / validation_count * 100
        rows.append(
            {
                "Answer": json.dumps(answer),
                "total_count": total_counts[answer],
                "train_count": train_counts[answer],
                "validation_count": validation_counts[answer],
                "total_percentage": round(total_percentage, 6),
                "train_percentage": round(train_percentage, 6),
                "validation_percentage": round(validation_percentage, 6),
                "validation_minus_total_pp": round(
                    validation_percentage - total_percentage, 6
                ),
            }
        )
    return rows


def _cross_split_hash_rows(
    duplicate_index: DuplicateIndex,
    validation_ids: set[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sha256, sample_ids in duplicate_index.hash_samples.items():
        if len(sample_ids) < 2:
            continue
        validation_samples = sorted(sample_ids & validation_ids)
        train_samples = sorted(sample_ids - validation_ids)
        if not validation_samples or not train_samples:
            continue
        metadata = duplicate_index.hash_metadata[sha256]
        rows.append(
            {
                "sha256": sha256,
                "uniform_candidate": sha256 in duplicate_index.uniform_hashes,
                "unique_sample_count": len(sample_ids),
                "train_sample_count": len(train_samples),
                "validation_sample_count": len(validation_samples),
                "representative_Id": metadata["Id"],
                "representative_filename": metadata["filename"],
                "width": metadata.get("width", ""),
                "height": metadata.get("height", ""),
                "mean_brightness": metadata.get("mean_brightness", ""),
            }
        )
    return sorted(
        rows,
        key=lambda row: (-int(row["unique_sample_count"]), str(row["sha256"])),
    )


def _cross_split_pair_distribution(
    pair_counts: Mapping[tuple[str, str], int], validation_ids: set[str]
) -> dict[str, int]:
    distribution: Counter[int] = Counter()
    for (first, second), shared_count in pair_counts.items():
        if (first in validation_ids) != (second in validation_ids):
            distribution[shared_count] += 1
    return {str(count): distribution[count] for count in sorted(distribution)}


def create_train_validation_split(
    *,
    data_dir: Path,
    eda_dir: Path,
    config: SplitConfig | None = None,
) -> SplitResult:
    """Create the fixed Train-only split and all leakage diagnostics."""

    config = config or SplitConfig()
    config.validate()
    samples = load_samples(data_dir / "train.csv", labeled=True)
    labels: dict[str, Permutation] = {}
    for sample in samples:
        if sample.answer is None:
            raise ValueError(f"Train sample {sample.sample_id!r} has no Answer.")
        labels[sample.sample_id] = sample.answer

    duplicate_index = load_duplicate_index(eda_dir / "image_metrics.csv", config)
    if set(labels) != set(duplicate_index.sample_hashes):
        raise ValueError("EDA image metrics IDs do not exactly match train.csv IDs.")

    informative_pair_counts = count_shared_hashes(
        duplicate_index.hash_samples,
        excluded_hashes=duplicate_index.uniform_hashes,
    )
    components = build_duplicate_components(
        labels,
        informative_pair_counts,
        min_shared_frames=config.min_shared_informative_frames,
    )
    validation_ids = assign_components_to_validation(
        components,
        labels,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
        search_trials=config.search_trials,
    )
    train_ids = set(labels) - validation_ids
    sample_to_component, component_sizes = _component_index(components)

    def split_rows(selected_ids: set[str], split_name: str) -> list[dict[str, object]]:
        return [
            {
                "Id": sample.sample_id,
                "Answer": json.dumps(sample.answer),
                "split": split_name,
                "duplicate_component_id": sample_to_component[sample.sample_id],
                "duplicate_component_size": component_sizes[
                    sample_to_component[sample.sample_id]
                ],
            }
            for sample in samples
            if sample.sample_id in selected_ids
        ]

    train_rows = split_rows(train_ids, "train")
    validation_rows = split_rows(validation_ids, "validation")
    label_rows = _build_label_rows(samples, validation_ids)
    cross_split_hash_rows = _cross_split_hash_rows(
        duplicate_index, validation_ids
    )
    informative_cross_rows = [
        row for row in cross_split_hash_rows if not bool(row["uniform_candidate"])
    ]
    uniform_cross_rows = [
        row for row in cross_split_hash_rows if bool(row["uniform_candidate"])
    ]
    affected_train_ids: set[str] = set()
    affected_validation_ids: set[str] = set()
    for sha256, sample_ids in duplicate_index.hash_samples.items():
        if sha256 not in {str(row["sha256"]) for row in cross_split_hash_rows}:
            continue
        affected_train_ids.update(sample_ids & train_ids)
        affected_validation_ids.update(sample_ids & validation_ids)

    label_max_pp_difference = max(
        abs(float(row["validation_minus_total_pp"])) for row in label_rows
    )
    multi_sample_components = [component for component in components if len(component) > 1]
    cross_pair_distribution = _cross_split_pair_distribution(
        informative_pair_counts, validation_ids
    )
    max_cross_shared_frames = max(
        (int(shared_count) for shared_count in cross_pair_distribution), default=0
    )

    summary: dict[str, object] = {
        "scope": "train_only",
        "config": asdict(config),
        "split": {
            "total_count": len(samples),
            "train_count": len(train_rows),
            "validation_count": len(validation_rows),
            "actual_validation_fraction": round(len(validation_rows) / len(samples), 8),
            "label_class_count": len(label_rows),
            "max_validation_vs_total_label_difference_pp": round(
                label_max_pp_difference, 6
            ),
        },
        "duplicate_components": {
            "component_count": len(components),
            "multi_sample_component_count": len(multi_sample_components),
            "samples_in_multi_sample_components": sum(
                len(component) for component in multi_sample_components
            ),
            "largest_component_size": max(len(component) for component in components),
            "grouping_rule": (
                "same component when a sample pair shares at least "
                f"{config.min_shared_informative_frames} non-uniform exact images"
            ),
        },
        "cross_split_exact_overlap": {
            "all_hash_count": len(cross_split_hash_rows),
            "informative_hash_count": len(informative_cross_rows),
            "uniform_hash_count": len(uniform_cross_rows),
            "affected_train_sample_count": len(affected_train_ids),
            "affected_validation_sample_count": len(affected_validation_ids),
            "informative_sample_pair_count_by_shared_frame_count": cross_pair_distribution,
            "max_informative_frames_shared_by_cross_split_pair": max_cross_shared_frames,
        },
    }

    return SplitResult(
        summary=summary,
        train_rows=train_rows,
        validation_rows=validation_rows,
        label_rows=label_rows,
        cross_split_hash_rows=cross_split_hash_rows,
    )


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_split_outputs(result: SplitResult, output_dir: Path) -> None:
    """Write fixed split IDs and diagnostics without modifying Train data."""

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result.summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    split_fields = (
        "Id",
        "Answer",
        "split",
        "duplicate_component_id",
        "duplicate_component_size",
    )
    _write_csv(output_dir / "train_ids.csv", result.train_rows, split_fields)
    _write_csv(
        output_dir / "validation_ids.csv", result.validation_rows, split_fields
    )
    _write_csv(
        output_dir / "label_distribution.csv",
        result.label_rows,
        (
            "Answer",
            "total_count",
            "train_count",
            "validation_count",
            "total_percentage",
            "train_percentage",
            "validation_percentage",
            "validation_minus_total_pp",
        ),
    )
    _write_csv(
        output_dir / "cross_split_exact_hashes.csv",
        result.cross_split_hash_rows,
        (
            "sha256",
            "uniform_candidate",
            "unique_sample_count",
            "train_sample_count",
            "validation_sample_count",
            "representative_Id",
            "representative_filename",
            "width",
            "height",
            "mean_brightness",
        ),
    )
