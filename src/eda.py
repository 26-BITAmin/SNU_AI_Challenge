"""Train-only exploratory data analysis for the SNU AI Challenge.

This module intentionally never opens ``test.csv`` or ``data/test``.  It
creates descriptive metrics and review-candidate flags; it does not remove or
modify any competition sample.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from PIL import Image, UnidentifiedImageError

from .data import Sample, load_samples


WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
TEMPORAL_CUES = (
    "followed by",
    "transitions to",
    "transitioning to",
    "shifts to",
    "moves from",
    "ends with",
    "ending with",
    "before",
    "after",
    "then",
    "finally",
    "next",
    "first",
    "later",
    "eventually",
    "meanwhile",
    "subsequently",
    "initially",
)

QUALITY_FLAG_COLUMNS = (
    "decode_error",
    "mixed_resolution",
    "low_resolution",
    "non_rgb_mode",
    "dark_frame",
    "bright_frame",
    "exact_duplicate_within_sample",
    "near_duplicate_within_sample",
    "cross_sample_exact_duplicate",
    "review_candidate",
)


@dataclass(frozen=True)
class EDAConfig:
    """Explicit thresholds used to create review-candidate flags."""

    low_resolution_min_side: int = 224
    dark_pixel_max: int = 16
    bright_pixel_min: int = 239
    uniform_fraction_threshold: float = 0.98
    near_duplicate_hamming_threshold: int = 4
    brightness_sample_size: int = 64

    def validate(self) -> None:
        if self.low_resolution_min_side < 1:
            raise ValueError("low_resolution_min_side must be positive.")
        if not 0 <= self.dark_pixel_max < self.bright_pixel_min <= 255:
            raise ValueError(
                "Pixel thresholds must satisfy 0 <= dark < bright <= 255."
            )
        if not 0.0 <= self.uniform_fraction_threshold <= 1.0:
            raise ValueError("uniform_fraction_threshold must be between 0 and 1.")
        if not 0 <= self.near_duplicate_hamming_threshold <= 64:
            raise ValueError(
                "near_duplicate_hamming_threshold must be between 0 and 64."
            )
        if self.brightness_sample_size < 8:
            raise ValueError("brightness_sample_size must be at least 8.")


@dataclass
class EDAResult:
    """All tabular outputs and the aggregate Train-only summary."""

    summary: dict[str, object]
    label_rows: list[dict[str, object]]
    caption_rows: list[dict[str, object]]
    image_rows: list[dict[str, object]]
    sample_quality_rows: list[dict[str, object]]
    resolution_rows: list[dict[str, object]]
    quality_flag_rows: list[dict[str, object]]
    duplicate_rows: list[dict[str, object]]


def analyze_caption(sample: Sample) -> dict[str, object]:
    """Compute format and temporal-cue metrics without copying the caption text."""

    sentence = sample.sentence
    words = WORD_PATTERN.findall(sentence)
    lowercase = sentence.lower()
    cues = [
        cue
        for cue in TEMPORAL_CUES
        if re.search(
            rf"(?<![A-Za-z0-9]){re.escape(cue)}(?![A-Za-z0-9])",
            lowercase,
        )
    ]
    sentence_count = max(1, len(re.findall(r"[.!?]+", sentence)))

    return {
        "Id": sample.sample_id,
        "char_count": len(sentence),
        "word_count": len(words),
        "unique_word_count": len({word.lower() for word in words}),
        "sentence_count": sentence_count,
        "newline_count": sentence.count("\n") + sentence.count("\r"),
        "has_temporal_cue": bool(cues),
        "temporal_cue_type_count": len(cues),
        "temporal_cues": "|".join(cues),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _average_hash(grayscale: Image.Image) -> str:
    resized = grayscale.resize((8, 8), Image.Resampling.BILINEAR)
    pixels = list(resized.get_flattened_data())
    mean_value = sum(pixels) / len(pixels)
    bits = 0
    for pixel in pixels:
        bits = (bits << 1) | int(pixel >= mean_value)
    return f"{bits:016x}"


def hash_hamming_distance(first: str, second: str) -> int:
    """Return the Hamming distance between two 64-bit hexadecimal hashes."""

    return (int(first, 16) ^ int(second, 16)).bit_count()


def analyze_image(
    path: Path,
    *,
    sample_id: str,
    input_index: int,
    relative_path: Path,
    config: EDAConfig,
) -> dict[str, object]:
    """Decode one Train image and compute lightweight technical metrics."""

    row: dict[str, object] = {
        "Id": sample_id,
        "input_index": input_index,
        "filename": path.name,
        "relative_path": relative_path.as_posix(),
        "file_size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "decode_error": False,
        "error_type": "",
        "format": "",
        "mode": "",
        "width": "",
        "height": "",
        "aspect_ratio": "",
        "mean_brightness": "",
        "brightness_std": "",
        "dark_pixel_fraction": "",
        "bright_pixel_fraction": "",
        "average_hash": "",
    }

    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            grayscale = image.convert("L")
            sampled = grayscale.resize(
                (config.brightness_sample_size, config.brightness_sample_size),
                Image.Resampling.BILINEAR,
            )
            pixels = list(sampled.get_flattened_data())
            mean_brightness = statistics.fmean(pixels)
            brightness_std = math.sqrt(
                statistics.fmean(
                    (pixel - mean_brightness) ** 2 for pixel in pixels
                )
            )
            dark_fraction = sum(
                pixel <= config.dark_pixel_max for pixel in pixels
            ) / len(pixels)
            bright_fraction = sum(
                pixel >= config.bright_pixel_min for pixel in pixels
            ) / len(pixels)

            row.update(
                {
                    "format": image.format or "",
                    "mode": image.mode,
                    "width": width,
                    "height": height,
                    "aspect_ratio": round(width / height, 6),
                    "mean_brightness": round(mean_brightness, 4),
                    "brightness_std": round(brightness_std, 4),
                    "dark_pixel_fraction": round(dark_fraction, 6),
                    "bright_pixel_fraction": round(bright_fraction, 6),
                    "average_hash": _average_hash(grayscale),
                }
            )
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        row["decode_error"] = True
        row["error_type"] = type(exc).__name__

    return row


def build_sample_quality_row(
    sample: Sample,
    caption_row: Mapping[str, object],
    image_rows: Sequence[Mapping[str, object]],
    config: EDAConfig,
) -> dict[str, object]:
    """Aggregate four image records into conservative sample-level flags."""

    decoded = [row for row in image_rows if not bool(row["decode_error"])]
    resolutions = {
        (int(row["width"]), int(row["height"])) for row in decoded
    }
    hashes = [str(row["sha256"]) for row in image_rows]
    average_hashes = [
        (str(row["sha256"]), str(row["average_hash"]))
        for row in decoded
        if row["average_hash"]
    ]

    exact_duplicate = len(hashes) != len(set(hashes))
    near_duplicate = any(
        first_sha != second_sha
        and hash_hamming_distance(first_hash, second_hash)
        <= config.near_duplicate_hamming_threshold
        for (first_sha, first_hash), (second_sha, second_hash) in combinations(
            average_hashes, 2
        )
    )
    low_resolution = any(
        min(int(row["width"]), int(row["height"]))
        < config.low_resolution_min_side
        for row in decoded
    )
    dark_frame = any(
        float(row["dark_pixel_fraction"]) >= config.uniform_fraction_threshold
        for row in decoded
    )
    bright_frame = any(
        float(row["bright_pixel_fraction"]) >= config.uniform_fraction_threshold
        for row in decoded
    )
    non_rgb_mode = any(str(row["mode"]) != "RGB" for row in decoded)
    decode_error = len(decoded) != 4
    mixed_resolution = len(resolutions) > 1

    review_candidate = any(
        (
            decode_error,
            mixed_resolution,
            low_resolution,
            non_rgb_mode,
            dark_frame,
            bright_frame,
            exact_duplicate,
            near_duplicate,
        )
    )

    return {
        "Id": sample.sample_id,
        "Answer": json.dumps(sample.answer),
        "No_ordering": sample.no_ordering,
        "caption_has_newline": int(caption_row["newline_count"]) > 0,
        "caption_has_temporal_cue": bool(caption_row["has_temporal_cue"]),
        "decode_error": decode_error,
        "mixed_resolution": mixed_resolution,
        "low_resolution": low_resolution,
        "non_rgb_mode": non_rgb_mode,
        "dark_frame": dark_frame,
        "bright_frame": bright_frame,
        "exact_duplicate_within_sample": exact_duplicate,
        "near_duplicate_within_sample": near_duplicate,
        "cross_sample_exact_duplicate": False,
        "review_candidate": review_candidate,
    }


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute a quantile of an empty sequence.")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
    )


def describe(values: Iterable[float]) -> dict[str, float]:
    """Return a small, deterministic five-number summary plus the mean."""

    sorted_values = sorted(float(value) for value in values)
    if not sorted_values:
        return {}
    return {
        "min": round(sorted_values[0], 4),
        "p25": round(_quantile(sorted_values, 0.25), 4),
        "median": round(_quantile(sorted_values, 0.5), 4),
        "mean": round(statistics.fmean(sorted_values), 4),
        "p75": round(_quantile(sorted_values, 0.75), 4),
        "max": round(sorted_values[-1], 4),
    }


def _build_duplicate_rows(
    image_rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], set[str], int]:
    by_hash: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in image_rows:
        by_hash[str(row["sha256"])].append(row)

    duplicate_rows: list[dict[str, object]] = []
    cross_sample_ids: set[str] = set()
    cross_sample_group_count = 0
    group_number = 0

    for sha256, members in sorted(by_hash.items()):
        if len(members) < 2:
            continue
        group_number += 1
        sample_ids = {str(member["Id"]) for member in members}
        is_cross_sample = len(sample_ids) > 1
        if is_cross_sample:
            cross_sample_group_count += 1
            cross_sample_ids.update(sample_ids)
        for member in members:
            duplicate_rows.append(
                {
                    "duplicate_group": group_number,
                    "sha256": sha256,
                    "cross_sample": is_cross_sample,
                    "Id": member["Id"],
                    "input_index": member["input_index"],
                    "filename": member["filename"],
                }
            )

    return duplicate_rows, cross_sample_ids, cross_sample_group_count


def analyze_train(
    data_dir: Path,
    *,
    config: EDAConfig | None = None,
    progress_every: int = 500,
    progress_callback: Callable[[int, int], None] | None = None,
) -> EDAResult:
    """Analyze only Train and return reproducible EDA tables in memory."""

    config = config or EDAConfig()
    config.validate()
    train_csv = data_dir / "train.csv"
    train_dir = data_dir / "train"
    samples = load_samples(train_csv, labeled=True)

    label_counts: Counter[tuple[int, int, int, int]] = Counter()
    caption_rows: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []

    for sample_number, sample in enumerate(samples, start=1):
        if sample.answer is None:
            raise ValueError(f"Train sample {sample.sample_id!r} has no Answer.")
        label_counts[sample.answer] += 1
        caption_row = analyze_caption(sample)
        caption_rows.append(caption_row)

        sample_image_rows: list[dict[str, object]] = []
        for input_index, path in enumerate(
            sample.image_paths(train_dir), start=1
        ):
            image_row = analyze_image(
                path,
                sample_id=sample.sample_id,
                input_index=input_index,
                relative_path=path.relative_to(data_dir),
                config=config,
            )
            image_rows.append(image_row)
            sample_image_rows.append(image_row)

        quality_rows.append(
            build_sample_quality_row(sample, caption_row, sample_image_rows, config)
        )
        if (
            progress_callback is not None
            and progress_every > 0
            and (sample_number % progress_every == 0 or sample_number == len(samples))
        ):
            progress_callback(sample_number, len(samples))

    duplicate_rows, cross_sample_ids, cross_sample_group_count = (
        _build_duplicate_rows(image_rows)
    )
    for row in quality_rows:
        cross_sample_duplicate = str(row["Id"]) in cross_sample_ids
        row["cross_sample_exact_duplicate"] = cross_sample_duplicate
        row["review_candidate"] = bool(row["review_candidate"]) or cross_sample_duplicate

    label_rows = [
        {
            "Answer": json.dumps(answer),
            "count": count,
            "percentage": round(count / len(samples) * 100, 6),
            "No_ordering": answer == (1, 2, 3, 4),
        }
        for answer, count in sorted(label_counts.items())
    ]

    decoded_images = [row for row in image_rows if not bool(row["decode_error"])]
    resolution_counts = Counter(
        (int(row["width"]), int(row["height"])) for row in decoded_images
    )
    resolution_rows = [
        {
            "width": width,
            "height": height,
            "resolution": f"{width}x{height}",
            "count": count,
            "percentage": round(count / len(image_rows) * 100, 6),
        }
        for (width, height), count in sorted(
            resolution_counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]

    quality_flag_counts = {
        column: sum(bool(row[column]) for row in quality_rows)
        for column in QUALITY_FLAG_COLUMNS
    }
    quality_flag_rows = [
        {
            "flag": column,
            "count": quality_flag_counts[column],
            "percentage": round(
                quality_flag_counts[column] / len(samples) * 100, 6
            ),
        }
        for column in QUALITY_FLAG_COLUMNS
    ]

    label_frequencies = list(label_counts.values())
    mode_counts = Counter(str(row["mode"]) for row in decoded_images)
    format_counts = Counter(str(row["format"]) for row in decoded_images)
    identity_count = label_counts[(1, 2, 3, 4)]
    caption_newline_count = sum(
        int(row["newline_count"]) > 0 for row in caption_rows
    )
    caption_no_cue_count = sum(
        not bool(row["has_temporal_cue"]) for row in caption_rows
    )

    summary: dict[str, object] = {
        "scope": "train_only",
        "sample_count": len(samples),
        "image_count": len(image_rows),
        "config": asdict(config),
        "label": {
            "class_count": len(label_counts),
            "min_class_count": min(label_frequencies),
            "max_class_count": max(label_frequencies),
            "max_to_min_ratio": round(
                max(label_frequencies) / min(label_frequencies), 6
            ),
            "identity_count": identity_count,
            "identity_percentage": round(identity_count / len(samples) * 100, 6),
        },
        "caption": {
            "char_count": describe(int(row["char_count"]) for row in caption_rows),
            "word_count": describe(int(row["word_count"]) for row in caption_rows),
            "newline_sample_count": caption_newline_count,
            "no_temporal_cue_sample_count": caption_no_cue_count,
        },
        "image": {
            "decoded_count": len(decoded_images),
            "decode_error_count": len(image_rows) - len(decoded_images),
            "unique_resolution_count": len(resolution_counts),
            "mode_counts": dict(sorted(mode_counts.items())),
            "format_counts": dict(sorted(format_counts.items())),
            "file_size_bytes": describe(
                int(row["file_size_bytes"]) for row in image_rows
            ),
            "mean_brightness": describe(
                float(row["mean_brightness"]) for row in decoded_images
            ),
            "top_resolutions": resolution_rows[:10],
        },
        "quality_flags": quality_flag_counts,
        "duplicates": {
            "exact_duplicate_group_count": len(
                {row["duplicate_group"] for row in duplicate_rows}
            ),
            "cross_sample_exact_duplicate_group_count": cross_sample_group_count,
            "images_in_exact_duplicate_groups": len(duplicate_rows),
        },
    }

    return EDAResult(
        summary=summary,
        label_rows=label_rows,
        caption_rows=caption_rows,
        image_rows=image_rows,
        sample_quality_rows=quality_rows,
        resolution_rows=resolution_rows,
        quality_flag_rows=quality_flag_rows,
        duplicate_rows=duplicate_rows,
    )


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_eda_outputs(result: EDAResult, output_dir: Path) -> None:
    """Write deterministic Train-only CSV/JSON artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result.summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    _write_csv(
        output_dir / "label_distribution.csv",
        result.label_rows,
        ("Answer", "count", "percentage", "No_ordering"),
    )
    _write_csv(
        output_dir / "caption_metrics.csv",
        result.caption_rows,
        (
            "Id",
            "char_count",
            "word_count",
            "unique_word_count",
            "sentence_count",
            "newline_count",
            "has_temporal_cue",
            "temporal_cue_type_count",
            "temporal_cues",
        ),
    )
    _write_csv(
        output_dir / "image_metrics.csv",
        result.image_rows,
        (
            "Id",
            "input_index",
            "filename",
            "relative_path",
            "file_size_bytes",
            "sha256",
            "decode_error",
            "error_type",
            "format",
            "mode",
            "width",
            "height",
            "aspect_ratio",
            "mean_brightness",
            "brightness_std",
            "dark_pixel_fraction",
            "bright_pixel_fraction",
            "average_hash",
        ),
    )
    _write_csv(
        output_dir / "sample_quality_flags.csv",
        result.sample_quality_rows,
        (
            "Id",
            "Answer",
            "No_ordering",
            "caption_has_newline",
            "caption_has_temporal_cue",
            *QUALITY_FLAG_COLUMNS,
        ),
    )
    _write_csv(
        output_dir / "resolution_distribution.csv",
        result.resolution_rows,
        ("width", "height", "resolution", "count", "percentage"),
    )
    _write_csv(
        output_dir / "quality_flag_counts.csv",
        result.quality_flag_rows,
        ("flag", "count", "percentage"),
    )
    _write_csv(
        output_dir / "exact_duplicate_groups.csv",
        result.duplicate_rows,
        (
            "duplicate_group",
            "sha256",
            "cross_sample",
            "Id",
            "input_index",
            "filename",
        ),
    )
