"""Run reproducible exploratory analysis on Train only."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.eda import EDAConfig, analyze_train, write_eda_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze SNU AI Challenge Train data only."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eda"))
    parser.add_argument("--low-resolution-min-side", type=int, default=224)
    parser.add_argument("--dark-pixel-max", type=int, default=16)
    parser.add_argument("--bright-pixel-min", type=int, default=239)
    parser.add_argument("--uniform-fraction-threshold", type=float, default=0.98)
    parser.add_argument("--near-duplicate-hamming-threshold", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EDAConfig(
        low_resolution_min_side=args.low_resolution_min_side,
        dark_pixel_max=args.dark_pixel_max,
        bright_pixel_min=args.bright_pixel_min,
        uniform_fraction_threshold=args.uniform_fraction_threshold,
        near_duplicate_hamming_threshold=args.near_duplicate_hamming_threshold,
    )

    print("Scope: TRAIN ONLY (Test content is not opened or analyzed).")
    result = analyze_train(
        args.data_dir.resolve(),
        config=config,
        progress_every=args.progress_every,
        progress_callback=lambda done, total: print(
            f"Processed {done:,}/{total:,} Train samples"
        ),
    )
    output_dir = args.output_dir.resolve()
    write_eda_outputs(result, output_dir)

    label = result.summary["label"]
    caption = result.summary["caption"]
    image = result.summary["image"]
    flags = result.summary["quality_flags"]
    duplicates = result.summary["duplicates"]
    assert isinstance(label, dict)
    assert isinstance(caption, dict)
    assert isinstance(image, dict)
    assert isinstance(flags, dict)
    assert isinstance(duplicates, dict)

    print("Train-only EDA completed.")
    print(f"samples: {result.summary['sample_count']:,}")
    print(f"images: {result.summary['image_count']:,}")
    print(
        "label classes: "
        f"{label['class_count']} (max/min={label['max_to_min_ratio']})"
    )
    print(
        "caption word count median: "
        f"{caption['word_count']['median']}"
    )
    print(
        "image decode errors: "
        f"{image['decode_error_count']}"
    )
    print(
        "review candidates: "
        f"{flags['review_candidate']:,} samples"
    )
    print(
        "cross-sample exact duplicate groups: "
        f"{duplicates['cross_sample_exact_duplicate_group_count']:,}"
    )
    print(f"outputs: {output_dir}")


if __name__ == "__main__":
    main()
