"""Create the fixed Train/Validation split and duplicate diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.split import SplitConfig, create_train_validation_split, write_split_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a reproducible Train-only validation split."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--eda-dir", type=Path, default=Path("outputs/eda"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/splits"))
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-shared-informative-frames", type=int, default=3)
    parser.add_argument("--search-trials", type=int, default=500)
    parser.add_argument("--uniform-fraction-threshold", type=float, default=0.98)
    parser.add_argument("--uniform-brightness-std-max", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SplitConfig(
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        min_shared_informative_frames=args.min_shared_informative_frames,
        search_trials=args.search_trials,
        uniform_fraction_threshold=args.uniform_fraction_threshold,
        uniform_brightness_std_max=args.uniform_brightness_std_max,
    )
    print("Scope: TRAIN ONLY (Test data is not opened or analyzed).")
    result = create_train_validation_split(
        data_dir=args.data_dir.resolve(),
        eda_dir=args.eda_dir.resolve(),
        config=config,
    )
    output_dir = args.output_dir.resolve()
    write_split_outputs(result, output_dir)

    split = result.summary["split"]
    components = result.summary["duplicate_components"]
    overlap = result.summary["cross_split_exact_overlap"]
    clean_validation = result.summary["clean_validation"]
    assert isinstance(split, dict)
    assert isinstance(components, dict)
    assert isinstance(overlap, dict)
    assert isinstance(clean_validation, dict)
    print("Train/Validation split created.")
    print(f"train: {split['train_count']:,}")
    print(f"validation: {split['validation_count']:,}")
    print(
        "max label proportion difference: "
        f"{split['max_validation_vs_total_label_difference_pp']:.6f} pp"
    )
    print(
        "duplicate components: "
        f"{components['multi_sample_component_count']:,} multi-sample, "
        f"largest={components['largest_component_size']}"
    )
    print(
        "max informative frames shared across split: "
        f"{overlap['max_informative_frames_shared_by_cross_split_pair']}"
    )
    print(
        "clean validation samples: "
        f"{clean_validation['sample_count']:,}/{split['validation_count']:,}"
    )
    print(f"outputs: {output_dir}")


if __name__ == "__main__":
    main()
