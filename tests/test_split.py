from __future__ import annotations

import unittest

from src.split import (
    DuplicateIndex,
    assign_components_to_validation,
    build_duplicate_components,
    count_shared_hashes,
    cross_split_affected_sample_ids,
)


class DuplicateComponentTests(unittest.TestCase):
    def test_groups_only_strong_informative_overlap(self) -> None:
        hash_samples = {
            "h1": {"A", "B"},
            "h2": {"A", "B"},
            "h3": {"A", "B"},
            "h4": {"C", "D"},
            "h5": {"C", "D"},
            "blank": {"A", "B", "C", "D", "E"},
        }
        pair_counts = count_shared_hashes(
            hash_samples, excluded_hashes={"blank"}
        )
        components = build_duplicate_components(
            ["A", "B", "C", "D", "E"],
            pair_counts,
            min_shared_frames=3,
        )

        self.assertIn(("A", "B"), components)
        self.assertIn(("C",), components)
        self.assertIn(("D",), components)
        self.assertIn(("E",), components)

    def test_component_assignment_is_atomic_and_reproducible(self) -> None:
        components = [("A", "B"), ("C",), ("D",), ("E",), ("F",)]
        labels = {
            "A": (1, 2, 3, 4),
            "B": (1, 2, 3, 4),
            "C": (1, 2, 3, 4),
            "D": (4, 3, 2, 1),
            "E": (4, 3, 2, 1),
            "F": (4, 3, 2, 1),
        }
        first = assign_components_to_validation(
            components,
            labels,
            validation_fraction=0.5,
            seed=7,
            search_trials=50,
        )
        second = assign_components_to_validation(
            components,
            labels,
            validation_fraction=0.5,
            seed=7,
            search_trials=50,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual("A" in first, "B" in first)

    def test_clean_validation_excludes_only_informative_cross_split_hashes(self) -> None:
        duplicate_index = DuplicateIndex(
            sample_hashes={
                "T1": {"shared", "blank"},
                "T2": {"train_only"},
                "V1": {"shared"},
                "V2": {"blank"},
                "V3": {"validation_only"},
            },
            hash_samples={
                "shared": {"T1", "V1"},
                "blank": {"T1", "V2"},
                "train_only": {"T2"},
                "validation_only": {"V3"},
            },
            hash_metadata={},
            uniform_hashes={"blank"},
        )

        affected_train, affected_validation = cross_split_affected_sample_ids(
            duplicate_index,
            {"V1", "V2", "V3"},
            excluded_hashes=duplicate_index.uniform_hashes,
        )

        self.assertEqual(affected_train, {"T1"})
        self.assertEqual(affected_validation, {"V1"})


if __name__ == "__main__":
    unittest.main()
