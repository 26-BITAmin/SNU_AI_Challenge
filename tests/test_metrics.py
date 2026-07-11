from __future__ import annotations

import unittest

from src.metrics import evaluate_predictions, exact_match_accuracy


class CompetitionMetricTests(unittest.TestCase):
    def test_exact_match_requires_all_four_positions(self) -> None:
        targets = [(1, 2, 3, 4), (3, 1, 2, 4)]
        predictions = [(1, 2, 3, 4), (3, 1, 4, 2)]
        self.assertEqual(exact_match_accuracy(targets, predictions), 0.5)

    def test_reverse_order_diagnostics(self) -> None:
        metrics = evaluate_predictions([(1, 2, 3, 4)], [(4, 3, 2, 1)])
        self.assertEqual(metrics["exact_match_accuracy"], 0.0)
        self.assertEqual(metrics["position_accuracy"], 0.0)
        self.assertEqual(metrics["pairwise_ordering_accuracy"], 0.0)
        self.assertEqual(metrics["mean_kendall_tau"], -1.0)

    def test_invalid_prediction_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_predictions([(1, 2, 3, 4)], [(1, 1, 3, 4)])


if __name__ == "__main__":
    unittest.main()
