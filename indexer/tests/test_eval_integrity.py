"""Adversarial integrity checks for caption-model evaluation evidence."""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

from evals.metrics import cluster_bootstrap_lower_bound
from evals.report import CandidateAggregate, decide_winner


def aggregate(model: str = "gpt-5.6-luna", **overrides: object) -> CandidateAggregate:
    values: dict[str, object] = {
        "model": model,
        "effort": "low",
        "schema_valid_rate": 1.0,
        "critical_hallucinations": 0,
        "concept_precision": 0.96,
        "concept_f1": 0.91,
        "ocr_character_f1": 0.91,
        "ndcg_at_10": 0.86,
        "recall_at_10": 0.91,
        "known_query_passed": True,
        "background_throughput_accepted": False,
        "lower_bounds": {
            "ndcg_at_10": -0.01,
            "recall_at_10": -0.01,
            "concept_f1": -0.01,
            "ocr_character_f1": -0.01,
        },
        "lanes": {},
        "latencies_seconds": (1.0,),
    }
    values.update(overrides)
    return CandidateAggregate(**values)  # type: ignore[arg-type]


class DecisionIntegrityTests(unittest.TestCase):
    def test_decision_requires_exactly_one_comparator(self) -> None:
        luna = aggregate()
        terra = aggregate("gpt-5.6-terra")

        with self.assertRaisesRegex(ValueError, "missing comparator"):
            decide_winner([luna], comparator="gpt-5.6-terra")
        with self.assertRaisesRegex(ValueError, "duplicate comparator"):
            decide_winner([luna, terra, terra], comparator="gpt-5.6-terra")

    def test_candidate_rejects_nonfinite_or_out_of_range_selection_inputs(self) -> None:
        valid = aggregate()
        invalid_fields = {
            "schema_valid_rate": math.nan,
            "concept_precision": math.inf,
            "concept_f1": -0.01,
            "ocr_character_f1": 1.01,
            "ndcg_at_10": math.nan,
            "recall_at_10": 1.01,
        }
        for field, value in invalid_fields.items():
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                replace(valid, **{field: value})

        invalid_bounds = dict(valid.lower_bounds)
        invalid_bounds["ndcg_at_10"] = math.nan
        with self.assertRaisesRegex(ValueError, "lower_bounds"):
            replace(valid, lower_bounds=invalid_bounds)

    def test_candidate_rejects_invalid_counts_and_latency_evidence(self) -> None:
        valid = aggregate()
        with self.assertRaisesRegex(ValueError, "critical_hallucinations"):
            replace(valid, critical_hallucinations=-1)
        with self.assertRaisesRegex(ValueError, "latencies_seconds"):
            replace(valid, latencies_seconds=(1.0, math.inf))


class PairedEvidenceIntegrityTests(unittest.TestCase):
    def test_bootstrap_rejects_asymmetric_cluster_sets(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical cluster sets"):
            cluster_bootstrap_lower_bound(
                {"target-a": [0.9]},
                {"target-a": [0.9], "target-b": [0.8]},
                seed=7,
            )

    def test_bootstrap_rejects_any_empty_paired_cluster(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            cluster_bootstrap_lower_bound(
                {"target-a": [0.9], "target-b": []},
                {"target-a": [0.9], "target-b": [0.8]},
                seed=7,
            )


if __name__ == "__main__":
    unittest.main()
