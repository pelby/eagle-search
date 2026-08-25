"""R2/R12 private fixture and aggregate decision tests; no Eagle data is used."""

from __future__ import annotations

import json
import unittest

from evals.fixture_builder import (
    build_private_manifest,
    private_manifest_summary,
    validate_private_manifest,
)
from evals.report import CandidateAggregate, decide_winner, render_anonymised_report


STRATA = ("diagram", "screenshot", "text_dense", "artwork", "edge_case")


def candidates(count_per_stratum: int = 70):
    return [
        {
            "image_hash": f"sha256:{STRATA.index(stratum) * 10_000 + index:064x}",
            "image_path": f"/private/Eagle/secret-{stratum}-{index}.png",
            "stratum": stratum,
        }
        for stratum in STRATA
        for index in range(count_per_stratum)
    ]


def aggregate(model: str, *, critical: int = 0, lower: float = -0.01) -> CandidateAggregate:
    return CandidateAggregate(
        model=model,
        effort="low",
        schema_valid_rate=1.0,
        critical_hallucinations=critical,
        concept_precision=0.97,
        concept_f1=0.93,
        ocr_character_f1=0.94,
        ndcg_at_10=0.90,
        recall_at_10=0.94,
        known_query_passed=True,
        lower_bounds={"ndcg_at_10": lower, "recall_at_10": lower, "concept_f1": lower, "ocr_character_f1": lower},
        lanes={
            "bm25": {"ndcg_at_10": 0.88, "recall_at_10": 0.92, "mrr": 0.8},
            "nomic": {"ndcg_at_10": 0.86, "recall_at_10": 0.91, "mrr": 0.77},
            "rrf": {"ndcg_at_10": 0.90, "recall_at_10": 0.94, "mrr": 0.82},
        },
        latencies_seconds=(1.3, 1.6, 2.2),
    )


class FixtureBuilderTests(unittest.TestCase):
    def test_deterministic_stratified_private_manifest_has_frozen_stages_and_seal(self) -> None:
        sealing = {
            "labels_hash": "sha256:" + "a" * 64,
            "gates_hash": "sha256:" + "b" * 64,
            "amendment_hash": "sha256:" + "c" * 64,
            "instrument_version": "selection-instrument-v1",
            "target_count": 60,
        }
        first = build_private_manifest(candidates(), snapshot_id="run-01", seed=71, sealing_inputs=sealing)
        second = build_private_manifest(candidates(), snapshot_id="run-01", seed=71, sealing_inputs=sealing)

        self.assertEqual(first, second)
        validate_private_manifest(first)
        fixtures = first["fixtures"]
        self.assertEqual(sum(f["stage"] == "A" and f["role"] == "target" for f in fixtures), 6)
        self.assertEqual(sum(f["stage"] == "B" and f["role"] == "target" for f in fixtures), 24)
        self.assertEqual(sum(f["stage"] == "B" and f["role"] == "distractor" for f in fixtures), 72)
        self.assertEqual(sum(f["stage"] == "C" and f["role"] == "target" for f in fixtures), 120)
        self.assertEqual(sum(f["stage"] == "C" and f["role"] == "distractor" for f in fixtures), 120)
        self.assertNotEqual(first["hidden_seal"], build_private_manifest(candidates(), snapshot_id="run-01", seed=71, sealing_inputs={**sealing, "gates_hash": "sha256:" + "d" * 64})["hidden_seal"])

    def test_builder_rejects_filename_or_caption_leakage_and_summary_is_anonymous(self) -> None:
        unsafe = candidates()
        unsafe[0]["filename"] = "client-secret.png"
        with self.assertRaises(ValueError):
            build_private_manifest(unsafe, snapshot_id="run", seed=7, sealing_inputs={"labels_hash": "x", "gates_hash": "y", "amendment_hash": "z", "instrument_version": "v", "target_count": 60})

        manifest = build_private_manifest(candidates(), snapshot_id="run", seed=7, sealing_inputs={"labels_hash": "x", "gates_hash": "y", "amendment_hash": "z", "instrument_version": "v", "target_count": 60})
        anonymous = json.dumps(private_manifest_summary(manifest))
        self.assertNotIn("/private/Eagle", anonymous)
        self.assertNotIn("secret-", anonymous)
        self.assertNotIn("image_path", anonymous)
        self.assertNotIn("image_hash", anonymous)


class AggregateReportTests(unittest.TestCase):
    def test_lowest_passing_tier_wins_and_report_keeps_lane_and_latency_metrics(self) -> None:
        result = decide_winner([aggregate("gpt-5.6-luna"), aggregate("gpt-5.6-terra")], comparator="gpt-5.6-terra")

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.winner.model, "gpt-5.6-luna")
        report = json.dumps(render_anonymised_report(result))
        self.assertIn("bm25", report)
        self.assertIn("p95_latency_seconds", report)
        self.assertNotIn("cost", report.casefold())

    def test_critical_failure_is_never_selected_and_inconclusive_uses_stronger_comparator(self) -> None:
        failed = aggregate("gpt-5.6-luna", critical=1)
        terra = aggregate("gpt-5.6-terra")
        result = decide_winner([failed, terra], comparator="gpt-5.6-terra")
        self.assertEqual(result.status, "passed")
        self.assertEqual(result.winner.model, "gpt-5.6-terra")

        inconclusive = aggregate("gpt-5.6-luna", lower=-0.05)
        fallback = decide_winner([inconclusive, terra], comparator="gpt-5.6-terra")
        self.assertEqual(fallback.status, "inconclusive")
        self.assertEqual(fallback.winner.model, "gpt-5.6-terra")
        self.assertIn("not proven non-inferior", fallback.reason)


if __name__ == "__main__":
    unittest.main()
