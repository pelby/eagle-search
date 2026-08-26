"""R2/R12 private fixture and aggregate decision tests; no Eagle data is used."""

from __future__ import annotations

import json
import unittest
from dataclasses import replace

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
        background_throughput_accepted=False,
        lower_bounds={"ndcg_at_10": lower, "recall_at_10": lower, "concept_f1": lower, "ocr_character_f1": lower},
        lanes={
            "bm25": {"ndcg_at_10": 0.88, "recall_at_10": 0.92, "mrr": 0.8},
            "semantic": {"ndcg_at_10": 0.86, "recall_at_10": 0.91, "mrr": 0.77},
            "rrf": {"ndcg_at_10": 0.90, "recall_at_10": 0.94, "mrr": 0.82},
        },
        latencies_seconds=tuple([1.3, 1.6, 2.2] * 14),
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
        for stratum in STRATA:
            self.assertNotIn(stratum, anonymous)

    def test_required_targets_are_included_once_and_count_toward_stratum_quota(self) -> None:
        anchored = candidates()
        b_anchor = anchored[0]
        b_anchor.update({"required_stage": "B", "required_role": "target"})
        c_anchor = anchored[70]
        c_anchor.update({"required_stage": "C", "required_role": "target"})

        manifest = build_private_manifest(
            anchored,
            snapshot_id="anchored",
            seed=71,
            sealing_inputs={
                "labels_hash": "labels",
                "gates_hash": "gates",
                "amendment_hash": "amendment",
                "instrument_version": "selection-instrument-v1",
                "target_count": 60,
            },
        )
        fixtures = manifest["fixtures"]

        for anchor, stage in ((b_anchor, "B"), (c_anchor, "C")):
            matches = [fixture for fixture in fixtures if fixture["image_hash"] == anchor["image_hash"]]
            self.assertEqual(len(matches), 1)
            self.assertEqual((matches[0]["stage"], matches[0]["role"]), (stage, "target"))
            self.assertEqual(
                set(matches[0]),
                {"fixture_id", "image_hash", "image_path", "stratum", "stage", "role"},
            )

        b_diagrams = [
            fixture
            for fixture in fixtures
            if fixture["stage"] == "B" and fixture["role"] == "target" and fixture["stratum"] == "diagram"
        ]
        self.assertEqual(len(b_diagrams), 5)

    def test_required_target_validation_rejects_partial_invalid_or_over_quota_anchors(self) -> None:
        seal = {
            "labels_hash": "labels",
            "gates_hash": "gates",
            "amendment_hash": "amendment",
            "instrument_version": "selection-instrument-v1",
            "target_count": 60,
        }
        malformed = (
            {"required_stage": "B"},
            {"required_stage": "A", "required_role": "target"},
            {"required_stage": "B", "required_role": "distractor"},
            {"required_stage": "B", "required_role": "target", "query_text": "must not enter the manifest"},
        )
        for extra in malformed:
            with self.subTest(extra=extra):
                records = candidates()
                records[0].update(extra)
                with self.assertRaises(ValueError):
                    build_private_manifest(records, snapshot_id="bad-anchor", seed=7, sealing_inputs=seal)

        over_quota = candidates()
        for record in over_quota[:6]:
            record.update({"required_stage": "B", "required_role": "target"})
        with self.assertRaisesRegex(ValueError, "quota"):
            build_private_manifest(over_quota, snapshot_id="too-many-anchors", seed=7, sealing_inputs=seal)


class AggregateReportTests(unittest.TestCase):
    def test_latency_gate_requires_42_caption_window_or_explicit_background_evidence(self) -> None:
        slow = replace(aggregate("gpt-5.6-luna"), latencies_seconds=(30.0,) * 42)
        documented = replace(slow, background_throughput_accepted=True)

        self.assertIn("latency", slow.absolute_failures())
        self.assertNotIn("latency", documented.absolute_failures())
        self.assertEqual(slow.projected_caption_window_seconds(), 1260.0)

    def test_lowest_passing_tier_wins_and_report_keeps_lane_and_latency_metrics(self) -> None:
        result = decide_winner([aggregate("gpt-5.6-luna"), aggregate("gpt-5.6-terra")], comparator="gpt-5.6-terra")

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.winner.model, "gpt-5.6-luna")
        report = json.dumps(render_anonymised_report(result))
        self.assertIn("bm25", report)
        self.assertIn("p95_latency_seconds", report)
        self.assertIn('"concept_precision": 0.97', report)
        self.assertIn('"concept_f1": 0.93', report)
        self.assertIn('"ocr_character_f1": 0.94', report)
        self.assertIn('"schema_valid_rate": 1.0', report)
        self.assertIn('"known_query_passed": true', report)
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

    def test_final_selection_independently_blocks_lane_regression_and_ocr_bearing_collapse(self) -> None:
        final = replace(
            aggregate("gpt-5.6-luna"),
            selection_mode="final",
            pooled_relevance_hash="sha256:" + "a" * 64,
            ocr_bearing_target_count=8,
            ocr_bearing_character_f1=0.94,
            lane_lower_bounds={
                "bm25": {"ndcg_at_10": -0.01, "recall_at_10": -0.01},
                "semantic": {"ndcg_at_10": -0.01, "recall_at_10": -0.01},
            },
            annotation_guide_version=2,
            concept_recall=0.84,
            lower_bounds={
                "ndcg_at_10": -0.01,
                "recall_at_10": -0.01,
                "concept_recall": -0.01,
                "ocr_character_f1": -0.01,
            },
        )
        self.assertEqual(final.absolute_failures(final_selection=True), ())

        # Extra useful caption terms make the legacy precision/F1 diagnostics
        # low, but cannot invalidate v2 atomic-concept coverage.
        diagnostic_only = replace(final, concept_precision=0.20, concept_f1=0.33)
        self.assertEqual(diagnostic_only.absolute_failures(final_selection=True), ())

        broken = replace(
            final,
            ocr_bearing_character_f1=0.80,
            lane_lower_bounds={
                **final.lane_lower_bounds,
                "semantic": {"ndcg_at_10": -0.04, "recall_at_10": -0.01},
            },
        )
        failures = broken.absolute_failures(final_selection=True)
        self.assertIn("ocr_bearing_character_f1", failures)
        self.assertIn("semantic_ndcg_at_10", failures)

        low_coverage = replace(final, concept_recall=0.79)
        self.assertIn("concept_recall", low_coverage.absolute_failures(final_selection=True))

    def test_v2_preliminary_report_uses_coverage_and_ocr_not_legacy_concept_gates(self) -> None:
        preliminary = replace(
            aggregate("gpt-5.6-luna"),
            concept_precision=0.20,
            concept_f1=0.33,
            concept_recall=0.79,
            annotation_guide_version=2,
            ocr_bearing_target_count=8,
            ocr_bearing_character_f1=0.80,
            lower_bounds={
                "ndcg_at_10": -0.01,
                "recall_at_10": -0.01,
                "concept_recall": -0.01,
                "ocr_character_f1": -0.01,
            },
        )

        failures = preliminary.absolute_failures()
        self.assertNotIn("concept_precision", failures)
        self.assertNotIn("concept_f1", failures)
        self.assertIn("concept_recall", failures)
        self.assertIn("ocr_bearing_character_f1", failures)
        self.assertEqual(preliminary.noninferiority_failures(), ())


if __name__ == "__main__":
    unittest.main()
