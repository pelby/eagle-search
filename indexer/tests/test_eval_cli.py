"""Private eval CLI-support tests without a live model or Eagle library."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evals.cli import EvalCliError, aggregate_report_json, create_manifest_json
from evals.runner import load_fixture_manifest


def candidates():
    strata = ("diagram", "screenshot", "text", "art", "edge")
    return [
        {"image_hash": f"sha256:{strata.index(stratum) * 10000 + index:064x}", "image_path": f"/private/{stratum}/{index}.png", "stratum": stratum}
        for stratum in strata
        for index in range(70)
    ]


def candidate_result(model: str):
    return {
        "model": model, "effort": "low", "schema_valid_rate": 1.0, "critical_hallucinations": 0,
        "concept_precision": 0.97, "concept_f1": 0.93, "ocr_character_f1": 0.94,
        "ndcg_at_10": 0.90, "recall_at_10": 0.94, "known_query_passed": True,
        "lower_bounds": {"ndcg_at_10": -0.01, "recall_at_10": -0.01, "concept_f1": -0.01, "ocr_character_f1": -0.01},
        "lanes": {"bm25": {"ndcg_at_10": 0.88}, "nomic": {"ndcg_at_10": 0.87}, "rrf": {"ndcg_at_10": 0.90}},
        "latencies_seconds": [1.0, 2.0],
    }


class EvalCliSupportTests(unittest.TestCase):
    def test_refuses_non_private_root_and_writes_only_when_explicitly_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidates_path = root / "candidates.json"
            candidates_path.write_text(json.dumps(candidates()), encoding="utf-8")
            output = root / "manifest.json"
            seal = {"labels_hash": "l", "gates_hash": "g", "amendment_hash": "a", "instrument_version": "v", "target_count": 60}

            with self.assertRaises(EvalCliError):
                create_manifest_json(candidates_path, output, snapshot_id="s", seed=1, sealing_inputs=seal)
            summary = create_manifest_json(candidates_path, output, snapshot_id="s", seed=1, sealing_inputs=seal, allowed_root=root)
            self.assertTrue(output.exists())
            self.assertNotIn("/private", json.dumps(summary))
            self.assertNotIn("image_hash", json.dumps(summary))
            self.assertNotIn("hidden_seal", summary)
            loaded = load_fixture_manifest(root, enforce_private_root=False)
            self.assertEqual(loaded.snapshot_id, "s")
            self.assertEqual(len(loaded.fixtures_for("A")), 6)

    def test_aggregates_existing_json_without_leaking_private_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "results.json"
            source.write_text(json.dumps({"candidates": [candidate_result("gpt-5.6-luna"), candidate_result("gpt-5.6-terra")], "image_path": "/private/nope", "caption": "secret"}), encoding="utf-8")
            output = root / "report.json"
            report = aggregate_report_json(source, output, comparator="gpt-5.6-terra", allowed_root=root)
            rendered = json.dumps(report)
            self.assertEqual(report["winner"]["model"], "gpt-5.6-luna")
            self.assertNotIn("/private", rendered)
            self.assertNotIn("secret", rendered)
            self.assertNotIn("cost", rendered.casefold())

    def test_rejects_malformed_candidate_instead_of_silently_omitting_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "results.json"
            source.write_text(json.dumps({"candidates": [candidate_result("gpt-5.6-luna"), "not-an-aggregate"]}), encoding="utf-8")
            with self.assertRaises(EvalCliError):
                aggregate_report_json(source, root / "report.json", comparator="gpt-5.6-luna", allowed_root=root)


if __name__ == "__main__":
    unittest.main()
