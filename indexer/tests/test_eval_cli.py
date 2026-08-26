"""Private eval CLI-support tests without a live model or Eagle library."""

from __future__ import annotations

import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evals.cli import (
    EvalCliError,
    _validated_caption_receipts,
    aggregate_report_json,
    create_manifest_json,
    score_stage_b_json,
)
from evals.runner import PrivateReceiptJournal, load_fixture_manifest
from evals.report import FROZEN_EVALUATION_THRESHOLDS
from evals.study import artifact_sha256
from src.captioning.codex_cli import CAPTION_INPUT_PREPARATION_VERSION
from src.captioning.prompt import CAPTION_PROMPT_VERSION
from src.contracts import CaptionReceiptV1, CaptionResultV1


def candidates():
    strata = ("diagram", "screenshot", "text", "art", "edge")
    return [
        {"image_hash": f"sha256:{strata.index(stratum) * 10000 + index:064x}", "image_path": f"/private/{stratum}/{index}.png", "stratum": stratum}
        for stratum in strata
        for index in range(70)
    ]


def candidate_result(model: str):
    return {
        "model": model, "effort": "low", "prompt_version": CAPTION_PROMPT_VERSION,
        "schema_valid_rate": 1.0, "critical_hallucinations": 0,
        "concept_precision": 0.97, "concept_f1": 0.93, "ocr_character_f1": 0.94,
        "ndcg_at_10": 0.90, "recall_at_10": 0.94, "known_query_passed": True,
        "background_throughput_accepted": False,
        "lower_bounds": {"ndcg_at_10": -0.01, "recall_at_10": -0.01, "concept_f1": -0.01, "ocr_character_f1": -0.01},
        "lanes": {"bm25": {"ndcg_at_10": 0.88}, "nomic": {"ndcg_at_10": 0.87}, "rrf": {"ndcg_at_10": 0.90}},
        "latencies_seconds": [1.0, 2.0] * 21,
    }


def sealed_report_evidence(root: Path, *, result_candidates=None):
    snapshot = root / "snapshot"
    gates = {
        "gates_version": 1,
        "guide_hash": "sha256:" + "1" * 64,
        "labels_hash": "sha256:" + "2" * 64,
        "query_artifact_hash": "sha256:" + "3" * 64,
        "embedding_contract_hash": "sha256:" + "4" * 64,
        "candidates": [
            f"gpt-5.6-luna:low:{CAPTION_PROMPT_VERSION}",
            f"gpt-5.6-terra:low:{CAPTION_PROMPT_VERSION}",
        ],
        "comparator": f"gpt-5.6-terra:low:{CAPTION_PROMPT_VERSION}",
        "prompt_version": CAPTION_PROMPT_VERSION,
        "input_preparation_version": CAPTION_INPUT_PREPARATION_VERSION,
        "thresholds": FROZEN_EVALUATION_THRESHOLDS,
    }
    gates_path = root / "gates.json"
    gates_path.write_text(json.dumps(gates), encoding="utf-8")
    candidates_path = root / "candidates.json"
    candidates_path.write_text(json.dumps(candidates()), encoding="utf-8")
    create_manifest_json(
        candidates_path,
        snapshot / "manifest.json",
        snapshot_id="sealed-report",
        seed=17,
        sealing_inputs={
            "labels_hash": gates["labels_hash"],
            "gates_hash": artifact_sha256(gates),
            "amendment_hash": "sha256:" + "5" * 64,
            "instrument_version": "selection-instrument-v1",
            "target_count": 120,
        },
        allowed_root=root,
    )
    manifest = load_fixture_manifest(snapshot, enforce_private_root=False)
    results = {
        "manifest_hash": manifest.manifest_hash,
        "guide_hash": gates["guide_hash"],
        "labels_hash": gates["labels_hash"],
        "query_artifact_hash": gates["query_artifact_hash"],
        "embedding_contract_hash": gates["embedding_contract_hash"],
        "gates_hash": artifact_sha256(gates),
        "comparator": gates["comparator"],
        "input_preparation_version": CAPTION_INPUT_PREPARATION_VERSION,
        "thresholds": FROZEN_EVALUATION_THRESHOLDS,
        "candidates": result_candidates or [
            candidate_result("gpt-5.6-luna"), candidate_result("gpt-5.6-terra")
        ],
    }
    return snapshot, gates_path, results


class EvalCliSupportTests(unittest.TestCase):
    def test_eval_journal_hardens_preexisting_directory_and_file_owner_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "snapshot"
            receipt_directory = snapshot / "caption-receipts"
            receipt_directory.mkdir(parents=True)
            receipt_directory.chmod(0o755)
            journal = PrivateReceiptJournal(snapshot, enforce_private_root=False)
            path = journal.put({
                "fixture_id": "f-1", "model": "gpt-5.6-luna", "effort": "low",
                "prompt_version": CAPTION_PROMPT_VERSION,
                "image_hash": "sha256:" + "a" * 64,
            })

            self.assertEqual(stat.S_IMODE(receipt_directory.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_sealed_score_envelope_flows_into_report_without_a_free_comparator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot, gates_path, _ = sealed_report_evidence(root)
            gates = json.loads(gates_path.read_text(encoding="utf-8"))
            manifest = load_fixture_manifest(snapshot, enforce_private_root=False)
            fixture = manifest.fixtures_for("B")[0]
            caption = CaptionResultV1.from_dict({
                "contract_version": 1, "image_type": "icon", "diagram_types": [],
                "subjects": ["circle"], "visual_style": [], "colours": [],
                "layout": [], "visible_text": [], "search_terms": ["ring"],
                "summary": "A circle icon.", "uncertainties": [],
            })
            receipt = CaptionReceiptV1.create(
                image_hash=fixture.image_hash, caption_result=caption, provider="codex-cli",
                model="gpt-5.6-luna", effort="low", prompt_version=CAPTION_PROMPT_VERSION,
                created_at="2026-08-26T00:00:00Z",
            )
            model_path = snapshot / "model-receipts" / fixture.image_hash / f"{receipt.receipt_id}.json"
            model_path.parent.mkdir(parents=True)
            model_path.write_text(json.dumps(receipt.to_dict()), encoding="utf-8")
            journal_path = snapshot / "caption-receipts" / "entry.json"
            journal_path.parent.mkdir(parents=True)
            journal_path.write_text(json.dumps({
                "fixture_id": fixture.fixture_id, "model": receipt.model,
                "effort": receipt.effort, "prompt_version": receipt.prompt_version,
                "image_hash": receipt.image_hash, "receipt_id": receipt.receipt_id,
                "caption_result": caption.to_dict(), "search_text": caption.search_text(),
                "manifest_hash": manifest.manifest_hash, "latency_seconds": 1.0,
            }), encoding="utf-8")
            for name, payload in {
                "guide.json": {}, "labels.json": {}, "queries.json": {},
                "embedding.json": {"model": "nomic-embed-text:v1.5", "dimensions": 768},
            }.items():
                (root / name).write_text(json.dumps(payload), encoding="utf-8")
            synthetic = {
                "study_version": 1, "stage": "B", "guide_hash": gates["guide_hash"],
                "labels_hash": gates["labels_hash"], "query_artifact_hash": gates["query_artifact_hash"],
                "embedding_contract_hash": gates["embedding_contract_hash"],
                "manifest_hash": manifest.manifest_hash,
                "candidates": [candidate_result("gpt-5.6-luna"), candidate_result("gpt-5.6-terra")],
            }
            results_path = root / "results.json"
            with patch("evals.cli.run_stage_b_study", return_value=synthetic):
                score_stage_b_json(
                    snapshot=snapshot, guide_path=root / "guide.json", labels_path=root / "labels.json",
                    query_path=root / "queries.json", embedding_contract_path=root / "embedding.json",
                    gates_path=gates_path, output_path=results_path, embed=lambda _text: [1.0] * 768,
                    allowed_root=root, embedder_model="nomic-embed-text:v1.5",
                )
            report = aggregate_report_json(
                results_path, root / "report.json", snapshot=snapshot,
                gates_path=gates_path, allowed_root=root,
            )

            self.assertEqual(report["winner"]["model"], "gpt-5.6-luna")
            self.assertNotIn("comparator", json.dumps(report))

    def test_scoring_cross_checks_journal_identity_against_immutable_model_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "snapshot"
            image_hash = "sha256:" + "a" * 64
            caption = CaptionResultV1.from_dict({
                "contract_version": 1, "image_type": "icon", "diagram_types": [],
                "subjects": ["circle"], "visual_style": [], "colours": [],
                "layout": [], "visible_text": [], "search_terms": ["ring"],
                "summary": "A circle icon.", "uncertainties": [],
            })
            receipt = CaptionReceiptV1.create(
                image_hash=image_hash, caption_result=caption, provider="codex-cli",
                model="gpt-5.6-luna", effort="low", prompt_version="caption-v3",
                created_at="2026-08-26T00:00:00Z",
            )
            model_path = snapshot / "model-receipts" / image_hash / f"{receipt.receipt_id}.json"
            model_path.parent.mkdir(parents=True)
            model_path.write_text(json.dumps(receipt.to_dict()), encoding="utf-8")
            journal_path = snapshot / "caption-receipts" / "entry.json"
            journal_path.parent.mkdir(parents=True)
            journal = {
                "fixture_id": "f-0001", "model": receipt.model, "effort": receipt.effort,
                "prompt_version": receipt.prompt_version, "image_hash": image_hash,
                "receipt_id": receipt.receipt_id, "caption_result": caption.to_dict(),
                "search_text": caption.search_text(), "manifest_hash": "sha256:" + "b" * 64,
                "latency_seconds": 1.0,
            }
            fake_hash = "sha256:" + "c" * 64
            fake_receipt_id = "sha256:" + "d" * 64
            for name, changes in {
                "model": {"model": "gpt-5.6-terra"},
                "effort": {"effort": "medium"},
                "prompt": {"prompt_version": "caption-v2"},
                "image": {"image_hash": fake_hash},
                "receipt": {"receipt_id": fake_receipt_id},
            }.items():
                with self.subTest(name=name):
                    candidate = {**journal, **changes}
                    alternate_path = (
                        snapshot / "model-receipts" / candidate["image_hash"]
                        / f'{candidate["receipt_id"]}.json'
                    )
                    alternate_path.parent.mkdir(parents=True, exist_ok=True)
                    alternate_path.write_text(json.dumps(receipt.to_dict()), encoding="utf-8")
                    journal_path.write_text(json.dumps(candidate), encoding="utf-8")
                    with self.assertRaisesRegex(EvalCliError, "identity"):
                        _validated_caption_receipts(snapshot, allowed_root=root)

            journal["caption_result"] = {**caption.to_dict(), "summary": "tampered"}
            journal_path.write_text(json.dumps(journal), encoding="utf-8")
            with self.assertRaisesRegex(EvalCliError, "content"):
                _validated_caption_receipts(snapshot, allowed_root=root)

    def test_stage_b_score_refuses_hand_authored_aggregate_or_missing_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, payload in {
                "manifest.json": {"snapshot_version": 1, "snapshot_id": "s", "fixtures": []},
                "guide.json": {}, "labels.json": {}, "queries.json": {}, "embedding.json": {}, "gates.json": {},
            }.items():
                (root / name).write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(EvalCliError):
                score_stage_b_json(
                    snapshot=root, guide_path=root / "guide.json", labels_path=root / "labels.json",
                    query_path=root / "queries.json", embedding_contract_path=root / "embedding.json",
                    gates_path=root / "gates.json",
                    output_path=root / "result.json", embed=lambda _text: [1.0], allowed_root=root,
                )
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

    def test_manifest_output_directory_and_file_are_owner_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidates_path = root / "candidates.json"
            candidates_path.write_text(json.dumps(candidates()), encoding="utf-8")
            output = root / "snapshot" / "nested" / "manifest.json"
            seal = {
                "labels_hash": "l",
                "gates_hash": "g",
                "amendment_hash": "a",
                "instrument_version": "v",
                "target_count": 60,
            }

            create_manifest_json(
                candidates_path,
                output,
                snapshot_id="private",
                seed=1,
                sealing_inputs=seal,
                allowed_root=root,
            )

            self.assertEqual(stat.S_IMODE(output.parent.parent.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(output.parent.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)

    def test_aggregates_existing_json_without_leaking_private_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot, gates_path, results = sealed_report_evidence(root)
            source = root / "results.json"
            source.write_text(json.dumps({**results, "image_path": "/private/nope", "caption": "secret"}), encoding="utf-8")
            output = root / "report.json"
            report = aggregate_report_json(
                source, output, snapshot=snapshot, gates_path=gates_path, allowed_root=root
            )
            rendered = json.dumps(report)
            self.assertEqual(report["winner"]["model"], "gpt-5.6-luna")
            self.assertNotIn("/private", rendered)
            self.assertNotIn("secret", rendered)
            self.assertNotIn("cost", rendered.casefold())

    def test_rejects_malformed_candidate_instead_of_silently_omitting_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot, gates_path, results = sealed_report_evidence(
                root,
                result_candidates=[candidate_result("gpt-5.6-luna"), "not-an-aggregate"],
            )
            source = root / "results.json"
            source.write_text(json.dumps(results), encoding="utf-8")
            with self.assertRaises(EvalCliError):
                aggregate_report_json(
                    source,
                    root / "report.json",
                    snapshot=snapshot,
                    gates_path=gates_path,
                    allowed_root=root,
                )


if __name__ == "__main__":
    unittest.main()
