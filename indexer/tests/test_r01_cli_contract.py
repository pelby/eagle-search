"""R1/R7/R9 contract tests for the consumer-neutral command surface."""

from __future__ import annotations

import io
import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from src import db
from src.captioning.codex_cli import CAPTION_INPUT_PREPARATION_VERSION, CAPTION_PROVIDER_NAME, CaptionItemFailure
from src.captioning.prompt import CAPTION_PROMPT_VERSION
from src.cli import CliRuntime, build_parser, main
from src.contracts import CaptionReceiptV1, CaptionResultV1
from src.persistence.receipts import FileReceiptStore
from evals.fixture_builder import build_private_manifest
from evals.report import FROZEN_EVALUATION_THRESHOLDS, FROZEN_FINAL_SELECTION_THRESHOLDS
from evals.study import artifact_sha256


class FakeEmbedder:
    model = "fake"

    def embed(self, texts):
        return [[1.0] + [0.0] * 767 for _ in texts]


class DownEmbedder(FakeEmbedder):
    def embed(self, texts):
        raise RuntimeError("offline")


class FakeCaptionProvider:
    provider_name = CAPTION_PROVIDER_NAME

    def caption(self, image_path, *, model, effort, receipt_store):
        digest = "sha256:" + hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
        result = CaptionResultV1.from_dict(
            {
                "contract_version": 1,
                "image_type": "illustration",
                "diagram_types": [],
                "subjects": ["robot"],
                "visual_style": [],
                "colours": [],
                "layout": [],
                "visible_text": [],
                "search_terms": ["classroom"],
                "summary": "A robot in a classroom.",
                "uncertainties": [],
            }
        )
        receipt = CaptionReceiptV1.create(
            image_hash=digest,
            caption_result=result,
            provider=self.provider_name,
            model=model,
            effort=effort,
            prompt_version=CAPTION_PROMPT_VERSION,
            created_at="2026-08-25T22:00:00Z",
        )
        receipt_store.put_immutable(receipt)
        return receipt


class CountingCaptionProvider(FakeCaptionProvider):
    def __init__(self) -> None:
        self.calls = 0

    def caption(self, image_path, *, model, effort, receipt_store):
        self.calls += 1
        return super().caption(image_path, model=model, effort=effort, receipt_store=receipt_store)


class OneItemFailureCaptionProvider(FakeCaptionProvider):
    def __init__(self) -> None:
        self.failed_once = False

    def caption(self, image_path, *, model, effort, receipt_store):
        if Path(image_path).name == "first.png" and not self.failed_once:
            self.failed_once = True
            raise CaptionItemFailure("transient fixture failure")
        return super().caption(image_path, model=model, effort=effort, receipt_store=receipt_store)


class WrongIdentityCaptionProvider(FakeCaptionProvider):
    def caption(self, image_path, *, model, effort, receipt_store):
        return super().caption(
            image_path,
            model="gpt-5.6-terra",
            effort=effort,
            receipt_store=receipt_store,
        )


class WrongProviderCaptionProvider(FakeCaptionProvider):
    provider_name = "not-codex"


class AmbiguousFirstNotesApi:
    def __init__(self) -> None:
        self.read_ids = []
        self.update_ids = []

    async def get_item(self, item_id):
        self.read_ids.append(item_id)
        return {"id": item_id, "annotation": "Human note", "lastModified": 1}

    async def update_item(self, item_id, *, annotation=None, tags=None, source=None):
        self.update_ids.append(item_id)
        raise RuntimeError("uncertain transport outcome")


class CliContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.home = Path(self.temporary.name)
        connection = db.init_db(self.home / "db.sqlite")
        db.upsert_image(
            connection,
            {
                "eagle_id": "fixture",
                "name": "Classroom scene",
                "ai_description": "A teacher presents beside an easel.",
                "thumbnail_path": "/tmp/fixture-thumb.png",
                "image_path": "/tmp/fixture.png",
            },
        )
        connection.close()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def invoke(self, arguments, *, embedder=None, caption_provider=None, eagle=None):
        stdout = io.StringIO()
        stderr = io.StringIO()
        runtime = CliRuntime(
            home=self.home,
            embedder=embedder or FakeEmbedder(),
            eagle=eagle,
            caption_provider=caption_provider,
        )
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(arguments, runtime=runtime)
        lines = stdout.getvalue().splitlines()
        payload = json.loads(lines[0]) if lines else None
        return exit_code, payload, lines, stderr.getvalue()

    def test_exact_search_emits_one_versioned_json_envelope(self) -> None:
        exit_code, payload, lines, stderr = self.invoke(
            ["search", "classroom", "--mode", "exact", "--limit", "5", "--no-log", "--json"]
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(lines), 1)
        self.assertEqual(stderr, "")
        self.assertEqual(payload["contract_version"], 1)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["results"][0]["eagle_id"], "fixture")

    def test_semantic_outage_is_a_successful_lexical_response_with_warning(self) -> None:
        exit_code, payload, _, _ = self.invoke(
            ["search", "classroom", "--mode", "automatic", "--limit", "5", "--no-log", "--json"],
            embedder=DownEmbedder(),
        )

        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["ok"])
        self.assertFalse(payload["retrieval"]["semantic_available"])
        self.assertEqual(payload["retrieval"]["mode"], "lexical")
        self.assertTrue(payload["retrieval"]["warnings"])

    def test_invalid_limit_is_stable_json_and_nonzero(self) -> None:
        exit_code, payload, lines, _ = self.invoke(
            ["search", "classroom", "--mode", "exact", "--limit", "0", "--json"]
        )

        self.assertEqual(exit_code, 2)
        self.assertEqual(len(lines), 1)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "invalid_request")

    def test_missing_status_file_reports_idle(self) -> None:
        exit_code, payload, lines, _ = self.invoke(["status", "--json"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(lines), 1)
        self.assertEqual(payload["state"], "idle")
        self.assertEqual(payload["contract_version"], 1)

    def test_operational_subcommands_are_present_on_the_canonical_cli(self) -> None:
        parser = build_parser()
        subparsers = next(
            action for action in parser._actions if action.__class__.__name__ == "_SubParsersAction"
        )
        self.assertTrue(
            {
                "index",
                "search",
                "watch-generated",
                "rebuild",
                "eval-manifest",
                "eval-captions",
                "eval-captions-c",
                "eval-report",
                "eval-score",
                "eval-score-c",
                "eval-report-c",
            }.issubset(subparsers.choices)
        )
        self.assertEqual(parser.parse_args(["index"]).model, "gpt-5.6-sol")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["eval-captions", "--snapshot", "fixture", "--stage", "C", "--model", "gpt-5.6-sol"])

    def test_stage_c_caption_refuses_tampered_approval_before_provider_call(self) -> None:
        snapshot = self.home / "evals" / "stage-c"
        snapshot.mkdir(parents=True)
        records = [
            {
                "image_hash": f"sha256:{index:064x}",
                "image_path": str(snapshot / f"private-{index}.png"),
                "stratum": f"stratum-{index % 5}",
            }
            for index in range(400)
        ]
        manifest = build_private_manifest(
            records,
            snapshot_id="stage-c-cli",
            seed=71,
            sealing_inputs={
                "labels_hash": "sha256:" + "1" * 64,
                "gates_hash": "sha256:" + "2" * 64,
                "amendment_hash": "sha256:" + "3" * 64,
                "instrument_version": "selection-instrument-v1",
                "target_count": 60,
            },
        )
        (snapshot / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        identities = [f"gpt-5.6-sol:low:{CAPTION_PROMPT_VERSION}"]
        effects = {
            "stage_b_effects_version": 2,
            "stage_b_results_hash": "sha256:" + "4" * 64,
            "stage_c_candidates": identities,
            "candidate_powered_target_counts": {identities[0]: 60},
            "powered_hidden_target_count": 60,
        }
        gates = {
            "stage_c_gates_version": 2,
            "snapshot_hash": artifact_sha256(manifest),
            "stage_b_effects_hash": artifact_sha256(effects),
            "guide_hash": "sha256:" + "5" * 64,
            "labels_hash": "sha256:" + "6" * 64,
            "query_artifact_hash": "sha256:" + "7" * 64,
            "embedding_contract_hash": "sha256:" + "8" * 64,
            "candidates": identities,
            "comparator": identities[0],
            "selection_algorithm": "stratified-shuffle-v1",
            "selection_seed": 42,
            "seed": 43,
            "prompt_version": CAPTION_PROMPT_VERSION,
            "input_preparation_version": CAPTION_INPUT_PREPARATION_VERSION,
            "thresholds": FROZEN_EVALUATION_THRESHOLDS,
            "final_selection_thresholds": FROZEN_FINAL_SELECTION_THRESHOLDS,
        }
        approval = {"approval_version": 1, "approved": False, "gates_hash": artifact_sha256(gates)}
        for name, payload in (("gates.json", gates), ("effects.json", effects), ("approval.json", approval)):
            (snapshot / name).write_text(json.dumps(payload), encoding="utf-8")
        provider = CountingCaptionProvider()

        exit_code, payload, _lines, _stderr = self.invoke(
            [
                "eval-captions-c", "--snapshot", str(snapshot), "--gates", str(snapshot / "gates.json"),
                "--approval", str(snapshot / "approval.json"), "--stage-b-effects", str(snapshot / "effects.json"),
                "--model", "gpt-5.6-sol", "--json",
            ],
            caption_provider=provider,
        )

        self.assertNotEqual(exit_code, 0)
        self.assertFalse(payload["ok"])
        self.assertEqual(provider.calls, 0)

    def test_eval_caption_stage_uses_private_resumable_journal(self) -> None:
        snapshot = self.home / "evals" / "fixture"
        snapshot.mkdir(parents=True)
        image = snapshot / "image.png"
        image.write_bytes(b"private image")
        image_hash = "sha256:" + hashlib.sha256(image.read_bytes()).hexdigest()
        (snapshot / "manifest.json").write_text(
            json.dumps(
                {
                    "snapshot_version": 1,
                    "snapshot_id": "fixture",
                    "fixtures": [
                        {
                            "fixture_id": "f-0001",
                            "image_hash": image_hash,
                            "image_path": str(image),
                            "stage": "A",
                            "stratum": "art",
                            "role": "target",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        first = self.invoke(
            [
                "eval-captions",
                "--snapshot",
                str(snapshot),
                "--stage",
                "A",
                "--model",
                "gpt-5.6-luna",
                "--json",
            ],
            caption_provider=FakeCaptionProvider(),
        )
        second = self.invoke(
            [
                "eval-captions",
                "--snapshot",
                str(snapshot),
                "--stage",
                "A",
                "--model",
                "gpt-5.6-luna",
                "--json",
            ],
            caption_provider=FakeCaptionProvider(),
        )

        self.assertEqual(first[0], 0)
        self.assertEqual(first[1]["created"], 1)
        self.assertEqual(second[1]["created"], 0)
        self.assertEqual(second[1]["completed_receipts"], 1)
        self.assertEqual(second[1]["completed"], 1)

    def test_eval_caption_stage_reports_and_retries_bounded_item_failure(self) -> None:
        snapshot = self.home / "evals" / "flaky-fixture"
        snapshot.mkdir(parents=True)
        fixtures = []
        for fixture_id in ("first", "later"):
            image = snapshot / f"{fixture_id}.png"
            image.write_bytes(f"private {fixture_id}".encode())
            fixtures.append(
                {
                    "fixture_id": fixture_id,
                    "image_hash": "sha256:" + hashlib.sha256(image.read_bytes()).hexdigest(),
                    "image_path": str(image),
                    "stage": "A",
                    "stratum": "art",
                    "role": "target",
                }
            )
        (snapshot / "manifest.json").write_text(
            json.dumps({"snapshot_version": 1, "snapshot_id": "flaky-fixture", "fixtures": fixtures}),
            encoding="utf-8",
        )
        provider = OneItemFailureCaptionProvider()
        arguments = [
            "eval-captions",
            "--snapshot",
            str(snapshot),
            "--stage",
            "A",
            "--model",
            "gpt-5.6-luna",
            "--effort",
            "low",
            "--json",
        ]

        first = self.invoke(arguments, caption_provider=provider)
        self.assertEqual(first[0], 1)
        self.assertFalse(first[1]["ok"])
        self.assertEqual({key: first[1][key] for key in ("created", "completed", "failed")}, {"created": 1, "completed": 1, "failed": 1})
        self.assertNotIn("private", json.dumps(first[1]))

        retried = self.invoke(arguments, caption_provider=provider)
        self.assertEqual(retried[0], 0)
        self.assertTrue(retried[1]["ok"])
        self.assertEqual({key: retried[1][key] for key in ("created", "completed", "failed")}, {"created": 1, "completed": 2, "failed": 0})

    def test_eval_caption_refuses_provider_receipt_for_a_different_candidate(self) -> None:
        snapshot = self.home / "evals" / "wrong-identity"
        snapshot.mkdir(parents=True)
        image = snapshot / "image.png"
        image.write_bytes(b"private image")
        image_hash = "sha256:" + hashlib.sha256(image.read_bytes()).hexdigest()
        (snapshot / "manifest.json").write_text(json.dumps({
            "snapshot_version": 1,
            "snapshot_id": "wrong-identity",
            "fixtures": [{
                "fixture_id": "f-0001", "image_hash": image_hash,
                "image_path": str(image), "stage": "A", "stratum": "art",
                "role": "target",
            }],
        }), encoding="utf-8")

        exit_code, payload, _, _ = self.invoke([
            "eval-captions", "--snapshot", str(snapshot), "--stage", "A",
            "--model", "gpt-5.6-luna", "--effort", "low", "--json",
        ], caption_provider=WrongIdentityCaptionProvider())

        self.assertEqual(exit_code, 1)
        self.assertEqual(payload["error"]["code"], "operation_failed")
        self.assertFalse((snapshot / "caption-receipts").exists())

        exit_code, payload, _, _ = self.invoke([
            "eval-captions", "--snapshot", str(snapshot), "--stage", "A",
            "--model", "gpt-5.6-luna", "--effort", "low", "--json",
        ], caption_provider=WrongProviderCaptionProvider())
        self.assertEqual(exit_code, 1)
        self.assertEqual(payload["error"]["code"], "operation_failed")

    def test_eval_paths_outside_private_root_are_refused(self) -> None:
        exit_code, payload, _, _ = self.invoke(
            [
                "eval-report",
                "--results",
                "/tmp/not-private.json",
                "--output",
                "/tmp/report.json",
                "--snapshot",
                "/tmp/snapshot",
                "--gates",
                "/tmp/gates.json",
                "--json",
            ]
        )
        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["error"]["code"], "invalid_request")

    def test_notes_apply_stops_after_ambiguous_item_and_persists_private_journal(self) -> None:
        store = FileReceiptStore(self.home / "captions")
        connection = db.init_db(self.home / "db.sqlite")
        for index, eagle_id in enumerate(("a-first", "b-second"), start=1):
            image_hash = f"sha256:{index:064x}"
            result = CaptionResultV1.from_dict(
                {
                    "contract_version": 1,
                    "image_type": "illustration",
                    "diagram_types": [],
                    "subjects": ["robot"],
                    "visual_style": [],
                    "colours": [],
                    "layout": [],
                    "visible_text": [],
                    "search_terms": ["teaching"],
                    "summary": "A robot presents beside an easel.",
                    "uncertainties": [],
                }
            )
            receipt = CaptionReceiptV1.create(
                image_hash=image_hash,
                caption_result=result,
                provider="fake",
                model="fake",
                effort="low",
                prompt_version="caption-v2",
                created_at="2026-08-25T22:00:00Z",
            )
            store.put_immutable(receipt)
            store.set_active(image_hash, receipt.receipt_id, "test")
            db.upsert_image(
                connection,
                {
                    "eagle_id": eagle_id,
                    "name": eagle_id,
                    "image_hash": image_hash,
                    "active_receipt_id": receipt.receipt_id,
                },
            )
        connection.close()
        api = AmbiguousFirstNotesApi()

        exit_code, payload, _, _ = self.invoke(
            ["notes-sync", "--apply", "--quiescent", "--json"],
            eagle=api,
        )

        self.assertEqual(exit_code, 1)
        self.assertFalse(payload["ok"])
        self.assertTrue(payload["stopped_early"])
        self.assertEqual(api.update_ids, ["a-first"])
        self.assertEqual(api.read_ids.count("b-second"), 1)
        journal = Path(payload["journal_path"])
        self.assertTrue(journal.is_file())
        persisted = json.loads(journal.read_text(encoding="utf-8"))
        self.assertEqual(persisted["state"], "stopped")
        self.assertEqual(len(persisted["initial_snapshots"]), 2)
        self.assertEqual(persisted["results"][0]["status"], "ambiguous")
        self.assertEqual(persisted["results"][0]["before_annotation"], "Human note")


if __name__ == "__main__":
    unittest.main()
