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
from src.cli import CliRuntime, build_parser, main
from src.contracts import CaptionReceiptV1, CaptionResultV1


class FakeEmbedder:
    model = "fake"

    def embed(self, texts):
        return [[1.0] + [0.0] * 767 for _ in texts]


class DownEmbedder(FakeEmbedder):
    def embed(self, texts):
        raise RuntimeError("offline")


class FakeCaptionProvider:
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
            provider="fake",
            model=model,
            effort=effort,
            prompt_version="caption-v1",
            created_at="2026-08-25T22:00:00Z",
        )
        receipt_store.put_immutable(receipt)
        return receipt


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

    def invoke(self, arguments, *, embedder=None, caption_provider=None):
        stdout = io.StringIO()
        stderr = io.StringIO()
        runtime = CliRuntime(
            home=self.home,
            embedder=embedder or FakeEmbedder(),
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
                "eval-report",
            }.issubset(subparsers.choices)
        )

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

    def test_eval_paths_outside_private_root_are_refused(self) -> None:
        exit_code, payload, _, _ = self.invoke(
            [
                "eval-report",
                "--results",
                "/tmp/not-private.json",
                "--output",
                "/tmp/report.json",
                "--comparator",
                "gpt-5.6-terra",
                "--json",
            ]
        )
        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["error"]["code"], "invalid_request")


if __name__ == "__main__":
    unittest.main()
