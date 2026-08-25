"""R2/R11/R12 tests for safe, mocked Codex CLI captioning."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.captioning.codex_cli import (
    CaptionGlobalFailure,
    CaptionItemFailure,
    CodexCliCaptionProvider,
)


VALID_CAPTION = {
    "contract_version": 1,
    "image_type": "illustration",
    "diagram_types": [],
    "subjects": ["robot"],
    "visual_style": ["3D"],
    "colours": ["navy"],
    "layout": ["centred"],
    "visible_text": [{"text": "hello", "legibility": "high"}],
    "search_terms": ["teaching", "classroom"],
    "summary": "A robot teaches in a classroom.",
    "uncertainties": [],
}


class MemoryReceiptStore:
    def __init__(self) -> None:
        self.receipts = []

    def put_immutable(self, receipt):
        self.receipts.append(receipt)
        return receipt.receipt_id


class CodexCaptionRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.image = Path(self.tmp.name) / "fixture.png"
        self.image.write_bytes(b"private fixture bytes")
        self.calls: list[tuple[list[str], dict]] = []
        self.store = MemoryReceiptStore()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _provider(self, results, *, retries: int = 1):
        iterator = iter(results)

        def fake_run(argv, **kwargs):
            self.calls.append((list(argv), kwargs))
            next_result = next(iterator)
            if isinstance(next_result, BaseException):
                raise next_result
            return next_result

        return CodexCliCaptionProvider(
            command="codex-test",
            run=fake_run,
            retries=retries,
            timeout_seconds=7,
            now=lambda: "2026-08-25T22:00:00Z",
        )

    def test_argv_is_safe_explicit_and_receipt_is_immutable(self) -> None:
        provider = self._provider(
            [subprocess.CompletedProcess([], 0, json.dumps(VALID_CAPTION), "")]
        )

        receipt = provider.caption(
            self.image,
            model="gpt-5.6-luna",
            effort="low",
            receipt_store=self.store,
        )

        argv, kwargs = self.calls[0]
        self.assertEqual(argv[:3], ["codex-test", "exec", "--ephemeral"])
        self.assertIn("--sandbox", argv)
        self.assertEqual(argv[argv.index("--sandbox") + 1], "read-only")
        self.assertEqual(argv[argv.index("-m") + 1], "gpt-5.6-luna")
        self.assertIn('model_reasoning_effort="low"', argv)
        self.assertEqual(argv[argv.index("-i") + 1], str(self.image))
        self.assertIn("--output-schema", argv)
        self.assertFalse(kwargs["shell"])
        self.assertEqual(kwargs["timeout"], 7)
        self.assertEqual(receipt.provider, "codex-cli")
        self.assertEqual(receipt.effort, "low")
        self.assertEqual(receipt.image_hash, "sha256:" + hashlib.sha256(self.image.read_bytes()).hexdigest())
        self.assertEqual(self.store.receipts, [receipt])

    def test_item_failures_retry_but_invalid_output_never_becomes_a_receipt(self) -> None:
        provider = self._provider(
            [
                subprocess.CompletedProcess([], 1, "", "temporary service failure"),
                subprocess.CompletedProcess([], 0, "not-json", ""),
            ]
        )

        with self.assertRaises(CaptionItemFailure):
            provider.caption(self.image, model="gpt-5.6-luna", effort="low", receipt_store=self.store)

        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.store.receipts, [])

    def test_auth_and_quota_stop_without_retry(self) -> None:
        provider = self._provider(
            [subprocess.CompletedProcess([], 1, "", "Please login: quota exhausted")],
            retries=3,
        )

        with self.assertRaises(CaptionGlobalFailure):
            provider.caption(self.image, model="gpt-5.6-terra", effort="low", receipt_store=self.store)

        self.assertEqual(len(self.calls), 1)

    def test_timeout_is_bounded_retryable_item_failure(self) -> None:
        provider = self._provider([subprocess.TimeoutExpired(["codex-test"], 7)], retries=0)

        with self.assertRaises(CaptionItemFailure):
            provider.caption(self.image, model="gpt-5.6-sol", effort="low", receipt_store=self.store)

    def test_unsupported_model_or_effort_never_invokes_process(self) -> None:
        provider = self._provider([])

        with self.assertRaises(ValueError):
            provider.caption(self.image, model="gpt-5.6-sol", effort="xhigh", receipt_store=self.store)

        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
