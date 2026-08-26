"""SVG preparation tests for the Codex caption provider; no live processes run."""

from __future__ import annotations

import hashlib
import json
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.captioning.codex_cli import CaptionItemFailure, CodexCliCaptionProvider


VALID_CAPTION = {
    "contract_version": 1,
    "image_type": "illustration",
    "diagram_types": [],
    "subjects": ["robot"],
    "visual_style": ["3D"],
    "colours": ["navy"],
    "layout": ["centred"],
    "visible_text": [],
    "search_terms": ["teaching", "classroom"],
    "summary": "A robot teaches in a classroom.",
    "uncertainties": [],
}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class MemoryReceiptStore:
    def __init__(self) -> None:
        self.receipts = []

    def put_immutable(self, receipt):
        self.receipts.append(receipt)
        return receipt.receipt_id


class EvalImagePreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.original = Path(self.temporary.name) / "eagle-cached.png"
        self.store = MemoryReceiptStore()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_svg_content_under_png_suffix_uses_private_real_png_and_original_hash(self) -> None:
        original_bytes = b'\xef\xbb\xbf<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"></svg>'
        self.original.write_bytes(original_bytes)
        conversion_calls = []
        codex_calls = []
        prepared_path: Path | None = None

        def convert_run(argv, **kwargs):
            conversion_calls.append((list(argv), kwargs))
            output = Path(argv[argv.index("--output") + 1])
            output.write_bytes(PNG_SIGNATURE + b"converted")
            return subprocess.CompletedProcess(argv, 0, "", "")

        def codex_run(argv, **kwargs):
            nonlocal prepared_path
            codex_calls.append((list(argv), kwargs))
            prepared_path = Path(argv[argv.index("-i") + 1])
            self.assertNotEqual(prepared_path, self.original)
            self.assertTrue(prepared_path.is_file())
            self.assertEqual(prepared_path.read_bytes()[:8], PNG_SIGNATURE)
            self.assertEqual(stat.S_IMODE(prepared_path.parent.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(prepared_path.stat().st_mode), 0o600)
            return subprocess.CompletedProcess(argv, 0, json.dumps(VALID_CAPTION), "")

        provider = CodexCliCaptionProvider(
            command="codex-test",
            run=codex_run,
            convert_run=convert_run,
            timeout_seconds=7,
            conversion_timeout_seconds=3,
            retries=0,
            now=lambda: "2026-08-26T12:00:00Z",
        )
        receipt = provider.caption(
            self.original,
            model="gpt-5.6-luna",
            effort="low",
            receipt_store=self.store,
        )

        conversion_argv, conversion_kwargs = conversion_calls[0]
        self.assertEqual(
            conversion_argv,
            [
                "rsvg-convert",
                "--format",
                "png",
                "--background-color",
                "white",
                "--output",
                str(prepared_path),
                str(self.original),
            ],
        )
        self.assertFalse(conversion_kwargs["shell"])
        self.assertEqual(conversion_kwargs["timeout"], 3)
        codex_argv, codex_kwargs = codex_calls[0]
        self.assertNotIn(str(self.original), codex_argv)
        self.assertFalse(codex_kwargs["shell"])
        self.assertEqual(codex_kwargs["timeout"], 7)
        self.assertEqual(receipt.image_hash, "sha256:" + hashlib.sha256(original_bytes).hexdigest())
        self.assertEqual(self.store.receipts, [receipt])
        self.assertIsNotNone(prepared_path)
        self.assertFalse(prepared_path.exists())
        self.assertFalse(prepared_path.parent.exists())

    def test_conversion_failure_is_item_scoped_and_never_invokes_codex(self) -> None:
        self.original.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>', encoding="utf-8")
        prepared_paths = []
        codex_calls = []

        def convert_run(argv, **_kwargs):
            prepared_paths.append(Path(argv[argv.index("--output") + 1]))
            return subprocess.CompletedProcess(argv, 1, "", "invalid SVG")

        def codex_run(argv, **kwargs):
            codex_calls.append((argv, kwargs))
            return subprocess.CompletedProcess(argv, 0, json.dumps(VALID_CAPTION), "")

        provider = CodexCliCaptionProvider(run=codex_run, convert_run=convert_run, retries=0)
        with self.assertRaisesRegex(CaptionItemFailure, "SVG conversion failed"):
            provider.caption(
                self.original,
                model="gpt-5.6-luna",
                effort="low",
                receipt_store=self.store,
            )

        self.assertEqual(codex_calls, [])
        self.assertEqual(self.store.receipts, [])
        self.assertFalse(prepared_paths[0].exists())
        self.assertFalse(prepared_paths[0].parent.exists())

    def test_successful_conversion_must_produce_png_bytes(self) -> None:
        self.original.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>', encoding="utf-8")

        def convert_run(argv, **_kwargs):
            Path(argv[argv.index("--output") + 1]).write_bytes(b"not a png")
            return subprocess.CompletedProcess(argv, 0, "", "")

        provider = CodexCliCaptionProvider(
            run=lambda *_args, **_kwargs: self.fail("Codex must not receive a fake PNG"),
            convert_run=convert_run,
            retries=0,
        )
        with self.assertRaisesRegex(CaptionItemFailure, "valid PNG"):
            provider.caption(
                self.original,
                model="gpt-5.6-luna",
                effort="low",
                receipt_store=self.store,
            )

    def test_svg_temporary_png_is_removed_after_codex_failure(self) -> None:
        self.original.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>', encoding="utf-8")
        prepared_paths = []

        def convert_run(argv, **_kwargs):
            output = Path(argv[argv.index("--output") + 1])
            prepared_paths.append(output)
            output.write_bytes(PNG_SIGNATURE + b"converted")
            return subprocess.CompletedProcess(argv, 0, "", "")

        provider = CodexCliCaptionProvider(
            run=lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 1, "", "item failure"),
            convert_run=convert_run,
            retries=0,
        )
        with self.assertRaises(CaptionItemFailure):
            provider.caption(
                self.original,
                model="gpt-5.6-luna",
                effort="low",
                receipt_store=self.store,
            )

        self.assertFalse(prepared_paths[0].exists())
        self.assertFalse(prepared_paths[0].parent.exists())

    def test_raster_content_uses_original_path_without_conversion(self) -> None:
        raster = self.original.with_suffix(".svg")
        raster.write_bytes(PNG_SIGNATURE + b"raster")
        codex_calls = []

        def codex_run(argv, **kwargs):
            codex_calls.append((list(argv), kwargs))
            return subprocess.CompletedProcess(argv, 0, json.dumps(VALID_CAPTION), "")

        provider = CodexCliCaptionProvider(
            run=codex_run,
            convert_run=lambda *_args, **_kwargs: self.fail("raster input must not be converted"),
            retries=0,
        )
        provider.caption(
            raster,
            model="gpt-5.6-luna",
            effort="low",
            receipt_store=self.store,
        )

        self.assertEqual(codex_calls[0][0][codex_calls[0][0].index("-i") + 1], str(raster))
        self.assertTrue(raster.exists())

    def test_jpeg_and_webp_content_under_png_suffix_keep_original_bytes_and_path(self) -> None:
        samples = {
            "jpeg": b"\xff\xd8\xff\xe0jpeg payload",
            "webp": b"RIFF\x10\x00\x00\x00WEBPwebp payload",
        }
        for name, content in samples.items():
            with self.subTest(name=name):
                image = Path(self.temporary.name) / f"{name}.png"
                image.write_bytes(content)
                store = MemoryReceiptStore()
                seen_paths = []

                def codex_run(argv, **_kwargs):
                    seen_paths.append(Path(argv[argv.index("-i") + 1]))
                    return subprocess.CompletedProcess(argv, 0, json.dumps(VALID_CAPTION), "")

                provider = CodexCliCaptionProvider(
                    run=codex_run,
                    convert_run=lambda *_args, **_kwargs: self.fail(f"{name} must not be converted"),
                    retries=0,
                )
                receipt = provider.caption(
                    image,
                    model="gpt-5.6-luna",
                    effort="low",
                    receipt_store=store,
                )

                self.assertEqual(seen_paths, [image])
                self.assertEqual(receipt.image_hash, "sha256:" + hashlib.sha256(content).hexdigest())


if __name__ == "__main__":
    unittest.main()
