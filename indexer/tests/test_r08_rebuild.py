from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from src import db
from src.contracts import CaptionReceiptV1, CaptionResultV1
from src.eagle.notes import render_caption_block
from src.persistence.receipts import FileReceiptStore
from src.rebuild import (
    RebuildRecord,
    ManagedNotesFallbackError,
    parse_managed_caption,
    rebuild_database,
)


def _hash(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _receipt(image_hash: str) -> CaptionReceiptV1:
    caption = CaptionResultV1.from_dict(
        {
            "contract_version": 1,
            "image_type": "illustration",
            "diagram_types": [],
            "subjects": ["robot teacher", "easel"],
            "visual_style": ["3D"],
            "colours": ["navy"],
            "layout": ["subject right"],
            "visible_text": [{"text": ">_", "legibility": "high"}],
            "search_terms": ["classroom", "teaching"],
            "summary": "A robot teaches beside an easel in a classroom.",
            "uncertainties": [],
        }
    )
    return CaptionReceiptV1.create(
        image_hash=image_hash,
        caption_result=caption,
        provider="fixture",
        model="fixture",
        effort="low",
        prompt_version="caption-v1",
        created_at="2026-08-25T12:00:00Z",
    )


class RebuildTests(unittest.TestCase):
    def test_rebuilds_disposable_db_from_receipt_without_model_calls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_hash = _hash(b"image")
            store = FileReceiptStore(root / "receipts")
            receipt = _receipt(image_hash)
            store.put_immutable(receipt)
            store.set_active(image_hash, receipt.receipt_id, "accepted")

            outcome = rebuild_database(
                root / "rebuilt.sqlite",
                records=[
                    RebuildRecord(
                        eagle_id="item",
                        name="Human name",
                        annotation="Human note bytes: café",
                        image_hash=image_hash,
                    )
                ],
                receipt_store=store,
            )

            self.assertEqual(outcome.receipts, 1)
            self.assertEqual(outcome.notes_fallbacks, 0)
            connection = db.init_db(root / "rebuilt.sqlite")
            row = db.search(connection, "classroom")[0]
            self.assertEqual(row["eagle_id"], "item")
            self.assertEqual(row["human_notes"], "Human note bytes: café")
            self.assertEqual(row["active_receipt_id"], receipt.receipt_id)
            self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            connection.close()

    def test_legacy_manifest_maps_eagle_item_back_to_synthetic_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual_hash = _hash(b"current thumbnail")
            legacy_hash = _hash(b"legacy evidence")
            store = FileReceiptStore(root / "receipts")
            receipt = _receipt(legacy_hash)
            store.put_immutable(receipt)
            manifest = {
                "manifest_version": 1,
                "entries": [
                    {
                        "eagle_id": "legacy-item",
                        "image_hash": legacy_hash,
                        "receipt_id": receipt.receipt_id,
                    }
                ],
            }

            outcome = rebuild_database(
                root / "rebuilt.sqlite",
                records=[RebuildRecord(eagle_id="legacy-item", name="Legacy", image_hash=actual_hash)],
                receipt_store=store,
                legacy_manifest=manifest,
            )

            self.assertEqual(outcome.legacy_receipts, 1)
            connection = db.init_db(root / "rebuilt.sqlite")
            self.assertEqual(db.search(connection, "classroom")[0]["eagle_id"], "legacy-item")
            connection.close()

    def test_notes_fallback_is_strict_separately_gated_and_degraded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_hash = _hash(b"notes-only")
            receipt = _receipt(image_hash)
            annotation = "Human preface\n\n" + render_caption_block(receipt)
            record = RebuildRecord(
                eagle_id="notes-item",
                name="Notes only",
                annotation=annotation,
                image_hash=image_hash,
            )

            without = rebuild_database(
                root / "without.sqlite",
                records=[record],
                receipt_store=FileReceiptStore(root / "empty"),
            )
            with_fallback = rebuild_database(
                root / "with.sqlite",
                records=[record],
                receipt_store=FileReceiptStore(root / "empty"),
                allow_notes_fallback=True,
            )

            self.assertEqual(without.pending, 1)
            self.assertEqual(with_fallback.notes_fallbacks, 1)
            connection = db.init_db(root / "with.sqlite")
            row = db.search(connection, "classroom")[0]
            self.assertEqual(row["caption_state"], "degraded-notes")
            self.assertEqual(row["active_receipt_id"], receipt.receipt_id)
            self.assertIn("Human preface", row["human_notes"])
            connection.close()

    def test_malformed_managed_notes_are_refused_not_partially_trusted(self) -> None:
        receipt = _receipt(_hash(b"malformed"))
        malformed = render_caption_block(receipt).replace("**Receipt:**", "**Wrong:**")
        with self.assertRaises(ManagedNotesFallbackError):
            parse_managed_caption(malformed)

    def test_existing_destination_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "derived.sqlite"
            destination.write_bytes(b"user bytes")
            with self.assertRaises(FileExistsError):
                rebuild_database(
                    destination,
                    records=[],
                    receipt_store=FileReceiptStore(Path(directory) / "receipts"),
                )
            self.assertEqual(destination.read_bytes(), b"user bytes")


if __name__ == "__main__":
    unittest.main()
