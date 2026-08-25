from __future__ import annotations

import sqlite3
import json
import tempfile
import unittest
from pathlib import Path

from src import db
from src.persistence.receipts import FileReceiptStore, export_legacy_receipts


class LegacyReceiptTests(unittest.TestCase):
    def test_nonblank_v1_descriptions_export_count_for_count_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            connection = db.init_db(root / "legacy.sqlite")
            db.upsert_image(connection, {"eagle_id": "one", "name": "One", "ai_description": "A classroom."})
            db.upsert_image(connection, {"eagle_id": "two", "name": "Two", "ai_description": ""})
            report = export_legacy_receipts(connection, FileReceiptStore(root / "receipts"))
            self.assertEqual(report["source_rows"], 1)
            self.assertEqual(report["receipt_count"], 1)
            self.assertEqual(export_legacy_receipts(connection, FileReceiptStore(root / "receipts"))["receipt_count"], 1)
            connection.close()

    def test_raw_v1_exports_before_migration_and_preserves_long_descriptions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "raw-v1.sqlite"
            connection = sqlite3.connect(path)
            connection.row_factory = sqlite3.Row
            connection.execute(
                "CREATE TABLE images (eagle_id TEXT PRIMARY KEY, thumbnail_path TEXT, "
                "ai_description TEXT, indexed_at TEXT)"
            )
            description = "legacy caption " * 120
            connection.execute(
                "INSERT INTO images VALUES (?,?,?,?)",
                ("one", "/tmp/one.png", description, "2026-08-25T00:00:00Z"),
            )
            connection.commit()

            receipts = root / "receipts"
            report = export_legacy_receipts(connection, FileReceiptStore(receipts))

            self.assertEqual(report, {"source_rows": 1, "distinct_eagle_ids": 1, "receipt_count": 1})
            receipt_files = list(receipts.glob("sha256:*/sha256:*.json"))
            self.assertEqual(len(receipt_files), 1)
            self.assertIn(description, receipt_files[0].read_text(encoding="utf-8"))
            manifest = json.loads((receipts / "legacy-manifest-v1.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["manifest_version"], 1)
            self.assertEqual(manifest["entries"][0]["eagle_id"], "one")
            self.assertEqual(manifest["entries"][0]["receipt_id"], receipt_files[0].stem)
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 0)
            connection.close()
