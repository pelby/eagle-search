from __future__ import annotations

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
