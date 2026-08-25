from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src import db


class WeightedFtsTests(unittest.TestCase):
    def test_name_outranks_low_weight_notes_and_quotes_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connection = db.init_db(Path(directory) / "db.sqlite")
            db.upsert_image(connection, {"eagle_id": "name", "name": "Classroom", "annotation": ""})
            db.upsert_image(connection, {"eagle_id": "notes", "name": "Other", "annotation": "classroom classroom classroom"})
            db.upsert_image(connection, {"eagle_id": "phrase", "name": "Other", "ai_description": "value based pricing", "created_at": 1})
            self.assertEqual(db.search(connection, "classroom")[0]["eagle_id"], "name")
            self.assertEqual([row["eagle_id"] for row in db.search(connection, '"value based pricing"')], ["phrase"])
            self.assertEqual(db.search(connection, "", limit=1)[0]["eagle_id"], "phrase")
            connection.close()
