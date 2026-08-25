from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src import db
from src.retrieval.backfill import backfill_embeddings


class FakeEmbedder:
    model = "fake"
    def embed(self, texts): return [[1.0] + [0.0] * 767 for _ in texts]


class BackfillTests(unittest.TestCase):
    def test_only_missing_or_stale_rows_are_embedded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connection = db.init_db(Path(directory) / "db.sqlite")
            db.upsert_image(connection, {"eagle_id": "one", "name": "one", "ai_description": "classroom"})
            embedder = FakeEmbedder()
            self.assertEqual(backfill_embeddings(connection, embedder)["embedded"], 1)
            self.assertEqual(backfill_embeddings(connection, embedder)["embedded"], 0)
            db.upsert_image(connection, {"eagle_id": "one", "name": "one", "ai_description": "lecture theatre"})
            self.assertEqual(backfill_embeddings(connection, embedder)["embedded"], 1)
            connection.close()
