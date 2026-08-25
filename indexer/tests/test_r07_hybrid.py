from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src import db
from src.retrieval.hybrid import hybrid_search


class FakeEmbedder:
    model = "fake"
    def embed(self, texts):
        return [[1.0] + [0.0] * 767 for _ in texts]


class DownEmbedder(FakeEmbedder):
    def embed(self, texts):
        raise RuntimeError("offline")


class HybridTests(unittest.TestCase):
    def test_rrf_floor_and_lexical_degradation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connection = db.init_db(Path(directory) / "db.sqlite")
            db.upsert_image(connection, {"eagle_id": "literal", "name": "classroom", "ai_description": "school"})
            db.upsert_image(connection, {"eagle_id": "near", "name": "other", "ai_description": "learning space"})
            db.store_embedding(connection, "near", FakeEmbedder.model, "visual", db.image_content_hash(connection, "near"), [1.0] + [0.0] * 767)
            db.store_embedding(connection, "literal", FakeEmbedder.model, "visual", db.image_content_hash(connection, "literal"), [0.1] + [0.995] + [0.0] * 766)
            response = hybrid_search(connection, "classroom", FakeEmbedder(), semantic_floor=0.6)
            self.assertEqual(response.results[0].eagle_id, "literal")
            self.assertIn("semantic", response.results[1].matched_by)
            fallback = hybrid_search(connection, "classroom", DownEmbedder())
            self.assertEqual(fallback.mode, "lexical")
            self.assertFalse(fallback.semantic_available)
            connection.close()
