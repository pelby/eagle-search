"""R1/R4/R5/R8 composition tests for the canonical indexing seam."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src import db
from src.contracts import CaptionReceiptV1, CaptionResultV1
from src.indexing import EmbeddedMetadata, index_library


class FakeEagle:
    def __init__(self, thumbnail: Path) -> None:
        self.thumbnail = thumbnail
        self.items = [
            {
                "id": "fixture",
                "name": "Robot at an easel",
                "tags": ["presentation"],
                "folders": ["folder"],
                "annotation": "Human note",
                "ext": "png",
                "width": 568,
                "height": 320,
                "btime": 1,
            }
        ]

    async def is_running(self) -> bool:
        return True

    async def list_items(self, *, limit: int = 10_000):
        return list(self.items[:limit])

    async def get_folder_map(self):
        return {"folder": "Teaching"}

    async def get_thumbnail_path(self, item_id: str):
        self.assert_item(item_id)
        return str(self.thumbnail)

    @staticmethod
    def assert_item(item_id: str) -> None:
        if item_id != "fixture":
            raise AssertionError(item_id)


class FakeEmbedder:
    model = "fake"

    def embed(self, texts):
        return [[1.0] + [0.0] * 767 for _ in texts]


class FakeCaptionProvider:
    def __init__(self) -> None:
        self.calls = 0

    def caption(self, image_path, *, model, effort, receipt_store):
        self.calls += 1
        result = CaptionResultV1.from_dict(
            {
                "contract_version": 1,
                "image_type": "3D illustration",
                "diagram_types": [],
                "subjects": ["robot presenter", "easel"],
                "visual_style": ["cinematic"],
                "colours": ["navy", "coral"],
                "layout": ["subject on right"],
                "visible_text": [{"text": ">_", "legibility": "high"}],
                "search_terms": ["classroom", "teaching", "presentation"],
                "summary": "A robot presenter points at an easel.",
                "uncertainties": [],
            }
        )
        from src.captioning.codex_cli import image_sha256

        receipt = CaptionReceiptV1.create(
            image_hash=image_sha256(Path(image_path)),
            caption_result=result,
            provider="fake",
            model=model,
            effort=effort,
            prompt_version="caption-v1",
            created_at="2026-08-25T20:00:00Z",
        )
        receipt_store.put_immutable(receipt)
        return receipt


class IndexingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.thumbnail = self.root / "fixture_thumbnail.png"
        self.thumbnail.write_bytes(b"stable thumbnail bytes")
        self.home = self.root / "state"
        self.eagle = FakeEagle(self.thumbnail)
        self.provider = FakeCaptionProvider()

    async def asyncTearDown(self) -> None:
        self.temporary.cleanup()

    async def test_new_image_is_captioned_receipted_embedded_and_searchable_once(self) -> None:
        first = await index_library(
            home=self.home,
            eagle=self.eagle,
            embedder=FakeEmbedder(),
            caption_provider=self.provider,
        )
        second = await index_library(
            home=self.home,
            eagle=self.eagle,
            embedder=FakeEmbedder(),
            caption_provider=self.provider,
        )

        self.assertEqual((first.queued, first.captioned, first.embedded), (1, 1, 1))
        self.assertEqual((second.queued, second.captioned), (0, 0))
        self.assertEqual(self.provider.calls, 1)
        connection = db.init_db(self.home / "db.sqlite")
        row = db.search(connection, "classroom")[0]
        self.assertEqual(row["eagle_id"], "fixture")
        self.assertEqual(row["human_notes"], "Human note")
        self.assertTrue(row["active_receipt_id"].startswith("sha256:"))
        self.assertEqual(connection.execute("SELECT count(*) FROM image_embeddings").fetchone()[0], 1)
        connection.close()

    async def test_embedded_description_is_used_without_a_model_call(self) -> None:
        outcome = await index_library(
            home=self.home,
            eagle=self.eagle,
            embedder=FakeEmbedder(),
            caption_provider=self.provider,
            metadata_reader=lambda _path: EmbeddedMetadata(
                prompt="A robot teaching in a classroom",
                description="A classroom presentation with a robot beside an easel.",
            ),
        )

        self.assertEqual((outcome.queued, outcome.captioned), (0, 0))
        self.assertEqual(self.provider.calls, 0)
        connection = db.init_db(self.home / "db.sqlite")
        row = connection.execute("SELECT * FROM images WHERE eagle_id='fixture'").fetchone()
        self.assertEqual(row["generation_prompt"], "A robot teaching in a classroom")
        self.assertIn("classroom presentation", row["visual_caption"])
        self.assertTrue(row["active_receipt_id"].startswith("sha256:"))
        connection.close()


if __name__ == "__main__":
    unittest.main()
