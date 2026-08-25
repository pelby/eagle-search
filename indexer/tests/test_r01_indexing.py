"""R1/R4/R5/R8 composition tests for the canonical indexing seam."""

from __future__ import annotations

import tempfile
import sqlite3
import unittest
from pathlib import Path

from src import db
from src.contracts import CaptionReceiptV1, CaptionResultV1
from src.indexing import EmbeddedMetadata, _plan_stale_cleanup, index_library
from src.captioning.codex_cli import image_sha256
from src.captioning.codex_cli import CaptionItemFailure
from src.persistence.receipts import FileReceiptStore
from src.eagle_api import EagleApiError


class FakeEagle:
    def __init__(self, thumbnail: Path) -> None:
        self.thumbnail = thumbnail
        self.thumbnail_calls = 0
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
                "mtime": 1,
                "size": len(thumbnail.read_bytes()),
            }
        ]

    async def is_running(self) -> bool:
        return True

    async def list_items(self, *, limit: int = 10_000):
        return list(self.items[:limit])

    async def get_folder_map(self):
        return {"folder": "Teaching"}

    async def get_thumbnail_path(self, item_id: str):
        self.thumbnail_calls += 1
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


class FailingCaptionProvider:
    def caption(self, image_path, *, model, effort, receipt_store):
        raise CaptionItemFailure("fixture cannot be decoded")


class TransientThumbnailEagle(FakeEagle):
    def __init__(self, thumbnail: Path) -> None:
        super().__init__(thumbnail)
        self.thumbnail_calls = 0

    async def get_thumbnail_path(self, item_id: str):
        self.thumbnail_calls += 1
        if self.thumbnail_calls == 1:
            raise EagleApiError("transient thumbnail route failure")
        self.assert_item(item_id)
        return str(self.thumbnail)


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
        self.assertEqual(self.eagle.thumbnail_calls, 1)
        connection = db.init_db(self.home / "db.sqlite")
        row = db.search(connection, "classroom")[0]
        self.assertEqual(row["eagle_id"], "fixture")
        self.assertEqual(row["human_notes"], "Human note")
        self.assertTrue(row["active_receipt_id"].startswith("sha256:"))
        self.assertEqual(connection.execute("SELECT count(*) FROM image_embeddings").fetchone()[0], 1)
        connection.close()

    async def test_transient_eagle_thumbnail_failure_is_retried(self) -> None:
        eagle = TransientThumbnailEagle(self.thumbnail)
        outcome = await index_library(
            home=self.home,
            eagle=eagle,
            embedder=FakeEmbedder(),
            caption_provider=self.provider,
        )

        self.assertEqual(outcome.captioned, 1)
        self.assertEqual(eagle.thumbnail_calls, 2)

    async def test_changed_eagle_source_refreshes_cached_thumbnail_and_caption(self) -> None:
        await index_library(
            home=self.home,
            eagle=self.eagle,
            embedder=FakeEmbedder(),
            caption_provider=self.provider,
        )
        self.thumbnail.write_bytes(b"replacement thumbnail bytes")
        self.eagle.items[0]["mtime"] = 456
        self.eagle.items[0]["size"] = len(self.thumbnail.read_bytes())

        outcome = await index_library(
            home=self.home,
            eagle=self.eagle,
            embedder=FakeEmbedder(),
            caption_provider=self.provider,
        )

        self.assertEqual(outcome.captioned, 1)
        self.assertEqual(self.provider.calls, 2)
        self.assertEqual(self.eagle.thumbnail_calls, 2)
        connection = db.init_db(self.home / "db.sqlite")
        row = connection.execute("SELECT * FROM images WHERE eagle_id='fixture'").fetchone()
        self.assertEqual(row["source_mtime"], 456)
        self.assertEqual(row["source_size"], len(self.thumbnail.read_bytes()))
        self.assertEqual(
            row["image_hash"],
            image_sha256(self.home / "thumbnails" / "fixture.png"),
        )
        connection.close()

    async def test_first_fingerprinted_run_refreshes_a_migrated_cached_thumbnail(self) -> None:
        await index_library(
            home=self.home,
            eagle=self.eagle,
            embedder=FakeEmbedder(),
            caption_provider=self.provider,
        )
        connection = db.init_db(self.home / "db.sqlite")
        with connection:
            connection.execute(
                "UPDATE images SET source_mtime=0,source_size=0 WHERE eagle_id='fixture'"
            )
        connection.close()
        self.thumbnail.write_bytes(b"bytes changed before fingerprint migration")
        self.eagle.items[0]["mtime"] = 789
        self.eagle.items[0]["size"] = len(self.thumbnail.read_bytes())

        outcome = await index_library(
            home=self.home,
            eagle=self.eagle,
            embedder=FakeEmbedder(),
            caption_provider=self.provider,
        )

        self.assertEqual(outcome.captioned, 1)
        self.assertEqual(self.eagle.thumbnail_calls, 2)

    async def test_changed_image_with_failed_recaption_never_serves_stale_caption(self) -> None:
        await index_library(
            home=self.home,
            eagle=self.eagle,
            embedder=FakeEmbedder(),
            caption_provider=self.provider,
        )
        self.thumbnail.write_bytes(b"changed undecodable bytes")
        self.eagle.items[0]["mtime"] = 999
        self.eagle.items[0]["size"] = len(self.thumbnail.read_bytes())

        outcome = await index_library(
            home=self.home,
            eagle=self.eagle,
            embedder=FakeEmbedder(),
            caption_provider=FailingCaptionProvider(),
        )

        self.assertEqual(outcome.failed, 1)
        connection = db.init_db(self.home / "db.sqlite")
        row = connection.execute("SELECT * FROM images WHERE eagle_id='fixture'").fetchone()
        self.assertEqual(row["visual_caption"], "")
        self.assertEqual(row["visual_search_text"], "")
        self.assertEqual(row["active_receipt_id"], "")
        self.assertEqual(db.search(connection, "classroom"), [])
        self.assertEqual(
            connection.execute(
                "SELECT count(*) FROM image_embeddings WHERE eagle_id='fixture'"
            ).fetchone()[0],
            0,
        )
        connection.close()

    async def test_full_index_removes_only_rows_absent_from_current_eagle_library(self) -> None:
        connection = db.init_db(self.home / "db.sqlite")
        db.upsert_image(
            connection,
            {"eagle_id": "deleted-in-eagle", "name": "Stale", "ai_description": "Old"},
        )
        connection.close()

        outcome = await index_library(
            home=self.home,
            eagle=self.eagle,
            embedder=FakeEmbedder(),
            caption_provider=self.provider,
        )

        self.assertEqual(outcome.removed, 1)
        connection = db.init_db(self.home / "db.sqlite")
        self.assertIsNone(
            connection.execute(
                "SELECT 1 FROM images WHERE eagle_id='deleted-in-eagle'"
            ).fetchone()
        )
        self.assertIsNotNone(
            connection.execute("SELECT 1 FROM images WHERE eagle_id='fixture'").fetchone()
        )
        connection.close()

    def test_stale_cleanup_requires_a_complete_nonempty_eagle_list(self) -> None:
        existing = {"kept", "stale"}
        self.assertEqual(
            _plan_stale_cleanup(existing, {"kept"}, list_limit=10_000),
            (["stale"], ""),
        )
        stale, empty_reason = _plan_stale_cleanup(
            existing,
            set(),
            list_limit=10_000,
        )
        self.assertEqual(stale, [])
        self.assertIn("no items", empty_reason)
        stale, capped_reason = _plan_stale_cleanup(
            existing | {f"item-{index}" for index in range(10_000)},
            {f"item-{index}" for index in range(10_000)},
            list_limit=10_000,
        )
        self.assertEqual(stale, [])
        self.assertIn("API limit", capped_reason)

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

    async def test_legacy_description_is_adopted_to_current_hash_without_model_call(self) -> None:
        self.home.mkdir(parents=True)
        raw = sqlite3.connect(self.home / "db.sqlite")
        raw.execute(
            """CREATE TABLE images (
                eagle_id TEXT PRIMARY KEY, name TEXT NOT NULL, tags TEXT DEFAULT '',
                annotation TEXT DEFAULT '', ai_description TEXT DEFAULT '',
                thumbnail_path TEXT DEFAULT '', image_path TEXT DEFAULT '',
                folder_name TEXT DEFAULT '', ext TEXT DEFAULT '', width INTEGER DEFAULT 0,
                height INTEGER DEFAULT 0, created_at INTEGER DEFAULT 0, indexed_at TEXT DEFAULT ''
            )"""
        )
        raw.execute(
            "INSERT INTO images (eagle_id,name,ai_description,thumbnail_path,indexed_at) VALUES (?,?,?,?,?)",
            (
                "fixture",
                "Robot at an easel",
                "A legacy classroom presentation caption.",
                str(self.thumbnail),
                "2026-08-01T00:00:00Z",
            ),
        )
        raw.commit()
        raw.close()

        outcome = await index_library(
            home=self.home,
            eagle=self.eagle,
            embedder=FakeEmbedder(),
            caption_provider=self.provider,
        )

        self.assertEqual(outcome.legacy_exported, 1)
        self.assertEqual(self.provider.calls, 0)
        digest = image_sha256(self.home / "thumbnails" / "fixture.png")
        receipt = FileReceiptStore(self.home / "captions").resolve_active(digest)
        self.assertIsNotNone(receipt)
        self.assertEqual(receipt.image_hash, digest)
        self.assertEqual(receipt.source, "legacy")
        connection = db.init_db(self.home / "db.sqlite")
        row = connection.execute("SELECT * FROM images WHERE eagle_id='fixture'").fetchone()
        self.assertEqual(row["active_receipt_id"], receipt.receipt_id)
        self.assertEqual(row["caption_state"], "complete")
        connection.close()


if __name__ == "__main__":
    unittest.main()
