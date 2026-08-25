"""R10 polling watcher: startup reconciliation and durable import hand-off."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src import db
from src.eagle.importer import FileIntentMetadataStore, GeneratedImageImporter
from src.persistence.jobs import SQLiteImportQueueStore
from src.watcher.core import GeneratedImageWatcher, WatcherAuthorityError


class FakeImporter:
    def __init__(self) -> None:
        self.enqueued: list[tuple[Path, str, str, tuple[str, ...], bool]] = []
        self.process_calls = 0

    def enqueue(self, path, *, prompt, source, tags, keep_existing_metadata=False):
        self.enqueued.append((Path(path), prompt, source, tuple(tags), keep_existing_metadata))
        return {"path": str(path)}

    async def process_one(self):
        self.process_calls += 1
        return type("Outcome", (), {"state": "idle"})()


class FakeEagle:
    def __init__(self) -> None:
        self.items: list[dict[str, str]] = []
        self.add_calls = 0

    async def list_recent(self, *, limit: int = 200):
        return self.items[-limit:]

    async def add_from_path(self, *, path, name, annotation, source, tags):
        self.add_calls += 1
        eagle_id = f"generated-{self.add_calls}"
        self.items.append({"id": eagle_id, "annotation": annotation})
        return eagle_id

    async def get_item(self, item_id: str):
        return next(item for item in self.items if item["id"] == item_id)


class WatcherTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        (self.folder / "classroom.png").write_bytes(b"image")
        (self.folder / "notes.txt").write_text("not an image", encoding="utf-8")
        (self.folder / "classroom.png.eagle-search.json").write_text(
            '{"prompt":"robot teaching","source":"imagegen","tags":["generated","classroom"]}',
            encoding="utf-8",
        )
        self.importer = FakeImporter()

    async def asyncTearDown(self) -> None:
        self.temp.cleanup()

    async def test_startup_scans_stable_supported_images_once_and_preserves_sidecar_metadata(self) -> None:
        watcher = GeneratedImageWatcher(
            folder=self.folder, importer=self.importer, eagle_auto_import_enabled=lambda: False,
            stable_check=lambda _path: True, eagle_available=lambda: True,
        )
        first = watcher.startup_reconcile()
        second = watcher.poll_once()

        self.assertEqual(first.enqueued, 1)
        self.assertEqual(second.enqueued, 0)
        self.assertEqual(self.importer.enqueued[0][1:4], ("robot teaching", "imagegen", ("generated", "classroom")))
        self.assertTrue(self.importer.enqueued[0][4])

    async def test_unavailable_eagle_does_not_claim_work_and_later_resumes(self) -> None:
        available = False
        watcher = GeneratedImageWatcher(
            folder=self.folder, importer=self.importer, eagle_auto_import_enabled=lambda: False,
            stable_check=lambda _path: True, eagle_available=lambda: available,
        )
        watcher.startup_reconcile()
        paused = await watcher.process_pending()
        self.assertFalse(paused.eagle_available)
        self.assertEqual(self.importer.process_calls, 0)
        available = True
        resumed = await watcher.process_pending()
        self.assertTrue(resumed.eagle_available)
        self.assertEqual(self.importer.process_calls, 1)

    async def test_restart_after_eagle_closed_resumes_one_durable_intent_without_duplicate_add(self) -> None:
        state = db.init_db(self.folder / "state.sqlite")
        queue = SQLiteImportQueueStore(state)
        eagle = FakeEagle()
        available = False
        importer = GeneratedImageImporter(
            queue=queue,
            metadata_store=FileIntentMetadataStore(self.folder / "intent-sidecars"),
            eagle=eagle,
            worker_id="watcher",
            overlapping_authority=lambda: False,
        )
        first = GeneratedImageWatcher(
            folder=self.folder, importer=importer, eagle_auto_import_enabled=lambda: False,
            stable_check=lambda _path: True, eagle_available=lambda: available,
        )
        first.startup_reconcile()
        self.assertEqual((await first.process_pending()).state, "paused")
        self.assertEqual(eagle.add_calls, 0)
        # A fresh process repeats mandatory startup reconciliation.  The durable
        # path/stat intent and metadata sidecar must converge rather than adding.
        second = GeneratedImageWatcher(
            folder=self.folder, importer=importer, eagle_auto_import_enabled=lambda: False,
            stable_check=lambda _path: True, eagle_available=lambda: available,
        )
        second.startup_reconcile()
        available = True
        result = await second.process_pending(maximum=2)
        self.assertEqual(result.processed, 1)
        self.assertEqual(eagle.add_calls, 1)
        self.assertEqual(state.execute("SELECT count(*) FROM pending_imports").fetchone()[0], 1)
        self.assertEqual(state.execute("SELECT state FROM pending_imports").fetchone()[0], "complete")
        state.close()

    async def test_overlapping_eagle_auto_import_refuses_before_scan_or_process(self) -> None:
        watcher = GeneratedImageWatcher(
            folder=self.folder, importer=self.importer, eagle_auto_import_enabled=lambda: True,
            stable_check=lambda _path: True, eagle_available=lambda: True,
        )
        with self.assertRaises(WatcherAuthorityError):
            watcher.startup_reconcile()
        with self.assertRaises(WatcherAuthorityError):
            await watcher.process_pending()
        self.assertEqual(self.importer.enqueued, [])
        self.assertEqual(self.importer.process_calls, 0)
