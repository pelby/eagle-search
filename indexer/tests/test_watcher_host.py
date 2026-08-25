"""Production host wiring for the generated-image watcher."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import tempfile
import unittest
from pathlib import Path

from src import db
from src.eagle_api import EagleApiError
from src.persistence.jobs import SQLiteImportQueueStore
from src.watcher.host import WatchGeneratedHost


class FakeEagle:
    def __init__(self, *, auto_import: bool = False, unavailable: bool = False) -> None:
        self.auto_import = auto_import
        self.unavailable = unavailable
        self.items: list[dict[str, str]] = []
        self.add_calls = 0

    async def application_info(self):
        if self.unavailable:
            raise EagleApiError("Eagle closed")
        return {
            "version": "4.0.0",
            "preferences": {"autoImport": {"enable": self.auto_import, "path": ""}},
        }

    async def list_recent(self, *, limit: int = 200):
        return self.items[-limit:]

    async def add_from_path(self, *, path, name, annotation, source, tags):
        self.add_calls += 1
        identifier = f"eagle-{self.add_calls}"
        self.items.append({"id": identifier, "annotation": annotation})
        return identifier

    async def get_item(self, item_id: str):
        return next(item for item in self.items if item["id"] == item_id)


class WatcherHostTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.folder = self.root / "generated"
        self.folder.mkdir()
        (self.folder / "classroom.png").write_bytes(b"image")

    async def asyncTearDown(self) -> None:
        self.temp.cleanup()

    async def test_once_startup_scans_processes_and_returns_json_safe_report(self) -> None:
        eagle = FakeEagle()
        report = await WatchGeneratedHost(
            home=self.root / "state", folder=self.folder, eagle=eagle,
            stable_check=lambda _path: True,
        ).once()
        self.assertTrue(report["ok"])
        self.assertEqual(report["startup"]["enqueued"], 1)
        self.assertEqual(report["processing"]["processed"], 1)
        self.assertEqual(eagle.add_calls, 1)

    async def test_auto_import_overlap_refuses_before_enqueue_or_claim(self) -> None:
        eagle = FakeEagle(auto_import=True)
        report = await WatchGeneratedHost(
            home=self.root / "state", folder=self.folder, eagle=eagle,
            stable_check=lambda _path: True,
        ).once()
        self.assertFalse(report["ok"])
        self.assertEqual(report["error"]["code"], "duplicate_import_authority")
        connection = db.init_db(self.root / "state" / "db.sqlite")
        self.assertEqual(connection.execute("SELECT count(*) FROM pending_imports").fetchone()[0], 0)
        connection.close()

    async def test_stale_claims_are_released_before_startup_work(self) -> None:
        home = self.root / "state"
        connection = db.init_db(home / "db.sqlite")
        queue = SQLiteImportQueueStore(connection)
        queue.enqueue("/tmp/already-queued.png", "11111111-1111-1111-1111-111111111111")
        queue.claim(worker_id="dead-worker")
        old = (datetime.now(timezone.utc) - timedelta(seconds=301)).isoformat()
        connection.execute("UPDATE pending_imports SET claimed_at=?", (old,)); connection.commit()
        connection.close()
        (self.folder / "classroom.png").unlink()

        report = await WatchGeneratedHost(
            home=home, folder=self.folder, eagle=FakeEagle(), stable_check=lambda _path: True,
            stale_claim_seconds=300,
        ).once()
        self.assertEqual(report["released_stale_claims"], 1)

    async def test_closed_eagle_returns_a_paused_report_without_claiming_work(self) -> None:
        report = await WatchGeneratedHost(
            home=self.root / "state", folder=self.folder, eagle=FakeEagle(unavailable=True),
            stable_check=lambda _path: True,
        ).once()
        self.assertTrue(report["ok"])
        self.assertEqual(report["state"], "paused")
        self.assertEqual(report["startup"], None)

