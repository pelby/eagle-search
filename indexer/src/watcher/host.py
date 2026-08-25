"""Production wiring for the generated-image polling watcher.

This module deliberately keeps the process boundary small: ``once`` is a
bounded, JSON-safe unit suitable for a CLI command or scheduler, while
``run`` repeats that unit for a long-lived host.  Eagle's application settings
are checked before SQLite is opened or any import intent can be enqueued.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from .. import db
from ..eagle.importer import FileIntentMetadataStore, GeneratedImageImporter, file_is_stable
from ..eagle_api import EagleApiError
from ..persistence.jobs import SQLiteImportQueueStore
from .core import GeneratedImageWatcher, ProcessReport, ScanReport, WatcherAuthorityError


class WatchGeneratedHostError(RuntimeError):
    """A long-running host cannot safely continue after an authority refusal."""


class EagleWatcherApi(Protocol):
    async def application_info(self) -> dict[str, Any]: ...

    async def list_recent(self, *, limit: int = 200) -> list[dict[str, Any]]: ...

    async def add_from_path(
        self,
        *,
        path: str,
        name: str,
        annotation: str,
        source: str,
        tags: tuple[str, ...],
    ) -> str: ...

    async def get_item(self, item_id: str) -> dict[str, Any]: ...


def auto_import_overlaps(info: dict[str, Any], folder: Path) -> bool:
    """Whether Eagle's own auto-import authority covers ``folder``.

    An enabled auto-import without a readable configured path is conservatively
    treated as an overlap.  That prevents two independent writers from adding
    the same generated image.
    """
    preferences = info.get("preferences", {})
    auto_import = preferences.get("autoImport", {}) if isinstance(preferences, dict) else {}
    if not isinstance(auto_import, dict) or str(auto_import.get("enable", "false")).casefold() != "true":
        return False
    configured = auto_import.get("path")
    if not isinstance(configured, str) or not configured:
        return True
    try:
        Path(folder).resolve().relative_to(Path(configured).expanduser().resolve())
        return True
    except ValueError:
        return False


def _scan_report(report: ScanReport) -> dict[str, Any]:
    return {
        "enqueued": report.enqueued,
        "skipped": report.skipped,
        "metadata_errors": list(report.metadata_errors),
    }


def _process_report(report: ProcessReport) -> dict[str, Any]:
    return {
        "eagle_available": report.eagle_available,
        "processed": report.processed,
        "state": report.state,
    }


class WatchGeneratedHost:
    """Own the production lifecycle around :class:`GeneratedImageWatcher`.

    It makes the one safe ordering explicit: inspect Eagle's auto-import
    settings, reject overlap, release stale claims, startup-scan, then claim
    and import at most ``maximum`` queued items.
    """

    def __init__(
        self,
        *,
        home: Path,
        folder: Path,
        eagle: EagleWatcherApi,
        worker_id: str = "watch-generated",
        stale_claim_seconds: int = 300,
        stable_check: Callable[[Path], bool] = file_is_stable,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        if stale_claim_seconds < 0:
            raise ValueError("stale_claim_seconds cannot be negative")
        if poll_interval_seconds < 0:
            raise ValueError("poll_interval_seconds cannot be negative")
        self.home = Path(home)
        self.folder = Path(folder)
        self.eagle = eagle
        self.worker_id = worker_id
        self.stale_claim_seconds = stale_claim_seconds
        self.stable_check = stable_check
        self.poll_interval_seconds = poll_interval_seconds

    async def once(self, *, maximum: int = 1) -> dict[str, Any]:
        """Run a bounded startup reconciliation and import pass.

        The return value contains only JSON-safe primitives so callers can
        print it directly as a canonical ``--once`` report.
        """
        if maximum < 1:
            raise ValueError("maximum must be positive")
        try:
            info = await self.eagle.application_info()
        except EagleApiError as error:
            return {
                "contract_version": 1,
                "ok": True,
                "state": "paused",
                "eagle_available": False,
                "startup": None,
                "processing": {"eagle_available": False, "processed": 0, "state": "paused"},
                "released_stale_claims": 0,
                "reason": str(error),
            }

        if auto_import_overlaps(info, self.folder):
            return {
                "contract_version": 1,
                "ok": False,
                "state": "refused",
                "eagle_available": True,
                "startup": None,
                "processing": None,
                "released_stale_claims": 0,
                "error": {
                    "code": "duplicate_import_authority",
                    "message": "Eagle auto-import overlaps the generated-image watcher folder",
                },
            }

        connection = db.init_db(self.home / "db.sqlite")
        try:
            queue = SQLiteImportQueueStore(connection)
            released_stale_claims = queue.release_stale(older_than_seconds=self.stale_claim_seconds)
            importer = GeneratedImageImporter(
                queue=queue,
                metadata_store=FileIntentMetadataStore(self.home / "import-intents"),
                eagle=self.eagle,
                worker_id=self.worker_id,
                # Settings were just inspected.  The core guard remains in
                # place on every enqueue and claim to make authority explicit.
                overlapping_authority=lambda: auto_import_overlaps(info, self.folder),
            )
            watcher = GeneratedImageWatcher(
                folder=self.folder,
                importer=importer,
                eagle_auto_import_enabled=lambda: auto_import_overlaps(info, self.folder),
                eagle_available=lambda: True,
                stable_check=self.stable_check,
            )
            startup = watcher.startup_reconcile()
            processing = await watcher.process_pending(maximum=maximum)
            return {
                "contract_version": 1,
                "ok": True,
                "state": processing.state,
                "eagle_available": processing.eagle_available,
                "startup": _scan_report(startup),
                "processing": _process_report(processing),
                "released_stale_claims": released_stale_claims,
            }
        except WatcherAuthorityError as error:
            # A future dynamic setting source could change between inspection
            # and work; retain a machine-readable refusal if it does.
            return {
                "contract_version": 1,
                "ok": False,
                "state": "refused",
                "eagle_available": True,
                "startup": None,
                "processing": None,
                "released_stale_claims": 0,
                "error": {"code": "duplicate_import_authority", "message": str(error)},
            }
        finally:
            connection.close()

    async def run(
        self,
        *,
        stop: Callable[[], bool],
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        maximum: int = 1,
    ) -> None:
        """Poll until ``stop``; reject an overlapping Eagle authority once."""
        while not stop():
            report = await self.once(maximum=maximum)
            if not report["ok"]:
                error = report.get("error", {})
                raise WatchGeneratedHostError(str(error.get("message", "watcher refused")))
            await sleep(self.poll_interval_seconds)
