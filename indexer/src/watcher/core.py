"""Polling-only generated-image watcher around the durable importer.

No filesystem event library is required.  The watcher only discovers stable,
regular supported images; ``GeneratedImageImporter`` owns durable intent and
Eagle commit/reconciliation semantics.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import stat
from typing import Any, Awaitable, Callable, Protocol

from ..eagle.importer import IMAGE_EXTENSIONS, GeneratedImageImporter, file_is_stable


class WatcherAuthorityError(RuntimeError):
    """The watcher must not coexist with Eagle's own generated-folder import."""


class WatcherMetadataError(ValueError):
    """A sidecar exists but cannot safely be represented as import metadata."""


class Importer(Protocol):
    def enqueue(
        self, path: Path, *, prompt: str, source: str, tags: tuple[str, ...],
        keep_existing_metadata: bool = False,
    ) -> object: ...

    async def process_one(self) -> object: ...


@dataclass(frozen=True)
class GeneratedImageMetadata:
    prompt: str = ""
    source: str = ""
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScanReport:
    enqueued: int
    skipped: int
    metadata_errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProcessReport:
    eagle_available: bool
    processed: int
    state: str = ""


def sidecar_path(image_path: Path) -> Path:
    """The explicit, non-image sidecar name for generated-image metadata."""
    return image_path.with_suffix(image_path.suffix + ".eagle-search.json")


def load_metadata_sidecar(image_path: Path) -> GeneratedImageMetadata:
    """Read prompt/source/tags without touching image bytes.

    Absence is normal.  A present malformed sidecar is rejected rather than
    silently stripping provenance during import.
    """
    path = sidecar_path(image_path)
    if not path.exists():
        return GeneratedImageMetadata()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WatcherMetadataError(f"invalid generated-image sidecar: {path}") from error
    if not isinstance(payload, dict) or set(payload) - {"prompt", "source", "tags"}:
        raise WatcherMetadataError(f"unsupported generated-image sidecar shape: {path}")
    prompt = payload.get("prompt", "")
    source = payload.get("source", "")
    tags = payload.get("tags", [])
    if not isinstance(prompt, str) or not isinstance(source, str) or not isinstance(tags, list):
        raise WatcherMetadataError(f"invalid generated-image sidecar values: {path}")
    if any(not isinstance(tag, str) for tag in tags):
        raise WatcherMetadataError(f"invalid generated-image sidecar tags: {path}")
    return GeneratedImageMetadata(prompt=prompt, source=source, tags=tuple(tags))


class GeneratedImageWatcher:
    """Startup-reconciling, single-authority polling loop.

    ``eagle_available`` is deliberately checked before claiming queue work. This
    leaves queued intents untouched while Eagle is closed, so a later poll can
    resume through the importer's durable marker reconciliation.
    """

    def __init__(
        self,
        *,
        folder: Path,
        importer: Importer,
        eagle_auto_import_enabled: Callable[[], bool],
        eagle_available: Callable[[], bool],
        stable_check: Callable[[Path], bool] = file_is_stable,
        metadata_loader: Callable[[Path], GeneratedImageMetadata] = load_metadata_sidecar,
    ) -> None:
        self.folder = Path(folder)
        self.importer = importer
        self.eagle_auto_import_enabled = eagle_auto_import_enabled
        self.eagle_available = eagle_available
        self.stable_check = stable_check
        self.metadata_loader = metadata_loader
        self._started = False
        self._seen: set[tuple[str, int, int]] = set()

    def _require_single_authority(self) -> None:
        if self.eagle_auto_import_enabled():
            raise WatcherAuthorityError(
                "generated-folder watcher refused: Eagle auto-import must be disabled or retargeted"
            )

    @staticmethod
    def _identity(path: Path) -> tuple[str, int, int] | None:
        try:
            details = path.stat()
        except OSError:
            return None
        if not stat.S_ISREG(details.st_mode):
            return None
        return (str(path.resolve()), int(details.st_size), int(details.st_mtime_ns))

    def _candidates(self) -> list[Path]:
        try:
            entries = sorted(self.folder.iterdir())
        except FileNotFoundError:
            return []
        return [path for path in entries if path.suffix.casefold() in IMAGE_EXTENSIONS]

    def _scan(self) -> ScanReport:
        self._require_single_authority()
        enqueued = 0
        skipped = 0
        errors: list[str] = []
        for path in self._candidates():
            identity = self._identity(path)
            if identity is None or identity in self._seen:
                skipped += 1
                continue
            if not self.stable_check(path):
                skipped += 1
                continue
            try:
                metadata = self.metadata_loader(path)
            except WatcherMetadataError as error:
                errors.append(str(error))
                continue
            self.importer.enqueue(
                path,
                prompt=metadata.prompt,
                source=metadata.source,
                tags=metadata.tags,
                # On restart this delegates unchanged-file dedupe to the durable
                # importer/intent store without ever overwriting provenance.
                keep_existing_metadata=True,
            )
            self._seen.add(identity)
            enqueued += 1
        return ScanReport(enqueued=enqueued, skipped=skipped, metadata_errors=tuple(errors))

    def startup_reconcile(self) -> ScanReport:
        """Mandatory first action: discover files made while the watcher was down."""
        report = self._scan()
        self._started = True
        return report

    def poll_once(self) -> ScanReport:
        """Poll for newly stable images; it cannot bypass startup reconciliation."""
        if not self._started:
            return self.startup_reconcile()
        return self._scan()

    def enqueue_explicit(self, path: Path, metadata: GeneratedImageMetadata) -> object:
        """Preserve metadata supplied directly by an image-generation workflow."""
        self._require_single_authority()
        identity = self._identity(Path(path))
        if identity is None or Path(path).suffix.casefold() not in IMAGE_EXTENSIONS:
            raise ValueError("generated import path must be a supported regular image")
        if not self.stable_check(Path(path)):
            raise ValueError("generated import path is not stable")
        result = self.importer.enqueue(
            Path(path), prompt=metadata.prompt, source=metadata.source, tags=metadata.tags,
            keep_existing_metadata=True,
        )
        self._seen.add(identity)
        return result

    async def process_pending(self, *, maximum: int = 1) -> ProcessReport:
        """Process only when Eagle is known available, never claiming work otherwise."""
        self._require_single_authority()
        if not self._started:
            self.startup_reconcile()
        if maximum < 1:
            raise ValueError("maximum must be positive")
        if not self.eagle_available():
            return ProcessReport(eagle_available=False, processed=0, state="paused")
        processed = 0
        last_state = "idle"
        for _ in range(maximum):
            if not self.eagle_available():
                return ProcessReport(eagle_available=False, processed=processed, state="paused")
            outcome = await self.importer.process_one()
            last_state = str(getattr(outcome, "state", "unknown"))
            if last_state == "idle":
                break
            processed += 1
        return ProcessReport(eagle_available=True, processed=processed, state=last_state)

    async def run(
        self,
        *,
        stop: Callable[[], bool],
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        interval_seconds: float = 1.0,
    ) -> None:
        """Small dependency-free long-running polling loop for a worker host."""
        self.startup_reconcile()
        while not stop():
            self.poll_once()
            await self.process_pending()
            await sleep(interval_seconds)
