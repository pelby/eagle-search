"""Durable, exact-once-oriented generated-image import workflow."""

from __future__ import annotations

import json
import os
import stat
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Protocol

from ..eagle_api import EagleAmbiguousCommitError
from ..ports import ImportIntent, ImportQueueStore
from .notes import import_intent_marker, render_generation_block


IMAGE_EXTENSIONS = {
    ".avif", ".gif", ".heic", ".jpeg", ".jpg", ".png", ".svg", ".tif", ".tiff", ".webp"
}
IMPORT_NAMESPACE = uuid.UUID("ec1683f0-ff39-4cab-8c58-839aaeb2f688")


class ImportAuthorityError(RuntimeError):
    """Raised when more than one component could import the generated folder."""


class ImportMetadataError(RuntimeError):
    """Raised when durable import metadata is absent or inconsistent."""


class ImportEagleApi(Protocol):
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


@dataclass(frozen=True)
class IntentMetadata:
    intent_id: str
    path: str
    prompt: str
    source: str
    tags: tuple[str, ...]
    add_attempted: bool = False


@dataclass(frozen=True)
class ImportOutcome:
    state: str
    intent_id: str = ""
    eagle_id: str = ""
    error: str = ""


class FileIntentMetadataStore:
    """Owner-only sidecars for prompt/source/tags absent from the queue port."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, intent_id: str) -> Path:
        try:
            canonical = str(uuid.UUID(intent_id))
        except (ValueError, AttributeError) as exc:
            raise ImportMetadataError("intent ID must be a UUID") from exc
        if canonical != intent_id.lower():
            raise ImportMetadataError("intent ID must be canonical UUID text")
        return self.root / f"{canonical}.json"

    def get(self, intent_id: str) -> IntentMetadata | None:
        path = self._path(intent_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        expected = {"intent_id", "path", "prompt", "source", "tags", "add_attempted"}
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ImportMetadataError(f"malformed import metadata: {path}")
        if payload["intent_id"] != intent_id or not isinstance(payload["tags"], list):
            raise ImportMetadataError(f"inconsistent import metadata: {path}")
        values = (payload["path"], payload["prompt"], payload["source"])
        if any(not isinstance(value, str) for value in values):
            raise ImportMetadataError(f"invalid import metadata strings: {path}")
        if any(not isinstance(tag, str) for tag in payload["tags"]):
            raise ImportMetadataError(f"invalid import metadata tags: {path}")
        if not isinstance(payload["add_attempted"], bool):
            raise ImportMetadataError(f"invalid import attempt flag: {path}")
        return IntentMetadata(
            intent_id=intent_id,
            path=payload["path"],
            prompt=payload["prompt"],
            source=payload["source"],
            tags=tuple(payload["tags"]),
            add_attempted=payload["add_attempted"],
        )

    def put_new(self, metadata: IntentMetadata, *, keep_existing: bool = False) -> IntentMetadata:
        existing = self.get(metadata.intent_id)
        if existing is not None:
            if keep_existing:
                return existing
            if existing != metadata:
                raise ImportMetadataError("intent metadata conflicts with the durable record")
            return existing
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.root, 0o700)
        except OSError:
            pass
        path = self._path(metadata.intent_id)
        payload = {**asdict(metadata), "tags": list(metadata.tags)}
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return metadata

    def set_attempted(self, intent_id: str, attempted: bool) -> IntentMetadata:
        metadata = self.get(intent_id)
        if metadata is None:
            raise ImportMetadataError(f"missing metadata for import intent {intent_id}")
        updated = replace(metadata, add_attempted=attempted)
        self._replace(updated)
        return updated

    def _replace(self, metadata: IntentMetadata) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self._path(metadata.intent_id)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=self.root)
        try:
            os.chmod(temporary, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    {**asdict(metadata), "tags": list(metadata.tags)},
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _default_stat_sample(path: Path) -> tuple[int, int, bool]:
    result = path.stat()
    return result.st_size, result.st_mtime_ns, stat.S_ISREG(result.st_mode)


def file_is_stable(
    path: Path,
    *,
    interval_seconds: float = 0.25,
    stat_sample: Callable[[Path], tuple[int, int, bool]] = _default_stat_sample,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Require two matching size/mtime samples of a regular file."""

    try:
        first = stat_sample(Path(path))
        if not first[2]:
            return False
        sleep(interval_seconds)
        second = stat_sample(Path(path))
    except (FileNotFoundError, OSError, StopIteration):
        return False
    return second[2] and first[:2] == second[:2]


class GeneratedImageImporter:
    """Persist intent before Eagle and reconcile every ambiguous outcome."""

    def __init__(
        self,
        *,
        queue: ImportQueueStore,
        metadata_store: FileIntentMetadataStore,
        eagle: ImportEagleApi,
        worker_id: str,
        overlapping_authority: Callable[[], bool],
        recent_limit: int = 500,
    ) -> None:
        self.queue = queue
        self.metadata_store = metadata_store
        self.eagle = eagle
        self.worker_id = worker_id
        self.overlapping_authority = overlapping_authority
        self.recent_limit = recent_limit

    def _require_single_authority(self) -> None:
        if self.overlapping_authority():
            raise ImportAuthorityError(
                "generated-folder import refused: another Eagle auto-import authority is enabled"
            )

    @staticmethod
    def _intent_id(path: Path) -> str:
        result = path.stat()
        evidence = f"{path.resolve()}\0{result.st_size}\0{result.st_mtime_ns}"
        return str(uuid.uuid5(IMPORT_NAMESPACE, evidence))

    def enqueue(
        self,
        path: Path,
        *,
        prompt: str,
        source: str,
        tags: tuple[str, ...],
        keep_existing_metadata: bool = False,
    ) -> ImportIntent:
        self._require_single_authority()
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        intent_id = self._intent_id(path)
        metadata = self.metadata_store.put_new(
            IntentMetadata(intent_id, str(path), prompt, source, tuple(tags)),
            keep_existing=keep_existing_metadata,
        )
        return self.queue.enqueue(metadata.path, metadata.intent_id)

    def reconcile_startup(
        self,
        folder: Path,
        *,
        stable_check: Callable[[Path], bool] = file_is_stable,
    ) -> list[ImportIntent]:
        self._require_single_authority()
        intents: list[ImportIntent] = []
        for path in sorted(Path(folder).iterdir()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS or not stable_check(path):
                continue
            intents.append(
                self.enqueue(
                    path,
                    prompt="",
                    source="",
                    tags=(),
                    keep_existing_metadata=True,
                )
            )
        return intents

    async def _find_by_marker(self, intent_id: str) -> str:
        marker = import_intent_marker(intent_id)
        matches: list[str] = []
        for item in await self.eagle.list_recent(limit=self.recent_limit):
            if not isinstance(item, dict):
                raise ImportMetadataError("Eagle recent-items response contains a non-object")
            eagle_id = item.get("id")
            annotation = item.get("annotation", "")
            if not isinstance(eagle_id, str) or not isinstance(annotation, str):
                raise ImportMetadataError("Eagle recent item has invalid ID or annotation")
            if marker in annotation:
                matches.append(eagle_id)
        if len(matches) > 1:
            raise ImportMetadataError(
                f"multiple Eagle items contain import intent {intent_id}; exact-once is ambiguous"
            )
        return matches[0] if matches else ""

    async def _verify_marker(self, eagle_id: str, intent_id: str) -> bool:
        item = await self.eagle.get_item(eagle_id)
        return (
            isinstance(item, dict)
            and item.get("id") == eagle_id
            and isinstance(item.get("annotation"), str)
            and import_intent_marker(intent_id) in item["annotation"]
        )

    def _acknowledge(self, intent_id: str, eagle_id: str, *, reconciled: bool) -> ImportOutcome:
        if reconciled:
            self.queue.reconcile(intent_id, eagle_id)
        self.queue.acknowledge(intent_id, eagle_id)
        return ImportOutcome(
            state="reconciled" if reconciled else "imported",
            intent_id=intent_id,
            eagle_id=eagle_id,
        )

    async def process_one(self) -> ImportOutcome:
        self._require_single_authority()
        intent = self.queue.claim(worker_id=self.worker_id)
        if intent is None:
            return ImportOutcome("idle")
        metadata = self.metadata_store.get(intent.intent_id)
        if metadata is None:
            return ImportOutcome("blocked", intent.intent_id, error="durable import metadata is missing")

        existing = await self._find_by_marker(intent.intent_id)
        if existing:
            if not await self._verify_marker(existing, intent.intent_id):
                return ImportOutcome("ambiguous", intent.intent_id, existing, "marker read-back failed")
            return self._acknowledge(intent.intent_id, existing, reconciled=True)

        if metadata.add_attempted:
            return ImportOutcome(
                "ambiguous",
                intent.intent_id,
                error="prior add outcome is unresolved; refusing a second add",
            )

        metadata = self.metadata_store.set_attempted(intent.intent_id, True)
        annotation = render_generation_block(
            prompt=metadata.prompt,
            source=metadata.source,
            tags=metadata.tags,
            intent_id=metadata.intent_id,
        )
        try:
            eagle_id = await self.eagle.add_from_path(
                path=metadata.path,
                name=Path(metadata.path).stem,
                annotation=annotation,
                source=metadata.source,
                tags=metadata.tags,
            )
        except EagleAmbiguousCommitError as exc:
            existing = await self._find_by_marker(intent.intent_id)
            if existing and await self._verify_marker(existing, intent.intent_id):
                return self._acknowledge(intent.intent_id, existing, reconciled=True)
            return ImportOutcome("ambiguous", intent.intent_id, error=str(exc))
        except Exception as exc:
            self.metadata_store.set_attempted(intent.intent_id, False)
            return ImportOutcome("failed", intent.intent_id, error=str(exc)[:500])

        if not await self._verify_marker(eagle_id, intent.intent_id):
            return ImportOutcome(
                "ambiguous",
                intent.intent_id,
                eagle_id,
                "returned Eagle ID did not preserve the import marker",
            )
        return self._acknowledge(intent.intent_id, eagle_id, reconciled=False)
