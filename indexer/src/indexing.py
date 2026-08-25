"""Shared indexing seam: Eagle metadata -> durable captions -> searchable projections."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Protocol

from . import db
from .captioning.codex_cli import (
    CaptionGlobalFailure,
    CodexCliCaptionProvider,
    image_sha256,
)
from .contracts import CaptionReceiptV1, CaptionResultV1
from .eagle_api import EagleApiError
from .persistence.jobs import SQLiteCaptionJobStore
from .persistence.receipts import FileReceiptStore, export_legacy_receipts
from .retrieval.backfill import backfill_embeddings
from .retrieval.embeddings import Embedder
from .worker.caption_state import CaptionStateOrchestrator, GlobalCaptionFailure
from .worker.lock import FileLock
from .worker.runtime import RunStatusFile, WorkerRuntime


class EagleIndexApi(Protocol):
    async def is_running(self) -> bool: ...

    async def list_items(self, *, limit: int = 10_000) -> list[dict[str, Any]]: ...

    async def get_folder_map(self) -> dict[str, str]: ...

    async def get_thumbnail_path(self, item_id: str) -> str | None: ...


class CaptionProvider(Protocol):
    def caption(
        self,
        image_path: Path,
        *,
        model: str,
        effort: str,
        receipt_store: FileReceiptStore,
    ) -> CaptionReceiptV1: ...


@dataclass(frozen=True)
class EmbeddedMetadata:
    prompt: str = ""
    description: str = ""


@dataclass(frozen=True)
class IndexOutcome:
    discovered: int
    updated: int
    unavailable: int
    removed: int
    queued: int
    captioned: int
    failed: int
    embedded: int
    semantic_error: str = ""
    legacy_exported: int = 0
    stale_cleanup_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"contract_version": 1, "ok": self.failed == 0, **self.__dict__}


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def extract_embedded_metadata(path: Path) -> EmbeddedMetadata:
    """Read common PNG/JPEG text fields without making Pillow a test dependency."""

    try:
        from PIL import Image  # type: ignore
    except ModuleNotFoundError:
        return EmbeddedMetadata()
    try:
        with Image.open(path) as image:
            raw = {str(key).casefold(): str(value) for key, value in image.info.items() if isinstance(value, (str, int, float))}
    except (OSError, ValueError):
        return EmbeddedMetadata()
    prompt = ""
    for key in ("prompt", "generation_prompt", "parameters", "comment"):
        if raw.get(key, "").strip():
            prompt = raw[key].strip()
            if key == "parameters" and "Negative prompt:" in prompt:
                prompt = prompt.split("Negative prompt:", 1)[0].strip()
            break
    description = ""
    for key in ("description", "caption", "ai_description"):
        if raw.get(key, "").strip():
            description = raw[key].strip()
            break
    return EmbeddedMetadata(prompt=prompt[:20_000], description=description[:20_000])


def preflight_legacy_export(database_path: Path, receipts_root: Path) -> dict[str, int]:
    """Export raw v1 descriptions before any migration transaction runs."""

    if not database_path.is_file():
        return {"source_rows": 0, "distinct_eagle_ids": 0, "receipt_count": 0}
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        if int(connection.execute("PRAGMA user_version").fetchone()[0]) != 0:
            return {"source_rows": 0, "distinct_eagle_ids": 0, "receipt_count": 0}
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='images'"
        ).fetchone()
        if table is None:
            return {"source_rows": 0, "distinct_eagle_ids": 0, "receipt_count": 0}
        count = int(
            connection.execute(
                "SELECT count(*) FROM images WHERE ai_description <> ''"
            ).fetchone()[0]
        )
        if count == 0:
            return {"source_rows": 0, "distinct_eagle_ids": 0, "receipt_count": 0}
        report = export_legacy_receipts(connection, FileReceiptStore(receipts_root))
        if report["source_rows"] != count or report["receipt_count"] != count:
            raise RuntimeError("legacy receipt preflight did not preserve every description")
        return report
    finally:
        connection.close()


def _legacy_receipt_refs(path: Path) -> dict[str, tuple[str, str]]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("legacy receipt manifest is unreadable") from error
    if not isinstance(payload, dict) or payload.get("manifest_version") != 1:
        raise RuntimeError("legacy receipt manifest version is unsupported")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError("legacy receipt manifest entries are missing")
    refs: dict[str, tuple[str, str]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"eagle_id", "image_hash", "receipt_id"}:
            raise RuntimeError("legacy receipt manifest entry is malformed")
        eagle_id = str(entry["eagle_id"])
        if not eagle_id or eagle_id in refs:
            raise RuntimeError("legacy receipt manifest Eagle IDs must be unique")
        refs[eagle_id] = (str(entry["image_hash"]), str(entry["receipt_id"]))
    return refs


def _plan_stale_cleanup(
    existing_ids: set[str],
    current_ids: set[str],
    *,
    list_limit: int,
) -> tuple[list[str], str]:
    """Delete derived rows only when Eagle's capped list proves completeness."""

    if not existing_ids:
        return [], ""
    if not current_ids:
        return [], "Eagle returned no items; stale-row cleanup was skipped"
    if len(current_ids) >= list_limit:
        return [], f"Eagle returned the {list_limit}-item API limit; stale-row cleanup was skipped"
    return sorted(existing_ids - current_ids), ""


def _derive_image_path(thumbnail_path: Path, extension: str) -> Path:
    suffix = "_thumbnail.png"
    name = thumbnail_path.name
    if name.endswith(suffix):
        return thumbnail_path.with_name(name[: -len(suffix)] + f".{extension or 'png'}")
    return thumbnail_path


def _embedded_receipt(
    *,
    image_hash: str,
    description: str,
) -> CaptionReceiptV1:
    result = CaptionResultV1.from_dict(
        {
            "contract_version": 1,
            "image_type": "embedded metadata",
            "diagram_types": [],
            "subjects": [],
            "visual_style": [],
            "colours": [],
            "layout": [],
            "visible_text": [],
            "search_terms": [],
            "summary": description,
            "uncertainties": ["embedded description provenance is unverified"],
        }
    )
    return CaptionReceiptV1.create(
        image_hash=image_hash,
        caption_result=result,
        provider="embedded-metadata",
        model="none",
        effort="none",
        prompt_version="embedded-v1",
        created_at=_now(),
        source="legacy",
    )


def _project_receipt(
    connection: sqlite3.Connection,
    eagle_id: str,
    receipt: CaptionReceiptV1,
) -> None:
    caption = receipt.caption_result
    visible = " ".join(entry.text for entry in caption.visible_text)
    aliases = " ".join(caption.search_terms)
    with connection:
        connection.execute(
            """UPDATE images SET ai_description=?, visual_caption=?, visible_text=?,
               visual_search_terms=?, visual_search_text=?, search_content_hash=?,
               caption_state='complete', caption_last_error='', caption_updated_at=?,
               active_receipt_id=?, active_receipt_hash=? WHERE eagle_id=?""",
            (
                caption.summary,
                caption.summary,
                visible,
                aliases,
                receipt.search_text,
                db.content_hash(receipt.search_text),
                _now(),
                receipt.receipt_id,
                receipt.receipt_id,
                eagle_id,
            ),
        )


async def index_library(
    *,
    home: Path,
    eagle: EagleIndexApi,
    embedder: Embedder,
    caption_provider: CaptionProvider | None = None,
    model: str = "gpt-5.6-sol",
    effort: str = "low",
    max_items: int | None = None,
    metadata_reader: Callable[[Path], EmbeddedMetadata] = extract_embedded_metadata,
) -> IndexOutcome:
    """Incrementally update local state. Notes are never mutated by this command."""

    home = Path(home)
    if max_items is not None and max_items < 1:
        raise ValueError("max_items must be positive")
    if not await eagle.is_running():
        raise RuntimeError("Eagle is not running")
    home.mkdir(parents=True, exist_ok=True)
    receipts_root = home / "captions"
    legacy = preflight_legacy_export(home / "db.sqlite", receipts_root)
    connection = db.init_db(home / "db.sqlite")
    receipt_store = FileReceiptStore(receipts_root)
    legacy_refs = _legacy_receipt_refs(receipts_root / "legacy-manifest-v1.json")
    job_store = SQLiteCaptionJobStore(connection)
    thumbnails = home / "thumbnails"
    thumbnails.mkdir(parents=True, exist_ok=True)
    provider = caption_provider or CodexCliCaptionProvider()
    list_limit = 10_000
    items = await eagle.list_items(limit=list_limit)
    folder_map = await eagle.get_folder_map()
    existing = {
        str(row["eagle_id"]): dict(row)
        for row in connection.execute("SELECT * FROM images")
    }
    selected = items[:max_items] if max_items is not None else items
    updated = 0
    removed = 0
    stale_cleanup_reason = ""
    queued = 0

    async def prepare(
        item: dict[str, Any],
    ) -> tuple[dict[str, Any], Path, Path, str, EmbeddedMetadata] | None:
        eagle_id = str(item["id"])
        destination = thumbnails / f"{eagle_id}.png"
        previous = existing.get(eagle_id, {})
        source_mtime = int(item.get("mtime", 0) or 0)
        source_size = int(item.get("size", 0) or 0)
        previous_mtime = int(previous.get("source_mtime", 0) or 0)
        previous_size = int(previous.get("source_size", 0) or 0)
        fingerprint_missing = bool(previous) and bool(source_mtime or source_size) and not (
            previous_mtime or previous_size
        )
        source_changed = bool(previous) and (
            fingerprint_missing
            or bool(previous_mtime and source_mtime and previous_mtime != source_mtime)
            or bool(previous_size and source_size and previous_size != source_size)
        )
        if destination.is_file() and previous and not source_changed:
            source = destination
        else:
            source_value: str | None = None
            for attempt in range(3):
                try:
                    source_value = await eagle.get_thumbnail_path(eagle_id)
                    break
                except EagleApiError:
                    if attempt == 2:
                        raise
                    await asyncio.sleep(0.15 * (attempt + 1))
            if not source_value:
                return None
            source = Path(source_value)
            if not source.is_file():
                return None
            if source != destination:
                shutil.copy2(source, destination)
        digest = image_sha256(destination)
        previous_hash = str(previous.get("image_hash", "") or "")
        changed = bool(previous_hash) and previous_hash != digest
        stored_image_path = str(previous.get("image_path", "") or "")
        derived = _derive_image_path(source, str(item.get("ext", "png")))
        image_path = Path(stored_image_path) if stored_image_path else derived
        prior_description = str(previous.get("ai_description", "") or "")
        should_read_metadata = changed or not prior_description
        metadata_path = image_path if image_path.is_file() else source
        metadata = metadata_reader(metadata_path) if should_read_metadata else EmbeddedMetadata()
        return item, destination, image_path, digest, metadata

    # Eagle's desktop API is local but not designed for high request fan-out.
    # Two readers keep discovery fast without destabilising its thumbnail route.
    semaphore = asyncio.Semaphore(2)

    async def bounded(item: dict[str, Any]):
        async with semaphore:
            return await prepare(item)

    prepared = await asyncio.gather(*(bounded(item) for item in selected))
    unavailable = sum(value is None for value in prepared)
    for value in prepared:
        if value is None:
            continue
        item, thumbnail, image_path, digest, metadata = value
        eagle_id = str(item["id"])
        previous = existing.get(eagle_id, {})
        prior_description = str(previous.get("ai_description", "") or "")
        folder_names = [folder_map.get(str(folder_id), "") for folder_id in item.get("folders", [])]
        changed = bool(previous.get("image_hash")) and previous.get("image_hash") != digest
        prior_caption_state = str(previous.get("caption_state", ""))
        caption_state = "pending" if changed else prior_caption_state or ("complete" if prior_description else "pending")
        if changed:
            with connection:
                connection.execute(
                    "DELETE FROM image_embeddings WHERE eagle_id=?",
                    (eagle_id,),
                )
        record = {
            "eagle_id": eagle_id,
            "name": str(item.get("name", "")),
            "tags": ", ".join(str(tag) for tag in item.get("tags", [])),
            "annotation": str(item.get("annotation", "")),
            "human_notes": str(item.get("annotation", "")),
            "ai_description": metadata.description if changed else prior_description or metadata.description,
            "embedded_description": metadata.description if changed else metadata.description or str(previous.get("embedded_description", "")),
            "generation_prompt": metadata.prompt if changed else metadata.prompt or str(previous.get("generation_prompt", "")),
            "visual_caption": metadata.description if changed else prior_description or metadata.description,
            "visible_text": "" if changed else str(previous.get("visible_text", "")),
            "visual_search_terms": "" if changed else str(previous.get("visual_search_terms", "")),
            "visual_search_text": "" if changed else str(previous.get("visual_search_text", "")),
            "thumbnail_path": str(thumbnail),
            "image_path": str(image_path),
            "folder_name": ", ".join(name for name in folder_names if name),
            "ext": str(item.get("ext", "")),
            "width": int(item.get("width", 0) or 0),
            "height": int(item.get("height", 0) or 0),
            "created_at": int(item.get("btime", 0) or 0),
            "image_hash": digest,
            "caption_state": caption_state,
            "caption_attempts": 0 if changed else int(previous.get("caption_attempts", 0) or 0),
            "caption_last_error": "" if changed else str(previous.get("caption_last_error", "")),
            "caption_updated_at": "" if changed else str(previous.get("caption_updated_at", "")),
            "active_receipt_id": "" if changed else str(previous.get("active_receipt_id", "")),
            "active_receipt_hash": "" if changed else str(previous.get("active_receipt_hash", "")),
            "source_mtime": int(item.get("mtime", 0) or 0),
            "source_size": int(item.get("size", 0) or 0),
        }
        db.upsert_image(connection, record)
        updated += 1
        unchanged_legacy = not previous.get("image_hash") and bool(prior_description)
        needs_caption = changed or (not prior_description and caption_state != "complete")
        if unchanged_legacy:
            reference = legacy_refs.get(eagle_id)
            if reference is None:
                raise RuntimeError(f"legacy receipt manifest is missing Eagle item {eagle_id}")
            legacy_receipt = receipt_store.get(*reference)
            if legacy_receipt is None:
                raise RuntimeError(f"legacy receipt is missing for Eagle item {eagle_id}")
            adopted = CaptionReceiptV1.create(
                image_hash=digest,
                caption_result=legacy_receipt.caption_result,
                provider=legacy_receipt.provider,
                model=legacy_receipt.model,
                effort=legacy_receipt.effort,
                prompt_version=legacy_receipt.prompt_version,
                created_at=legacy_receipt.created_at,
                source="legacy",
            )
            receipt_store.put_immutable(adopted)
            receipt_store.set_active(digest, adopted.receipt_id, "legacy receipt adopted to current image hash")
            _project_receipt(connection, eagle_id, adopted)
            needs_caption = False
        if metadata.description and (changed or not prior_description):
            receipt = _embedded_receipt(image_hash=digest, description=metadata.description)
            receipt_store.put_immutable(receipt)
            receipt_store.set_active(digest, receipt.receipt_id, "embedded metadata")
            _project_receipt(connection, eagle_id, receipt)
            needs_caption = False
        if needs_caption and not unchanged_legacy:
            job_store.enqueue(eagle_id, digest, str(thumbnail))
            queued += 1

    if max_items is None:
        current_ids = {str(item["id"]) for item in items}
        stale_ids, stale_cleanup_reason = _plan_stale_cleanup(
            set(existing),
            current_ids,
            list_limit=list_limit,
        )
        if stale_ids:
            with connection:
                connection.executemany(
                    "DELETE FROM images WHERE eagle_id=?",
                    ((eagle_id,) for eagle_id in stale_ids),
                )
            removed = len(stale_ids)

    counts = job_store.counts()
    total = counts.get("pending", 0) + counts.get("failed", 0)
    captioned = 0
    failed = 0
    runtime = WorkerRuntime(
        lock=FileLock(home / "worker.lock"),
        status=RunStatusFile(home / "run-status.json"),
    )

    def work(report):
        nonlocal captioned, failed
        orchestrator = CaptionStateOrchestrator(job_store, worker_id="index-worker")

        def process(job):
            try:
                receipt = provider.caption(
                    Path(job.thumbnail_path),
                    model=model,
                    effort=effort,
                    receipt_store=receipt_store,
                )
            except CaptionGlobalFailure as error:
                raise GlobalCaptionFailure(str(error)) from error
            receipt_store.set_active(job.image_hash, receipt.receipt_id, "index worker")
            _project_receipt(connection, job.eagle_id, receipt)
            return receipt.receipt_id

        while captioned + failed < total:
            outcome = orchestrator.run_batch(process, limit=1)
            if outcome.claimed == 0:
                break
            captioned += outcome.completed
            failed += outcome.failed
            report(
                completed=captioned,
                failed=failed,
                pending=max(0, total - captioned - failed),
                last_error=outcome.last_error,
            )
            if outcome.stopped_globally:
                break

    runtime.run(
        stage="caption",
        total=total,
        work=work,
        provider="codex-cli",
        model=model,
        semantic_available=False,
    )
    semantic = backfill_embeddings(connection, embedder, batch_size=32, limit=10_000)
    final_status = RunStatusFile(home / "run-status.json")
    status = final_status.read()
    final_status.write(
        type(status)(
            **{
                **status.__dict__,
                "semantic_available": semantic["error"] is None,
                "last_error": status.last_error or str(semantic["error"] or "")[:500],
            }
        )
    )
    connection.close()
    return IndexOutcome(
        discovered=len(items),
        updated=updated,
        unavailable=unavailable,
        removed=removed,
        queued=queued,
        captioned=captioned,
        failed=failed,
        embedded=int(semantic["embedded"]),
        semantic_error=str(semantic["error"] or ""),
        legacy_exported=int(legacy["receipt_count"]),
        stale_cleanup_reason=stale_cleanup_reason,
    )
