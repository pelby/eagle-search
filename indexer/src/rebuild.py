"""Rebuild disposable SQLite search state from durable local evidence.

Receipts are authoritative. A strict managed-Notes parser is available only
behind an explicit gate and is marked as degraded provenance in the projection.
This module never calls a caption model and never mutates Eagle.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
from typing import Any, Iterable, Mapping, Protocol

from . import db
from .contracts import CaptionReceiptV1, SHA256_ID_RE
from .eagle.notes import CAPTION_END, CAPTION_START
from .persistence.receipts import FileReceiptStore
from .captioning.codex_cli import image_sha256
from .indexing import _derive_image_path, extract_embedded_metadata


class ManagedNotesFallbackError(ValueError):
    """A managed caption block exists but cannot be trusted as structured data."""


@dataclass(frozen=True)
class ManagedCaptionProjection:
    summary: str
    visible_text: tuple[str, ...]
    search_terms: tuple[str, ...]
    receipt_id: str

    @property
    def search_text(self) -> str:
        return " ".join((self.summary, *self.visible_text, *self.search_terms)).strip()


@dataclass(frozen=True)
class RebuildRecord:
    eagle_id: str
    name: str
    tags: str = ""
    annotation: str = ""
    generation_prompt: str = ""
    embedded_description: str = ""
    thumbnail_path: str = ""
    image_path: str = ""
    folder_name: str = ""
    ext: str = ""
    width: int = 0
    height: int = 0
    created_at: int = 0
    image_hash: str = ""


@dataclass(frozen=True)
class RebuildOutcome:
    total: int
    receipts: int
    legacy_receipts: int
    notes_fallbacks: int
    pending: int
    database_path: str

    def to_dict(self) -> dict[str, Any]:
        return {"contract_version": 1, "ok": True, **self.__dict__}


class EagleRebuildApi(Protocol):
    async def list_items(self, *, limit: int = 10_000) -> list[dict[str, Any]]: ...

    async def get_folder_map(self) -> dict[str, str]: ...

    async def get_thumbnail_path(self, item_id: str) -> str | None: ...


_BLOCK_PATTERN = re.compile(
    rf"{re.escape(CAPTION_START)}\n"
    r"\*\*AI visual description:\*\* (?P<summary>.*?)\n"
    r"\*\*Visible text:\*\* (?P<visible>.*?)\n"
    r"\*\*Search aliases:\*\* (?P<aliases>.*?)\n"
    r"\*\*Receipt:\*\* `(?P<receipt>sha256:[a-f0-9]{64})`\n"
    rf"{re.escape(CAPTION_END)}",
    re.DOTALL,
)


def parse_managed_caption(annotation: str) -> ManagedCaptionProjection | None:
    """Extract only the exact v1 managed block; ignore all human prose."""

    if not isinstance(annotation, str):
        raise ManagedNotesFallbackError("annotation must be text")
    marker_prefix = "<!-- eagle-search:caption:"
    has_marker = marker_prefix in annotation
    if not has_marker:
        return None
    if annotation.count(CAPTION_START) != 1 or annotation.count(CAPTION_END) != 1:
        raise ManagedNotesFallbackError("managed caption markers must form exactly one v1 block")
    start = annotation.index(CAPTION_START)
    end = annotation.index(CAPTION_END, start) + len(CAPTION_END)
    block = annotation[start:end]
    match = _BLOCK_PATTERN.fullmatch(block)
    if match is None:
        raise ManagedNotesFallbackError("managed caption block has an unsupported shape")
    summary = match.group("summary").strip()
    if not summary or len(summary) > 20_000:
        raise ManagedNotesFallbackError("managed caption summary is blank or too long")
    visible_raw = match.group("visible").strip()
    aliases_raw = match.group("aliases").strip()
    visible = () if visible_raw == "(none)" else tuple(part.strip() for part in visible_raw.split("|") if part.strip())
    aliases = () if aliases_raw == "(none)" else tuple(part.strip() for part in aliases_raw.split(",") if part.strip())
    if len(visible) > 40 or len(aliases) > 12:
        raise ManagedNotesFallbackError("managed caption block exceeds v1 field limits")
    receipt_id = match.group("receipt")
    if not SHA256_ID_RE.fullmatch(receipt_id):
        raise ManagedNotesFallbackError("managed caption receipt ID is invalid")
    return ManagedCaptionProjection(summary, visible, aliases, receipt_id)


def load_legacy_manifest(path: Path) -> dict[str, Any]:
    """Load and validate the minimal legacy-ID-to-receipt mapping."""

    if not Path(path).is_file():
        return {"manifest_version": 1, "entries": []}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("legacy receipt manifest is unreadable") from error
    if not isinstance(payload, dict) or payload.get("manifest_version") != 1:
        raise ValueError("legacy receipt manifest version is unsupported")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("legacy receipt manifest entries must be an array")
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"eagle_id", "image_hash", "receipt_id"}:
            raise ValueError("legacy receipt manifest entry has an unsupported shape")
        eagle_id = entry["eagle_id"]
        if not isinstance(eagle_id, str) or not eagle_id or eagle_id in seen:
            raise ValueError("legacy receipt manifest Eagle IDs must be unique text")
        if not SHA256_ID_RE.fullmatch(str(entry["image_hash"])) or not SHA256_ID_RE.fullmatch(str(entry["receipt_id"])):
            raise ValueError("legacy receipt manifest contains an invalid digest")
        seen.add(eagle_id)
    return payload


def _legacy_index(manifest: Mapping[str, Any] | None) -> dict[str, tuple[str, str]]:
    if manifest is None:
        return {}
    entries = manifest.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("legacy receipt manifest entries must be an array")
    result: dict[str, tuple[str, str]] = {}
    for raw in entries:
        if not isinstance(raw, Mapping):
            raise ValueError("legacy receipt manifest entry must be an object")
        eagle_id = str(raw.get("eagle_id", ""))
        image_hash = str(raw.get("image_hash", ""))
        receipt_id = str(raw.get("receipt_id", ""))
        if not eagle_id or eagle_id in result:
            raise ValueError("legacy receipt manifest Eagle IDs must be unique")
        if not SHA256_ID_RE.fullmatch(image_hash) or not SHA256_ID_RE.fullmatch(receipt_id):
            raise ValueError("legacy receipt manifest contains an invalid digest")
        result[eagle_id] = (image_hash, receipt_id)
    return result


def _base_projection(record: RebuildRecord) -> dict[str, Any]:
    if not record.eagle_id or not isinstance(record.eagle_id, str):
        raise ValueError("rebuild record requires an Eagle ID")
    return {
        "eagle_id": record.eagle_id,
        "name": record.name,
        "tags": record.tags,
        "annotation": record.annotation,
        "human_notes": record.annotation,
        "generation_prompt": record.generation_prompt,
        "embedded_description": record.embedded_description,
        "thumbnail_path": record.thumbnail_path,
        "image_path": record.image_path,
        "folder_name": record.folder_name,
        "ext": record.ext,
        "width": record.width,
        "height": record.height,
        "created_at": record.created_at,
        "image_hash": record.image_hash,
        "caption_state": "pending",
    }


def _project_receipt(values: dict[str, Any], receipt: CaptionReceiptV1) -> None:
    caption = receipt.caption_result
    values.update(
        {
            "ai_description": caption.summary,
            "visual_caption": caption.summary,
            "visible_text": " | ".join(item.text for item in caption.visible_text),
            "visual_search_terms": ", ".join(caption.search_terms),
            "visual_search_text": receipt.search_text,
            "caption_state": "complete",
            "active_receipt_id": receipt.receipt_id,
            "active_receipt_hash": receipt.receipt_id,
        }
    )


def _project_notes(values: dict[str, Any], projection: ManagedCaptionProjection) -> None:
    values.update(
        {
            "ai_description": projection.summary,
            "visual_caption": projection.summary,
            "visible_text": " | ".join(projection.visible_text),
            "visual_search_terms": ", ".join(projection.search_terms),
            "visual_search_text": projection.search_text,
            "caption_state": "degraded-notes",
            "caption_last_error": "rebuilt from managed Notes because immutable receipt was unavailable",
            "active_receipt_id": projection.receipt_id,
            "active_receipt_hash": "",
        }
    )


def rebuild_database(
    destination: Path,
    *,
    records: Iterable[RebuildRecord],
    receipt_store: FileReceiptStore,
    legacy_manifest: Mapping[str, Any] | None = None,
    allow_notes_fallback: bool = False,
) -> RebuildOutcome:
    """Create a new derived database atomically; never overwrite an existing file."""

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"rebuild destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.rebuild-", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    connection: sqlite3.Connection | None = None
    receipts = legacy_receipts = notes_fallbacks = pending = total = 0
    legacy = _legacy_index(legacy_manifest)
    try:
        connection = db.init_db(temporary)
        for record in records:
            total += 1
            values = _base_projection(record)
            receipt = receipt_store.resolve_active(record.image_hash) if record.image_hash else None
            used_legacy = False
            if receipt is None and record.eagle_id in legacy:
                legacy_hash, receipt_id = legacy[record.eagle_id]
                receipt = receipt_store.get(legacy_hash, receipt_id)
                if receipt is None:
                    raise RuntimeError(f"legacy manifest points to a missing receipt for {record.eagle_id}")
                used_legacy = True
            if receipt is not None:
                _project_receipt(values, receipt)
                receipts += 1
                legacy_receipts += int(used_legacy)
            elif allow_notes_fallback:
                projection = parse_managed_caption(record.annotation)
                if projection is None:
                    pending += 1
                else:
                    _project_notes(values, projection)
                    notes_fallbacks += 1
            else:
                pending += 1
            db.upsert_image(connection, values)
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise sqlite3.DatabaseError(f"rebuilt database failed integrity check: {integrity}")
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.close()
        connection = None
        os.replace(temporary, destination)
    except BaseException:
        if connection is not None:
            connection.close()
        temporary.unlink(missing_ok=True)
        Path(f"{temporary}-wal").unlink(missing_ok=True)
        Path(f"{temporary}-shm").unlink(missing_ok=True)
        raise
    return RebuildOutcome(total, receipts, legacy_receipts, notes_fallbacks, pending, str(destination))


async def snapshot_eagle_records(
    eagle: EagleRebuildApi,
    *,
    thumbnail_root: Path,
    limit: int = 10_000,
) -> list[RebuildRecord]:
    """Read current Eagle metadata and cache thumbnails without mutating Eagle."""

    if not 1 <= limit <= 10_000:
        raise ValueError("rebuild item limit must be between 1 and 10000")
    thumbnail_root = Path(thumbnail_root)
    thumbnail_root.mkdir(parents=True, exist_ok=True)
    items = await eagle.list_items(limit=limit)
    folder_map = await eagle.get_folder_map()
    records: list[RebuildRecord] = []
    for item in items:
        eagle_id = str(item["id"])
        source_value = await eagle.get_thumbnail_path(eagle_id)
        if not source_value:
            continue
        source = Path(source_value)
        if not source.is_file():
            continue
        cached = thumbnail_root / f"{eagle_id}.png"
        shutil.copy2(source, cached)
        original = _derive_image_path(source, str(item.get("ext", "png")))
        image_path = original if original.is_file() else source
        embedded = extract_embedded_metadata(image_path)
        folders = ", ".join(
            folder_map.get(str(folder_id), "")
            for folder_id in item.get("folders", [])
            if folder_map.get(str(folder_id), "")
        )
        records.append(
            RebuildRecord(
                eagle_id=eagle_id,
                name=str(item.get("name", "")),
                tags=", ".join(str(tag) for tag in item.get("tags", [])),
                annotation=str(item.get("annotation", "")),
                generation_prompt=embedded.prompt,
                embedded_description=embedded.description,
                thumbnail_path=str(cached),
                image_path=str(image_path),
                folder_name=folders,
                ext=str(item.get("ext", "")),
                width=int(item.get("width", 0) or 0),
                height=int(item.get("height", 0) or 0),
                created_at=int(item.get("btime", 0) or 0),
                image_hash=image_sha256(cached),
            )
        )
    return records
