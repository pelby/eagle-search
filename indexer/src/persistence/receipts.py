"""Create-only caption receipt files and legacy receipt export."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any

from ..contracts import CaptionReceiptV1, CaptionResultV1


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


class FileReceiptStore:
    """Filesystem implementation of ReceiptStore; tests must pass an explicit root."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _directory(self, image_hash: str) -> Path:
        return self.root / image_hash

    def _path(self, image_hash: str, receipt_id: str) -> Path:
        return self._directory(image_hash) / f"{receipt_id}.json"

    def put_immutable(self, receipt: CaptionReceiptV1) -> str:
        path = self._path(receipt.image_hash, receipt.receipt_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = receipt.to_dict()
        if path.exists():
            existing = CaptionReceiptV1.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if existing.to_dict() != payload:
                raise RuntimeError("immutable receipt path contains different content")
            return receipt.receipt_id
        # O_EXCL gives create-only semantics even if two workers race.
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            return self.put_immutable(receipt)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        return receipt.receipt_id

    def get(self, image_hash: str, receipt_id: str) -> CaptionReceiptV1 | None:
        path = self._path(image_hash, receipt_id)
        if not path.exists():
            return None
        return CaptionReceiptV1.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def resolve_active(self, image_hash: str) -> CaptionReceiptV1 | None:
        directory = self._directory(image_hash)
        pointer = directory / "active.json"
        if pointer.exists():
            try:
                receipt_id = str(json.loads(pointer.read_text(encoding="utf-8"))["receipt_id"])
                resolved = self.get(image_hash, receipt_id)
                if resolved is not None:
                    return resolved
            except (KeyError, OSError, ValueError, json.JSONDecodeError):
                pass
        # Missing/corrupt pointer recovery is deterministic: latest immutable
        # receipt timestamp, then receipt id. The caller can persist its reason.
        receipts = []
        if directory.exists():
            for path in directory.glob("sha256:*.json"):
                try:
                    receipts.append(CaptionReceiptV1.from_dict(json.loads(path.read_text(encoding="utf-8"))))
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
        return max(receipts, key=lambda item: (item.created_at, item.receipt_id), default=None)

    def set_active(self, image_hash: str, receipt_id: str, reason: str) -> None:
        if self.get(image_hash, receipt_id) is None:
            raise KeyError("cannot activate a missing receipt")
        _atomic_json(self._directory(image_hash) / "active.json", {
            "receipt_id": receipt_id, "reason": reason,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })


def _legacy_hash(row: sqlite3.Row) -> str:
    existing = str(row["image_hash"] or "") if "image_hash" in row.keys() else ""
    if existing.startswith("sha256:") and len(existing) == 71:
        return existing
    evidence = "\0".join((str(row["eagle_id"]), str(row["thumbnail_path"]), str(row["ai_description"])))
    return "sha256:" + sha256(evidence.encode("utf-8")).hexdigest()


def export_legacy_receipts(connection: sqlite3.Connection, store: FileReceiptStore) -> dict[str, int]:
    """Export every nonblank v1 description before declaring SQLite disposable."""
    rows = connection.execute(
        "SELECT eagle_id, thumbnail_path, ai_description, indexed_at, image_hash FROM images WHERE ai_description <> '' ORDER BY eagle_id"
    ).fetchall()
    ids: set[str] = set()
    receipt_ids: set[str] = set()
    for row in rows:
        text = str(row["ai_description"])
        caption = CaptionResultV1.from_dict({
            "contract_version": 1, "image_type": "legacy", "diagram_types": [], "subjects": [],
            "visual_style": [], "colours": [], "layout": [], "visible_text": [], "search_terms": [],
            "summary": text, "uncertainties": ["legacy provider and prompt are unverified"],
        })
        receipt = CaptionReceiptV1.create(
            image_hash=_legacy_hash(row), caption_result=caption, provider="legacy-unverified",
            model="unknown", effort="unknown", prompt_version="legacy-v1",
            created_at=str(row["indexed_at"] or "1970-01-01T00:00:00+00:00"), source="legacy",
        )
        store.put_immutable(receipt)
        store.set_active(receipt.image_hash, receipt.receipt_id, "legacy export")
        ids.add(str(row["eagle_id"])); receipt_ids.add(receipt.receipt_id)
    report = {"source_rows": len(rows), "distinct_eagle_ids": len(ids), "receipt_count": len(receipt_ids)}
    if len(rows) != len(ids) or len(ids) != len(receipt_ids):
        raise RuntimeError("legacy receipt export count mismatch")
    _atomic_json(store.root / "legacy-manifest-v1.json", report)
    return report
