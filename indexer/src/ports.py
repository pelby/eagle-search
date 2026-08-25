"""Frozen persistence ports used by parallel Eagle Search packages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .contracts import CaptionReceiptV1


@dataclass(frozen=True)
class CaptionJob:
    eagle_id: str
    image_hash: str
    thumbnail_path: str
    claim_token: str
    attempts: int


@dataclass(frozen=True)
class ImportIntent:
    intent_id: str
    path: str
    state: str
    eagle_id: str = ""
    attempts: int = 0


class ReceiptStore(Protocol):
    def put_immutable(self, receipt: CaptionReceiptV1) -> str: ...

    def get(self, image_hash: str, receipt_id: str) -> CaptionReceiptV1 | None: ...

    def resolve_active(self, image_hash: str) -> CaptionReceiptV1 | None: ...

    def set_active(self, image_hash: str, receipt_id: str, reason: str) -> None: ...


class CaptionJobStore(Protocol):
    def claim(self, *, limit: int, worker_id: str) -> list[CaptionJob]: ...

    def mark_complete(self, eagle_id: str, claim_token: str, receipt_id: str) -> None: ...

    def mark_failed(self, eagle_id: str, claim_token: str, error: str) -> None: ...

    def release_stale(self, *, older_than_seconds: int) -> int: ...

    def counts(self) -> dict[str, int]: ...


class ImportQueueStore(Protocol):
    def enqueue(self, path: str, intent_id: str) -> ImportIntent: ...

    def claim(self, *, worker_id: str) -> ImportIntent | None: ...

    def reconcile(self, intent_id: str, eagle_id: str) -> None: ...

    def acknowledge(self, intent_id: str, eagle_id: str) -> None: ...
