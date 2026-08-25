"""Durable status and exclusive execution for background workers."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, TypeVar

from ..contracts import RunStatusV1
from .lock import FileLock, LockUnavailable


T = TypeVar("T")


class WorkerAlreadyRunning(RuntimeError):
    """Raised when a second worker attempts to own the same run."""


class RunStatusFile:
    """Atomically persist the latest versioned worker status."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def write(self, status: RunStatusV1) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(status.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def read(self) -> RunStatusV1:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.pop("contract_version", None) != 1:
            raise ValueError("unsupported or missing RunStatusV1 contract version")
        expected = set(RunStatusV1.__dataclass_fields__)
        if set(payload) != expected:
            raise ValueError("RunStatusV1 fields are incomplete or unexpected")
        return RunStatusV1(**payload)


class WorkerRuntime:
    """Run one bounded worker operation while publishing recoverable status."""

    def __init__(
        self,
        *,
        lock: FileLock,
        status: RunStatusFile,
        run_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.lock = lock
        self.status = status
        self.run_id_factory = run_id_factory or (lambda: str(uuid.uuid4()))

    def run(
        self,
        *,
        stage: str,
        total: int,
        work: Callable[[Callable[..., None]], T],
        provider: str = "",
        model: str = "",
        semantic_available: bool = False,
    ) -> T:
        if total < 0:
            raise ValueError("total cannot be negative")
        run_id = self.run_id_factory()
        current = RunStatusV1(
            run_id=run_id,
            state="running",
            stage=stage,
            total=total,
            completed=0,
            pending=total,
            failed=0,
            provider=provider,
            model=model,
            semantic_available=semantic_available,
        )
        try:
            self.lock.acquire()
        except LockUnavailable as exc:
            raise WorkerAlreadyRunning(str(exc)) from exc
        try:
            self.status.write(current)

            def report(*, completed: int, pending: int, failed: int, last_error: str = "") -> None:
                nonlocal current
                if min(completed, pending, failed) < 0:
                    raise ValueError("worker counts cannot be negative")
                if completed + pending + failed > total:
                    raise ValueError("worker counts cannot exceed total")
                current = RunStatusV1(
                    run_id=run_id,
                    state="running",
                    stage=stage,
                    total=total,
                    completed=completed,
                    pending=pending,
                    failed=failed,
                    provider=provider,
                    model=model,
                    semantic_available=semantic_available,
                    last_error=last_error[:500],
                )
                self.status.write(current)

            result = work(report)
            current = RunStatusV1(
                **{
                    **current.__dict__,
                    "state": "complete",
                }
            )
            self.status.write(current)
            return result
        except Exception as exc:
            failed_status = RunStatusV1(
                **{
                    **current.__dict__,
                    "state": "failed",
                    "last_error": str(exc)[:500],
                }
            )
            self.status.write(failed_status)
            raise
        finally:
            self.lock.release()
