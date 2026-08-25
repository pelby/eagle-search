"""Small non-blocking, process-safe file lock."""

from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import IO


class LockUnavailable(RuntimeError):
    """Raised when another process owns a worker lock."""


class FileLock:
    """Own an advisory lock for the lifetime of an open file descriptor."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._handle: IO[str] | None = None

    def acquire(self) -> "FileLock":
        if self._handle is not None:
            raise RuntimeError(f"lock is already held by this object: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise LockUnavailable(f"lock is already held: {self.path}") from exc
        handle.seek(0)
        handle.truncate()
        json.dump(
            {
                "pid": os.getpid(),
                "acquired_at": datetime.now(timezone.utc).isoformat(),
            },
            handle,
            sort_keys=True,
        )
        handle.flush()
        os.fsync(handle.fileno())
        self._handle = handle
        return self

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def __enter__(self) -> "FileLock":
        return self.acquire()

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.release()
