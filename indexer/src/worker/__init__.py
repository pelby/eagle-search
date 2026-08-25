"""Process-safe worker primitives for Eagle Search."""

from .caption_state import CaptionBatchOutcome, CaptionStateOrchestrator, GlobalCaptionFailure
from .lock import FileLock, LockUnavailable
from .runtime import RunStatusFile, WorkerAlreadyRunning, WorkerRuntime

__all__ = [
    "CaptionBatchOutcome",
    "CaptionStateOrchestrator",
    "FileLock",
    "GlobalCaptionFailure",
    "LockUnavailable",
    "RunStatusFile",
    "WorkerAlreadyRunning",
    "WorkerRuntime",
]
