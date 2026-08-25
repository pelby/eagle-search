"""Retry-safe orchestration over the frozen CaptionJobStore port."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..contracts import SHA256_ID_RE
from ..ports import CaptionJob, CaptionJobStore


class GlobalCaptionFailure(RuntimeError):
    """A provider-wide auth/quota failure that must stop the batch."""


@dataclass(frozen=True)
class CaptionBatchOutcome:
    claimed: int
    completed: int
    failed: int
    stopped_globally: bool
    last_error: str = ""


class CaptionStateOrchestrator:
    """Translate one caption attempt into explicit store state transitions."""

    def __init__(self, store: CaptionJobStore, *, worker_id: str) -> None:
        self.store = store
        self.worker_id = worker_id

    def run_batch(
        self,
        process: Callable[[CaptionJob], str],
        *,
        limit: int,
    ) -> CaptionBatchOutcome:
        if limit < 1:
            raise ValueError("limit must be positive")
        jobs = self.store.claim(limit=limit, worker_id=self.worker_id)
        completed = 0
        failed = 0
        stopped_globally = False
        last_error = ""
        for job in jobs:
            try:
                receipt_id = process(job)
                if not isinstance(receipt_id, str) or not SHA256_ID_RE.fullmatch(receipt_id):
                    raise ValueError("blank receipt or invalid receipt ID")
                self.store.mark_complete(job.eagle_id, job.claim_token, receipt_id)
                completed += 1
            except GlobalCaptionFailure as exc:
                last_error = str(exc)[:500]
                self.store.mark_failed(job.eagle_id, job.claim_token, last_error)
                failed += 1
                stopped_globally = True
                break
            except Exception as exc:
                last_error = str(exc)[:500]
                self.store.mark_failed(job.eagle_id, job.claim_token, last_error)
                failed += 1
        return CaptionBatchOutcome(
            claimed=len(jobs),
            completed=completed,
            failed=failed,
            stopped_globally=stopped_globally,
            last_error=last_error,
        )
