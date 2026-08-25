"""R5 red-first tests for retryable caption job orchestration."""

from __future__ import annotations

import unittest
from dataclasses import replace

from src.ports import CaptionJob
from src.worker.caption_state import CaptionStateOrchestrator, GlobalCaptionFailure


RECEIPT_ID = f"sha256:{'a' * 64}"


class FakeCaptionJobStore:
    def __init__(self, jobs: list[CaptionJob]) -> None:
        self.available = list(jobs)
        self.completed: list[tuple[str, str, str]] = []
        self.failed: list[tuple[str, str, str]] = []
        self.claimed: list[str] = []

    def claim(self, *, limit: int, worker_id: str) -> list[CaptionJob]:
        jobs = self.available[:limit]
        self.available = self.available[limit:]
        self.claimed.extend(job.eagle_id for job in jobs)
        return jobs

    def mark_complete(self, eagle_id: str, claim_token: str, receipt_id: str) -> None:
        self.completed.append((eagle_id, claim_token, receipt_id))

    def mark_failed(self, eagle_id: str, claim_token: str, error: str) -> None:
        self.failed.append((eagle_id, claim_token, error))

    def release_stale(self, *, older_than_seconds: int) -> int:
        return 0

    def counts(self) -> dict[str, int]:
        return {
            "pending": len(self.available),
            "complete": len(self.completed),
            "failed": len(self.failed),
            "blocked": 0,
        }


def job(eagle_id: str, *, attempts: int = 1) -> CaptionJob:
    return CaptionJob(
        eagle_id=eagle_id,
        image_hash=f"sha256:{'b' * 64}",
        thumbnail_path=f"/fixtures/{eagle_id}.png",
        claim_token=f"claim-{eagle_id}-{attempts}",
        attempts=attempts,
    )


class CaptionStateTests(unittest.TestCase):
    def test_blank_receipt_is_failed_not_complete_and_can_be_retried(self) -> None:
        first = job("blank")
        store = FakeCaptionJobStore([first])
        runner = CaptionStateOrchestrator(store, worker_id="worker-1")

        outcome = runner.run_batch(lambda _job: "", limit=1)

        self.assertEqual(outcome.completed, 0)
        self.assertEqual(outcome.failed, 1)
        self.assertEqual(store.completed, [])
        self.assertIn("blank receipt", store.failed[0][2])

        retry = replace(first, claim_token="claim-blank-2", attempts=2)
        store.available.append(retry)
        retry_outcome = runner.run_batch(lambda _job: RECEIPT_ID, limit=1)

        self.assertEqual(retry_outcome.completed, 1)
        self.assertEqual(store.completed[-1], ("blank", "claim-blank-2", RECEIPT_ID))

    def test_local_failure_does_not_prevent_later_jobs_completing(self) -> None:
        store = FakeCaptionJobStore([job("bad"), job("good")])
        runner = CaptionStateOrchestrator(store, worker_id="worker-1")

        def process(item: CaptionJob) -> str:
            if item.eagle_id == "bad":
                raise ValueError("invalid caption JSON")
            return RECEIPT_ID

        outcome = runner.run_batch(process, limit=2)

        self.assertEqual((outcome.completed, outcome.failed, outcome.stopped_globally), (1, 1, False))
        self.assertEqual([entry[0] for entry in store.failed], ["bad"])
        self.assertEqual([entry[0] for entry in store.completed], ["good"])

    def test_global_auth_failure_stops_without_stamping_remaining_jobs(self) -> None:
        store = FakeCaptionJobStore([job("first"), job("second"), job("third")])
        runner = CaptionStateOrchestrator(store, worker_id="worker-1")
        processed: list[str] = []

        def process(item: CaptionJob) -> str:
            processed.append(item.eagle_id)
            raise GlobalCaptionFailure("ChatGPT authentication expired")

        outcome = runner.run_batch(process, limit=3)

        self.assertEqual(processed, ["first"])
        self.assertTrue(outcome.stopped_globally)
        self.assertEqual(len(store.failed), 1)
        self.assertEqual(store.completed, [])


if __name__ == "__main__":
    unittest.main()
