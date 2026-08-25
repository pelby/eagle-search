"""R9 red-first tests for a process-safe worker and durable status."""

from __future__ import annotations

import multiprocessing
import os
import tempfile
import time
import unittest
from pathlib import Path

from src.worker.lock import FileLock, LockUnavailable
from src.worker.runtime import RunStatusFile, WorkerAlreadyRunning, WorkerRuntime


def _hold_lock(path: str, ready: multiprocessing.synchronize.Event, release: multiprocessing.synchronize.Event) -> None:
    with FileLock(Path(path)):
        ready.set()
        release.wait(3)


def _crash_with_lock(path: str, ready: multiprocessing.synchronize.Event) -> None:
    lock = FileLock(Path(path))
    lock.acquire()
    ready.set()
    os._exit(17)


class WorkerLockTests(unittest.TestCase):
    def test_second_process_cannot_enter_same_worker_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "worker.lock")
            ready = multiprocessing.Event()
            release = multiprocessing.Event()
            process = multiprocessing.Process(target=_hold_lock, args=(path, ready, release))
            process.start()
            self.assertTrue(ready.wait(2))
            try:
                with self.assertRaises(LockUnavailable):
                    FileLock(Path(path)).acquire()
            finally:
                release.set()
                process.join(3)
                if process.is_alive():
                    process.terminate()

    def test_os_releases_lock_after_worker_crash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "worker.lock")
            ready = multiprocessing.Event()
            process = multiprocessing.Process(target=_crash_with_lock, args=(path, ready))
            process.start()
            self.assertTrue(ready.wait(2))
            process.join(3)
            self.assertFalse(process.is_alive())

            lock = FileLock(Path(path))
            lock.acquire()
            lock.release()


class WorkerRuntimeTests(unittest.TestCase):
    def test_status_progresses_atomically_to_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status = RunStatusFile(root / "status.json")
            runtime = WorkerRuntime(
                lock=FileLock(root / "worker.lock"),
                status=status,
                run_id_factory=lambda: "run-1",
            )
            observed: list[str] = []

            def work(report):
                observed.append(status.read().state)
                report(completed=1, pending=1, failed=0)
                observed.append(status.read().state)
                report(completed=2, pending=0, failed=0)
                return "done"

            result = runtime.run(stage="caption", total=2, work=work, provider="codex-cli", model="luna")

            self.assertEqual(result, "done")
            self.assertEqual(observed, ["running", "running"])
            final = status.read()
            self.assertEqual((final.state, final.completed, final.pending, final.failed), ("complete", 2, 0, 0))

    def test_runtime_refuses_duplicate_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            held = FileLock(root / "worker.lock")
            held.acquire()
            try:
                runtime = WorkerRuntime(
                    lock=FileLock(root / "worker.lock"),
                    status=RunStatusFile(root / "status.json"),
                )
                with self.assertRaises(WorkerAlreadyRunning):
                    runtime.run(stage="caption", total=1, work=lambda _report: None)
            finally:
                held.release()

    def test_failure_status_is_bounded_and_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status = RunStatusFile(root / "status.json")
            runtime = WorkerRuntime(lock=FileLock(root / "worker.lock"), status=status)

            with self.assertRaisesRegex(RuntimeError, "boom"):
                runtime.run(
                    stage="caption",
                    total=1,
                    work=lambda _report: (_ for _ in ()).throw(RuntimeError("boom" + "x" * 2_000)),
                )

            final = status.read()
            self.assertEqual(final.state, "failed")
            self.assertTrue(final.last_error.startswith("boom"))
            self.assertLessEqual(len(final.last_error), 500)


if __name__ == "__main__":
    unittest.main()
