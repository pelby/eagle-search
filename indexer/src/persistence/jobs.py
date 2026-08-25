"""SQLite implementations of the frozen worker-facing persistence ports."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import sqlite3
from uuid import uuid4

from ..ports import CaptionJob, ImportIntent


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None = None) -> str:
    return (value or _now()).isoformat()


class SQLiteCaptionJobStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def enqueue(self, eagle_id: str, image_hash: str, thumbnail_path: str) -> None:
        with self.connection:
            self.connection.execute(
                """INSERT INTO caption_jobs(eagle_id,image_hash,thumbnail_path,state,updated_at)
                VALUES (?,? ,?,'pending',?) ON CONFLICT(eagle_id) DO UPDATE SET
                image_hash=excluded.image_hash, thumbnail_path=excluded.thumbnail_path,
                state=CASE WHEN caption_jobs.image_hash<>excluded.image_hash THEN 'pending' ELSE caption_jobs.state END,
                updated_at=excluded.updated_at""", (eagle_id, image_hash, thumbnail_path, _iso())
            )

    def claim(self, *, limit: int, worker_id: str) -> list[CaptionJob]:
        if limit < 1:
            raise ValueError("limit must be positive")
        token = f"{worker_id}:{uuid4()}"
        with self.connection:
            rows = self.connection.execute(
                """SELECT eagle_id,image_hash,thumbnail_path,attempts FROM caption_jobs
                WHERE state IN ('pending','failed') AND (next_attempt_at='' OR next_attempt_at<=?)
                ORDER BY updated_at,eagle_id LIMIT ?""", (_iso(), limit)
            ).fetchall()
            for row in rows:
                self.connection.execute(
                    "UPDATE caption_jobs SET state='claimed',claim_token=?,claimed_at=?,attempts=attempts+1,updated_at=? WHERE eagle_id=?",
                    (token, _iso(), _iso(), row["eagle_id"]),
                )
        return [CaptionJob(str(row["eagle_id"]), str(row["image_hash"]), str(row["thumbnail_path"]), token, int(row["attempts"]) + 1) for row in rows]

    def _claimed(self, eagle_id: str, token: str) -> None:
        row = self.connection.execute("SELECT 1 FROM caption_jobs WHERE eagle_id=? AND state='claimed' AND claim_token=?", (eagle_id, token)).fetchone()
        if row is None:
            raise ValueError("caption job is not claimed by this token")

    def mark_complete(self, eagle_id: str, claim_token: str, receipt_id: str) -> None:
        with self.connection:
            self._claimed(eagle_id, claim_token)
            self.connection.execute("UPDATE caption_jobs SET state='complete',active_receipt_id=?,claim_token='',claimed_at='',last_error='',updated_at=? WHERE eagle_id=?", (receipt_id, _iso(), eagle_id))

    def mark_failed(self, eagle_id: str, claim_token: str, error: str) -> None:
        with self.connection:
            self._claimed(eagle_id, claim_token)
            self.connection.execute("UPDATE caption_jobs SET state='failed',claim_token='',claimed_at='',last_error=?,next_attempt_at=?,updated_at=? WHERE eagle_id=?", (error[:500], _iso(), _iso(), eagle_id))

    def release_stale(self, *, older_than_seconds: int) -> int:
        cutoff = _iso(_now() - timedelta(seconds=older_than_seconds))
        with self.connection:
            cursor = self.connection.execute("UPDATE caption_jobs SET state='pending',claim_token='',claimed_at='',updated_at=? WHERE state='claimed' AND claimed_at<?", (_iso(), cutoff))
        return int(cursor.rowcount)

    def counts(self) -> dict[str, int]:
        rows = self.connection.execute("SELECT state,count(*) AS count FROM caption_jobs GROUP BY state").fetchall()
        return {str(row["state"]): int(row["count"]) for row in rows}


class SQLiteImportQueueStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def enqueue(self, path: str, intent_id: str) -> ImportIntent:
        with self.connection:
            self.connection.execute(
                "INSERT INTO pending_imports(intent_id,path,state,updated_at) VALUES (?,?,'pending',?) "
                "ON CONFLICT(path) DO NOTHING", (intent_id, path, _iso())
            )
        row = self.connection.execute("SELECT * FROM pending_imports WHERE intent_id=?", (intent_id,)).fetchone()
        if row is None:
            row = self.connection.execute("SELECT * FROM pending_imports WHERE path=?", (path,)).fetchone()
        if row is None:
            raise RuntimeError("import intent disappeared")
        return ImportIntent(intent_id, str(row["path"]), str(row["state"]), str(row["eagle_id"]), int(row["attempts"]))

    def claim(self, *, worker_id: str) -> ImportIntent | None:
        token = f"{worker_id}:{uuid4()}"
        with self.connection:
            row = self.connection.execute("SELECT * FROM pending_imports WHERE state IN ('pending','reconciled') ORDER BY updated_at,intent_id LIMIT 1").fetchone()
            if row is None:
                return None
            self.connection.execute("UPDATE pending_imports SET state='claimed',claim_token=?,attempts=attempts+1,claimed_at=?,updated_at=? WHERE intent_id=?", (token, _iso(), _iso(), row["intent_id"]))
        return ImportIntent(str(row["intent_id"]), str(row["path"]), "claimed", str(row["eagle_id"]), int(row["attempts"]) + 1)

    def reconcile(self, intent_id: str, eagle_id: str) -> None:
        with self.connection:
            self.connection.execute("UPDATE pending_imports SET state='reconciled',eagle_id=?,claim_token='',updated_at=? WHERE intent_id=?", (eagle_id, _iso(), intent_id))

    def acknowledge(self, intent_id: str, eagle_id: str) -> None:
        with self.connection:
            self.connection.execute("UPDATE pending_imports SET state='complete',eagle_id=?,claim_token='',updated_at=? WHERE intent_id=?", (eagle_id, _iso(), intent_id))
