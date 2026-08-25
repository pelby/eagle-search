"""Private, bounded local search-feedback storage."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import sqlite3
from typing import Sequence


class SearchFeedbackStore:
    def __init__(self, connection: sqlite3.Connection, *, retention_days: int = 90) -> None:
        self.connection = connection
        self.retention_days = retention_days

    def record(self, query_text: str, returned_ids: Sequence[str], *, selected_eagle_id: str = "", no_result: bool = False, retrieval_mode: str = "automatic", enabled: bool = True) -> int | None:
        if not enabled:
            return None
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO search_feedback(query_text,retrieval_mode,returned_ids_json,selected_eagle_id,no_result,created_at) VALUES (?,?,?,?,?,?)",
                (query_text, retrieval_mode, json.dumps(list(returned_ids), separators=(",", ":")), selected_eagle_id, int(no_result), datetime.now(timezone.utc).isoformat()),
            )
        self.purge_expired()
        return int(cursor.lastrowid)

    def purge_expired(self) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.retention_days)).isoformat()
        with self.connection:
            cursor = self.connection.execute("DELETE FROM search_feedback WHERE created_at < ?", (cutoff,))
        return int(cursor.rowcount)
