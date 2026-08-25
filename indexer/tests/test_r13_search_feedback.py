from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src import db
from src.persistence.feedback import SearchFeedbackStore


class FeedbackTests(unittest.TestCase):
    def test_local_feedback_can_be_disabled_and_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connection = db.init_db(Path(directory) / "db.sqlite")
            store = SearchFeedbackStore(connection)
            self.assertIsNone(store.record("classroom", ["one"], enabled=False))
            store.record("classroom", ["one"], selected_eagle_id="one")
            old = (datetime.now(timezone.utc) - timedelta(days=91)).isoformat()
            connection.execute("UPDATE search_feedback SET created_at = ?", (old,)); connection.commit()
            self.assertEqual(store.purge_expired(), 1)
            connection.close()
