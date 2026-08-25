from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from src import db


class MigrationTests(unittest.TestCase):
    def test_v1_database_is_adopted_idempotently_and_fts_is_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.sqlite"
            legacy = sqlite3.connect(path)
            legacy.execute("CREATE TABLE images (eagle_id TEXT PRIMARY KEY, name TEXT NOT NULL, tags TEXT DEFAULT '', annotation TEXT DEFAULT '', ai_description TEXT DEFAULT '', thumbnail_path TEXT DEFAULT '', image_path TEXT DEFAULT '', folder_name TEXT DEFAULT '', ext TEXT DEFAULT '', width INTEGER DEFAULT 0, height INTEGER DEFAULT 0, created_at INTEGER DEFAULT 0, indexed_at TEXT DEFAULT '')")
            legacy.execute("INSERT INTO images(eagle_id, name, ai_description) VALUES ('one', 'One', 'classroom teaching')")
            legacy.commit(); legacy.close()
            connection = db.init_db(path)
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], db.SCHEMA_VERSION)
            self.assertTrue(db.search(connection, "classroom"))
            self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            connection.close()
            reopened = db.init_db(path)
            self.assertEqual(reopened.execute("PRAGMA user_version").fetchone()[0], db.SCHEMA_VERSION)
            reopened.close()
