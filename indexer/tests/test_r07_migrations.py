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
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(images)")
            }
            self.assertTrue({"source_mtime", "source_size"}.issubset(columns))
            self.assertTrue(db.search(connection, "classroom"))
            self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            connection.close()
            reopened = db.init_db(path)
            self.assertEqual(reopened.execute("PRAGMA user_version").fetchone()[0], db.SCHEMA_VERSION)
            reopened.close()

    def test_failed_migration_three_rolls_back_every_new_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken-v2.sqlite"
            raw = sqlite3.connect(path)
            raw.execute("CREATE TABLE image_embeddings (bad INTEGER)")
            raw.execute("PRAGMA user_version = 2")
            raw.commit()
            raw.close()

            with self.assertRaises(sqlite3.DatabaseError):
                db.init_db(path)

            inspected = sqlite3.connect(path)
            tables = {
                row[0]
                for row in inspected.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            self.assertEqual(tables, {"image_embeddings"})
            self.assertEqual(inspected.execute("PRAGMA user_version").fetchone()[0], 2)
            inspected.close()
