"""SQLite database with FTS5 for image search."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path.home() / ".eagle-search" / "db.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    eagle_id    TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    tags        TEXT DEFAULT '',
    annotation  TEXT DEFAULT '',
    ai_description TEXT DEFAULT '',
    thumbnail_path TEXT DEFAULT '',
    image_path  TEXT DEFAULT '',
    folder_name TEXT DEFAULT '',
    ext         TEXT DEFAULT '',
    width       INTEGER DEFAULT 0,
    height      INTEGER DEFAULT 0,
    created_at  INTEGER DEFAULT 0,
    indexed_at  TEXT DEFAULT ''
);

CREATE VIRTUAL TABLE IF NOT EXISTS images_fts USING fts5(
    name, tags, annotation, ai_description,
    content=images,
    content_rowid=rowid
);

CREATE TRIGGER IF NOT EXISTS images_fts_insert AFTER INSERT ON images BEGIN
    INSERT INTO images_fts(rowid, name, tags, annotation, ai_description)
    VALUES (new.rowid, new.name, new.tags, new.annotation, new.ai_description);
END;

CREATE TRIGGER IF NOT EXISTS images_fts_delete AFTER DELETE ON images BEGIN
    INSERT INTO images_fts(images_fts, rowid, name, tags, annotation, ai_description)
    VALUES ('delete', old.rowid, old.name, old.tags, old.annotation, old.ai_description);
END;

CREATE TRIGGER IF NOT EXISTS images_fts_update AFTER UPDATE ON images BEGIN
    INSERT INTO images_fts(images_fts, rowid, name, tags, annotation, ai_description)
    VALUES ('delete', old.rowid, old.name, old.tags, old.annotation, old.ai_description);
    INSERT INTO images_fts(rowid, name, tags, annotation, ai_description)
    VALUES (new.rowid, new.name, new.tags, new.annotation, new.ai_description);
END;
"""


def init_db(path: Path | None = None) -> sqlite3.Connection:
    """Create database and tables if they don't exist."""
    db_path = path or DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def get_indexed_ids(conn: sqlite3.Connection) -> set[str]:
    """Return set of eagle_ids already in the database."""
    rows = conn.execute("SELECT eagle_id FROM images").fetchall()
    return {row["eagle_id"] for row in rows}


def upsert_image(conn: sqlite3.Connection, data: dict) -> None:
    """Insert or update an image record."""
    conn.execute(
        """INSERT INTO images (
            eagle_id, name, tags, annotation, ai_description,
            thumbnail_path, image_path, folder_name, ext,
            width, height, created_at, indexed_at
        ) VALUES (
            :eagle_id, :name, :tags, :annotation, :ai_description,
            :thumbnail_path, :image_path, :folder_name, :ext,
            :width, :height, :created_at, :indexed_at
        ) ON CONFLICT(eagle_id) DO UPDATE SET
            name=:name, tags=:tags, annotation=:annotation,
            ai_description=:ai_description, thumbnail_path=:thumbnail_path,
            image_path=:image_path, folder_name=:folder_name, ext=:ext,
            width=:width, height=:height, created_at=:created_at,
            indexed_at=:indexed_at
        """,
        {
            "eagle_id": data["eagle_id"],
            "name": data.get("name", ""),
            "tags": data.get("tags", ""),
            "annotation": data.get("annotation", ""),
            "ai_description": data.get("ai_description", ""),
            "thumbnail_path": data.get("thumbnail_path", ""),
            "image_path": data.get("image_path", ""),
            "folder_name": data.get("folder_name", ""),
            "ext": data.get("ext", ""),
            "width": data.get("width", 0),
            "height": data.get("height", 0),
            "created_at": data.get("created_at", 0),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    conn.commit()


def search(conn: sqlite3.Connection, query: str, limit: int = 20) -> list[dict]:
    """Search images using FTS5. Returns list of dicts sorted by relevance."""
    safe_query = query.replace("'", "''").replace('"', '""')

    # Try AND match first (all words must appear)
    rows = conn.execute(
        """SELECT i.*, images_fts.rank
        FROM images_fts
        JOIN images i ON images_fts.rowid = i.rowid
        WHERE images_fts MATCH ?
        ORDER BY images_fts.rank
        LIMIT ?""",
        (safe_query, limit),
    ).fetchall()

    if not rows:
        # Fall back to OR match
        or_query = " OR ".join(safe_query.split())
        rows = conn.execute(
            """SELECT i.*, images_fts.rank
            FROM images_fts
            JOIN images i ON images_fts.rowid = i.rowid
            WHERE images_fts MATCH ?
            ORDER BY images_fts.rank
            LIMIT ?""",
            (or_query, limit),
        ).fetchall()

    return [dict(row) for row in rows]


def stats(conn: sqlite3.Connection) -> dict:
    """Return index statistics."""
    total = conn.execute("SELECT count(*) as n FROM images").fetchone()["n"]
    last = conn.execute(
        "SELECT max(indexed_at) as t FROM images"
    ).fetchone()["t"]
    described = conn.execute(
        "SELECT count(*) as n FROM images WHERE ai_description != ''"
    ).fetchone()["n"]
    return {"total": total, "last_indexed": last or "never", "described": described}
