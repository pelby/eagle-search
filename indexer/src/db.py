"""Versioned local SQLite state for Eagle Search."""

from __future__ import annotations

from array import array
from datetime import datetime, timezone
from hashlib import sha256
import math
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable, Sequence

DB_PATH = Path.home() / ".eagle-search" / "db.sqlite"
SCHEMA_VERSION = 5
VECTOR_DIMENSIONS = 768
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
PHRASE_RE = re.compile(r'"([^"\n]+)"')


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table})")}


def _add_column(connection: sqlite3.Connection, name: str, definition: str) -> None:
    if name not in _columns(connection, "images"):
        connection.execute(f"ALTER TABLE images ADD COLUMN {name} {definition}")


def _migration_one(connection: sqlite3.Connection) -> None:
    """Adopt the v1 images table without discarding its public columns."""
    connection.execute(
        """CREATE TABLE IF NOT EXISTS images (
            eagle_id TEXT PRIMARY KEY, name TEXT NOT NULL, tags TEXT DEFAULT '',
            annotation TEXT DEFAULT '', ai_description TEXT DEFAULT '',
            thumbnail_path TEXT DEFAULT '', image_path TEXT DEFAULT '',
            folder_name TEXT DEFAULT '', ext TEXT DEFAULT '', width INTEGER DEFAULT 0,
            height INTEGER DEFAULT 0, created_at INTEGER DEFAULT 0, indexed_at TEXT DEFAULT ''
        )"""
    )
    for name, definition in (
        ("human_notes", "TEXT DEFAULT ''"), ("generation_prompt", "TEXT DEFAULT ''"),
        ("embedded_description", "TEXT DEFAULT ''"), ("visual_caption", "TEXT DEFAULT ''"),
        ("visible_text", "TEXT DEFAULT ''"), ("visual_search_terms", "TEXT DEFAULT ''"),
        ("visual_search_text", "TEXT DEFAULT ''"), ("image_hash", "TEXT DEFAULT ''"),
        ("search_content_hash", "TEXT DEFAULT ''"),
        ("caption_state", "TEXT NOT NULL DEFAULT 'pending'"),
        ("caption_attempts", "INTEGER NOT NULL DEFAULT 0"),
        ("caption_last_error", "TEXT DEFAULT ''"), ("caption_updated_at", "TEXT DEFAULT ''"),
        ("active_receipt_id", "TEXT DEFAULT ''"), ("active_receipt_hash", "TEXT DEFAULT ''"),
    ):
        _add_column(connection, name, definition)
    # v1 fields are retained verbatim and projected into explicit v2 roles.
    connection.execute("UPDATE images SET human_notes = annotation WHERE human_notes = ''")
    connection.execute("UPDATE images SET embedded_description = ai_description WHERE embedded_description = ''")
    connection.execute("UPDATE images SET visual_caption = ai_description WHERE visual_caption = ''")
    connection.execute("UPDATE images SET visual_search_text = visual_caption WHERE visual_search_text = ''")
    connection.execute("UPDATE images SET caption_state = 'complete' WHERE ai_description <> ''")


def _migration_two(connection: sqlite3.Connection) -> None:
    for trigger in ("images_fts_insert", "images_fts_delete", "images_fts_update"):
        connection.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    connection.execute("DROP TABLE IF EXISTS images_fts")
    fields = "name, tags, visible_text, visual_caption, visual_search_terms, annotation, generation_prompt, ai_description"
    connection.execute(
        f"CREATE VIRTUAL TABLE images_fts USING fts5({fields}, content='images', content_rowid='rowid')"
    )
    new_fields = ", ".join(f"new.{field}" for field in fields.split(", "))
    old_fields = ", ".join(f"old.{field}" for field in fields.split(", "))
    connection.execute(
        f"CREATE TRIGGER images_fts_insert AFTER INSERT ON images BEGIN "
        f"INSERT INTO images_fts(rowid, {fields}) VALUES (new.rowid, {new_fields}); END"
    )
    connection.execute(
        f"CREATE TRIGGER images_fts_delete AFTER DELETE ON images BEGIN "
        f"INSERT INTO images_fts(images_fts, rowid, {fields}) VALUES ('delete', old.rowid, {old_fields}); END"
    )
    connection.execute(
        f"CREATE TRIGGER images_fts_update AFTER UPDATE ON images BEGIN "
        f"INSERT INTO images_fts(images_fts, rowid, {fields}) VALUES ('delete', old.rowid, {old_fields}); "
        f"INSERT INTO images_fts(rowid, {fields}) VALUES (new.rowid, {new_fields}); END"
    )
    connection.execute("INSERT INTO images_fts(images_fts) VALUES ('rebuild')")


def _migration_three(connection: sqlite3.Connection) -> None:
    statements = (
        """CREATE TABLE IF NOT EXISTS image_embeddings (
            eagle_id TEXT NOT NULL REFERENCES images(eagle_id) ON DELETE CASCADE,
            model TEXT NOT NULL, vector_kind TEXT NOT NULL, dimensions INTEGER NOT NULL,
            content_hash TEXT NOT NULL, vector BLOB NOT NULL, embedded_at TEXT NOT NULL,
            PRIMARY KEY(eagle_id, model, vector_kind)
        )""",
        """CREATE TABLE IF NOT EXISTS caption_jobs (
            eagle_id TEXT PRIMARY KEY REFERENCES images(eagle_id) ON DELETE CASCADE,
            image_hash TEXT NOT NULL, thumbnail_path TEXT NOT NULL, state TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0, claim_token TEXT DEFAULT '', claimed_at TEXT DEFAULT '',
            next_attempt_at TEXT DEFAULT '', last_error TEXT DEFAULT '', active_receipt_id TEXT DEFAULT '',
            updated_at TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS pending_imports (
            intent_id TEXT PRIMARY KEY, path TEXT NOT NULL UNIQUE, state TEXT NOT NULL,
            eagle_id TEXT DEFAULT '', attempts INTEGER NOT NULL DEFAULT 0, claim_token TEXT DEFAULT '',
            claimed_at TEXT DEFAULT '', updated_at TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS index_runs (
            run_id TEXT PRIMARY KEY, state TEXT NOT NULL, stage TEXT NOT NULL,
            started_at TEXT NOT NULL, finished_at TEXT DEFAULT '', details_json TEXT DEFAULT ''
        )""",
        """CREATE TABLE IF NOT EXISTS search_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT, query_text TEXT NOT NULL, retrieval_mode TEXT NOT NULL,
            returned_ids_json TEXT NOT NULL, selected_eagle_id TEXT DEFAULT '',
            no_result INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL
        )""",
        "CREATE INDEX IF NOT EXISTS image_embeddings_lookup ON image_embeddings(model, vector_kind, content_hash)",
        "CREATE INDEX IF NOT EXISTS caption_jobs_claim ON caption_jobs(state, next_attempt_at, updated_at)",
        "CREATE INDEX IF NOT EXISTS pending_imports_claim ON pending_imports(state, updated_at)",
        "CREATE INDEX IF NOT EXISTS search_feedback_retention ON search_feedback(created_at)",
    )
    for statement in statements:
        connection.execute(statement)


def _migration_four(connection: sqlite3.Connection) -> None:
    columns = _columns(connection, "pending_imports")
    if "last_error" not in columns:
        connection.execute("ALTER TABLE pending_imports ADD COLUMN last_error TEXT DEFAULT ''")


def _migration_five(connection: sqlite3.Connection) -> None:
    _add_column(connection, "source_mtime", "INTEGER NOT NULL DEFAULT 0")
    _add_column(connection, "source_size", "INTEGER NOT NULL DEFAULT 0")


_MIGRATIONS = {
    1: _migration_one,
    2: _migration_two,
    3: _migration_three,
    4: _migration_four,
    5: _migration_five,
}


def _migrate(connection: sqlite3.Connection) -> None:
    current = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if current > SCHEMA_VERSION:
        raise RuntimeError(f"database schema {current} is newer than supported {SCHEMA_VERSION}")
    for version in range(current + 1, SCHEMA_VERSION + 1):
        try:
            connection.execute("BEGIN IMMEDIATE")
            _MIGRATIONS[version](connection)
            connection.execute(f"PRAGMA user_version = {version}")
            connection.commit()
        except Exception:
            connection.rollback()
            raise


def init_db(path: Path | None = None) -> sqlite3.Connection:
    """Open/migrate a database while retaining the v1 connection API."""
    db_path = path or DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path))
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 3000")
        connection.execute("PRAGMA journal_mode = WAL")
        _migrate(connection)
        return connection
    except Exception:
        connection.close()
        raise


def get_indexed_ids(connection: sqlite3.Connection) -> set[str]:
    return {str(row["eagle_id"]) for row in connection.execute("SELECT eagle_id FROM images")}


def _string(data: dict[str, Any], name: str, default: str = "") -> str:
    value = data.get(name, default)
    return value if isinstance(value, str) else str(value)


def content_hash(text: str) -> str:
    return "sha256:" + sha256(text.encode("utf-8")).hexdigest()


def _semantic_text(values: Iterable[str]) -> str:
    return " ".join(value.strip() for value in values if value and value.strip())


def upsert_image(connection: sqlite3.Connection, data: dict[str, Any]) -> None:
    """Upsert metadata and FTS projection atomically; v1 fields remain accepted."""
    ai_description = _string(data, "ai_description")
    annotation = _string(data, "annotation")
    visual_caption = _string(data, "visual_caption", ai_description)
    visible_text = _string(data, "visible_text")
    visual_search_terms = _string(data, "visual_search_terms")
    visual_search_text = _string(data, "visual_search_text", _semantic_text((visual_caption, visible_text, visual_search_terms)))
    values = {
        "eagle_id": _string(data, "eagle_id"), "name": _string(data, "name"), "tags": _string(data, "tags"),
        "annotation": annotation, "ai_description": ai_description, "thumbnail_path": _string(data, "thumbnail_path"),
        "image_path": _string(data, "image_path"), "folder_name": _string(data, "folder_name"), "ext": _string(data, "ext"),
        "width": int(data.get("width", 0) or 0), "height": int(data.get("height", 0) or 0),
        "created_at": int(data.get("created_at", 0) or 0), "indexed_at": _now(),
        "human_notes": _string(data, "human_notes", annotation), "generation_prompt": _string(data, "generation_prompt"),
        "embedded_description": _string(data, "embedded_description"), "visual_caption": visual_caption,
        "visible_text": visible_text, "visual_search_terms": visual_search_terms, "visual_search_text": visual_search_text,
        "image_hash": _string(data, "image_hash"), "search_content_hash": content_hash(visual_search_text),
        "caption_state": _string(data, "caption_state", "complete" if ai_description else "pending"),
        "caption_attempts": int(data.get("caption_attempts", 0) or 0), "caption_last_error": _string(data, "caption_last_error"),
        "caption_updated_at": _string(data, "caption_updated_at"), "active_receipt_id": _string(data, "active_receipt_id"),
        "active_receipt_hash": _string(data, "active_receipt_hash"),
        "source_mtime": int(data.get("source_mtime", 0) or 0),
        "source_size": int(data.get("source_size", 0) or 0),
    }
    columns = ", ".join(values)
    placeholders = ", ".join(f":{key}" for key in values)
    updates = ", ".join(f"{key}=excluded.{key}" for key in values if key != "eagle_id")
    with connection:
        connection.execute(
            f"INSERT INTO images ({columns}) VALUES ({placeholders}) ON CONFLICT(eagle_id) DO UPDATE SET {updates}", values
        )


def _query_parts(query: str) -> tuple[list[str], list[str]]:
    phrases = [match.group(1).strip() for match in PHRASE_RE.finditer(query) if match.group(1).strip()]
    return TOKEN_RE.findall(PHRASE_RE.sub(" ", query).casefold()), phrases


def _expression(tokens: Sequence[str], phrases: Sequence[str], joiner: str) -> str | None:
    pieces = [f'"{token}"*' for token in tokens]
    pieces.extend(f'"{phrase.replace(chr(34), "")}"' for phrase in phrases)
    return f" {joiner} ".join(pieces) if pieces else None


def weighted_lexical_search(connection: sqlite3.Connection, query: str, limit: int = 30) -> list[dict[str, Any]]:
    if limit < 1 or limit > 100:
        raise ValueError("limit must be between 1 and 100")
    tokens, phrases = _query_parts(query)
    if not tokens and not phrases:
        return [dict(row) for row in connection.execute("SELECT * FROM images ORDER BY created_at DESC, eagle_id ASC LIMIT ?", (limit,))]
    for joiner in ("AND", "OR"):
        expression = _expression(tokens, phrases, joiner)
        if expression is None:
            return []
        rows = connection.execute(
            """SELECT i.*, bm25(images_fts, 12.0, 10.0, 10.0, 9.0, 8.0, 2.0, 1.0, 4.0) AS rank
               FROM images_fts JOIN images AS i ON images_fts.rowid = i.rowid
               WHERE images_fts MATCH ? ORDER BY rank ASC, i.eagle_id ASC LIMIT ?""", (expression, limit)
        ).fetchall()
        if rows or joiner == "OR":
            return [dict(row) for row in rows]
    return []


def search(connection: sqlite3.Connection, query: str, limit: int = 20) -> list[dict[str, Any]]:
    return weighted_lexical_search(connection, query, limit)


def image_content_hash(connection: sqlite3.Connection, eagle_id: str) -> str:
    row = connection.execute("SELECT search_content_hash FROM images WHERE eagle_id = ?", (eagle_id,)).fetchone()
    if row is None:
        raise KeyError(eagle_id)
    return str(row["search_content_hash"])


def image_semantic_text(connection: sqlite3.Connection, eagle_id: str) -> str:
    row = connection.execute("SELECT visual_search_text FROM images WHERE eagle_id = ?", (eagle_id,)).fetchone()
    if row is None:
        raise KeyError(eagle_id)
    return str(row["visual_search_text"])


def _validated_vector(vector: Sequence[float]) -> list[float]:
    if len(vector) != VECTOR_DIMENSIONS or any(not math.isfinite(float(value)) for value in vector):
        raise ValueError(f"vectors must contain {VECTOR_DIMENSIONS} finite values")
    return [float(value) for value in vector]


def store_embedding(connection: sqlite3.Connection, eagle_id: str, model: str, vector_kind: str, source_hash: str, vector: Sequence[float]) -> None:
    packed = array("f", _validated_vector(vector)).tobytes()
    with connection:
        connection.execute(
            """INSERT INTO image_embeddings(eagle_id, model, vector_kind, dimensions, content_hash, vector, embedded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(eagle_id, model, vector_kind) DO UPDATE SET
               dimensions=excluded.dimensions, content_hash=excluded.content_hash, vector=excluded.vector, embedded_at=excluded.embedded_at""",
            (eagle_id, model, vector_kind, VECTOR_DIMENSIONS, source_hash, packed, _now()),
        )


def current_embeddings(connection: sqlite3.Connection, model: str, vector_kind: str = "visual") -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(
        """SELECT e.eagle_id, e.vector FROM image_embeddings e JOIN images i ON i.eagle_id=e.eagle_id
           WHERE e.model=? AND e.vector_kind=? AND e.dimensions=? AND e.content_hash=i.search_content_hash""",
        (model, vector_kind, VECTOR_DIMENSIONS),
    )]


def missing_embedding_rows(connection: sqlite3.Connection, model: str, vector_kind: str = "visual", limit: int = 1000) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(
        """SELECT i.eagle_id, i.visual_search_text, i.search_content_hash FROM images i
           LEFT JOIN image_embeddings e ON e.eagle_id=i.eagle_id AND e.model=? AND e.vector_kind=?
           WHERE i.visual_search_text<>'' AND (e.eagle_id IS NULL OR e.content_hash<>i.search_content_hash OR e.dimensions<>?)
           ORDER BY i.eagle_id LIMIT ?""", (model, vector_kind, VECTOR_DIMENSIONS, limit)
    )]


def stats(connection: sqlite3.Connection) -> dict[str, Any]:
    total = int(connection.execute("SELECT count(*) FROM images").fetchone()[0])
    described = int(connection.execute("SELECT count(*) FROM images WHERE ai_description != ''").fetchone()[0])
    last = connection.execute("SELECT max(indexed_at) FROM images").fetchone()[0]
    return {"total": total, "last_indexed": last or "never", "described": described}
