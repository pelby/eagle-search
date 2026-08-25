"""Incremental local text-embedding backfill."""

from __future__ import annotations

import sqlite3
from typing import Any

from .. import db
from .embeddings import Embedder


def backfill_embeddings(connection: sqlite3.Connection, embedder: Embedder, *, batch_size: int = 32, limit: int = 1000) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rows = db.missing_embedding_rows(connection, embedder.model, limit=limit)
    embedded = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        try:
            vectors = embedder.embed([str(row["visual_search_text"]) for row in batch])
            if len(vectors) != len(batch):
                raise RuntimeError("embedder returned a wrong-sized batch")
            for row, vector in zip(batch, vectors, strict=True):
                db.store_embedding(connection, str(row["eagle_id"]), embedder.model, "visual", str(row["search_content_hash"]), vector)
                embedded += 1
        except Exception as error:
            return {"embedded": embedded, "pending": len(rows) - embedded, "error": f"{type(error).__name__}: {error}"}
    return {"embedded": embedded, "pending": 0, "error": None}
