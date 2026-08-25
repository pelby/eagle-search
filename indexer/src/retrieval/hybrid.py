"""Weighted lexical + local semantic retrieval with deterministic RRF."""

from __future__ import annotations

from array import array
from collections import defaultdict
import sqlite3
import re
from typing import Sequence

from .. import db
from ..contracts import SearchResponseV1, SearchResultV1
from .embeddings import Embedder, cosine_similarity

RRF_K = 60
SEMANTIC_FLOOR = 0.60
_QUERY_PROFILES = {
    "classroom": (
        "presentation teaching easel",
        "teaching illustration presentation scene art demonstration easel",
    ),
    "workshop": ("facilitation whiteboard sticky notes",),
    "meeting": ("presentation conference team discussion",),
    "diagram": ("flowchart schematic process model",),
}


def semantic_query_texts(query: str) -> tuple[str, ...]:
    """Return bounded sense profiles for local semantic retrieval only."""

    tokens = set(re.findall(r"\w+", query.casefold()))
    base = query.strip()
    variants = [base]
    for token in sorted(tokens):
        for profile in _QUERY_PROFILES.get(token, ()):
            variants.append(" ".join((base, profile)).strip())
    return tuple(dict.fromkeys(variant for variant in variants if variant))


def semantic_query_text(query: str) -> str:
    """Return the primary expansion for diagnostics and compatibility."""

    variants = semantic_query_texts(query)
    return variants[1] if len(variants) > 1 else (variants[0] if variants else "")


def _rrf(rankings: Sequence[Sequence[str]]) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, eagle_id in enumerate(ranking, start=1):
            scores[eagle_id] += 1.0 / (RRF_K + rank)
    return dict(scores)


def _result(row: dict, score: float, matched_by: tuple[str, ...]) -> SearchResultV1:
    return SearchResultV1(
        eagle_id=str(row["eagle_id"]), name=str(row["name"]), thumbnail_path=str(row["thumbnail_path"]),
        image_path=str(row["image_path"]), score=score, matched_by=matched_by,
        tags=str(row["tags"]), annotation=str(row["annotation"]), ai_description=str(row["ai_description"]),
        folder_name=str(row["folder_name"]), ext=str(row["ext"]), width=int(row["width"]),
        height=int(row["height"]), created_at=int(row["created_at"]),
    )


def hybrid_search(connection: sqlite3.Connection, query: str, embedder: Embedder, *, limit: int = 30, mode: str = "automatic", semantic_floor: float = SEMANTIC_FLOOR) -> SearchResponseV1:
    if limit < 1 or limit > 100:
        raise ValueError("limit must be between 1 and 100")
    if mode not in {"automatic", "exact", "best"}:
        raise ValueError("unknown retrieval mode")
    lexical_rows = db.weighted_lexical_search(connection, query, max(100, limit))
    if not query.strip():
        return SearchResponseV1(True, query, limit, "browse", True, (), tuple(_result(row, 0.0, ("browse",)) for row in lexical_rows[:limit]))
    quoted = '"' in query
    if mode == "exact" or quoted:
        return SearchResponseV1(True, query, limit, "lexical", True, (), tuple(_result(row, 1.0 / (RRF_K + rank), ("lexical",)) for rank, row in enumerate(lexical_rows[:limit], start=1)))
    lexical_ids = [str(row["eagle_id"]) for row in lexical_rows]
    row_by_id = {str(row["eagle_id"]): row for row in lexical_rows}
    try:
        query_vectors = embedder.embed(list(semantic_query_texts(query)))
        semantic_scored: list[tuple[str, float]] = []
        for row in db.current_embeddings(connection, embedder.model):
            try:
                vector = array("f"); vector.frombytes(row["vector"])
                semantic_scored.append(
                    (
                        str(row["eagle_id"]),
                        max(
                            cosine_similarity(query_vector, vector)
                            for query_vector in query_vectors
                        ),
                    )
                )
            except ValueError:
                continue
        semantic_scored.sort(key=lambda item: (-item[1], item[0]))
        semantic_ids = [eagle_id for eagle_id, _ in semantic_scored[:100]]
        semantic_score = dict(semantic_scored)
    except Exception as error:
        return SearchResponseV1(True, query, limit, "lexical", False, (f"semantic unavailable: {type(error).__name__}",), tuple(_result(row, 1.0 / (RRF_K + rank), ("lexical",)) for rank, row in enumerate(lexical_rows[:limit], start=1)))
    scores = _rrf((lexical_ids, semantic_ids))
    selected: list[tuple[str, float, tuple[str, ...]]] = []
    for eagle_id, score in scores.items():
        lexical = eagle_id in row_by_id
        semantic = eagle_id in semantic_score
        if not lexical and semantic_score[eagle_id] < semantic_floor:
            continue
        if eagle_id not in row_by_id:
            row = connection.execute("SELECT * FROM images WHERE eagle_id=?", (eagle_id,)).fetchone()
            if row is None:
                continue
            row_by_id[eagle_id] = dict(row)
        selected.append((eagle_id, score, tuple(kind for kind, enabled in (("lexical", lexical), ("semantic", semantic)) if enabled)))
    selected.sort(key=lambda item: (-item[1], item[0]))
    return SearchResponseV1(True, query, limit, "hybrid", True, (), tuple(_result(row_by_id[eagle_id], score, matched_by) for eagle_id, score, matched_by in selected[:limit]))
