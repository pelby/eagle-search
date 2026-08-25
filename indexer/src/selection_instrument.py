"""Frozen, untuned retrieval instrument used only for caption-model selection."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
BM25_K1 = 1.2
BM25_B = 0.75
RRF_K = 60


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        raise ValueError("vectors must be non-empty and have identical dimensions")
    if any(not math.isfinite(value) for value in (*left, *right)):
        raise ValueError("vectors must contain only finite values")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("zero vectors do not have a cosine similarity")
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], *, k: int = RRF_K
) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


@dataclass(frozen=True)
class SelectionDocument:
    item_id: str
    search_text: str


@dataclass(frozen=True)
class SelectionRanking:
    bm25_ids: list[str]
    semantic_ids: list[str]
    fused_ids: list[str]
    fused_scores: dict[str, float]


class SelectionInstrument:
    """No tuned weights, relevance floor or production ranking parameters."""

    def _bm25(self, query: str, documents: Sequence[SelectionDocument]) -> list[str]:
        query_terms = tokenize(query)
        if not query_terms or not documents:
            return []
        tokenized = [tokenize(document.search_text) for document in documents]
        average_length = sum(len(tokens) for tokens in tokenized) / len(tokenized)
        if average_length == 0:
            return []
        document_frequency = Counter(
            term for tokens in tokenized for term in set(tokens)
        )
        scores: list[tuple[str, float]] = []
        document_count = len(documents)
        for document, tokens in zip(documents, tokenized, strict=True):
            frequencies = Counter(tokens)
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                df = document_frequency[term]
                idf = math.log(1.0 + (document_count - df + 0.5) / (df + 0.5))
                denominator = frequency + BM25_K1 * (
                    1.0 - BM25_B + BM25_B * len(tokens) / average_length
                )
                score += idf * (frequency * (BM25_K1 + 1.0)) / denominator
            if score > 0:
                scores.append((document.item_id, score))
        return [item_id for item_id, _ in sorted(scores, key=lambda item: (-item[1], item[0]))]

    def rank(
        self,
        query: str,
        documents: Sequence[SelectionDocument],
        *,
        document_embeddings: Mapping[str, Sequence[float]] | None = None,
        query_embedding: Sequence[float] | None = None,
        limit: int = 10,
    ) -> SelectionRanking:
        bm25_ids = self._bm25(query, documents)
        semantic_scores: list[tuple[str, float]] = []
        if document_embeddings is not None and query_embedding is not None:
            for document in documents:
                vector = document_embeddings.get(document.item_id)
                if vector is None:
                    continue
                semantic_scores.append(
                    (document.item_id, cosine_similarity(query_embedding, vector))
                )
        semantic_ids = [
            item_id
            for item_id, _ in sorted(
                semantic_scores, key=lambda item: (-item[1], item[0])
            )
        ]
        lanes = [lane for lane in (bm25_ids, semantic_ids) if lane]
        fused = reciprocal_rank_fusion(lanes) if lanes else []
        return SelectionRanking(
            bm25_ids=bm25_ids[:limit],
            semantic_ids=semantic_ids[:limit],
            fused_ids=[item_id for item_id, _ in fused[:limit]],
            fused_scores=dict(fused[:limit]),
        )
