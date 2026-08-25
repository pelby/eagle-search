"""Local lexical and semantic retrieval adapters."""

from .backfill import backfill_embeddings
from .embeddings import OllamaEmbedder, cosine_similarity
from .hybrid import hybrid_search

__all__ = ["OllamaEmbedder", "backfill_embeddings", "cosine_similarity", "hybrid_search"]
