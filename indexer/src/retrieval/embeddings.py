"""Bounded local Ollama adapter and strict vector helpers."""

from __future__ import annotations

import json
import math
from typing import Protocol, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen

from ..db import VECTOR_DIMENSIONS


class Embedder(Protocol):
    model: str
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class OllamaEmbedder:
    model = "nomic-embed-text:v1.5"

    def __init__(self, model: str = model, endpoint: str = "http://127.0.0.1:11434/api/embed", timeout_seconds: int = 15) -> None:
        self.model = model
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        request = Request(
            self.endpoint,
            data=json.dumps({"model": self.model, "input": list(texts)}).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, URLError, TimeoutError, json.JSONDecodeError) as error:
            raise RuntimeError(f"local embedding unavailable: {type(error).__name__}") from error
        vectors = payload.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise RuntimeError("local embedding returned an invalid batch")
        result: list[list[float]] = []
        for vector in vectors:
            if not isinstance(vector, list) or len(vector) != VECTOR_DIMENSIONS:
                raise RuntimeError("local embedding returned invalid dimensions")
            values = [float(value) for value in vector]
            if any(not math.isfinite(value) for value in values):
                raise RuntimeError("local embedding returned non-finite values")
            result.append(values)
        return result


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("vectors must be non-empty and have identical dimensions")
    if any(not math.isfinite(float(value)) for value in (*left, *right)):
        raise ValueError("vectors must be finite")
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if not left_norm or not right_norm:
        raise ValueError("zero vectors do not have cosine similarity")
    return sum(float(a) * float(b) for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
