"""Aggregate frozen caption-evaluation metrics and apply the quality-first rule."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Mapping, Sequence


_MODEL_ORDER = {
    ("gpt-5.6-luna", "low"): 0,
    ("gpt-5.6-luna", "medium"): 1,
    ("gpt-5.6-terra", "low"): 2,
    ("gpt-5.6-sol", "low"): 3,
}
_MARGINS = {"ndcg_at_10": -0.03, "recall_at_10": -0.03, "concept_f1": -0.04, "ocr_character_f1": -0.02}


@dataclass(frozen=True)
class CandidateAggregate:
    model: str
    effort: str
    schema_valid_rate: float
    critical_hallucinations: int
    concept_precision: float
    concept_f1: float
    ocr_character_f1: float
    ndcg_at_10: float
    recall_at_10: float
    known_query_passed: bool
    lower_bounds: Mapping[str, float]
    lanes: Mapping[str, Mapping[str, float]]
    latencies_seconds: Sequence[float]

    def tier(self) -> int:
        try:
            return _MODEL_ORDER[(self.model, self.effort)]
        except KeyError as error:
            raise ValueError(f"unknown caption tier {self.model}/{self.effort}") from error

    def absolute_failures(self) -> tuple[str, ...]:
        failures: list[str] = []
        if self.schema_valid_rate != 1.0:
            failures.append("schema_valid_rate")
        if self.critical_hallucinations:
            failures.append("critical_hallucinations")
        if self.concept_precision < 0.95:
            failures.append("concept_precision")
        if self.concept_f1 < 0.90:
            failures.append("concept_f1")
        if self.ocr_character_f1 < 0.90:
            failures.append("ocr_character_f1")
        if self.ndcg_at_10 < 0.85:
            failures.append("ndcg_at_10")
        if self.recall_at_10 < 0.90:
            failures.append("recall_at_10")
        if not self.known_query_passed:
            failures.append("known_query")
        return tuple(failures)

    def noninferiority_failures(self) -> tuple[str, ...]:
        return tuple(metric for metric, margin in _MARGINS.items() if self.lower_bounds.get(metric, -math.inf) < margin)

    def latency_summary(self) -> dict[str, float | None]:
        samples = sorted(value for value in self.latencies_seconds if math.isfinite(value) and value >= 0)
        if not samples:
            return {"median_latency_seconds": None, "p95_latency_seconds": None}
        return {
            "median_latency_seconds": median(samples),
            "p95_latency_seconds": samples[max(0, math.ceil(len(samples) * 0.95) - 1)],
        }


@dataclass(frozen=True)
class Decision:
    status: str
    winner: CandidateAggregate | None
    reason: str
    candidates: tuple[CandidateAggregate, ...]


def _selection_key(candidate: CandidateAggregate) -> tuple[int, float]:
    latency = candidate.latency_summary()["p95_latency_seconds"]
    return candidate.tier(), float(latency) if latency is not None else math.inf


def decide_winner(candidates: Sequence[CandidateAggregate], *, comparator: str) -> Decision:
    """Select only a proven lower tier; an inconclusive result retains comparator.

    This intentionally does not calculate or report token/subscription cost.
    """

    if not candidates:
        raise ValueError("at least one caption candidate is required")
    comparator_candidate = next((candidate for candidate in candidates if candidate.model == comparator), None)
    proven = [candidate for candidate in candidates if not candidate.absolute_failures() and not candidate.noninferiority_failures()]
    comparator_tier = comparator_candidate.tier() if comparator_candidate is not None else math.inf
    inconclusive_smaller = [
        candidate
        for candidate in candidates
        if candidate.tier() < comparator_tier and not candidate.absolute_failures() and candidate.noninferiority_failures()
    ]
    if inconclusive_smaller and comparator_candidate is not None and not comparator_candidate.absolute_failures():
        return Decision(
            "inconclusive",
            comparator_candidate,
            "smaller tier not proven non-inferior; retained the stronger comparator",
            tuple(candidates),
        )
    if proven:
        return Decision("passed", min(proven, key=_selection_key), "lowest tier cleared absolute and non-inferiority gates", tuple(candidates))
    absolute_passers = [candidate for candidate in candidates if not candidate.absolute_failures()]
    if comparator_candidate is not None and comparator_candidate in absolute_passers:
        return Decision(
            "inconclusive",
            comparator_candidate,
            "smaller tier not proven non-inferior; retained the stronger comparator",
            tuple(candidates),
        )
    return Decision("failed", None, "no candidate cleared the frozen absolute gates", tuple(candidates))


def render_anonymised_report(decision: Decision) -> dict[str, object]:
    """Render aggregate evidence only: never fixture paths, hashes, captions or costs."""

    return {
        "report_version": 1,
        "status": decision.status,
        "reason": decision.reason,
        "winner": None if decision.winner is None else {"model": decision.winner.model, "effort": decision.winner.effort},
        "candidates": [
            {
                "model": candidate.model,
                "effort": candidate.effort,
                "absolute_failures": list(candidate.absolute_failures()),
                "noninferiority_failures": list(candidate.noninferiority_failures()),
                "retrieval_lanes": {lane: dict(metrics) for lane, metrics in sorted(candidate.lanes.items())},
                "latency": candidate.latency_summary(),
            }
            for candidate in sorted(decision.candidates, key=_selection_key)
        ],
    }
