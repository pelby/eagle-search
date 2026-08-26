"""Aggregate frozen caption-evaluation metrics and apply the quality-first rule."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean, median
from typing import Mapping, Sequence


_MODEL_ORDER = {
    ("gpt-5.6-luna", "low"): 0,
    ("gpt-5.6-luna", "medium"): 1,
    ("gpt-5.6-terra", "low"): 2,
    ("gpt-5.6-sol", "low"): 3,
}
NONINFERIORITY_MARGINS = {
    "ndcg_at_10": -0.03,
    "recall_at_10": -0.03,
    "concept_f1": -0.04,
    "ocr_character_f1": -0.02,
}
FROZEN_EVALUATION_THRESHOLDS = {
    "absolute": {
        "schema_valid_rate": 1.0,
        "critical_hallucinations": 0,
        "concept_precision": 0.95,
        "concept_f1": 0.90,
        "ocr_character_f1": 0.90,
        "ndcg_at_10": 0.85,
        "recall_at_10": 0.90,
        "known_query_passed": True,
    },
    "noninferiority_margins": NONINFERIORITY_MARGINS,
    "bootstrap_resamples": 10_000,
    "latency": {
        "caption_count": 42,
        "max_seconds": 900,
        "documented_background_result_allowed": True,
    },
}


def _require_finite_range(name: str, value: object, *, minimum: float, maximum: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} must be finite and between {minimum} and {maximum}")


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
    background_throughput_accepted: bool
    lower_bounds: Mapping[str, float]
    lanes: Mapping[str, Mapping[str, float]]
    latencies_seconds: Sequence[float]

    def __post_init__(self) -> None:
        rate_fields = (
            "schema_valid_rate",
            "concept_precision",
            "concept_f1",
            "ocr_character_f1",
            "ndcg_at_10",
            "recall_at_10",
        )
        for field in rate_fields:
            _require_finite_range(field, getattr(self, field), minimum=0.0, maximum=1.0)
        if (
            isinstance(self.critical_hallucinations, bool)
            or not isinstance(self.critical_hallucinations, int)
            or self.critical_hallucinations < 0
        ):
            raise ValueError("critical_hallucinations must be a non-negative integer")
        if not isinstance(self.known_query_passed, bool):
            raise ValueError("known_query_passed must be a boolean")
        if not isinstance(self.background_throughput_accepted, bool):
            raise ValueError("background_throughput_accepted must be a boolean")
        if not isinstance(self.lower_bounds, Mapping):
            raise ValueError("lower_bounds must be a mapping")
        for metric, value in self.lower_bounds.items():
            _require_finite_range(f"lower_bounds[{metric!r}]", value, minimum=-1.0, maximum=1.0)
        if not isinstance(self.lanes, Mapping):
            raise ValueError("lanes must be a mapping")
        for lane, metrics in self.lanes.items():
            if not isinstance(metrics, Mapping):
                raise ValueError(f"lanes[{lane!r}] must be a mapping")
            for metric, value in metrics.items():
                _require_finite_range(f"lanes[{lane!r}][{metric!r}]", value, minimum=0.0, maximum=1.0)
        for value in self.latencies_seconds:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("latencies_seconds must contain only finite non-negative numbers")

    def tier(self) -> int:
        try:
            return _MODEL_ORDER[(self.model, self.effort)]
        except KeyError as error:
            raise ValueError(f"unknown caption tier {self.model}/{self.effort}") from error

    def absolute_failures(self) -> tuple[str, ...]:
        gates = FROZEN_EVALUATION_THRESHOLDS["absolute"]
        failures: list[str] = []
        if self.schema_valid_rate != gates["schema_valid_rate"]:
            failures.append("schema_valid_rate")
        if self.critical_hallucinations != gates["critical_hallucinations"]:
            failures.append("critical_hallucinations")
        if self.concept_precision < gates["concept_precision"]:
            failures.append("concept_precision")
        if self.concept_f1 < gates["concept_f1"]:
            failures.append("concept_f1")
        if self.ocr_character_f1 < gates["ocr_character_f1"]:
            failures.append("ocr_character_f1")
        if self.ndcg_at_10 < gates["ndcg_at_10"]:
            failures.append("ndcg_at_10")
        if self.recall_at_10 < gates["recall_at_10"]:
            failures.append("recall_at_10")
        if self.known_query_passed is not gates["known_query_passed"]:
            failures.append("known_query")
        projected = self.projected_caption_window_seconds()
        latency_gate = FROZEN_EVALUATION_THRESHOLDS["latency"]
        if (
            (projected is None or projected > latency_gate["max_seconds"])
            and not self.background_throughput_accepted
        ):
            failures.append("latency")
        return tuple(failures)

    def noninferiority_failures(self) -> tuple[str, ...]:
        return tuple(
            metric
            for metric, margin in NONINFERIORITY_MARGINS.items()
            if self.lower_bounds.get(metric, -math.inf) < margin
        )

    def latency_summary(self) -> dict[str, float | None]:
        samples = sorted(value for value in self.latencies_seconds if math.isfinite(value) and value >= 0)
        if not samples:
            return {"median_latency_seconds": None, "p95_latency_seconds": None}
        return {
            "median_latency_seconds": median(samples),
            "p95_latency_seconds": samples[max(0, math.ceil(len(samples) * 0.95) - 1)],
        }

    def projected_caption_window_seconds(self) -> float | None:
        """Project the frozen 42-caption window from all measured item latencies."""

        samples = [value for value in self.latencies_seconds if math.isfinite(value) and value >= 0]
        count = int(FROZEN_EVALUATION_THRESHOLDS["latency"]["caption_count"])
        if len(samples) < count:
            return None
        return fmean(samples) * count


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
    comparator_candidates = [candidate for candidate in candidates if candidate.model == comparator]
    if not comparator_candidates:
        raise ValueError(f"missing comparator candidate {comparator!r}")
    if len(comparator_candidates) > 1:
        raise ValueError(f"duplicate comparator candidate {comparator!r}")
    comparator_candidate = comparator_candidates[0]
    proven = [candidate for candidate in candidates if not candidate.absolute_failures() and not candidate.noninferiority_failures()]
    comparator_tier = comparator_candidate.tier()
    inconclusive_smaller = [
        candidate
        for candidate in candidates
        if candidate.tier() < comparator_tier and not candidate.absolute_failures() and candidate.noninferiority_failures()
    ]
    if inconclusive_smaller and not comparator_candidate.absolute_failures():
        return Decision(
            "inconclusive",
            comparator_candidate,
            "smaller tier not proven non-inferior; retained the stronger comparator",
            tuple(candidates),
        )
    if proven:
        return Decision("passed", min(proven, key=_selection_key), "lowest tier cleared absolute and non-inferiority gates", tuple(candidates))
    absolute_passers = [candidate for candidate in candidates if not candidate.absolute_failures()]
    if comparator_candidate in absolute_passers:
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
                "quality": {
                    "schema_valid_rate": candidate.schema_valid_rate,
                    "critical_hallucinations": candidate.critical_hallucinations,
                    "concept_precision": candidate.concept_precision,
                    "concept_f1": candidate.concept_f1,
                    "ocr_character_f1": candidate.ocr_character_f1,
                    "ndcg_at_10": candidate.ndcg_at_10,
                    "recall_at_10": candidate.recall_at_10,
                    "known_query_passed": candidate.known_query_passed,
                    "projected_42_caption_seconds": candidate.projected_caption_window_seconds(),
                    "background_throughput_accepted": candidate.background_throughput_accepted,
                    "noninferiority_lower_bounds": dict(sorted(candidate.lower_bounds.items())),
                },
                "retrieval_lanes": {lane: dict(metrics) for lane, metrics in sorted(candidate.lanes.items())},
                "latency": candidate.latency_summary(),
            }
            for candidate in sorted(decision.candidates, key=_selection_key)
        ],
    }
