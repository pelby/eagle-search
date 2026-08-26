"""Aggregate frozen caption-evaluation metrics and apply the quality-first rule."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
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
FINAL_NONINFERIORITY_MARGINS = {
    "ndcg_at_10": -0.03,
    "recall_at_10": -0.03,
    "concept_recall": -0.04,
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

# These protections apply only once a complete blind pooled-relevance artifact
# has been sealed.  Keeping them separate preserves frozen preliminary Stage B
# gates while making final selection stricter.
FROZEN_FINAL_SELECTION_THRESHOLDS = {
    "absolute": {"concept_recall": 0.80},
    "ocr_bearing": {"minimum_target_count": 1, "character_f1": 0.90},
    "lane_noninferiority_margins": {
        "bm25": {"ndcg_at_10": -0.03, "recall_at_10": -0.03},
        "semantic": {"ndcg_at_10": -0.03, "recall_at_10": -0.03},
    },
}
_STRATUM_RATE_FIELDS = {
    "concept_precision",
    "concept_recall",
    "concept_f1",
    "ocr_character_f1",
    "type_accuracy",
}
_STRATUM_FIELDS = _STRATUM_RATE_FIELDS | {"target_count", "retrieval_lanes"}
_RETRIEVAL_LANES = {"bm25", "semantic", "rrf"}
_RETRIEVAL_METRICS = {"ndcg_at_10", "recall_at_10", "mrr"}


def _require_finite_range(name: str, value: object, *, minimum: float, maximum: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} must be finite and between {minimum} and {maximum}")


def _validate_anonymous_strata(value: object) -> None:
    """Accept only the closed numeric study aggregate that is safe to report."""

    if not isinstance(value, Mapping):
        raise ValueError("strata must be a mapping")
    for stratum, metrics in value.items():
        if not isinstance(stratum, str) or not stratum.strip() or len(stratum) > 128:
            raise ValueError("stratum name is invalid")
        if not isinstance(metrics, Mapping) or set(metrics) != _STRATUM_FIELDS:
            raise ValueError(f"strata[{stratum!r}] has an invalid aggregate schema")
        target_count = metrics["target_count"]
        if isinstance(target_count, bool) or not isinstance(target_count, int) or target_count < 1:
            raise ValueError(f"strata[{stratum!r}].target_count must be a positive integer")
        for field in _STRATUM_RATE_FIELDS:
            _require_finite_range(
                f"strata[{stratum!r}][{field!r}]",
                metrics[field],
                minimum=0.0,
                maximum=1.0,
            )
        lanes = metrics["retrieval_lanes"]
        if not isinstance(lanes, Mapping) or set(lanes) != _RETRIEVAL_LANES:
            raise ValueError(f"strata[{stratum!r}].retrieval_lanes has an invalid schema")
        for lane, lane_metrics in lanes.items():
            if not isinstance(lane_metrics, Mapping) or set(lane_metrics) != _RETRIEVAL_METRICS:
                raise ValueError(f"strata[{stratum!r}].retrieval_lanes[{lane!r}] has an invalid schema")
            for metric, score in lane_metrics.items():
                _require_finite_range(
                    f"strata[{stratum!r}].retrieval_lanes[{lane!r}][{metric!r}]",
                    score,
                    minimum=0.0,
                    maximum=1.0,
                )


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
    lane_lower_bounds: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    ocr_bearing_target_count: int = 0
    ocr_bearing_character_f1: float | None = None
    strata: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    selection_mode: str = "preliminary"
    pooled_relevance_hash: str | None = None
    concept_recall: float | None = None
    annotation_guide_version: int = 1

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
        expected_bounds = (
            set(FINAL_NONINFERIORITY_MARGINS)
            if self.annotation_guide_version == 2
            else set(NONINFERIORITY_MARGINS)
        )
        if set(self.lower_bounds) != expected_bounds:
            raise ValueError("lower_bounds must contain only the frozen metrics")
        for metric, value in self.lower_bounds.items():
            _require_finite_range(f"lower_bounds[{metric!r}]", value, minimum=-1.0, maximum=1.0)
        if not isinstance(self.lanes, Mapping):
            raise ValueError("lanes must be a mapping")
        if self.lanes and set(self.lanes) != _RETRIEVAL_LANES:
            raise ValueError("lanes must contain only the frozen retrieval lanes")
        for lane, metrics in self.lanes.items():
            if not isinstance(metrics, Mapping) or set(metrics) != _RETRIEVAL_METRICS:
                raise ValueError(f"lanes[{lane!r}] must contain only the frozen retrieval metrics")
            for metric, value in metrics.items():
                _require_finite_range(f"lanes[{lane!r}][{metric!r}]", value, minimum=0.0, maximum=1.0)
        for value in self.latencies_seconds:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("latencies_seconds must contain only finite non-negative numbers")
        if not isinstance(self.lane_lower_bounds, Mapping):
            raise ValueError("lane_lower_bounds must be a mapping")
        if self.lane_lower_bounds and set(self.lane_lower_bounds) != {"bm25", "semantic"}:
            raise ValueError("lane_lower_bounds must contain only the frozen lanes")
        for lane, metrics in self.lane_lower_bounds.items():
            if not isinstance(metrics, Mapping) or set(metrics) != {"ndcg_at_10", "recall_at_10"}:
                raise ValueError(f"lane_lower_bounds[{lane!r}] must contain only frozen metrics")
            for metric, value in metrics.items():
                _require_finite_range(f"lane_lower_bounds[{lane!r}][{metric!r}]", value, minimum=-1.0, maximum=1.0)
        if isinstance(self.ocr_bearing_target_count, bool) or not isinstance(self.ocr_bearing_target_count, int) or self.ocr_bearing_target_count < 0:
            raise ValueError("ocr_bearing_target_count must be a non-negative integer")
        if self.ocr_bearing_character_f1 is not None:
            _require_finite_range("ocr_bearing_character_f1", self.ocr_bearing_character_f1, minimum=0.0, maximum=1.0)
        _validate_anonymous_strata(self.strata)
        if self.selection_mode not in {"preliminary", "final"}:
            raise ValueError("selection_mode must be preliminary or final")
        if self.pooled_relevance_hash is not None and (
            not isinstance(self.pooled_relevance_hash, str) or not self.pooled_relevance_hash.startswith("sha256:")
        ):
            raise ValueError("pooled_relevance_hash must be a SHA-256 identifier")
        if self.concept_recall is not None:
            _require_finite_range("concept_recall", self.concept_recall, minimum=0.0, maximum=1.0)
        if self.annotation_guide_version not in {1, 2}:
            raise ValueError("annotation_guide_version must be 1 or 2")

    def tier(self) -> int:
        try:
            return _MODEL_ORDER[(self.model, self.effort)]
        except KeyError as error:
            raise ValueError(f"unknown caption tier {self.model}/{self.effort}") from error

    def absolute_failures(self, *, final_selection: bool = False) -> tuple[str, ...]:
        gates = FROZEN_EVALUATION_THRESHOLDS["absolute"]
        failures: list[str] = []
        if self.schema_valid_rate != gates["schema_valid_rate"]:
            failures.append("schema_valid_rate")
        if self.critical_hallucinations != gates["critical_hallucinations"]:
            failures.append("critical_hallucinations")
        # v2 measures atomically labelled pixel concepts as recall/coverage.
        # The legacy precision/F1 values stay in reports for diagnostics, but
        # cannot reject a v2 final candidate for emitting useful extra terms.
        if self.annotation_guide_version != 2:
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
        if self.annotation_guide_version == 2:
            final_gates = FROZEN_FINAL_SELECTION_THRESHOLDS
            if self.concept_recall is None or self.concept_recall < final_gates["absolute"]["concept_recall"]:
                failures.append("concept_recall")
            ocr_gate = final_gates["ocr_bearing"]
            if (
                self.ocr_bearing_target_count < ocr_gate["minimum_target_count"]
                or self.ocr_bearing_character_f1 is None
                or self.ocr_bearing_character_f1 < ocr_gate["character_f1"]
            ):
                failures.append("ocr_bearing_character_f1")
        if final_selection:
            if self.selection_mode != "final" or self.pooled_relevance_hash is None:
                failures.append("pooled_relevance")
            if self.annotation_guide_version != 2:
                failures.append("annotation_guide_v2")
            final_gates = FROZEN_FINAL_SELECTION_THRESHOLDS
            for lane, metrics in final_gates["lane_noninferiority_margins"].items():
                for metric, margin in metrics.items():
                    if self.lane_lower_bounds.get(lane, {}).get(metric, -math.inf) < margin:
                        failures.append(f"{lane}_{metric}")
        return tuple(failures)

    def noninferiority_failures(self, *, final_selection: bool = False) -> tuple[str, ...]:
        margins = (
            FINAL_NONINFERIORITY_MARGINS
            if final_selection or self.annotation_guide_version == 2
            else NONINFERIORITY_MARGINS
        )
        return tuple(
            metric
            for metric, margin in margins.items()
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
    selection_mode: str = "preliminary"


def _selection_key(candidate: CandidateAggregate) -> tuple[int, float]:
    latency = candidate.latency_summary()["p95_latency_seconds"]
    return candidate.tier(), float(latency) if latency is not None else math.inf


def decide_winner(
    candidates: Sequence[CandidateAggregate], *, comparator: str, final_selection: bool = False
) -> Decision:
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
    failures = lambda candidate: candidate.absolute_failures(final_selection=final_selection)
    mode = "final" if final_selection else "preliminary"
    noninferior = lambda candidate: candidate.noninferiority_failures(final_selection=final_selection)
    proven = [candidate for candidate in candidates if not failures(candidate) and not noninferior(candidate)]
    comparator_tier = comparator_candidate.tier()
    inconclusive_smaller = [
        candidate
        for candidate in candidates
        if candidate.tier() < comparator_tier and not failures(candidate) and noninferior(candidate)
    ]
    if inconclusive_smaller and not failures(comparator_candidate):
        return Decision(
            "inconclusive",
            comparator_candidate,
            "smaller tier not proven non-inferior; retained the stronger comparator",
            tuple(candidates), mode,
        )
    if proven:
        return Decision("passed", min(proven, key=_selection_key), "lowest tier cleared absolute and non-inferiority gates", tuple(candidates), mode)
    absolute_passers = [candidate for candidate in candidates if not failures(candidate)]
    if comparator_candidate in absolute_passers:
        return Decision(
            "inconclusive",
            comparator_candidate,
            "smaller tier not proven non-inferior; retained the stronger comparator",
            tuple(candidates), mode,
        )
    return Decision("failed", None, "no candidate cleared the frozen absolute gates", tuple(candidates), mode)


def render_anonymised_report(decision: Decision) -> dict[str, object]:
    """Render aggregate evidence only: never fixture paths, hashes, captions or costs."""

    return {
        "report_version": 1,
        "selection_mode": decision.selection_mode,
        "status": decision.status,
        "reason": decision.reason,
        "winner": None if decision.winner is None else {"model": decision.winner.model, "effort": decision.winner.effort},
        "candidates": [
            {
                "model": candidate.model,
                "effort": candidate.effort,
                "absolute_failures": list(candidate.absolute_failures(final_selection=decision.selection_mode == "final")),
                "noninferiority_failures": list(candidate.noninferiority_failures(final_selection=decision.selection_mode == "final")),
                "quality": {
                    "schema_valid_rate": candidate.schema_valid_rate,
                    "critical_hallucinations": candidate.critical_hallucinations,
                    "concept_precision": candidate.concept_precision,
                    "concept_recall": candidate.concept_recall,
                    "concept_f1": candidate.concept_f1,
                    "ocr_character_f1": candidate.ocr_character_f1,
                    "ocr_bearing_target_count": candidate.ocr_bearing_target_count,
                    "ocr_bearing_character_f1": candidate.ocr_bearing_character_f1,
                    "ndcg_at_10": candidate.ndcg_at_10,
                    "recall_at_10": candidate.recall_at_10,
                    "known_query_passed": candidate.known_query_passed,
                    "projected_42_caption_seconds": candidate.projected_caption_window_seconds(),
                    "background_throughput_accepted": candidate.background_throughput_accepted,
                    "noninferiority_lower_bounds": dict(sorted(candidate.lower_bounds.items())),
                },
                "retrieval_lanes": {lane: dict(metrics) for lane, metrics in sorted(candidate.lanes.items())},
                "lane_noninferiority_lower_bounds": {
                    lane: dict(metrics) for lane, metrics in sorted(candidate.lane_lower_bounds.items())
                },
                "strata": {
                    f"stratum-{index:02d}": dict(metrics)
                    for index, (_stratum, metrics) in enumerate(sorted(candidate.strata.items()), start=1)
                },
                "latency": candidate.latency_summary(),
            }
            for candidate in sorted(decision.candidates, key=_selection_key)
        ],
    }
