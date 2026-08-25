"""Deterministic, dependency-free metrics for the frozen caption selection study."""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from statistics import fmean, stdev
from typing import Mapping, Sequence


_NORMALISE_RE = re.compile(r"[^\w]+", re.UNICODE)


def ndcg_at_k(ranking: Sequence[str], grades: Mapping[str, int], k: int = 10) -> float:
    """Graded nDCG with gains ``2**grade - 1`` and no post-hoc rank tuning."""

    if k <= 0:
        raise ValueError("k must be positive")

    def dcg(ids: Sequence[str]) -> float:
        return sum(
            (2 ** max(0, grades.get(item_id, 0)) - 1) / math.log2(rank + 1)
            for rank, item_id in enumerate(ids[:k], start=1)
        )

    ideal = sorted(grades, key=lambda item_id: (-grades[item_id], item_id))
    denominator = dcg(ideal)
    return dcg(ranking) / denominator if denominator else 0.0


def recall_at_k(ranking: Sequence[str], grades: Mapping[str, int], k: int = 10) -> float:
    intended = {item_id for item_id, grade in grades.items() if grade > 0}
    return len(intended.intersection(ranking[:k])) / len(intended) if intended else 0.0


def reciprocal_rank(ranking: Sequence[str], grades: Mapping[str, int]) -> float:
    for rank, item_id in enumerate(ranking, start=1):
        if grades.get(item_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def _normalise(value: str) -> str:
    return _NORMALISE_RE.sub("", value.casefold())


def _lcs_length(left: str, right: str) -> int:
    prior = [0] * (len(right) + 1)
    for character in left:
        current = [0]
        for index, other in enumerate(right, start=1):
            current.append(prior[index - 1] + 1 if character == other else max(prior[index], current[-1]))
        prior = current
    return prior[-1]


def _levenshtein(left: str, right: str) -> int:
    row = list(range(len(right) + 1))
    for i, character in enumerate(left, start=1):
        next_row = [i]
        for j, other in enumerate(right, start=1):
            next_row.append(min(row[j] + 1, next_row[j - 1] + 1, row[j - 1] + (character != other)))
        row = next_row
    return row[-1]


def ocr_character_f1(predicted: str, expected: str) -> float:
    predicted, expected = _normalise(predicted), _normalise(expected)
    if not predicted and not expected:
        return 1.0
    overlap = _lcs_length(predicted, expected)
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(expected) if expected else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def normalised_character_error_rate(predicted: str, expected: str) -> float:
    predicted, expected = _normalise(predicted), _normalise(expected)
    if not expected:
        return 0.0 if not predicted else 1.0
    return _levenshtein(predicted, expected) / len(expected)


@dataclass(frozen=True)
class ConceptScores:
    precision: float
    recall: float
    f1: float


def concept_scores(predicted: Sequence[str], expected_groups: Sequence[set[str]]) -> ConceptScores:
    """Score predeclared alias groups, never a post-hoc embedding similarity."""

    normalised_predicted = {_normalise(value) for value in predicted if _normalise(value)}
    groups = [{_normalise(alias) for alias in group} for group in expected_groups]
    matched_groups = sum(bool(group.intersection(normalised_predicted)) for group in groups)
    recognised_terms = set().union(*groups) if groups else set()
    matched_predicted = sum(value in recognised_terms for value in normalised_predicted)
    precision = matched_predicted / len(normalised_predicted) if normalised_predicted else 0.0
    recall = matched_groups / len(groups) if groups else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return ConceptScores(precision, recall, f1)


def critical_hallucination_count(flags: Sequence[bool]) -> int:
    """Count only labeler-predeclared critical flags."""

    return sum(bool(flag) for flag in flags)


def cluster_bootstrap_lower_bound(
    candidate: Mapping[str, Sequence[float]],
    comparator: Mapping[str, Sequence[float]],
    *,
    seed: int,
    resamples: int = 10_000,
) -> float:
    """One-sided 95% paired cluster-bootstrap lower bound.

    A target image, not each of its queries, is the resampling unit.
    """

    clusters = sorted(set(candidate).intersection(comparator))
    if not clusters or resamples <= 0:
        raise ValueError("paired clusters and positive resamples are required")
    differences = {
        cluster: fmean(candidate[cluster]) - fmean(comparator[cluster])
        for cluster in clusters
        if candidate[cluster] and comparator[cluster]
    }
    if not differences:
        raise ValueError("every paired cluster is empty")
    cluster_ids = sorted(differences)
    rng = random.Random(seed)
    samples = sorted(
        fmean(differences[rng.choice(cluster_ids)] for _ in cluster_ids)
        for _ in range(resamples)
    )
    return samples[max(0, math.ceil(0.05 * resamples) - 1)]


def _simulated_power(
    differences: Sequence[float],
    *,
    target_count: int,
    margin: float,
    seed: int,
    simulations: int,
) -> float:
    rng = random.Random(seed + target_count)
    successes = 0
    for _ in range(simulations):
        sample = [rng.choice(differences) for _ in range(target_count)]
        mean = fmean(sample)
        standard_error = stdev(sample) / math.sqrt(target_count) if len(set(sample)) > 1 else 0.0
        if mean - 1.6448536269514722 * standard_error >= -margin:
            successes += 1
    return successes / simulations


def required_target_count(
    development_differences: Sequence[float],
    *,
    margin: float,
    seed: int,
    simulations: int = 10_000,
    minimum: int = 60,
    maximum: int = 120,
) -> int | None:
    """Return a powered bounded hidden size, else explicitly report infeasibility."""

    if not development_differences or minimum <= 0 or maximum < minimum or simulations <= 0:
        raise ValueError("valid development differences, bounds and simulations are required")
    for count in range(minimum, maximum + 1):
        if _simulated_power(development_differences, target_count=count, margin=margin, seed=seed, simulations=simulations) >= 0.80:
            return count
    return None
