"""Sealed hidden Stage C scoring; Stage B inputs are never mutated."""

from __future__ import annotations

import random
from typing import Any, Callable, Mapping, Sequence

from .fixture_builder import validate_private_manifest
from .report import FROZEN_EVALUATION_THRESHOLDS, FROZEN_FINAL_SELECTION_THRESHOLDS
from .study import CandidateSpec, StudyValidationError, artifact_sha256, run_stage_b_study


class StageCValidationError(ValueError):
    """A hidden-set artifact is unsealed, inconsistent, or incomplete."""


_GATE_KEYS = {
    "stage_c_gates_version",
    "snapshot_hash",
    "stage_b_effects_hash",
    "guide_hash",
    "labels_hash",
    "query_artifact_hash",
    "embedding_contract_hash",
    "candidates",
    "comparator",
    "selection_algorithm",
    "selection_seed",
    "seed",
    "prompt_version",
    "input_preparation_version",
    "thresholds",
    "final_selection_thresholds",
}
_EFFECT_KEYS = {
    "stage_b_effects_version",
    "stage_b_results_hash",
    "stage_c_candidates",
    "candidate_powered_target_counts",
    "powered_hidden_target_count",
}
_APPROVAL_KEYS = {"approval_version", "approved", "gates_hash"}


def validate_stage_c_approval(
    approval: Mapping[str, Any],
    *,
    gates: Mapping[str, Any],
) -> None:
    """Require the exact versioned human approval before hidden pixels open."""

    if (
        set(approval) != _APPROVAL_KEYS
        or approval.get("approval_version") != 1
        or approval.get("approved") is not True
        or approval.get("gates_hash") != artifact_sha256(gates)
    ):
        raise StageCValidationError("hash-bound human audit approval is required")


def _spec(value: Any) -> CandidateSpec:
    if not isinstance(value, str) or value.count(":") != 2:
        raise StageCValidationError("candidate identity is invalid")
    try:
        return CandidateSpec(*value.split(":"))
    except StudyValidationError as error:
        raise StageCValidationError("candidate identity is invalid") from error


def build_stage_c_effects(
    stage_b_results: Mapping[str, Any], *, stage_c_candidates: Sequence[str]
) -> dict[str, Any]:
    """Derive Stage C power only from the finalists that will enter Stage C.

    An eliminated candidate with an infeasible power estimate cannot block a
    viable finalist.  The returned artifact is separately hash-bound by the
    Stage C gates, while retaining the exact Stage B result hash for custody.
    """

    if not isinstance(stage_b_results.get("candidates"), list):
        raise StageCValidationError("Stage B results must contain candidate aggregates")
    selected = list(stage_c_candidates)
    if not selected or len(set(selected)) != len(selected):
        raise StageCValidationError("Stage C finalists must be nonempty and unique")
    for identity in selected:
        _spec(identity)
    counts: dict[str, int] = {}
    available: dict[str, Mapping[str, Any]] = {}
    for aggregate in stage_b_results["candidates"]:
        if not isinstance(aggregate, Mapping):
            raise StageCValidationError("Stage B candidate aggregate is invalid")
        identity = ":".join(
            str(aggregate.get(field, ""))
            for field in ("model", "effort", "prompt_version")
        )
        if identity in available:
            raise StageCValidationError("Stage B candidate identities are duplicated")
        available[identity] = aggregate
    if not set(selected).issubset(available):
        raise StageCValidationError("Stage C finalist is absent from Stage B results")
    for identity in selected:
        count = available[identity].get("powered_hidden_target_count")
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or not 60 <= count <= 120
        ):
            raise StageCValidationError("every Stage C finalist needs a feasible powered target count")
        counts[identity] = count
    return {
        "stage_b_effects_version": 2,
        "stage_b_results_hash": artifact_sha256(stage_b_results),
        "stage_c_candidates": selected,
        "candidate_powered_target_counts": counts,
        "powered_hidden_target_count": max(counts.values()),
    }


def _stage_c_corpus(
    *,
    snapshot: Mapping[str, Any],
    gates: Mapping[str, Any],
    stage_b_effects: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], set[str]]:
    """Deterministically select powered targets and retain all C distractors."""

    try:
        validate_private_manifest(snapshot)
    except (TypeError, ValueError) as error:
        raise StageCValidationError("a valid sealed private snapshot is required") from error
    if set(gates) != _GATE_KEYS or gates.get("stage_c_gates_version") != 2:
        raise StageCValidationError("strict Stage C gates v2 are required")
    if (
        gates.get("thresholds") != FROZEN_EVALUATION_THRESHOLDS
        or gates.get("final_selection_thresholds") != FROZEN_FINAL_SELECTION_THRESHOLDS
    ):
        raise StageCValidationError("Stage C thresholds do not match the frozen runtime contract")
    if artifact_sha256(snapshot) != gates.get("snapshot_hash"):
        raise StageCValidationError("Stage C gates do not bind the exact snapshot")
    if (
        set(stage_b_effects) != _EFFECT_KEYS
        or stage_b_effects.get("stage_b_effects_version") != 2
        or artifact_sha256(stage_b_effects) != gates.get("stage_b_effects_hash")
    ):
        raise StageCValidationError("hash-bound Stage B powered effects are required")
    if (
        stage_b_effects.get("stage_c_candidates") != gates.get("candidates")
        or not isinstance(stage_b_effects.get("stage_b_results_hash"), str)
        or not stage_b_effects["stage_b_results_hash"].startswith("sha256:")
        or not isinstance(stage_b_effects.get("candidate_powered_target_counts"), Mapping)
        or set(stage_b_effects["candidate_powered_target_counts"]) != set(gates.get("candidates", []))
    ):
        raise StageCValidationError("Stage B power effects do not bind the exact Stage C finalists")
    powered_counts = stage_b_effects["candidate_powered_target_counts"]
    target_count = stage_b_effects.get("powered_hidden_target_count")
    if (
        not isinstance(target_count, int)
        or isinstance(target_count, bool)
        or not 60 <= target_count <= 120
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 60 <= value <= 120
            for value in powered_counts.values()
        )
        or target_count != max(powered_counts.values())
        or gates.get("selection_algorithm") != "stratified-shuffle-v1"
        or not isinstance(gates.get("selection_seed"), int)
        or isinstance(gates.get("selection_seed"), bool)
    ):
        raise StageCValidationError("powered Stage C target selection is invalid")

    fixtures = snapshot["fixtures"]
    targets = [fixture for fixture in fixtures if fixture["stage"] == "C" and fixture["role"] == "target"]
    if target_count > len(targets):
        raise StageCValidationError("powered target count exceeds sealed Stage C targets")
    pools: dict[str, list[Mapping[str, Any]]] = {}
    for target in targets:
        pools.setdefault(target["stratum"], []).append(target)
    rng = random.Random(gates["selection_seed"])
    for pool in pools.values():
        pool.sort(key=lambda fixture: fixture["fixture_id"])
        rng.shuffle(pool)

    selected: list[Mapping[str, Any]] = []
    while len(selected) < target_count:
        for stratum in sorted(pools):
            if pools[stratum] and len(selected) < target_count:
                selected.append(pools[stratum].pop())
    selected_ids = {fixture["fixture_id"] for fixture in selected}
    corpus = [
        {**fixture, "stage": "B"}
        for fixture in fixtures
        if fixture["stage"] == "C"
        and (fixture["role"] == "distractor" or fixture["fixture_id"] in selected_ids)
    ]
    return corpus, selected_ids


def run_stage_c_study(
    *,
    snapshot: Mapping[str, Any],
    gates: Mapping[str, Any],
    approval: Mapping[str, Any],
    stage_b_effects: Mapping[str, Any],
    guide: Mapping[str, Any],
    labels: Mapping[str, Any],
    queries: Mapping[str, Any],
    embedding_contract: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    embed: Callable[[str], Sequence[float]],
    pooled_relevance: Mapping[str, Any] | None = None,
    expected_pooled_relevance_hash: str = "",
    blind_pool: Mapping[str, Any] | None = None,
    expected_blind_pool_hash: str = "",
    expected_packet_evidence_hash: str = "",
    selection_mode: str = "preliminary",
    power_simulations: int = 10_000,
) -> dict[str, Any]:
    """Score a deterministic hidden corpus only after a hash-bound approval."""

    corpus, selected_ids = _stage_c_corpus(
        snapshot=snapshot,
        gates=gates,
        stage_b_effects=stage_b_effects,
    )
    validate_stage_c_approval(approval, gates=gates)
    raw_candidates = gates.get("candidates")
    if not isinstance(raw_candidates, list):
        raise StageCValidationError("Stage C candidates must be an array")
    specs = [_spec(value) for value in raw_candidates]
    if not specs or len(set(specs)) != len(specs):
        raise StageCValidationError("Stage C candidates must be nonempty and unique")
    comparator = _spec(gates.get("comparator"))
    if comparator not in specs:
        raise StageCValidationError("comparator is not a sealed candidate")
    if any(spec.prompt_version != gates.get("prompt_version") for spec in specs):
        raise StageCValidationError("candidate prompts do not match the Stage C gates")

    corpus_ids = {fixture["fixture_id"] for fixture in corpus}
    candidate_keys = {spec.receipt_key() for spec in specs}
    selected_receipts = [
        receipt
        for receipt in receipts
        if isinstance(receipt, Mapping)
        and receipt.get("fixture_id") in corpus_ids
        and (
            receipt.get("model"),
            receipt.get("effort"),
            receipt.get("prompt_version"),
        )
        in candidate_keys
    ]
    try:
        result = run_stage_b_study(
            guide=guide,
            labels=labels,
            expected_guide_hash=gates["guide_hash"],
            expected_labels_hash=gates["labels_hash"],
            fixtures=corpus,
            receipts=selected_receipts,
            query_artifact=queries,
            expected_query_hash=gates["query_artifact_hash"],
            pooled_relevance_artifact=pooled_relevance,
            expected_pooled_relevance_hash=expected_pooled_relevance_hash,
            blind_pool_artifact=blind_pool,
            expected_blind_pool_hash=expected_blind_pool_hash,
            expected_packet_evidence_hash=expected_packet_evidence_hash,
            selection_mode=selection_mode,
            embedding_contract=embedding_contract,
            expected_embedding_contract_hash=gates["embedding_contract_hash"],
            embed=embed,
            candidates=specs,
            comparator=comparator,
            manifest_hash=gates["snapshot_hash"],
            seed=gates["seed"],
            power_simulations=power_simulations,
        )
    except (StudyValidationError, KeyError, TypeError, ValueError) as error:
        raise StageCValidationError(str(error)) from error
    return {
        **result,
        "stage": "C",
        "stage_c_gates_version": 2,
        "final_selection_thresholds": FROZEN_FINAL_SELECTION_THRESHOLDS,
        "stage_c_gates_hash": artifact_sha256(gates),
        "approval_hash": artifact_sha256(approval),
        "selected_target_count": len(selected_ids),
        "completed": len(selected_receipts),
    }
