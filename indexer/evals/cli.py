"""Pure CLI-support functions for private evaluation artifacts.

The shared application CLI owns argument parsing; this module supplies safe,
testable operations it can call after resolving user arguments.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .fixture_builder import build_private_manifest, private_manifest_summary
from .report import (
    CandidateAggregate,
    FROZEN_EVALUATION_THRESHOLDS,
    FROZEN_FINAL_SELECTION_THRESHOLDS,
    decide_winner,
    render_anonymised_report,
)
from .runner import PRIVATE_EVAL_ROOT, load_fixture_manifest
from .stage_c import StageCValidationError, run_stage_c_study, validate_stage_c_approval
from .study import CandidateSpec, StudyValidationError, artifact_sha256, run_stage_b_study
from src.contracts import CaptionReceiptV1, ContractError
from src.captioning.codex_cli import CAPTION_INPUT_PREPARATION_VERSION, CAPTION_PROVIDER_NAME
from src.captioning.prompt import CAPTION_PROMPT_VERSION
from src.db import VECTOR_DIMENSIONS


class EvalCliError(ValueError):
    """A safe, stable private-evaluation CLI-support failure."""


_V1_GATE_KEYS = {
    "gates_version",
    "guide_hash",
    "labels_hash",
    "query_artifact_hash",
    "embedding_contract_hash",
    "candidates",
    "comparator",
    "prompt_version",
    "input_preparation_version",
    "thresholds",
}
_V2_GATE_KEYS = _V1_GATE_KEYS | {"final_selection_thresholds"}


def _root(path: Path | None) -> Path:
    return (path or PRIVATE_EVAL_ROOT).expanduser().resolve()


def _private_path(path: Path, *, allowed_root: Path | None) -> Path:
    resolved = Path(path).expanduser().resolve()
    root = _root(allowed_root)
    if not resolved.is_relative_to(root):
        raise EvalCliError(f"private evaluation files must be under {root}")
    return resolved


def _load_json(path: Path, *, allowed_root: Path | None) -> Any:
    resolved = _private_path(path, allowed_root=allowed_root)
    try:
        with resolved.open(encoding="utf-8") as source:
            return json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise EvalCliError("private evaluation JSON could not be read") from error


def _make_owner_only_directories(directory: Path, *, root: Path) -> None:
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not root.is_dir():
        raise EvalCliError("private evaluation root must be a directory")
    os.chmod(root, 0o700)
    current = root
    for part in directory.relative_to(root).parts:
        current = current / part
        current.mkdir(mode=0o700, exist_ok=True)
        if not current.is_dir():
            raise EvalCliError("private evaluation output parent must be a directory")
        os.chmod(current, 0o700)


def _write_json(path: Path, payload: Mapping[str, Any], *, allowed_root: Path | None) -> None:
    resolved = _private_path(path, allowed_root=allowed_root)
    _make_owner_only_directories(resolved.parent, root=_root(allowed_root))
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
    except Exception:
        os.close(descriptor)
        raise
    with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
        destination.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(resolved)
    os.chmod(resolved, 0o600)


def create_manifest_json(
    candidates_path: Path,
    output_path: Path,
    *,
    snapshot_id: str,
    seed: int,
    sealing_inputs: Mapping[str, Any],
    allowed_root: Path | None = None,
) -> dict[str, Any]:
    """Build and privately persist a deterministic fixture manifest.

    The returned value is anonymised; the full manifest stays only at output_path.
    """

    candidates = _load_json(candidates_path, allowed_root=allowed_root)
    if not isinstance(candidates, list):
        raise EvalCliError("fixture candidates JSON must be an array")
    try:
        manifest = build_private_manifest(candidates, snapshot_id=snapshot_id, seed=seed, sealing_inputs=sealing_inputs)
    except (TypeError, ValueError) as error:
        raise EvalCliError("fixture manifest could not be built") from error
    _write_json(output_path, manifest, allowed_root=allowed_root)
    # The seal is deliberately retained in the private manifest, but a digest
    # is still identifying material and therefore not part of the CLI's
    # anonymous return payload.
    return {
        key: value
        for key, value in private_manifest_summary(manifest).items()
        if key != "hidden_seal"
    }


def _candidate(payload: Mapping[str, Any]) -> CandidateAggregate:
    required_fields = {
        "model", "effort", "schema_valid_rate", "critical_hallucinations", "concept_precision", "concept_f1",
        "ocr_character_f1", "ndcg_at_10", "recall_at_10", "known_query_passed",
        "background_throughput_accepted", "lower_bounds", "lanes", "latencies_seconds",
    }
    optional_fields = {
        "lane_lower_bounds", "ocr_bearing_target_count", "ocr_bearing_character_f1",
        "strata", "selection_mode", "pooled_relevance_hash", "concept_recall", "annotation_guide_version",
    }
    if not required_fields.issubset(payload):
        raise EvalCliError("evaluation candidate is missing aggregate metrics")
    try:
        fields = required_fields | (optional_fields & set(payload))
        return CandidateAggregate(**{field: payload[field] for field in fields})
    except (TypeError, ValueError) as error:
        raise EvalCliError("evaluation candidate has invalid aggregate metrics") from error


def _candidate_spec(value: str) -> CandidateSpec:
    parts = value.split(":")
    if len(parts) != 3:
        raise EvalCliError("candidate must be model:effort:prompt-version")
    try:
        return CandidateSpec(*parts)
    except StudyValidationError as error:
        raise EvalCliError("candidate identity is invalid") from error


def _sealed_gates(
    *,
    raw_manifest: Mapping[str, Any],
    gates_path: Path,
    allowed_root: Path | None,
) -> tuple[dict[str, Any], list[CandidateSpec], CandidateSpec, str]:
    sealing = raw_manifest.get("hidden_sealing_inputs")
    if not isinstance(sealing, Mapping):
        raise EvalCliError("sealed private manifest is required")
    gates = _load_json(gates_path, allowed_root=allowed_root)
    if not isinstance(gates, Mapping):
        raise EvalCliError("sealed gates artifact is invalid")
    gate_version = gates.get("gates_version")
    if gate_version == 1:
        expected_keys = _V1_GATE_KEYS
    elif gate_version == 2:
        expected_keys = _V2_GATE_KEYS
    else:
        raise EvalCliError("sealed gates artifact is invalid")
    if set(gates) != expected_keys:
        raise EvalCliError("sealed gates artifact is invalid")
    gates_hash = artifact_sha256(gates)
    if gates_hash != sealing.get("gates_hash") or gates.get("labels_hash") != sealing.get("labels_hash"):
        raise EvalCliError("sealed gates do not match the manifest")
    if sealing.get("instrument_version") != "selection-instrument-v1":
        raise EvalCliError("sealed selection instrument is unsupported")
    if gates.get("prompt_version") != CAPTION_PROMPT_VERSION:
        raise EvalCliError("sealed caption prompt does not match the runtime")
    if gates.get("input_preparation_version") != CAPTION_INPUT_PREPARATION_VERSION:
        raise EvalCliError("sealed image preparation does not match the runtime")
    if gates.get("thresholds") != FROZEN_EVALUATION_THRESHOLDS:
        raise EvalCliError("sealed decision thresholds do not match the runtime")
    if gate_version == 2 and gates.get("final_selection_thresholds") != FROZEN_FINAL_SELECTION_THRESHOLDS:
        raise EvalCliError("sealed final-selection thresholds do not match the runtime")
    if not all(
        isinstance(gates.get(key), str) and gates[key]
        for key in ("guide_hash", "labels_hash", "query_artifact_hash", "embedding_contract_hash", "comparator")
    ) or not isinstance(gates.get("candidates"), list):
        raise EvalCliError("sealed gates artifact is incomplete")
    specs = [_candidate_spec(value) for value in gates["candidates"]]
    if not specs or len(set(specs)) != len(specs):
        raise EvalCliError("sealed gates candidates must be nonempty and unique")
    chosen = _candidate_spec(gates["comparator"])
    if any(spec.prompt_version != CAPTION_PROMPT_VERSION for spec in specs) or chosen not in specs:
        raise EvalCliError("sealed gates candidate identities are inconsistent")
    return dict(gates), specs, chosen, gates_hash


def aggregate_report_json(
    results_path: Path,
    output_path: Path,
    *,
    snapshot: Path,
    gates_path: Path,
    allowed_root: Path | None = None,
    final_selection: bool = False,
) -> dict[str, Any]:
    """Decide only from a sealed score envelope and its exact gate contract."""

    if final_selection:
        raise EvalCliError(
            "final selection reports are emitted only by eval-score while it validates and scores the sealed raw evidence"
        )

    raw_manifest = _load_json(Path(snapshot) / "manifest.json", allowed_root=allowed_root)
    if not isinstance(raw_manifest, Mapping):
        raise EvalCliError("sealed private manifest is invalid")
    try:
        manifest = load_fixture_manifest(Path(snapshot), enforce_private_root=False)
    except (OSError, ValueError) as error:
        raise EvalCliError("sealed private manifest is invalid") from error
    gates, specs, comparator, gates_hash = _sealed_gates(
        raw_manifest=raw_manifest,
        gates_path=gates_path,
        allowed_root=allowed_root,
    )
    raw = _load_json(results_path, allowed_root=allowed_root)
    if not isinstance(raw, Mapping) or not isinstance(raw.get("candidates"), list):
        raise EvalCliError("evaluation results JSON must contain candidates")
    if any(not isinstance(candidate, Mapping) for candidate in raw["candidates"]):
        raise EvalCliError("evaluation results candidates must be aggregate objects")
    if any(candidate.get("background_throughput_accepted") is not False for candidate in raw["candidates"]):
        raise EvalCliError("background throughput requires a separately sealed amendment")
    expected_evidence = {
        "manifest_hash": manifest.manifest_hash,
        "guide_hash": gates["guide_hash"],
        "labels_hash": gates["labels_hash"],
        "query_artifact_hash": gates["query_artifact_hash"],
        "embedding_contract_hash": gates["embedding_contract_hash"],
        "gates_hash": gates_hash,
        "comparator": gates["comparator"],
        "input_preparation_version": CAPTION_INPUT_PREPARATION_VERSION,
        "thresholds": FROZEN_EVALUATION_THRESHOLDS,
    }
    if any(raw.get(key) != value for key, value in expected_evidence.items()):
        raise EvalCliError("evaluation results do not match the sealed decision evidence")
    expected_mode = "final" if final_selection else "preliminary"
    if raw.get("selection_mode", "preliminary") != expected_mode:
        raise EvalCliError("final selection requires a matching final score envelope")
    if final_selection and (
        not isinstance(raw.get("pooled_relevance_hash"), str)
        or not str(raw["pooled_relevance_hash"]).startswith("sha256:")
        or not isinstance(raw.get("blind_pool_hash"), str)
        or not str(raw["blind_pool_hash"]).startswith("sha256:")
    ):
        raise EvalCliError("final selection requires hash-bound blind-pool and pooled-relevance evidence")
    if final_selection and raw.get("annotation_guide_version") != 2:
        raise EvalCliError("final selection requires annotation guide v2")
    if final_selection and (
        gates.get("gates_version") != 2
        or raw.get("gates_version") != 2
        or raw.get("final_selection_thresholds") != gates.get("final_selection_thresholds")
        or raw.get("final_selection_thresholds") != FROZEN_FINAL_SELECTION_THRESHOLDS
    ):
        raise EvalCliError("final selection requires exact sealed final-selection thresholds")
    result_identities = {
        (candidate.get("model"), candidate.get("effort"), candidate.get("prompt_version"))
        for candidate in raw["candidates"]
    }
    if result_identities != {spec.receipt_key() for spec in specs} or len(raw["candidates"]) != len(specs):
        raise EvalCliError("evaluation result candidates do not match the sealed gates")
    decision = decide_winner(
        [_candidate(candidate) for candidate in raw["candidates"]],
        comparator=comparator.model,
        final_selection=final_selection,
    )
    report = render_anonymised_report(decision)
    _write_json(output_path, report, allowed_root=allowed_root)
    return report


def _validated_caption_receipts(
    snapshot: Path,
    *,
    allowed_root: Path | None,
) -> list[dict[str, Any]]:
    """Cross-check every successful eval journal against its immutable receipt.

    The journal contains fixture and latency metadata, while the model receipt is
    the checksummed authority for the image, provider, model, effort, prompt and
    structured caption.  A caller can therefore never relabel one model's output
    as another candidate merely by changing journal fields.
    """

    snapshot = _private_path(snapshot, allowed_root=allowed_root)
    journal_root = _private_path(snapshot / "caption-receipts", allowed_root=allowed_root)
    model_root = _private_path(snapshot / "model-receipts", allowed_root=allowed_root)
    receipts: list[dict[str, Any]] = []
    if not journal_root.is_dir():
        return receipts
    for path in sorted(journal_root.glob("*.json")):
        if path.name.endswith(".failed.json"):
            continue
        journal = _load_json(path, allowed_root=allowed_root)
        required = {
            "fixture_id", "model", "effort", "prompt_version", "image_hash",
            "receipt_id", "caption_result", "search_text", "manifest_hash",
            "latency_seconds",
        }
        if not isinstance(journal, Mapping) or not required.issubset(journal):
            raise EvalCliError("caption journal entry is incomplete")
        model_path = model_root / str(journal["image_hash"]) / f'{journal["receipt_id"]}.json'
        raw_model = _load_json(model_path, allowed_root=allowed_root)
        try:
            model_receipt = CaptionReceiptV1.from_dict(raw_model)
        except (ContractError, TypeError, ValueError) as error:
            raise EvalCliError("immutable model receipt is invalid") from error
        journal_identity = (
            journal["image_hash"], journal["model"], journal["effort"],
            journal["prompt_version"], journal["receipt_id"],
        )
        receipt_identity = (
            model_receipt.image_hash, model_receipt.model, model_receipt.effort,
            model_receipt.prompt_version, model_receipt.receipt_id,
        )
        if journal_identity != receipt_identity:
            raise EvalCliError("caption journal identity does not match immutable model receipt")
        if model_receipt.provider != CAPTION_PROVIDER_NAME or model_receipt.source != "vision":
            raise EvalCliError("caption model receipt is not authenticated Codex vision evidence")
        if (
            journal["caption_result"] != model_receipt.caption_result.to_dict()
            or journal["search_text"] != model_receipt.search_text
        ):
            raise EvalCliError("caption journal content does not match immutable model receipt")
        receipts.append(dict(journal))
    return receipts


def score_stage_b_json(
    *,
    snapshot: Path,
    guide_path: Path,
    labels_path: Path,
    query_path: Path,
    embedding_contract_path: Path,
    gates_path: Path,
    output_path: Path,
    embed: Callable[[str], Sequence[float]],
    allowed_root: Path | None = None,
    embedder_model: str = "",
    selection_mode: str = "preliminary",
    pooled_relevance_path: Path | None = None,
    expected_pooled_relevance_hash: str = "",
    blind_pool_path: Path | None = None,
    expected_blind_pool_hash: str = "",
    expected_packet_evidence_hash: str = "",
) -> dict[str, Any]:
    """Score completed private Stage B receipts; never accept aggregate metrics."""
    raw_manifest = _load_json(Path(snapshot) / "manifest.json", allowed_root=allowed_root)
    if not isinstance(raw_manifest, Mapping) or not isinstance(raw_manifest.get("hidden_sealing_inputs"), Mapping):
        raise EvalCliError("sealed private manifest is required for Stage B scoring")
    try:
        manifest = load_fixture_manifest(Path(snapshot), enforce_private_root=False)
    except (OSError, ValueError) as error:
        raise EvalCliError("sealed private manifest is invalid") from error
    gates, specs, chosen, gates_hash = _sealed_gates(
        raw_manifest=raw_manifest,
        gates_path=gates_path,
        allowed_root=allowed_root,
    )
    if selection_mode == "final" and gates["gates_version"] != 2:
        raise EvalCliError("final Stage B scoring requires sealed gates v2")
    guide = _load_json(guide_path, allowed_root=allowed_root)
    labels = _load_json(labels_path, allowed_root=allowed_root)
    queries = _load_json(query_path, allowed_root=allowed_root)
    pooled_relevance = (
        _load_json(pooled_relevance_path, allowed_root=allowed_root)
        if pooled_relevance_path is not None
        else None
    )
    blind_pool = (
        _load_json(blind_pool_path, allowed_root=allowed_root)
        if blind_pool_path is not None
        else None
    )
    contract = _load_json(embedding_contract_path, allowed_root=allowed_root)
    if not isinstance(contract, Mapping) or contract.get("model") != embedder_model or contract.get("dimensions") != VECTOR_DIMENSIONS:
        raise EvalCliError("production embedder does not match the frozen embedding contract")
    receipts = _validated_caption_receipts(Path(snapshot), allowed_root=allowed_root)
    if not receipts:
        raise EvalCliError("Stage B scoring requires completed caption receipts")
    try:
        result = run_stage_b_study(
            guide=guide, labels=labels, expected_guide_hash=gates["guide_hash"], expected_labels_hash=gates["labels_hash"],
            fixtures=raw_manifest.get("fixtures", []), receipts=receipts,
            query_artifact=queries, expected_query_hash=gates["query_artifact_hash"],
            pooled_relevance_artifact=pooled_relevance,
            expected_pooled_relevance_hash=expected_pooled_relevance_hash,
            blind_pool_artifact=blind_pool,
            expected_blind_pool_hash=expected_blind_pool_hash,
            expected_packet_evidence_hash=expected_packet_evidence_hash,
            selection_mode=selection_mode,
            embedding_contract=contract, expected_embedding_contract_hash=gates["embedding_contract_hash"],
            embed=embed, candidates=specs, comparator=chosen,
            manifest_hash=manifest.manifest_hash, seed=int(raw_manifest["seed"]),
        )
    except (StudyValidationError, TypeError, ValueError) as error:
        raise EvalCliError("sealed Stage B study could not be scored") from error
    result["created"] = 0
    result["gates_hash"] = gates_hash
    result["gates_version"] = gates["gates_version"]
    result["comparator"] = gates["comparator"]
    result["input_preparation_version"] = CAPTION_INPUT_PREPARATION_VERSION
    result["thresholds"] = FROZEN_EVALUATION_THRESHOLDS
    if gates["gates_version"] == 2:
        result["final_selection_thresholds"] = FROZEN_FINAL_SELECTION_THRESHOLDS
    stage_b_ids = {fixture.fixture_id for fixture in manifest.fixtures_for("B")}
    wanted = {spec.receipt_key() for spec in specs}
    result["completed"] = sum(
        isinstance(receipt, Mapping) and receipt.get("fixture_id") in stage_b_ids
        and (receipt.get("model"), receipt.get("effort"), receipt.get("prompt_version")) in wanted
        for receipt in receipts
    )
    if selection_mode == "final":
        decision = decide_winner(
            [_candidate(candidate) for candidate in result["candidates"]],
            comparator=chosen.model,
            final_selection=True,
        )
        result["decision_report"] = render_anonymised_report(decision)
    _write_json(output_path, result, allowed_root=allowed_root)
    return result


def _stage_c_gate_inputs(
    *,
    snapshot: Path,
    gates_path: Path,
    approval_path: Path,
    allowed_root: Path | None,
) -> tuple[dict[str, Any], dict[str, Any], Mapping[str, Any], str]:
    raw_manifest = _load_json(Path(snapshot) / "manifest.json", allowed_root=allowed_root)
    gates = _load_json(gates_path, allowed_root=allowed_root)
    approval = _load_json(approval_path, allowed_root=allowed_root)
    if not isinstance(raw_manifest, Mapping) or not isinstance(gates, Mapping) or not isinstance(approval, Mapping):
        raise EvalCliError("sealed Stage C manifest, gates and approval are required")
    try:
        manifest = load_fixture_manifest(Path(snapshot), enforce_private_root=False)
    except (OSError, ValueError) as error:
        raise EvalCliError("sealed private manifest is invalid") from error
    if artifact_sha256(raw_manifest) != manifest.manifest_hash:
        raise EvalCliError("Stage C manifest hash is inconsistent")
    if (
        gates.get("stage_c_gates_version") != 2
        or gates.get("snapshot_hash") != manifest.manifest_hash
        or gates.get("prompt_version") != CAPTION_PROMPT_VERSION
        or gates.get("input_preparation_version") != CAPTION_INPUT_PREPARATION_VERSION
        or gates.get("thresholds") != FROZEN_EVALUATION_THRESHOLDS
        or gates.get("final_selection_thresholds") != FROZEN_FINAL_SELECTION_THRESHOLDS
    ):
        raise EvalCliError("Stage C gates or human approval do not match the runtime seal")
    try:
        validate_stage_c_approval(approval, gates=gates)
    except StageCValidationError as error:
        raise EvalCliError("Stage C gates or human approval do not match the runtime seal") from error
    return dict(raw_manifest), dict(gates), approval, manifest.manifest_hash


def score_stage_c_json(
    *,
    snapshot: Path,
    guide_path: Path,
    labels_path: Path,
    query_path: Path,
    embedding_contract_path: Path,
    gates_path: Path,
    approval_path: Path,
    stage_b_effects_path: Path,
    output_path: Path,
    embed: Callable[[str], Sequence[float]],
    allowed_root: Path | None = None,
    embedder_model: str = "",
    selection_mode: str = "preliminary",
    pooled_relevance_path: Path | None = None,
    expected_pooled_relevance_hash: str = "",
    blind_pool_path: Path | None = None,
    expected_blind_pool_hash: str = "",
    expected_packet_evidence_hash: str = "",
) -> dict[str, Any]:
    """Score the deterministic hidden Stage C corpus from immutable receipts."""

    raw_manifest, gates, approval, manifest_hash = _stage_c_gate_inputs(
        snapshot=snapshot,
        gates_path=gates_path,
        approval_path=approval_path,
        allowed_root=allowed_root,
    )
    guide = _load_json(guide_path, allowed_root=allowed_root)
    labels = _load_json(labels_path, allowed_root=allowed_root)
    queries = _load_json(query_path, allowed_root=allowed_root)
    effects = _load_json(stage_b_effects_path, allowed_root=allowed_root)
    contract = _load_json(embedding_contract_path, allowed_root=allowed_root)
    pooled_relevance = (
        _load_json(pooled_relevance_path, allowed_root=allowed_root)
        if pooled_relevance_path is not None
        else None
    )
    blind_pool = (
        _load_json(blind_pool_path, allowed_root=allowed_root)
        if blind_pool_path is not None
        else None
    )
    if (
        not isinstance(guide, Mapping)
        or not isinstance(labels, Mapping)
        or not isinstance(queries, Mapping)
        or not isinstance(effects, Mapping)
        or not isinstance(contract, Mapping)
    ):
        raise EvalCliError("Stage C study artifacts are invalid")
    if contract.get("model") != embedder_model or contract.get("dimensions") != VECTOR_DIMENSIONS:
        raise EvalCliError("production embedder does not match the frozen embedding contract")
    receipts = _validated_caption_receipts(Path(snapshot), allowed_root=allowed_root)
    if not receipts:
        raise EvalCliError("Stage C scoring requires completed caption receipts")
    try:
        result = run_stage_c_study(
            snapshot=raw_manifest,
            gates=gates,
            approval=approval,
            stage_b_effects=effects,
            guide=guide,
            labels=labels,
            queries=queries,
            embedding_contract=contract,
            receipts=receipts,
            embed=embed,
            pooled_relevance=pooled_relevance,
            expected_pooled_relevance_hash=expected_pooled_relevance_hash,
            blind_pool=blind_pool,
            expected_blind_pool_hash=expected_blind_pool_hash,
            expected_packet_evidence_hash=expected_packet_evidence_hash,
            selection_mode=selection_mode,
        )
    except (StageCValidationError, TypeError, ValueError) as error:
        raise EvalCliError("sealed Stage C study could not be scored") from error
    result["created"] = 0
    result["manifest_hash"] = manifest_hash
    result["gates_hash"] = artifact_sha256(gates)
    result["comparator"] = gates["comparator"]
    result["input_preparation_version"] = CAPTION_INPUT_PREPARATION_VERSION
    result["thresholds"] = FROZEN_EVALUATION_THRESHOLDS
    result["stage_c_gates_version"] = 2
    result["final_selection_thresholds"] = FROZEN_FINAL_SELECTION_THRESHOLDS
    if selection_mode == "final":
        comparator = _candidate_spec(str(gates["comparator"]))
        decision = decide_winner(
            [_candidate(candidate) for candidate in result["candidates"]],
            comparator=comparator.model,
            final_selection=True,
        )
        result["decision_report"] = render_anonymised_report(decision)
    _write_json(output_path, result, allowed_root=allowed_root)
    return result


def aggregate_stage_c_report_json(
    results_path: Path,
    output_path: Path,
    *,
    snapshot: Path,
    gates_path: Path,
    approval_path: Path,
    allowed_root: Path | None = None,
    final_selection: bool = False,
) -> dict[str, Any]:
    """Decide from Stage C evidence without accepting a free comparator."""

    if final_selection:
        raise EvalCliError(
            "final Stage C reports are emitted only by eval-score-c while it validates and scores the sealed raw evidence"
        )

    _manifest, gates, approval, manifest_hash = _stage_c_gate_inputs(
        snapshot=snapshot,
        gates_path=gates_path,
        approval_path=approval_path,
        allowed_root=allowed_root,
    )
    raw = _load_json(results_path, allowed_root=allowed_root)
    if not isinstance(raw, Mapping) or not isinstance(raw.get("candidates"), list):
        raise EvalCliError("Stage C results JSON must contain candidates")
    if any(not isinstance(candidate, Mapping) for candidate in raw["candidates"]):
        raise EvalCliError("Stage C result candidates must be aggregate objects")
    if any(candidate.get("background_throughput_accepted") is not False for candidate in raw["candidates"]):
        raise EvalCliError("background throughput requires a separately sealed amendment")
    expected_mode = "final" if final_selection else "preliminary"
    expected = {
        "stage": "C",
        "manifest_hash": manifest_hash,
        "stage_c_gates_hash": artifact_sha256(gates),
        "approval_hash": artifact_sha256(approval),
        "guide_hash": gates["guide_hash"],
        "labels_hash": gates["labels_hash"],
        "query_artifact_hash": gates["query_artifact_hash"],
        "embedding_contract_hash": gates["embedding_contract_hash"],
        "gates_hash": artifact_sha256(gates),
        "comparator": gates["comparator"],
        "selection_mode": expected_mode,
        "input_preparation_version": CAPTION_INPUT_PREPARATION_VERSION,
        "thresholds": FROZEN_EVALUATION_THRESHOLDS,
        "stage_c_gates_version": 2,
        "final_selection_thresholds": FROZEN_FINAL_SELECTION_THRESHOLDS,
    }
    if any(raw.get(key) != value for key, value in expected.items()):
        raise EvalCliError("Stage C results do not match the sealed decision evidence")
    if final_selection and (
        not isinstance(raw.get("pooled_relevance_hash"), str)
        or not raw["pooled_relevance_hash"].startswith("sha256:")
        or not isinstance(raw.get("blind_pool_hash"), str)
        or not raw["blind_pool_hash"].startswith("sha256:")
    ):
        raise EvalCliError("final Stage C selection requires blind-pool and pooled relevance evidence")
    if final_selection and raw.get("annotation_guide_version") != 2:
        raise EvalCliError("final Stage C selection requires annotation guide v2")
    specs = [_candidate_spec(value) for value in gates.get("candidates", [])]
    comparator = _candidate_spec(gates.get("comparator", ""))
    identities = {
        (candidate.get("model"), candidate.get("effort"), candidate.get("prompt_version"))
        for candidate in raw["candidates"]
    }
    if not specs or comparator not in specs or identities != {spec.receipt_key() for spec in specs}:
        raise EvalCliError("Stage C result candidates do not match the sealed gates")
    decision = decide_winner(
        [_candidate(candidate) for candidate in raw["candidates"]],
        comparator=comparator.model,
        final_selection=final_selection,
    )
    report = render_anonymised_report(decision)
    _write_json(output_path, report, allowed_root=allowed_root)
    return report
