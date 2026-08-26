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
    decide_winner,
    render_anonymised_report,
)
from .runner import PRIVATE_EVAL_ROOT, load_fixture_manifest
from .study import CandidateSpec, StudyValidationError, artifact_sha256, run_stage_b_study
from src.contracts import CaptionReceiptV1, ContractError
from src.captioning.codex_cli import CAPTION_INPUT_PREPARATION_VERSION, CAPTION_PROVIDER_NAME
from src.captioning.prompt import CAPTION_PROMPT_VERSION
from src.db import VECTOR_DIMENSIONS


class EvalCliError(ValueError):
    """A safe, stable private-evaluation CLI-support failure."""


_GATE_KEYS = {
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
    fields = {
        "model", "effort", "schema_valid_rate", "critical_hallucinations", "concept_precision", "concept_f1",
        "ocr_character_f1", "ndcg_at_10", "recall_at_10", "known_query_passed",
        "background_throughput_accepted", "lower_bounds", "lanes", "latencies_seconds",
    }
    if not fields.issubset(payload):
        raise EvalCliError("evaluation candidate is missing aggregate metrics")
    try:
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
    if not isinstance(gates, Mapping) or set(gates) != _GATE_KEYS or gates.get("gates_version") != 1:
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
) -> dict[str, Any]:
    """Decide only from a sealed score envelope and its exact gate contract."""

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
    result_identities = {
        (candidate.get("model"), candidate.get("effort"), candidate.get("prompt_version"))
        for candidate in raw["candidates"]
    }
    if result_identities != {spec.receipt_key() for spec in specs} or len(raw["candidates"]) != len(specs):
        raise EvalCliError("evaluation result candidates do not match the sealed gates")
    decision = decide_winner(
        [_candidate(candidate) for candidate in raw["candidates"]],
        comparator=comparator.model,
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
    guide = _load_json(guide_path, allowed_root=allowed_root)
    labels = _load_json(labels_path, allowed_root=allowed_root)
    queries = _load_json(query_path, allowed_root=allowed_root)
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
            embedding_contract=contract, expected_embedding_contract_hash=gates["embedding_contract_hash"],
            embed=embed, candidates=specs, comparator=chosen,
            manifest_hash=manifest.manifest_hash, seed=int(raw_manifest["seed"]),
        )
    except (StudyValidationError, TypeError, ValueError) as error:
        raise EvalCliError("sealed Stage B study could not be scored") from error
    result["created"] = 0
    result["gates_hash"] = gates_hash
    result["comparator"] = gates["comparator"]
    result["input_preparation_version"] = CAPTION_INPUT_PREPARATION_VERSION
    result["thresholds"] = FROZEN_EVALUATION_THRESHOLDS
    stage_b_ids = {fixture.fixture_id for fixture in manifest.fixtures_for("B")}
    wanted = {spec.receipt_key() for spec in specs}
    result["completed"] = sum(
        isinstance(receipt, Mapping) and receipt.get("fixture_id") in stage_b_ids
        and (receipt.get("model"), receipt.get("effort"), receipt.get("prompt_version")) in wanted
        for receipt in receipts
    )
    _write_json(output_path, result, allowed_root=allowed_root)
    return result
