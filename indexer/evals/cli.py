"""Pure CLI-support functions for private evaluation artifacts.

The shared application CLI owns argument parsing; this module supplies safe,
testable operations it can call after resolving user arguments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .fixture_builder import build_private_manifest, private_manifest_summary
from .report import CandidateAggregate, decide_winner, render_anonymised_report
from .runner import PRIVATE_EVAL_ROOT


class EvalCliError(ValueError):
    """A safe, stable private-evaluation CLI-support failure."""


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


def _write_json(path: Path, payload: Mapping[str, Any], *, allowed_root: Path | None) -> None:
    resolved = _private_path(path, allowed_root=allowed_root)
    resolved.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(resolved)


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
        "ocr_character_f1", "ndcg_at_10", "recall_at_10", "known_query_passed", "lower_bounds", "lanes", "latencies_seconds",
    }
    if not fields.issubset(payload):
        raise EvalCliError("evaluation candidate is missing aggregate metrics")
    try:
        return CandidateAggregate(**{field: payload[field] for field in fields})
    except (TypeError, ValueError) as error:
        raise EvalCliError("evaluation candidate has invalid aggregate metrics") from error


def aggregate_report_json(
    results_path: Path,
    output_path: Path,
    *,
    comparator: str,
    allowed_root: Path | None = None,
) -> dict[str, Any]:
    """Aggregate private existing result JSON and write only its anonymous report."""

    raw = _load_json(results_path, allowed_root=allowed_root)
    if not isinstance(raw, Mapping) or not isinstance(raw.get("candidates"), list):
        raise EvalCliError("evaluation results JSON must contain candidates")
    if any(not isinstance(candidate, Mapping) for candidate in raw["candidates"]):
        raise EvalCliError("evaluation results candidates must be aggregate objects")
    decision = decide_winner([_candidate(candidate) for candidate in raw["candidates"]], comparator=comparator)
    report = render_anonymised_report(decision)
    _write_json(output_path, report, allowed_root=allowed_root)
    return report
