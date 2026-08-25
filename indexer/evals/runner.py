"""Private manifest, resume and candidate-isolated selection-evaluation helpers."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Iterable, Mapping, Sequence

from src.selection_instrument import SelectionDocument, SelectionInstrument
from src.captioning.prompt import CAPTION_PROMPT_VERSION

from .metrics import ndcg_at_k, recall_at_k, reciprocal_rank


PRIVATE_EVAL_ROOT = Path.home() / ".eagle-search" / "evals"


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def seal_hidden_manifest(payload: Mapping[str, Any]) -> str:
    """Hash frozen seed, labels, gates and target selection before unsealing results."""

    return "sha256:" + hashlib.sha256(_canonical_json(payload)).hexdigest()


def completed_run_keys(receipts: Iterable[Mapping[str, str]]) -> set[tuple[str, str, str, str]]:
    """Return content-addressed resume keys; model outputs never share a cache entry."""

    required = ("fixture_id", "model", "prompt_version", "image_hash")
    keys: set[tuple[str, str, str, str]] = set()
    for receipt in receipts:
        if not all(receipt.get(name) for name in required):
            continue
        keys.add(tuple(receipt[name] for name in required))
    return keys


@dataclass(frozen=True)
class CandidateRetrieval:
    ndcg_at_10: float
    recall_at_10: float
    mrr: float


def evaluate_candidate_retrieval(
    documents_by_candidate: Mapping[str, Sequence[SelectionDocument]],
    queries: Sequence[Mapping[str, Any]],
) -> dict[str, CandidateRetrieval]:
    """Rank every candidate only against its own completely captioned corpus."""

    instrument = SelectionInstrument()
    results: dict[str, CandidateRetrieval] = {}
    for candidate, documents in documents_by_candidate.items():
        ndcgs: list[float] = []
        recalls: list[float] = []
        mrrs: list[float] = []
        for query in queries:
            grades = query["grades"]
            ranking = instrument.rank(str(query["query"]), documents, limit=10).fused_ids
            ndcgs.append(ndcg_at_k(ranking, grades))
            recalls.append(recall_at_k(ranking, grades))
            mrrs.append(reciprocal_rank(ranking, grades))
        results[candidate] = CandidateRetrieval(
            ndcg_at_10=fmean(ndcgs) if ndcgs else 0.0,
            recall_at_10=fmean(recalls) if recalls else 0.0,
            mrr=fmean(mrrs) if mrrs else 0.0,
        )
    return results


def assert_private_snapshot(path: Path) -> Path:
    """Reject a snapshot path outside the user-local private evaluation root."""

    resolved = Path(path).expanduser().resolve()
    root = PRIVATE_EVAL_ROOT.expanduser().resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"evaluation snapshots must live under {root}")
    return resolved


class PrivateReceiptJournal:
    """Atomic local-only resume record; callers must place it in a private snapshot."""

    def __init__(self, snapshot: Path, *, enforce_private_root: bool = True) -> None:
        self.snapshot = assert_private_snapshot(snapshot) if enforce_private_root else Path(snapshot).resolve()
        self.receipt_dir = self.snapshot / "caption-receipts"

    def _path_for(self, fixture_id: str, model: str, prompt_version: str, image_hash: str) -> Path:
        digest = hashlib.sha256("\0".join((fixture_id, model, prompt_version, image_hash)).encode()).hexdigest()
        return self.receipt_dir / f"{digest}.json"

    def load(self) -> list[dict[str, Any]]:
        if not self.receipt_dir.exists():
            return []
        loaded = []
        for entry in sorted(self.receipt_dir.glob("*.json")):
            with entry.open(encoding="utf-8") as receipt_file:
                loaded.append(json.load(receipt_file))
        return loaded

    def put(self, receipt: Mapping[str, Any]) -> Path:
        required = ("fixture_id", "model", "prompt_version", "image_hash")
        if not all(receipt.get(name) for name in required):
            raise ValueError("private eval receipt is missing its resume identity")
        destination = self._path_for(*(str(receipt[name]) for name in required))
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if destination.exists():
            return destination
        temporary = destination.with_suffix(".tmp")
        temporary.write_bytes(_canonical_json(dict(receipt)))
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
        return destination


def resume_missing(
    fixtures: Iterable[Mapping[str, str]],
    *,
    model: str,
    prompt_version: str,
    completed: Iterable[Mapping[str, str]],
    caption: Callable[[Mapping[str, str]], Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Invoke a supplied local caption function only for missing fixture/model/hash tuples."""

    seen = completed_run_keys(completed)
    output: list[Mapping[str, Any]] = []
    for fixture in fixtures:
        key = (fixture["fixture_id"], model, prompt_version, fixture["image_hash"])
        if key not in seen:
            output.append(caption(fixture))
    return output


@dataclass(frozen=True)
class EvalFixture:
    """A local-only fixture record.  Its path never belongs in a Git artifact."""

    fixture_id: str
    image_hash: str
    image_path: str
    stage: str
    stratum: str = ""
    role: str = "target"


@dataclass(frozen=True)
class FixtureManifest:
    snapshot_id: str
    fixtures: tuple[EvalFixture, ...]
    manifest_hash: str

    def fixtures_for(self, stage: str) -> tuple[EvalFixture, ...]:
        if stage not in {"A", "B", "C"}:
            raise ValueError("stage must be A, B or C")
        return tuple(fixture for fixture in self.fixtures if fixture.stage == stage)


def load_fixture_manifest(snapshot: Path, *, enforce_private_root: bool = True) -> FixtureManifest:
    """Load an owner-local manifest and return its content hash for reporting/sealing."""

    root = assert_private_snapshot(snapshot) if enforce_private_root else Path(snapshot).resolve()
    manifest_path = root / "manifest.json"
    with manifest_path.open(encoding="utf-8") as manifest_file:
        payload = json.load(manifest_file)
    allowed_envelopes = (
        {"snapshot_version", "snapshot_id", "fixtures"},
        {
            "snapshot_version",
            "snapshot_id",
            "seed",
            "fixtures",
            "hidden_sealing_inputs",
            "hidden_seal",
        },
    )
    if set(payload) not in allowed_envelopes:
        raise ValueError("private fixture manifest has unexpected keys")
    if "hidden_seal" in payload:
        # Import lazily: fixture_builder imports this module's seal helper.
        from .fixture_builder import validate_private_manifest

        validate_private_manifest(payload)
    if payload["snapshot_version"] != 1 or not isinstance(payload["snapshot_id"], str):
        raise ValueError("unsupported private fixture manifest version")
    if not isinstance(payload["fixtures"], list):
        raise ValueError("private fixture manifest fixtures must be a list")
    fixtures: list[EvalFixture] = []
    seen_ids: set[str] = set()
    for raw in payload["fixtures"]:
        allowed = {"fixture_id", "image_hash", "image_path", "stage", "stratum", "role"}
        if not isinstance(raw, dict) or set(raw) - allowed or not {"fixture_id", "image_hash", "image_path", "stage"}.issubset(raw):
            raise ValueError("private fixture has malformed fields")
        fixture = EvalFixture(
            fixture_id=str(raw["fixture_id"]),
            image_hash=str(raw["image_hash"]),
            image_path=str(raw["image_path"]),
            stage=str(raw["stage"]),
            stratum=str(raw.get("stratum", "")),
            role=str(raw.get("role", "target")),
        )
        if not fixture.fixture_id or fixture.fixture_id in seen_ids or fixture.stage not in {"A", "B", "C"}:
            raise ValueError("private fixture ids must be unique and stages must be A, B or C")
        seen_ids.add(fixture.fixture_id)
        fixtures.append(fixture)
    return FixtureManifest(
        snapshot_id=payload["snapshot_id"],
        fixtures=tuple(fixtures),
        manifest_hash="sha256:" + hashlib.sha256(_canonical_json(payload)).hexdigest(),
    )


def run_caption_stage(
    manifest: FixtureManifest,
    *,
    stage: str,
    models: Sequence[str],
    journal: PrivateReceiptJournal,
    caption: Callable[[EvalFixture, str], Mapping[str, Any]],
    prompt_version: str = CAPTION_PROMPT_VERSION,
) -> list[dict[str, Any]]:
    """Run only missing fixture/model work and persist a local atomic resume receipt.

    ``caption`` is injected so the orchestration stays testable and never calls a
    real model from ordinary unit tests.  The actual CLI adapter is supplied by the
    integration seam.
    """

    completed = completed_run_keys(journal.load())
    created: list[dict[str, Any]] = []
    for fixture in manifest.fixtures_for(stage):
        for model in models:
            identity = (fixture.fixture_id, model, prompt_version, fixture.image_hash)
            if identity in completed:
                continue
            payload = dict(caption(fixture, model))
            receipt = {
                "fixture_id": fixture.fixture_id,
                "model": model,
                "prompt_version": prompt_version,
                "image_hash": fixture.image_hash,
                "manifest_hash": manifest.manifest_hash,
                **payload,
            }
            journal.put(receipt)
            completed.add(identity)
            created.append(receipt)
    return created
