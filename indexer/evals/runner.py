"""Private manifest, resume and candidate-isolated selection-evaluation helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Iterable, Mapping, Sequence

from src.selection_instrument import SelectionDocument, SelectionInstrument
from src.captioning.codex_cli import CaptionItemFailure
from src.captioning.codex_cli import CAPTION_PROVIDER_NAME
from src.captioning.prompt import CAPTION_PROMPT_VERSION
from src.contracts import CaptionReceiptV1, ContractError

from .metrics import ndcg_at_k, recall_at_k, reciprocal_rank


PRIVATE_EVAL_ROOT = Path.home() / ".eagle-search" / "evals"


def _write_private_json(destination: Path, payload: Mapping[str, Any]) -> None:
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(destination.parent, 0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        os.chmod(destination, 0o600)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def seal_hidden_manifest(payload: Mapping[str, Any]) -> str:
    """Hash frozen seed, labels, gates and target selection before unsealing results."""

    return "sha256:" + hashlib.sha256(_canonical_json(payload)).hexdigest()


def completed_run_keys(receipts: Iterable[Mapping[str, Any]]) -> set[tuple[str, str, str, str, str]]:
    """Return successful resume keys, including the requested reasoning effort.

    Receipts written before effort became part of the journal identity retain an
    empty effort component.  ``run_caption_stage`` recognises that legacy key so
    old valid receipts remain resumable rather than being discarded.
    """

    required = ("fixture_id", "model", "prompt_version", "image_hash")
    keys: set[tuple[str, str, str, str, str]] = set()
    for receipt in receipts:
        if receipt.get("state") == "failed":
            continue
        if not all(receipt.get(name) for name in required):
            continue
        keys.add(
            (
                str(receipt["fixture_id"]),
                str(receipt["model"]),
                str(receipt.get("effort", "")),
                str(receipt["prompt_version"]),
                str(receipt["image_hash"]),
            )
        )
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

    def _path_for(self, fixture_id: str, model: str, effort: str, prompt_version: str, image_hash: str) -> Path:
        digest = hashlib.sha256("\0".join((fixture_id, model, effort, prompt_version, image_hash)).encode()).hexdigest()
        return self.receipt_dir / f"{digest}.json"

    def _failed_path_for(self, fixture_id: str, model: str, effort: str, prompt_version: str, image_hash: str) -> Path:
        return self._path_for(fixture_id, model, effort, prompt_version, image_hash).with_suffix(".failed.json")

    def load(self) -> list[dict[str, Any]]:
        if not self.receipt_dir.exists():
            return []
        loaded = []
        for entry in sorted(self.receipt_dir.glob("*.json")):
            with entry.open(encoding="utf-8") as receipt_file:
                loaded.append(json.load(receipt_file))
        return loaded

    def put(self, receipt: Mapping[str, Any]) -> Path:
        required = ("fixture_id", "model", "effort", "prompt_version", "image_hash")
        if not all(receipt.get(name) for name in required):
            raise ValueError("private eval receipt is missing its resume identity")
        destination = self._path_for(*(str(receipt[name]) for name in required))
        failed_destination = self._failed_path_for(*(str(receipt[name]) for name in required))
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(destination.parent, 0o700)
        if destination.exists():
            failed_destination.unlink(missing_ok=True)
            return destination
        _write_private_json(destination, dict(receipt))
        failed_destination.unlink(missing_ok=True)
        return destination

    def put_failed_attempt(self, failure: Mapping[str, Any]) -> Path:
        """Persist one latest retryable failure per identity without marking it complete."""

        required = ("fixture_id", "model", "effort", "prompt_version", "image_hash")
        if not all(failure.get(name) for name in required) or failure.get("state") != "failed":
            raise ValueError("private eval failure is missing its retry identity")
        successful_destination = self._path_for(*(str(failure[name]) for name in required))
        destination = self._failed_path_for(*(str(failure[name]) for name in required))
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(destination.parent, 0o700)
        if successful_destination.exists():
            return successful_destination
        _write_private_json(destination, dict(failure))
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


@dataclass(frozen=True)
class CaptionStageOutcome:
    """Local result counts with no image paths, hashes, or caption content."""

    created: tuple[dict[str, Any], ...]
    failed: tuple[dict[str, Any], ...]


def reseal_authenticated_receipt_journal(
    *,
    source_snapshot: Path,
    destination_snapshot: Path,
    source_manifest: FixtureManifest,
    destination_manifest: FixtureManifest,
    candidate: tuple[str, str, str],
    enforce_private_root: bool = True,
) -> int:
    """Copy only exact authenticated receipt evidence into a new private journal.

    This deliberately has no caption-provider dependency.  A receipt may be
    reused only when fixture ID/image hash, candidate identity and prompt are
    identical; the destination receives a new manifest binding while retaining
    the immutable receipt ID and caption content exactly.
    """

    source_root = assert_private_snapshot(source_snapshot) if enforce_private_root else Path(source_snapshot).resolve()
    destination_root = assert_private_snapshot(destination_snapshot) if enforce_private_root else Path(destination_snapshot).resolve()
    if source_root == destination_root:
        raise ValueError("receipt resealing requires a distinct destination snapshot")
    if source_manifest.manifest_hash == destination_manifest.manifest_hash:
        raise ValueError("receipt resealing requires a new destination manifest")
    fixture_identities = {
        fixture.fixture_id: (
            fixture.image_hash,
            fixture.stage,
            fixture.role,
            fixture.stratum,
        )
        for fixture in source_manifest.fixtures
    }
    if fixture_identities != {
        fixture.fixture_id: (
            fixture.image_hash,
            fixture.stage,
            fixture.role,
            fixture.stratum,
        )
        for fixture in destination_manifest.fixtures
    }:
        raise ValueError("receipt resealing requires exact fixture IDs, hashes, stages, roles and strata")
    fixture_hashes = {
        fixture_id: identity[0]
        for fixture_id, identity in fixture_identities.items()
    }
    source_journal = PrivateReceiptJournal(source_root, enforce_private_root=False)
    destination_journal = PrivateReceiptJournal(destination_root, enforce_private_root=False)
    model, effort, prompt_version = candidate
    copied = 0
    for journal in source_journal.load():
        if journal.get("state") == "failed" or (journal.get("model"), journal.get("effort"), journal.get("prompt_version")) != candidate:
            continue
        fixture_id, image_hash, receipt_id = journal.get("fixture_id"), journal.get("image_hash"), journal.get("receipt_id")
        if not isinstance(fixture_id, str) or fixture_hashes.get(fixture_id) != image_hash or not isinstance(receipt_id, str):
            raise ValueError("source receipt journal is not bound to its source manifest")
        if journal.get("manifest_hash") != source_manifest.manifest_hash:
            raise ValueError("source receipt journal manifest binding is invalid")
        receipt_path = source_root / "model-receipts" / image_hash / f"{receipt_id}.json"
        try:
            immutable = CaptionReceiptV1.from_dict(json.loads(receipt_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, ContractError, TypeError, AttributeError) as error:
            raise ValueError("authenticated immutable receipt is unavailable or invalid") from error
        if (
            immutable.provider != CAPTION_PROVIDER_NAME
            or immutable.source != "vision"
            or (immutable.model, immutable.effort, immutable.prompt_version) != candidate
            or immutable.image_hash != image_hash
            or journal.get("caption_result") != immutable.caption_result.to_dict()
            or journal.get("search_text") != immutable.search_text
        ):
            raise ValueError("source journal does not exactly match authenticated immutable receipt")
        destination_receipt_path = destination_root / "model-receipts" / image_hash / f"{receipt_id}.json"
        immutable_payload = immutable.to_dict()
        if destination_receipt_path.exists():
            try:
                existing = json.loads(destination_receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError("destination immutable receipt is invalid") from error
            if existing != immutable_payload:
                raise ValueError("destination immutable receipt conflicts with authenticated evidence")
        else:
            _write_private_json(destination_receipt_path, immutable_payload)
        destination_journal.put({**journal, "manifest_hash": destination_manifest.manifest_hash})
        copied += 1
    if copied == 0:
        raise ValueError("no authenticated receipts matched the requested candidate")
    return copied


def run_caption_stage(
    manifest: FixtureManifest,
    *,
    stage: str,
    models: Sequence[str],
    effort: str,
    journal: PrivateReceiptJournal,
    caption: Callable[[EvalFixture, str], Mapping[str, Any]],
    prompt_version: str = CAPTION_PROMPT_VERSION,
) -> CaptionStageOutcome:
    """Run only missing fixture/model work and persist a local atomic resume receipt.

    ``caption`` is injected so the orchestration stays testable and never calls a
    real model from ordinary unit tests.  The actual CLI adapter is supplied by the
    integration seam.
    """

    completed = completed_run_keys(journal.load())
    created: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for fixture in manifest.fixtures_for(stage):
        for model in models:
            identity = (fixture.fixture_id, model, effort, prompt_version, fixture.image_hash)
            legacy_identity = (fixture.fixture_id, model, "", prompt_version, fixture.image_hash)
            # Historical journals were produced at the then-only/default low
            # effort.  They can safely satisfy a low run, but must never mask a
            # newly requested medium-effort experiment.
            if identity in completed or (effort == "low" and legacy_identity in completed):
                continue
            try:
                payload = dict(caption(fixture, model))
            except CaptionItemFailure as error:
                failure = {
                    "fixture_id": fixture.fixture_id,
                    "model": model,
                    "effort": effort,
                    "prompt_version": prompt_version,
                    "image_hash": fixture.image_hash,
                    "manifest_hash": manifest.manifest_hash,
                    "state": "failed",
                    "error": " ".join(str(error).split())[:500] or "caption item failed",
                }
                journal.put_failed_attempt(failure)
                failed.append(failure)
                continue
            receipt = {
                **payload,
                "fixture_id": fixture.fixture_id,
                "model": model,
                "effort": effort,
                "prompt_version": prompt_version,
                "image_hash": fixture.image_hash,
                "manifest_hash": manifest.manifest_hash,
            }
            journal.put(receipt)
            completed.add(identity)
            created.append(receipt)
    return CaptionStageOutcome(tuple(created), tuple(failed))
