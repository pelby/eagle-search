"""Build a private, candidate-blind pooled-relevance grading packet.

Only opaque fixture IDs and pixels leave the sealed manifest. Candidate
identity, retrieval order, captions, model output and pre-existing relevance
grades are deliberately excluded from every rendered artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from src.captioning.codex_cli import CaptionItemFailure, CodexCliCaptionProvider

from .fixture_builder import validate_private_manifest
from .runner import PRIVATE_EVAL_ROOT
from .stage_c import StageCValidationError, _stage_c_corpus
from .study import (
    CandidateSpec,
    StudyValidationError,
    artifact_sha256,
    candidate_set_hash,
    validate_blind_pool_artifact,
    validate_frozen_query_artifact,
)


class RelevancePacketError(ValueError):
    """The sealed inputs or private packet destination are unsafe or invalid."""


_PACKET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_FIXTURE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HASH = re.compile(r"^sha256:[a-f0-9]{64}$")
_CORPUS_KEYS = {"relevance_corpus_version", "manifest_hash", "stage", "fixture_ids"}
_STAGE_C_APPROVAL_KEYS = {"approval_version", "approved", "gates_hash"}
_THUMBNAIL = (300, 220)
_CELL = (320, 260)
_COLUMNS = 4
_HEADER_HEIGHT = 62
_RUBRIC = {
    "3": "directly and strongly satisfies the query",
    "2": "clearly useful but partial or less central",
    "1": "weakly related or only one minor aspect matches",
    "0": "irrelevant, contradicted, or unsupported by visible pixels",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _resolve_private_root(path: Path | None, *, allowed_root: Path | None) -> tuple[Path, Path]:
    allowed = Path(allowed_root or PRIVATE_EVAL_ROOT).expanduser().resolve()
    root = Path(path or allowed).expanduser().resolve()
    if not root.is_relative_to(allowed):
        raise RelevancePacketError(f"private relevance packets must stay under {allowed}")
    return root, allowed


def _secure_root(root: Path, *, allowed_root: Path) -> Path:
    allowed_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not allowed_root.is_dir():
        raise RelevancePacketError("allowed private packet root must be a directory")
    os.chmod(allowed_root, 0o700)
    current = allowed_root
    for part in root.relative_to(allowed_root).parts:
        current = current / part
        current.mkdir(mode=0o700, exist_ok=True)
        if not current.is_dir():
            raise RelevancePacketError("private packet root must be a directory")
        os.chmod(current, 0o700)
    return root


def _validate_corpus_artifact(
    artifact: Mapping[str, Any],
    *,
    expected_hash: str,
    manifest_hash: str,
    fixtures: list[Mapping[str, Any]],
) -> tuple[str, set[str]]:
    if artifact_sha256(artifact) != expected_hash:
        raise RelevancePacketError("relevance corpus artifact SHA-256 does not match the packet seal")
    if set(artifact) != _CORPUS_KEYS or artifact.get("relevance_corpus_version") != 1:
        raise RelevancePacketError("relevance corpus artifact has an unsupported envelope")
    if artifact.get("manifest_hash") != manifest_hash:
        raise RelevancePacketError("relevance corpus artifact is bound to another manifest")
    stage = artifact.get("stage")
    raw_ids = artifact.get("fixture_ids")
    if stage not in {"B", "C"} or not isinstance(raw_ids, list) or not raw_ids:
        raise RelevancePacketError("relevance corpus stage or fixture IDs are invalid")
    if any(not isinstance(item, str) or not _FIXTURE_ID.fullmatch(item) for item in raw_ids):
        raise RelevancePacketError("relevance corpus contains an invalid fixture ID")
    if raw_ids != sorted(set(raw_ids)):
        raise RelevancePacketError("relevance corpus fixture IDs must be sorted and unique")

    allowed_ids = set(raw_ids)
    by_id = {str(fixture["fixture_id"]): fixture for fixture in fixtures}
    if any(fixture_id not in by_id or by_id[fixture_id]["stage"] != stage for fixture_id in allowed_ids):
        raise RelevancePacketError("relevance corpus crosses its sealed manifest stage")
    if stage == "B":
        expected = {fixture_id for fixture_id, fixture in by_id.items() if fixture["stage"] == "B"}
        if allowed_ids != expected:
            raise RelevancePacketError("Stage B relevance corpus must contain exactly every Stage B fixture")
    else:
        distractors = {
            fixture_id for fixture_id, fixture in by_id.items()
            if fixture["stage"] == "C" and fixture["role"] == "distractor"
        }
        target_count = sum(
            by_id[fixture_id]["role"] == "target" for fixture_id in allowed_ids
        )
        if not distractors.issubset(allowed_ids) or not 60 <= target_count <= 120:
            raise RelevancePacketError(
                "Stage C relevance corpus requires all sealed distractors and 60 through 120 separately selected targets"
            )
    return str(stage), allowed_ids


def _validate_stage_c_selection(
    *,
    manifest: Mapping[str, Any],
    allowed_ids: set[str],
    gates: Mapping[str, Any] | None,
    expected_gates_hash: str,
    stage_b_effects: Mapping[str, Any] | None,
    expected_effects_hash: str,
    approval: Mapping[str, Any] | None,
    expected_approval_hash: str,
    query_artifact_hash: str,
    expected_candidate_hash: str,
) -> dict[str, str]:
    evidence = (
        (gates, expected_gates_hash, "Stage C gates"),
        (stage_b_effects, expected_effects_hash, "Stage B effects"),
        (approval, expected_approval_hash, "Stage C approval"),
    )
    for artifact, expected_hash, name in evidence:
        if (
            not isinstance(artifact, Mapping)
            or not isinstance(expected_hash, str)
            or not _HASH.fullmatch(expected_hash)
            or artifact_sha256(artifact) != expected_hash
        ):
            raise RelevancePacketError(f"exact hash-bound {name} are required")
    assert gates is not None and stage_b_effects is not None and approval is not None
    if (
        set(approval) != _STAGE_C_APPROVAL_KEYS
        or approval.get("approval_version") != 1
        or approval.get("approved") is not True
        or approval.get("gates_hash") != expected_gates_hash
    ):
        raise RelevancePacketError("Stage C approval does not bind the exact sealed gates")
    if gates.get("query_artifact_hash") != query_artifact_hash:
        raise RelevancePacketError("Stage C query artifact does not match the exact sealed gates")
    if any(
        not isinstance(gates.get(name), str) or not _HASH.fullmatch(str(gates.get(name)))
        for name in ("snapshot_hash", "guide_hash", "labels_hash", "query_artifact_hash")
    ):
        raise RelevancePacketError("Stage C gates require exact manifest, guide, labels and query bindings")
    raw_candidates = gates.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise RelevancePacketError("Stage C gates require exact candidate identities")
    try:
        gated_candidates = [
            CandidateSpec(*value.split(":"))
            for value in raw_candidates
            if isinstance(value, str) and value.count(":") == 2
        ]
    except StudyValidationError as error:
        raise RelevancePacketError("Stage C gates contain an invalid candidate identity") from error
    if len(gated_candidates) != len(raw_candidates) or len(set(gated_candidates)) != len(gated_candidates):
        raise RelevancePacketError("Stage C gates contain invalid or duplicate candidate identities")
    if candidate_set_hash(gated_candidates) != expected_candidate_hash:
        raise RelevancePacketError("candidate set does not match the exact sealed Stage C candidates")
    try:
        derived_corpus, _selected_ids = _stage_c_corpus(
            snapshot=manifest,
            gates=gates,
            stage_b_effects=stage_b_effects,
        )
    except StageCValidationError as error:
        raise RelevancePacketError(str(error)) from error
    derived_ids = {str(fixture["fixture_id"]) for fixture in derived_corpus}
    if allowed_ids != derived_ids:
        raise RelevancePacketError("relevance corpus does not match the exact sealed Stage C selection")
    return {
        "stage_c_gates_hash": expected_gates_hash,
        "stage_b_effects_hash": expected_effects_hash,
        "stage_c_approval_hash": expected_approval_hash,
        "snapshot_hash": str(gates["snapshot_hash"]),
        "guide_hash": str(gates["guide_hash"]),
        "labels_hash": str(gates["labels_hash"]),
        "query_artifact_hash": query_artifact_hash,
        "candidate_set_hash": expected_candidate_hash,
    }


def _write_bytes(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    os.chmod(path, 0o600)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    _write_bytes(path, data)


def _query_inputs(
    queries: tuple[Mapping[str, Any], ...],
    *,
    pool_items: Mapping[str, tuple[str, ...]],
    corpus_ids: set[str],
) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in queries:
        if not isinstance(raw, Mapping):
            raise RelevancePacketError("frozen retrieval query must be an object")
        query_id = raw.get("query_id")
        query_text = raw.get("query")
        grades = raw.get("grades")
        target_fixture_id = raw.get("target_fixture_id")
        if (
            not isinstance(query_id, str)
            or not query_id.strip()
            or query_id in seen
            or not isinstance(query_text, str)
            or not query_text.strip()
            or not isinstance(grades, Mapping)
            or target_fixture_id not in corpus_ids
        ):
            raise RelevancePacketError("frozen retrieval query identity, text or judgments are invalid")
        frozen_ids: set[str] = set()
        for fixture_id, grade in grades.items():
            if (
                not isinstance(fixture_id, str)
                or fixture_id not in corpus_ids
                or not isinstance(grade, int)
                or isinstance(grade, bool)
                or not 0 <= grade <= 3
            ):
                raise RelevancePacketError("frozen retrieval judgments reference invalid corpus items")
            frozen_ids.add(fixture_id)
        # Existing grade values are never copied. Only the identifier union is
        # re-graded so the final adjudication can satisfy the study validator.
        fixture_ids = sorted(frozen_ids | set(pool_items[query_id]))
        if not fixture_ids:
            raise RelevancePacketError("a pooled-relevance query cannot be empty")
        parsed.append({"query_id": query_id, "query": query_text, "fixture_ids": fixture_ids})
        seen.add(query_id)
    return sorted(parsed, key=lambda item: item["query_id"])


def _render_sheet(
    path: Path,
    *,
    query_id: str,
    query_text: str,
    fixture_ids: list[str],
    paths_by_id: Mapping[str, Path],
) -> None:
    rows = math.ceil(len(fixture_ids) / _COLUMNS)
    sheet = Image.new("RGB", (_COLUMNS * _CELL[0], _HEADER_HEIGHT + rows * _CELL[1]), "white")
    draw = ImageDraw.Draw(sheet)
    # Reuse the caption provider's local-only input preparation so SVG bytes
    # cached under misleading ``.png`` names are rasterised onto the same white
    # background as the evaluated captions. No Codex/provider method is called.
    preparer = CodexCliCaptionProvider()
    header = f"Query {query_id}: " + "\n".join(textwrap.wrap(query_text, width=150)[:2])
    draw.text((10, 8), header, fill="black")
    for index, fixture_id in enumerate(fixture_ids):
        column, row = index % _COLUMNS, index // _COLUMNS
        left = column * _CELL[0]
        top = _HEADER_HEIGHT + row * _CELL[1]
        draw.rectangle((left, top, left + _CELL[0] - 1, top + _CELL[1] - 1), outline="#b0b0b0")
        try:
            with preparer._prepared_image(paths_by_id[fixture_id]) as prepared:  # noqa: SLF001
                with Image.open(prepared) as source:
                    source.seek(0)
                    pixels = ImageOps.exif_transpose(source).convert("RGB")
                    pixels.thumbnail(_THUMBNAIL, Image.Resampling.LANCZOS)
                    x = left + (_CELL[0] - pixels.width) // 2
                    y = top + 8 + (_THUMBNAIL[1] - pixels.height) // 2
                    sheet.paste(pixels, (x, y))
        except (CaptionItemFailure, OSError, UnidentifiedImageError, Image.DecompressionBombError, KeyError) as error:
            raise RelevancePacketError(f"image for opaque fixture {fixture_id} could not be rendered") from error
        draw.text((left + 8, top + 234), fixture_id, fill="black")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as destination:
            sheet.save(destination, format="PNG", compress_level=9, optimize=False)
            destination.flush()
            os.fsync(destination.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    finally:
        sheet.close()
    os.chmod(path, 0o600)


def _grading_schema(
    *,
    manifest_hash: str,
    query_hash: str,
    candidate_hash: str,
    pool_hash: str,
    packet_evidence_hash: str,
    queries: list[dict[str, Any]],
) -> dict[str, Any]:
    query_properties: dict[str, Any] = {}
    for query in queries:
        identifiers = query["fixture_ids"]
        query_properties[query["query_id"]] = {
            "type": "object",
            "additionalProperties": False,
            "required": identifiers,
            "properties": {
                fixture_id: {"type": "integer", "enum": [0, 1, 2, 3]}
                for fixture_id in identifiers
            },
        }
    constants = {
        "pooled_relevance_version": {"const": 2},
        "manifest_hash": {"const": manifest_hash},
        "base_query_artifact_hash": {"const": query_hash},
        "candidate_set_hash": {"const": candidate_hash},
        "blind_pool_hash": {"const": pool_hash},
        "packet_evidence_hash": {"const": packet_evidence_hash},
        "blinded": {"const": True},
        "adjudication_complete": {"const": True},
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": [*constants, "query_grades"],
        "properties": {
            **constants,
            "query_grades": {
                "type": "object",
                "additionalProperties": False,
                "required": [query["query_id"] for query in queries],
                "properties": query_properties,
            },
        },
    }


def _prompt() -> str:
    return """PRIVATE BLIND POOLED-RELEVANCE GRADING

Inspect each contact sheet against its printed query. Judge each opaque fixture
from pixels alone. The cells are deliberately candidate-blind and their order
has no quality meaning. Do not infer a generating model, caption, retrieval
position, source, or prior judgment.

Assign exactly one integer to every listed fixture:
3 = directly and strongly satisfies the query
2 = clearly useful but partial or less central
1 = weakly related or only one minor aspect matches
0 = irrelevant, contradicted, or unsupported by visible pixels

Use the same standard across all queries. Do not reward visual quality unless
the query asks for it. Do not infer hidden intent or unreadable text. Return
JSON only, conforming exactly to grading-schema.json: no omitted fixtures, no
extra fixtures, no explanations, and no additional fields.
"""


def build_relevance_packet(
    *,
    manifest: Mapping[str, Any],
    expected_manifest_hash: str,
    blind_pool: Mapping[str, Any],
    expected_blind_pool_hash: str,
    query_artifact: Mapping[str, Any],
    expected_query_artifact_hash: str,
    expected_candidate_set_hash: str,
    corpus_artifact: Mapping[str, Any],
    expected_corpus_artifact_hash: str,
    stage_c_gates: Mapping[str, Any] | None = None,
    expected_stage_c_gates_hash: str = "",
    stage_b_effects: Mapping[str, Any] | None = None,
    expected_stage_b_effects_hash: str = "",
    stage_c_approval: Mapping[str, Any] | None = None,
    expected_stage_c_approval_hash: str = "",
    private_root: Path | None = None,
    allowed_root: Path | None = None,
    packet_name: str,
) -> dict[str, str]:
    """Validate exact study seals, then atomically render a blind packet.

    The packet directory is created only after all identifier/hash validation
    succeeds and is atomically renamed into place after every sheet is written.
    """

    if not _PACKET_NAME.fullmatch(packet_name):
        raise RelevancePacketError("packet_name must be a single safe identifier")
    resolved_root, resolved_allowed_root = _resolve_private_root(
        private_root,
        allowed_root=allowed_root,
    )
    if not all(
        isinstance(value, str) and _HASH.fullmatch(value)
        for value in (
            expected_manifest_hash,
            expected_blind_pool_hash,
            expected_query_artifact_hash,
            expected_candidate_set_hash,
            expected_corpus_artifact_hash,
        )
    ):
        raise RelevancePacketError("all relevance packet bindings must be SHA-256 identifiers")
    try:
        validate_private_manifest(manifest)
        if artifact_sha256(manifest) != expected_manifest_hash:
            raise RelevancePacketError("private manifest SHA-256 does not match the packet seal")
        frozen_queries = validate_frozen_query_artifact(
            query_artifact,
            expected_artifact_hash=expected_query_artifact_hash,
        )
        fixtures = manifest["fixtures"]
        corpus_stage, corpus_ids = _validate_corpus_artifact(
            corpus_artifact,
            expected_hash=expected_corpus_artifact_hash,
            manifest_hash=expected_manifest_hash,
            fixtures=fixtures,
        )
        if corpus_stage == "C":
            stage_c_selection_evidence = _validate_stage_c_selection(
                manifest=manifest,
                allowed_ids=corpus_ids,
                gates=stage_c_gates,
                expected_gates_hash=expected_stage_c_gates_hash,
                stage_b_effects=stage_b_effects,
                expected_effects_hash=expected_stage_b_effects_hash,
                approval=stage_c_approval,
                expected_approval_hash=expected_stage_c_approval_hash,
                query_artifact_hash=expected_query_artifact_hash,
                expected_candidate_hash=expected_candidate_set_hash,
            )
        else:
            if any(
                value not in (None, "")
                for value in (
                    stage_c_gates,
                    expected_stage_c_gates_hash,
                    stage_b_effects,
                    expected_stage_b_effects_hash,
                    stage_c_approval,
                    expected_stage_c_approval_hash,
                )
            ):
                raise RelevancePacketError("Stage B packet cannot carry Stage C selection evidence")
            stage_c_selection_evidence = None
        frozen_pool = validate_blind_pool_artifact(
            blind_pool,
            expected_artifact_hash=expected_blind_pool_hash,
            manifest_hash=expected_manifest_hash,
            base_query_artifact=frozen_queries,
            expected_candidate_set_hash=expected_candidate_set_hash,
            corpus_ids=corpus_ids,
        )
    except RelevancePacketError:
        raise
    except (KeyError, TypeError, ValueError, StudyValidationError) as error:
        raise RelevancePacketError(str(error)) from error

    queries = _query_inputs(
        frozen_queries.queries,
        pool_items=frozen_pool.query_items,
        corpus_ids=corpus_ids,
    )
    manifest_by_id = {str(fixture["fixture_id"]): Path(str(fixture["image_path"])) for fixture in fixtures}
    manifest_hash_by_id = {str(fixture["fixture_id"]): str(fixture["image_hash"]) for fixture in fixtures}
    required_ids = {fixture_id for query in queries for fixture_id in query["fixture_ids"]}
    if any(not manifest_by_id[fixture_id].is_file() for fixture_id in required_ids):
        raise RelevancePacketError("a pooled-relevance source image is missing")
    try:
        if any(
            _file_sha256(manifest_by_id[fixture_id]) != manifest_hash_by_id[fixture_id]
            for fixture_id in required_ids
        ):
            raise RelevancePacketError("pooled-relevance source image bytes do not match the sealed manifest")
    except OSError as error:
        raise RelevancePacketError("pooled-relevance source image bytes could not be verified") from error

    root = _secure_root(resolved_root, allowed_root=resolved_allowed_root)
    destination = root / packet_name
    if destination.exists():
        raise RelevancePacketError("private relevance packet destination already exists")
    with tempfile.TemporaryDirectory(prefix=".relevance-packet-", dir=root, ignore_cleanup_errors=True) as temporary:
        staging = Path(temporary)
        os.chmod(staging, 0o700)
        packet_queries: list[dict[str, Any]] = []
        for index, query in enumerate(queries, start=1):
            filename = f"contact-{index:03d}.png"
            contact_sheet = staging / filename
            _render_sheet(
                contact_sheet,
                query_id=query["query_id"],
                query_text=query["query"],
                fixture_ids=query["fixture_ids"],
                paths_by_id=manifest_by_id,
            )
            packet_queries.append(
                {
                    **query,
                    "contact_sheet": filename,
                    "contact_sheet_hash": _file_sha256(contact_sheet),
                }
            )

        prompt_bytes = _prompt().encode("utf-8")
        _write_bytes(staging / "PROMPT.txt", prompt_bytes)
        grading_contract = {
            "grading_contract_version": 1,
            "pooled_relevance_version": 2,
            "grade_values": [0, 1, 2, 3],
            "rubric": _RUBRIC,
            "queries": [
                {"query_id": query["query_id"], "fixture_ids": query["fixture_ids"]}
                for query in queries
            ],
        }
        evidence = {
            "packet_evidence_version": 1,
            "manifest_hash": expected_manifest_hash,
            "base_query_artifact_hash": expected_query_artifact_hash,
            "candidate_set_hash": expected_candidate_set_hash,
            "blind_pool_hash": expected_blind_pool_hash,
            "corpus_artifact_hash": expected_corpus_artifact_hash,
            "corpus_stage": corpus_stage,
            "stage_c_selection_evidence": stage_c_selection_evidence,
            "prompt_hash": _file_sha256(staging / "PROMPT.txt"),
            "grading_contract_hash": artifact_sha256(grading_contract),
            "query_sheets": [
                {
                    "query_id": query["query_id"],
                    "fixture_ids": query["fixture_ids"],
                    "contact_sheet_hash": query["contact_sheet_hash"],
                }
                for query in packet_queries
            ],
        }
        packet_evidence_hash = artifact_sha256(evidence)
        schema = _grading_schema(
            manifest_hash=expected_manifest_hash,
            query_hash=expected_query_artifact_hash,
            candidate_hash=expected_candidate_set_hash,
            pool_hash=expected_blind_pool_hash,
            packet_evidence_hash=packet_evidence_hash,
            queries=queries,
        )
        _write_json(staging / "grading-schema.json", schema)
        packet = {
            "relevance_packet_version": 2,
            "manifest_hash": expected_manifest_hash,
            "base_query_artifact_hash": expected_query_artifact_hash,
            "candidate_set_hash": expected_candidate_set_hash,
            "blind_pool_hash": expected_blind_pool_hash,
            "corpus_artifact_hash": expected_corpus_artifact_hash,
            "corpus_stage": corpus_stage,
            "stage_c_selection_evidence": stage_c_selection_evidence,
            "blinded": True,
            "rubric": _RUBRIC,
            "packet_evidence": evidence,
            "packet_evidence_hash": packet_evidence_hash,
            "prompt": "PROMPT.txt",
            "prompt_hash": _file_sha256(staging / "PROMPT.txt"),
            "grading_schema": "grading-schema.json",
            "grading_schema_hash": _file_sha256(staging / "grading-schema.json"),
            "queries": packet_queries,
        }
        packet_hash = artifact_sha256(packet)
        _write_json(staging / "packet.json", packet)
        os.replace(staging, destination)
        os.chmod(destination, 0o700)
    return {
        "packet_directory": str(destination),
        "packet_hash": packet_hash,
        "packet_evidence_hash": packet_evidence_hash,
    }
