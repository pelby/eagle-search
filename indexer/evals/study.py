"""Pure scoring for a sealed, local-only caption-model study.

The caller owns private storage and model invocation.  This module accepts only
already-created receipts and identifier-keyed adjudicated labels, validates the
two frozen annotation artifacts before examining a caption, and returns an
aggregate that can be passed directly to :mod:`evals.report`.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re
from statistics import fmean
from typing import Any, Callable, Mapping, Sequence

from src.contracts import CaptionResultV1, ContractError
from src.selection_instrument import SelectionDocument, SelectionInstrument

from .metrics import (
    ConceptScores,
    cluster_bootstrap_lower_bound,
    concept_scores,
    critical_hallucination_count,
    exact_ocr_character_f1,
    ndcg_at_k,
    ocr_character_f1,
    recall_at_k,
    reciprocal_rank,
    required_target_count,
    token_aware_concept_recall,
    token_phrase_matches,
)


class StudyValidationError(ValueError):
    """A private study input is malformed, unsealed, or internally inconsistent."""


_V1_GUIDE_KEYS = {
    "annotation_guide_version",
    "label_schema_version",
    "concept_fields",
    "ocr_legibilities",
    "type_metric",
    "critical_metric",
}
_V2_GUIDE_KEYS = _V1_GUIDE_KEYS | {"concept_metric", "semantic_query_policy"}
_LABEL_ENVELOPE_KEYS = {"labels_version", "labels"}
_LABEL_KEYS = {
    "concept_alias_groups",
    "ocr_text",
    "image_type_aliases",
    "critical_absent_terms",
}
_CONCEPT_FIELDS = {
    "diagram_types",
    "subjects",
    "visual_style",
    "colours",
    "layout",
    "search_terms",
}
_FIXTURE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_NORMALISE_RE = re.compile(r"[^\w]+", re.UNICODE)
_V1_NONINFERIORITY_MARGINS = {
    "ndcg_at_10": -0.03,
    "recall_at_10": -0.03,
    "concept_f1": -0.04,
    "ocr_character_f1": -0.02,
}
_V2_NONINFERIORITY_MARGINS = {
    "ndcg_at_10": -0.03,
    "recall_at_10": -0.03,
    "concept_recall": -0.04,
    "ocr_character_f1": -0.02,
}
_EMBEDDING_CONTRACT_KEYS = {
    "embedding_contract_version",
    "model",
    "dimensions",
    "selection_instrument_version",
}
_QUERY_ARTIFACT_KEYS = {"query_artifact_version", "queries"}
_POOLED_RELEVANCE_KEYS = {
    "pooled_relevance_version",
    "manifest_hash",
    "base_query_artifact_hash",
    "candidate_set_hash",
    "blind_pool_hash",
    "packet_evidence_hash",
    "blinded",
    "adjudication_complete",
    "query_grades",
}
_BLIND_POOL_KEYS = {
    "blind_pool_version",
    "manifest_hash",
    "base_query_artifact_hash",
    "candidate_set_hash",
    "ranking_lane",
    "query_items",
}
_SELECTION_INSTRUMENT_VERSION = "selection-instrument-v1"
_LANE_NONINFERIORITY_MARGINS = {
    "bm25": {"ndcg_at_10": -0.03, "recall_at_10": -0.03},
    "semantic": {"ndcg_at_10": -0.03, "recall_at_10": -0.03},
}


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise StudyValidationError("study artifact must be canonical JSON") from error


def artifact_sha256(payload: Mapping[str, Any]) -> str:
    """Return the canonical SHA-256 identifier used to seal a private artifact."""
    return "sha256:" + sha256(_canonical_json(payload)).hexdigest()


def _require_fixture_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _FIXTURE_ID_RE.fullmatch(value):
        raise StudyValidationError(f"{name} must be an identifier-only fixture ID")
    return value


def _strings(value: Any, name: str, *, permit_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or (not permit_empty and not item.strip()) for item in value):
        raise StudyValidationError(f"{name} must be an array of nonblank strings")
    if len(set(value)) != len(value):
        raise StudyValidationError(f"{name} cannot contain duplicate strings")
    return tuple(value)


def _normalise(value: str) -> str:
    return _NORMALISE_RE.sub("", value.casefold())


@dataclass(frozen=True)
class AnnotationLabel:
    concept_alias_groups: tuple[frozenset[str], ...]
    ocr_text: str
    image_type_aliases: frozenset[str]
    critical_absent_terms: tuple[str, ...]


@dataclass(frozen=True)
class FrozenAnnotations:
    concept_fields: tuple[str, ...]
    ocr_legibilities: frozenset[str]
    labels: Mapping[str, AnnotationLabel]
    guide_hash: str
    labels_hash: str
    annotation_guide_version: int
    concept_metric: str
    critical_metric: str


@dataclass(frozen=True)
class CandidateSpec:
    """A prompt is part of the private receipt identity, not a shared cache key."""

    model: str
    effort: str
    prompt_version: str

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value.strip() for value in (self.model, self.effort, self.prompt_version)):
            raise StudyValidationError("candidate model, effort and prompt version must be nonblank strings")

    def receipt_key(self) -> tuple[str, str, str]:
        return self.model, self.effort, self.prompt_version


@dataclass(frozen=True)
class FrozenEmbeddingContract:
    model: str
    dimensions: int
    contract_hash: str


@dataclass(frozen=True)
class FrozenQueryArtifact:
    queries: tuple[Mapping[str, Any], ...]
    artifact_hash: str


@dataclass(frozen=True)
class FrozenBlindPool:
    query_items: Mapping[str, tuple[str, ...]]
    artifact_hash: str


def candidate_set_hash(candidates: Sequence[CandidateSpec]) -> str:
    """Hash candidate identity without putting captions or model output in an audit artifact."""

    identities = sorted(
        (candidate.model, candidate.effort, candidate.prompt_version)
        for candidate in candidates
    )
    return artifact_sha256(
        {
            "candidate_set_version": 1,
            "candidates": [
                {"model": model, "effort": effort, "prompt_version": prompt_version}
                for model, effort, prompt_version in identities
            ],
        }
    )


def _known_query_target_passed(ranking: Sequence[str], target_fixture_id: str) -> bool:
    """A frozen regression passes only when its exact target remains in top ten."""

    return target_fixture_id in ranking[:10]


def _label(raw: Any, fixture_id: str) -> AnnotationLabel:
    if not isinstance(raw, Mapping) or set(raw) != _LABEL_KEYS:
        raise StudyValidationError(f"label {fixture_id} has unexpected fields")
    groups = raw["concept_alias_groups"]
    if not isinstance(groups, list):
        raise StudyValidationError(f"label {fixture_id} concept_alias_groups must be an array")
    aliases: list[frozenset[str]] = []
    for index, group in enumerate(groups):
        values = _strings(group, f"label {fixture_id} concept_alias_groups[{index}]")
        normalised = frozenset(_normalise(value) for value in values)
        if not normalised or "" in normalised:
            raise StudyValidationError(f"label {fixture_id} contains an empty concept alias")
        aliases.append(frozenset(values))
    if len({frozenset(_normalise(value) for value in group) for group in aliases}) != len(aliases):
        raise StudyValidationError(f"label {fixture_id} duplicates a concept alias group")
    ocr_text = raw["ocr_text"]
    if not isinstance(ocr_text, str):
        raise StudyValidationError(f"label {fixture_id} ocr_text must be a string")
    image_type_aliases = frozenset(_normalise(value) for value in _strings(raw["image_type_aliases"], f"label {fixture_id} image_type_aliases"))
    if not image_type_aliases or "" in image_type_aliases:
        raise StudyValidationError(f"label {fixture_id} requires image type aliases")
    critical = _strings(raw["critical_absent_terms"], f"label {fixture_id} critical_absent_terms")
    if any(not _normalise(value) for value in critical):
        raise StudyValidationError(f"label {fixture_id} contains an empty critical term")
    return AnnotationLabel(tuple(aliases), ocr_text, image_type_aliases, critical)


def validate_frozen_annotations(
    guide: Mapping[str, Any],
    labels: Mapping[str, Any],
    *,
    expected_guide_hash: str,
    expected_labels_hash: str,
) -> FrozenAnnotations:
    """Validate strict identifier-only labels after verifying their frozen hashes.

    Hash comparison occurs first so a changed annotation guide cannot affect
    parsing or scoring while pretending to be part of the sealed experiment.
    """
    if artifact_sha256(guide) != expected_guide_hash or artifact_sha256(labels) != expected_labels_hash:
        raise StudyValidationError("annotation artifact SHA-256 does not match the frozen study seal")
    guide_version = guide.get("annotation_guide_version")
    if guide_version == 1:
        expected_keys = _V1_GUIDE_KEYS
    elif guide_version == 2:
        expected_keys = _V2_GUIDE_KEYS
    else:
        raise StudyValidationError("unsupported annotation guide version")
    if set(guide) != expected_keys or guide["label_schema_version"] != 1:
        raise StudyValidationError("annotation guide has unexpected fields")
    concept_fields = _strings(guide["concept_fields"], "annotation guide concept_fields")
    if not concept_fields or set(concept_fields) - _CONCEPT_FIELDS:
        raise StudyValidationError("annotation guide concept_fields are unsupported")
    legibilities = frozenset(_strings(guide["ocr_legibilities"], "annotation guide ocr_legibilities"))
    if not legibilities or not legibilities.issubset({"high", "medium", "low"}):
        raise StudyValidationError("annotation guide ocr_legibilities are unsupported")
    if guide["type_metric"] != "normalised-alias":
        raise StudyValidationError("annotation guide names an unsupported metric")
    if guide_version == 1 and guide["critical_metric"] != "normalised-substring":
        raise StudyValidationError("annotation guide names an unsupported metric")
    if guide_version == 2 and (
        guide["critical_metric"] != "token-phrase"
        or guide["concept_metric"] != "token-aware-alias-recall"
        or guide["semantic_query_policy"] != "separate-frozen-query-artifact"
    ):
        raise StudyValidationError("annotation guide names an unsupported v2 metric or query policy")
    if set(labels) != _LABEL_ENVELOPE_KEYS or labels["labels_version"] != 1 or not isinstance(labels["labels"], Mapping):
        raise StudyValidationError("adjudicated labels have an invalid envelope")
    parsed: dict[str, AnnotationLabel] = {}
    for fixture_id, raw in labels["labels"].items():
        fixture_id = _require_fixture_id(fixture_id, "label key")
        if fixture_id in parsed:
            raise StudyValidationError("duplicate adjudicated fixture label")
        parsed[fixture_id] = _label(raw, fixture_id)
    if not parsed:
        raise StudyValidationError("adjudicated labels cannot be empty")
    return FrozenAnnotations(
        tuple(concept_fields), legibilities, parsed, expected_guide_hash, expected_labels_hash,
        guide_version,
        "normalised-alias-precision-f1" if guide_version == 1 else str(guide["concept_metric"]),
        str(guide["critical_metric"]),
    )


def validate_frozen_embedding_contract(
    contract: Mapping[str, Any], *, expected_contract_hash: str
) -> FrozenEmbeddingContract:
    """Verify the exact local text-embedding contract before any vector is made."""
    if artifact_sha256(contract) != expected_contract_hash:
        raise StudyValidationError("embedding contract SHA-256 does not match the frozen study seal")
    if set(contract) != _EMBEDDING_CONTRACT_KEYS or contract["embedding_contract_version"] != 1:
        raise StudyValidationError("embedding contract has an unsupported envelope")
    model = contract["model"]
    dimensions = contract["dimensions"]
    if not isinstance(model, str) or not model.strip() or not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions < 1:
        raise StudyValidationError("embedding contract model and dimensions are invalid")
    if contract["selection_instrument_version"] != _SELECTION_INSTRUMENT_VERSION:
        raise StudyValidationError("embedding contract does not pin the frozen selection instrument")
    return FrozenEmbeddingContract(model, dimensions, expected_contract_hash)


def validate_frozen_query_artifact(
    artifact: Mapping[str, Any], *, expected_artifact_hash: str
) -> FrozenQueryArtifact:
    """Verify the private retrieval-query artifact before looking at caption receipts."""
    if artifact_sha256(artifact) != expected_artifact_hash:
        raise StudyValidationError("query artifact SHA-256 does not match the frozen study seal")
    if set(artifact) != _QUERY_ARTIFACT_KEYS or artifact["query_artifact_version"] != 1 or not isinstance(artifact["queries"], list):
        raise StudyValidationError("query artifact has an unsupported envelope")
    return FrozenQueryArtifact(tuple(artifact["queries"]), expected_artifact_hash)


def build_blind_pool_artifact(
    *,
    manifest_hash: str,
    base_query_artifact: FrozenQueryArtifact,
    candidates: Sequence[CandidateSpec],
    candidate_rankings: Mapping[CandidateSpec, Mapping[str, Sequence[str]]],
    corpus_ids: set[str],
) -> dict[str, Any]:
    """Seal the opaque union of each candidate's fixed-RRF top ten.

    Candidate identity and rank are deliberately discarded before the artifact
    leaves scoring; only query IDs and deduplicated corpus fixture IDs remain.
    """

    query_ids = {str(query.get("query_id")) for query in base_query_artifact.queries}
    if set(candidate_rankings) != set(candidates):
        raise StudyValidationError("blind pool rankings must cover every sealed candidate")
    pooled: dict[str, list[str]] = {}
    for query_id in sorted(query_ids):
        identifiers: set[str] = set()
        for candidate in candidates:
            rankings = candidate_rankings[candidate]
            if set(rankings) != query_ids:
                raise StudyValidationError("blind pool rankings must cover every frozen query")
            ranking = rankings[query_id]
            if isinstance(ranking, (str, bytes)) or not isinstance(ranking, Sequence) or not 1 <= len(ranking) <= 10:
                raise StudyValidationError("blind pool ranking must contain one through ten fixture IDs")
            for fixture_id in ranking:
                fixture_id = _require_fixture_id(fixture_id, "blind pool fixture_id")
                if fixture_id not in corpus_ids:
                    raise StudyValidationError("blind pool ranking references an item outside the corpus")
                identifiers.add(fixture_id)
        pooled[query_id] = sorted(identifiers)
    return {
        "blind_pool_version": 1,
        "manifest_hash": manifest_hash,
        "base_query_artifact_hash": base_query_artifact.artifact_hash,
        "candidate_set_hash": candidate_set_hash(candidates),
        "ranking_lane": "rrf",
        "query_items": pooled,
    }


def validate_blind_pool_artifact(
    artifact: Mapping[str, Any],
    *,
    expected_artifact_hash: str,
    manifest_hash: str,
    base_query_artifact: FrozenQueryArtifact,
    expected_candidate_set_hash: str,
    corpus_ids: set[str],
) -> FrozenBlindPool:
    """Validate an opaque pool before it can support final relevance grades."""

    if artifact_sha256(artifact) != expected_artifact_hash:
        raise StudyValidationError("blind pool artifact SHA-256 does not match the final-selection seal")
    if set(artifact) != _BLIND_POOL_KEYS or artifact.get("blind_pool_version") != 1:
        raise StudyValidationError("blind pool artifact has an unsupported envelope")
    if artifact.get("manifest_hash") != manifest_hash:
        raise StudyValidationError("blind pool artifact is bound to another manifest")
    if artifact.get("base_query_artifact_hash") != base_query_artifact.artifact_hash:
        raise StudyValidationError("blind pool artifact is bound to another query artifact")
    if artifact.get("candidate_set_hash") != expected_candidate_set_hash or artifact.get("ranking_lane") != "rrf":
        raise StudyValidationError("blind pool artifact has incompatible candidate or ranking evidence")
    query_ids = {str(query.get("query_id")) for query in base_query_artifact.queries}
    raw_items = artifact.get("query_items")
    if not isinstance(raw_items, Mapping) or set(raw_items) != query_ids:
        raise StudyValidationError("blind pool artifact must cover every frozen query exactly once")
    parsed: dict[str, tuple[str, ...]] = {}
    for query_id, items in raw_items.items():
        if isinstance(items, (str, bytes)) or not isinstance(items, list) or not 1 <= len(items) <= len(corpus_ids):
            raise StudyValidationError("blind pool items are invalid")
        checked = tuple(_require_fixture_id(item, "blind pool fixture_id") for item in items)
        if tuple(sorted(set(checked))) != checked or any(item not in corpus_ids for item in checked):
            raise StudyValidationError("blind pool items must be sorted unique corpus fixture IDs")
        parsed[query_id] = checked
    return FrozenBlindPool(parsed, expected_artifact_hash)


def validate_pooled_relevance_adjudication(
    artifact: Mapping[str, Any],
    *,
    expected_artifact_hash: str,
    manifest_hash: str,
    base_query_artifact: FrozenQueryArtifact,
    expected_candidate_set_hash: str,
    blind_pool: FrozenBlindPool,
    corpus_ids: set[str],
    expected_packet_evidence_hash: str,
) -> FrozenQueryArtifact:
    """Apply only a complete, blinded, hash-bound relevance adjudication.

    The artifact contains opaque fixture identifiers and integer grades only.  It
    never carries candidate identities, ranks, captions, paths, or pixels.
    """

    if artifact_sha256(artifact) != expected_artifact_hash:
        raise StudyValidationError("pooled relevance artifact SHA-256 does not match the final-selection seal")
    if set(artifact) != _POOLED_RELEVANCE_KEYS or artifact.get("pooled_relevance_version") != 2:
        raise StudyValidationError("pooled relevance artifact has an unsupported envelope")
    if artifact.get("manifest_hash") != manifest_hash:
        raise StudyValidationError("pooled relevance artifact is bound to another manifest")
    if artifact.get("base_query_artifact_hash") != base_query_artifact.artifact_hash:
        raise StudyValidationError("pooled relevance artifact is bound to another query artifact")
    if artifact.get("candidate_set_hash") != expected_candidate_set_hash:
        raise StudyValidationError("pooled relevance artifact is bound to another candidate set")
    if artifact.get("blind_pool_hash") != blind_pool.artifact_hash:
        raise StudyValidationError("pooled relevance artifact is bound to another blind pool")
    if (
        not isinstance(expected_packet_evidence_hash, str)
        or not expected_packet_evidence_hash.startswith("sha256:")
        or artifact.get("packet_evidence_hash") != expected_packet_evidence_hash
    ):
        raise StudyValidationError("pooled relevance artifact is not bound to the sealed grading packet evidence")
    if artifact.get("blinded") is not True or artifact.get("adjudication_complete") is not True:
        raise StudyValidationError("pooled relevance artifact must record complete blinded adjudication")
    raw_grades = artifact.get("query_grades")
    if not isinstance(raw_grades, Mapping):
        raise StudyValidationError("pooled relevance artifact query grades are invalid")

    base_by_id = {str(query.get("query_id")): query for query in base_query_artifact.queries}
    if set(raw_grades) != set(base_by_id):
        raise StudyValidationError("pooled relevance artifact must cover every frozen query exactly once")
    merged: list[dict[str, Any]] = []
    for query_id, query in base_by_id.items():
        grades = raw_grades[query_id]
        if not isinstance(grades, Mapping) or not grades:
            raise StudyValidationError("pooled relevance grades must be nonempty mappings")
        required = set(query["grades"]) | set(blind_pool.query_items[query_id])
        if not required.issubset(grades):
            raise StudyValidationError("pooled relevance artifact omits a frozen or pooled relevance judgment")
        checked: dict[str, int] = {}
        for fixture_id, grade in grades.items():
            fixture_id = _require_fixture_id(fixture_id, "pooled relevance fixture_id")
            if fixture_id not in corpus_ids or not isinstance(grade, int) or isinstance(grade, bool) or not 0 <= grade <= 3:
                raise StudyValidationError("pooled relevance grades are invalid")
            checked[fixture_id] = grade
        if not any(grade > 0 for grade in checked.values()):
            raise StudyValidationError("pooled relevance query needs at least one relevant grade")
        merged.append({**query, "grades": checked})
    return FrozenQueryArtifact(tuple(merged), expected_artifact_hash)


def _stage_b_fixtures(
    fixtures: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, ...], tuple[str, ...], set[str], dict[str, str]]:
    identifiers: set[str] = set()
    targets: list[str] = []
    all_stage_b: list[str] = []
    target_strata: dict[str, str] = {}
    for fixture in fixtures:
        if not isinstance(fixture, Mapping):
            raise StudyValidationError("fixture must be an object")
        fixture_id = _require_fixture_id(fixture.get("fixture_id"), "fixture_id")
        if fixture_id in identifiers:
            raise StudyValidationError("fixture IDs must be unique")
        identifiers.add(fixture_id)
        stage, role = fixture.get("stage"), fixture.get("role")
        if stage not in {"A", "B", "C"} or role not in {"target", "distractor"}:
            raise StudyValidationError("fixture stage and role are invalid")
        if stage == "B":
            all_stage_b.append(fixture_id)
            if role == "target":
                targets.append(fixture_id)
                stratum = fixture.get("stratum", "unstratified")
                if not isinstance(stratum, str) or not stratum.strip():
                    raise StudyValidationError("Stage B target stratum is invalid")
                target_strata[fixture_id] = stratum
    if not targets or not all_stage_b:
        raise StudyValidationError("Stage B requires target and corpus fixtures")
    return tuple(all_stage_b), tuple(targets), identifiers, target_strata


@dataclass(frozen=True)
class _CaptionedFixture:
    document: SelectionDocument
    result: CaptionResultV1 | None
    latency_seconds: float | None


def _receipt_candidate(raw: Mapping[str, Any]) -> tuple[str, str, str]:
    fields = ("model", "effort", "prompt_version")
    values = tuple(raw.get(field) for field in fields)
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise StudyValidationError("caption receipt is missing candidate identity")
    return values  # type: ignore[return-value]


def _receipt_latency(raw: Mapping[str, Any]) -> float | None:
    value = raw.get("latency_seconds")
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise StudyValidationError("caption receipt latency_seconds must be a finite nonnegative number")
    return float(value)


def _candidate_documents(
    *,
    fixtures: Sequence[str],
    known_fixture_ids: set[str],
    receipts: Sequence[Mapping[str, Any]],
    candidate: CandidateSpec,
    manifest_hash: str,
) -> dict[str, _CaptionedFixture]:
    stage_fixture_ids = set(fixtures)
    received: dict[str, _CaptionedFixture] = {}
    for raw in receipts:
        if not isinstance(raw, Mapping):
            raise StudyValidationError("caption receipt must be an object")
        fixture_id = _require_fixture_id(raw.get("fixture_id"), "caption receipt fixture_id")
        if fixture_id not in known_fixture_ids:
            raise StudyValidationError("caption receipt references an unknown fixture")
        if fixture_id not in stage_fixture_ids or _receipt_candidate(raw) != candidate.receipt_key():
            continue
        if raw.get("manifest_hash") != manifest_hash:
            raise StudyValidationError("caption receipt does not belong to this frozen manifest")
        if fixture_id in received:
            raise StudyValidationError("candidate has duplicate caption receipts for a fixture")
        parsed: CaptionResultV1 | None
        try:
            parsed = CaptionResultV1.from_dict(raw.get("caption_result"))
        except (ContractError, TypeError, AttributeError):
            parsed = None
        received[fixture_id] = _CaptionedFixture(
            document=SelectionDocument(fixture_id, parsed.search_text() if parsed is not None else ""),
            result=parsed,
            latency_seconds=_receipt_latency(raw),
        )
    if set(received) != stage_fixture_ids:
        missing = len(stage_fixture_ids - set(received))
        extra = len(set(received) - stage_fixture_ids)
        raise StudyValidationError(f"candidate Stage B receipt coverage is incomplete (missing={missing}, extra={extra})")
    return received


def _predicted_concepts(result: CaptionResultV1, fields: Sequence[str]) -> list[str]:
    values: list[str] = []
    for field in fields:
        values.extend(str(value) for value in getattr(result, field))
    return values


def _predicted_ocr(result: CaptionResultV1, legibilities: frozenset[str]) -> str:
    return " ".join(entry.text for entry in result.visible_text if entry.legibility in legibilities)


def _contains_critical_term(result: CaptionResultV1, terms: Sequence[str], *, metric: str) -> bool:
    searchable = result.search_text()
    if metric == "token-phrase":
        return any(token_phrase_matches(searchable, term) for term in terms)
    return any(_normalise(term) in _normalise(searchable) for term in terms)


def _queries_by_target(
    queries: Sequence[Mapping[str, Any]], *, target_ids: set[str], corpus_ids: set[str]
) -> tuple[tuple[dict[str, Any], ...], set[str]]:
    parsed: list[dict[str, Any]] = []
    known_queries: set[str] = set()
    seen_ids: set[str] = set()
    for raw in queries:
        if not isinstance(raw, Mapping) or set(raw) - {"query_id", "target_fixture_id", "query", "grades", "known_query"}:
            raise StudyValidationError("retrieval query has unexpected fields")
        query_id = raw.get("query_id")
        target = raw.get("target_fixture_id")
        query = raw.get("query")
        grades = raw.get("grades")
        if not isinstance(query_id, str) or not query_id.strip() or query_id in seen_ids:
            raise StudyValidationError("retrieval query IDs must be unique nonblank strings")
        if target not in target_ids or not isinstance(query, str) or not query.strip() or not isinstance(grades, Mapping):
            raise StudyValidationError("retrieval query target, text, or grades are invalid")
        converted: dict[str, int] = {}
        for fixture_id, grade in grades.items():
            fixture_id = _require_fixture_id(fixture_id, "retrieval grade fixture_id")
            if fixture_id not in corpus_ids or not isinstance(grade, int) or isinstance(grade, bool) or not 0 <= grade <= 3:
                raise StudyValidationError("retrieval grades must reference Stage B corpus IDs with integer grades from zero to three")
            converted[fixture_id] = grade
        if not any(grade > 0 for grade in converted.values()):
            raise StudyValidationError("retrieval query needs at least one relevant grade")
        known = raw.get("known_query", False)
        if not isinstance(known, bool):
            raise StudyValidationError("known_query must be boolean when present")
        parsed.append({"query_id": query_id, "target_fixture_id": target, "query": query, "grades": converted, "known_query": known})
        seen_ids.add(query_id)
        if known:
            known_queries.add(query_id)
    if not parsed or set(entry["target_fixture_id"] for entry in parsed) != target_ids or not known_queries:
        raise StudyValidationError("retrieval queries must cover every Stage B target and include a known query")
    return tuple(parsed), known_queries


def _mean(values: Sequence[float]) -> float:
    return fmean(values) if values else 0.0


def _vector(value: Any, *, dimensions: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != dimensions:
        raise StudyValidationError(f"{name} has the wrong embedding dimensions")
    vector: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool) or not math.isfinite(item):
            raise StudyValidationError(f"{name} embedding values must be finite numbers")
        vector.append(float(item))
    if not any(vector):
        raise StudyValidationError(f"{name} embedding cannot be a zero vector")
    return tuple(vector)


def _embed_queries(
    queries: Sequence[Mapping[str, Any]], *, embed: Callable[[str], Sequence[float]], dimensions: int
) -> dict[str, tuple[float, ...]]:
    """Embed every frozen query once; vectors are shared only because the text is frozen."""
    vectors: dict[str, tuple[float, ...]] = {}
    for query in queries:
        query_id = str(query["query_id"])
        if query_id in vectors:
            raise StudyValidationError("frozen query artifact has duplicate query IDs")
        try:
            vectors[query_id] = _vector(embed(str(query["query"])), dimensions=dimensions, name=f"query {query_id}")
        except StudyValidationError:
            raise
        except Exception as error:
            raise StudyValidationError(f"embedding query {query_id} failed") from error
    if len(vectors) != len(queries):
        raise StudyValidationError("frozen query embedding count is incomplete")
    return vectors


def _embed_documents(
    captions: Mapping[str, _CaptionedFixture], *, candidate: CandidateSpec, embed: Callable[[str], Sequence[float]], dimensions: int
) -> dict[str, tuple[float, ...]]:
    """Embed a complete candidate-owned corpus; never reuse another candidate's vectors."""
    vectors: dict[str, tuple[float, ...]] = {}
    for fixture_id in sorted(captions):
        try:
            vectors[fixture_id] = _vector(
                embed(captions[fixture_id].document.search_text),
                dimensions=dimensions,
                name=f"candidate {candidate.model}/{candidate.effort} fixture {fixture_id}",
            )
        except StudyValidationError:
            raise
        except Exception as error:
            raise StudyValidationError(f"embedding candidate corpus fixture {fixture_id} failed") from error
    if set(vectors) != set(captions):
        raise StudyValidationError("candidate document embedding count is incomplete")
    return vectors


def _candidate_scores(
    *,
    candidate: CandidateSpec,
    captions: Mapping[str, _CaptionedFixture],
    annotations: FrozenAnnotations,
    targets: Sequence[str],
    target_strata: Mapping[str, str],
    queries: Sequence[Mapping[str, Any]],
    known_queries: set[str],
    document_embeddings: Mapping[str, Sequence[float]],
    query_embeddings: Mapping[str, Sequence[float]],
) -> tuple[dict[str, Any], dict[str, dict[str, list[float]]], dict[str, tuple[str, ...]]]:
    noninferiority_margins = (
        _V2_NONINFERIORITY_MARGINS
        if annotations.annotation_guide_version == 2
        else _V1_NONINFERIORITY_MARGINS
    )
    lane_target_metrics = tuple(
        f"{lane}_{metric}"
        for lane in _LANE_NONINFERIORITY_MARGINS
        for metric in _LANE_NONINFERIORITY_MARGINS[lane]
    )
    target_values: dict[str, dict[str, list[float]]] = {
        metric: {fixture_id: [] for fixture_id in targets}
        for metric in (*noninferiority_margins, "type_accuracy", *lane_target_metrics)
    }
    strata_values: dict[str, dict[str, list[float]]] = {
        stratum: {
            metric: []
            for metric in ("concept_precision", "concept_recall", "concept_f1", "ocr_character_f1", "type_accuracy")
        }
        for stratum in sorted(set(target_strata.values()))
    }
    strata_lanes: dict[str, dict[str, dict[str, list[float]]]] = {
        stratum: {
            lane: {"ndcg_at_10": [], "recall_at_10": [], "mrr": []}
            for lane in ("bm25", "semantic", "rrf")
        }
        for stratum in strata_values
    }
    concept_precision: list[float] = []
    concept_recall: list[float] = []
    concept_f1: list[float] = []
    ocr_f1: list[float] = []
    ocr_bearing_f1: list[float] = []
    type_accuracy: list[float] = []
    critical_flags: list[bool] = []
    for fixture_id in targets:
        label = annotations.labels[fixture_id]
        result = captions[fixture_id].result
        if result is None:
            concepts = ConceptScores(0.0, 0.0, 0.0)
            ocr = 0.0
            image_type = 0.0
            critical = False
        else:
            concepts = concept_scores(_predicted_concepts(result, annotations.concept_fields), [set(group) for group in label.concept_alias_groups])
            ocr_metric = exact_ocr_character_f1 if annotations.annotation_guide_version == 2 else ocr_character_f1
            ocr = ocr_metric(_predicted_ocr(result, annotations.ocr_legibilities), label.ocr_text)
            image_type = float(_normalise(result.image_type) in label.image_type_aliases)
            critical = _contains_critical_term(result, label.critical_absent_terms, metric=annotations.critical_metric)
        recall = (
            token_aware_concept_recall(
                _predicted_concepts(result, annotations.concept_fields), [set(group) for group in label.concept_alias_groups]
            )
            if result is not None and annotations.annotation_guide_version == 2
            else concepts.recall
        )
        concept_precision.append(concepts.precision)
        concept_recall.append(recall)
        concept_f1.append(concepts.f1)
        ocr_f1.append(ocr)
        type_accuracy.append(image_type)
        critical_flags.append(critical)
        stratum = target_strata[fixture_id]
        strata_values[stratum]["concept_precision"].append(concepts.precision)
        strata_values[stratum]["concept_recall"].append(recall)
        strata_values[stratum]["concept_f1"].append(concepts.f1)
        strata_values[stratum]["ocr_character_f1"].append(ocr)
        strata_values[stratum]["type_accuracy"].append(image_type)
        if _normalise(label.ocr_text):
            ocr_bearing_f1.append(ocr)
        if "concept_f1" in target_values:
            target_values["concept_f1"][fixture_id].append(concepts.f1)
        if "concept_recall" in target_values:
            target_values["concept_recall"][fixture_id].append(recall)
        target_values["ocr_character_f1"][fixture_id].append(ocr)
        target_values["type_accuracy"][fixture_id].append(image_type)

    ranking = SelectionInstrument()
    documents = [captions[fixture_id].document for fixture_id in sorted(captions)]
    lane_values: dict[str, dict[str, list[float]]] = {
        lane: {"ndcg_at_10": [], "recall_at_10": [], "mrr": []}
        for lane in ("bm25", "semantic", "rrf")
    }
    known_passed = True
    blind_pool_rankings: dict[str, tuple[str, ...]] = {}
    for query in queries:
        query_id = str(query["query_id"])
        if query_id not in query_embeddings:
            raise StudyValidationError("frozen query embedding is missing")
        ranked = ranking.rank(
            query["query"],
            documents,
            document_embeddings=document_embeddings,
            query_embedding=query_embeddings[query_id],
            limit=10,
        )
        rankings = {"bm25": ranked.bm25_ids, "semantic": ranked.semantic_ids, "rrf": ranked.fused_ids}
        blind_pool_rankings[query_id] = tuple(ranked.fused_ids)
        for lane, identifiers in rankings.items():
            lane_values[lane]["ndcg_at_10"].append(ndcg_at_k(identifiers, query["grades"]))
            lane_values[lane]["recall_at_10"].append(recall_at_k(identifiers, query["grades"]))
            lane_values[lane]["mrr"].append(reciprocal_rank(identifiers, query["grades"]))
        target_id = query["target_fixture_id"]
        stratum = target_strata[target_id]
        target_values["ndcg_at_10"][target_id].append(lane_values["rrf"]["ndcg_at_10"][-1])
        target_values["recall_at_10"][target_id].append(lane_values["rrf"]["recall_at_10"][-1])
        for lane in _LANE_NONINFERIORITY_MARGINS:
            for metric in _LANE_NONINFERIORITY_MARGINS[lane]:
                value = lane_values[lane][metric][-1]
                target_values[f"{lane}_{metric}"][target_id].append(value)
                strata_lanes[stratum][lane][metric].append(value)
            strata_lanes[stratum][lane]["mrr"].append(lane_values[lane]["mrr"][-1])
        for metric in ("ndcg_at_10", "recall_at_10", "mrr"):
            strata_lanes[stratum]["rrf"][metric].append(lane_values["rrf"][metric][-1])
        if query_id in known_queries and not _known_query_target_passed(
            rankings["rrf"],
            target_id,
        ):
            known_passed = False
    schema_valid = sum(caption.result is not None for caption in captions.values()) / len(captions)
    latencies = [caption.latency_seconds for caption in captions.values() if caption.latency_seconds is not None]
    aggregate = {
        "model": candidate.model,
        "effort": candidate.effort,
        "schema_valid_rate": schema_valid,
        "critical_hallucinations": critical_hallucination_count(critical_flags),
        "concept_precision": _mean(concept_precision),
        "concept_recall": _mean(concept_recall),
        "concept_f1": _mean(concept_f1),
        "ocr_character_f1": _mean(ocr_f1),
        "ocr_bearing_target_count": len(ocr_bearing_f1),
        "ocr_bearing_character_f1": _mean(ocr_bearing_f1) if ocr_bearing_f1 else None,
        "type_accuracy": _mean(type_accuracy),
        # RRF is the frozen primary retrieval score; lane values remain visible
        # for diagnosing lexical versus semantic regressions.
        "ndcg_at_10": _mean(lane_values["rrf"]["ndcg_at_10"]),
        "recall_at_10": _mean(lane_values["rrf"]["recall_at_10"]),
        "known_query_passed": known_passed,
        # A documented override is a separate, sealed operational amendment.
        # Pure study scoring never grants it from candidate output.
        "background_throughput_accepted": False,
        "lower_bounds": {},
        "lanes": {
            lane: {metric: _mean(values) for metric, values in metrics.items()}
            for lane, metrics in lane_values.items()
        },
        "strata": {
            stratum: {
                "target_count": len(strata_values[stratum]["concept_f1"]),
                **{metric: _mean(values) for metric, values in strata_values[stratum].items()},
                "retrieval_lanes": {
                    lane: {metric: _mean(values) for metric, values in metrics.items()}
                    for lane, metrics in strata_lanes[stratum].items()
                },
            }
            for stratum in sorted(strata_values)
        },
        "lane_lower_bounds": {},
        "latencies_seconds": latencies,
        "prompt_version": candidate.prompt_version,
        "annotation_guide_version": annotations.annotation_guide_version,
        "per_target": target_values,
    }
    return aggregate, target_values, blind_pool_rankings


def _lower_bounds_and_power(
    *,
    candidate: Mapping[str, dict[str, list[float]]],
    comparator: Mapping[str, dict[str, list[float]]],
    seed: int,
    power_simulations: int,
    margins: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, int | None], int | None]:
    lower_bounds: dict[str, float] = {}
    powered: dict[str, int | None] = {}
    for index, (metric, margin) in enumerate(margins.items()):
        candidate_values = candidate[metric]
        comparator_values = comparator[metric]
        if set(candidate_values) != set(comparator_values) or any(not values for values in candidate_values.values()) or any(not values for values in comparator_values.values()):
            raise StudyValidationError("paired target metrics are incomplete")
        lower_bounds[metric] = cluster_bootstrap_lower_bound(
            candidate_values, comparator_values, seed=seed + index, resamples=10_000
        )
        differences = [fmean(candidate_values[fixture_id]) - fmean(comparator_values[fixture_id]) for fixture_id in sorted(candidate_values)]
        powered[metric] = required_target_count(
            differences,
            margin=abs(margin),
            seed=seed + 100 + index,
            simulations=power_simulations,
        )
    return lower_bounds, powered, None if any(value is None for value in powered.values()) else max(value for value in powered.values() if value is not None)


def _lane_lower_bounds(
    *,
    candidate: Mapping[str, dict[str, list[float]]],
    comparator: Mapping[str, dict[str, list[float]]],
    seed: int,
) -> dict[str, dict[str, float]]:
    """Return independent paired bounds for the lexical and semantic lanes."""

    bounds: dict[str, dict[str, float]] = {}
    offset = 0
    for lane, metrics in _LANE_NONINFERIORITY_MARGINS.items():
        bounds[lane] = {}
        for metric in metrics:
            key = f"{lane}_{metric}"
            candidate_values, comparator_values = candidate[key], comparator[key]
            if (
                set(candidate_values) != set(comparator_values)
                or any(not values for values in candidate_values.values())
                or any(not values for values in comparator_values.values())
            ):
                raise StudyValidationError("paired lane metrics are incomplete")
            bounds[lane][metric] = cluster_bootstrap_lower_bound(
                candidate_values, comparator_values, seed=seed + offset, resamples=10_000
            )
            offset += 1
    return bounds


def run_stage_b_study(
    *,
    guide: Mapping[str, Any],
    labels: Mapping[str, Any],
    expected_guide_hash: str,
    expected_labels_hash: str,
    fixtures: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    query_artifact: Mapping[str, Any] | None = None,
    expected_query_hash: str = "",
    pooled_relevance_artifact: Mapping[str, Any] | None = None,
    expected_pooled_relevance_hash: str = "",
    blind_pool_artifact: Mapping[str, Any] | None = None,
    expected_blind_pool_hash: str = "",
    expected_packet_evidence_hash: str = "",
    selection_mode: str = "preliminary",
    embedding_contract: Mapping[str, Any] | None = None,
    expected_embedding_contract_hash: str = "",
    embed: Callable[[str], Sequence[float]] | None = None,
    candidates: Sequence[CandidateSpec],
    comparator: CandidateSpec,
    manifest_hash: str,
    seed: int,
    power_simulations: int = 10_000,
) -> dict[str, Any]:
    """Score a complete Stage B corpus with sealed lexical, semantic and RRF lanes.

    ``embed`` is injected: production can bind its local text embedder while
    tests use a deterministic fake.  Omitting any embedding evidence is a hard
    failure, never an implicit lexical-only evaluation.
    """
    if not isinstance(seed, int) or power_simulations <= 0:
        raise StudyValidationError("study seed and positive power simulations are required")
    if selection_mode not in {"preliminary", "final"}:
        raise StudyValidationError("selection mode must be preliminary or final")
    if selection_mode == "preliminary" and (
        pooled_relevance_artifact is not None
        or expected_pooled_relevance_hash
        or blind_pool_artifact is not None
        or expected_blind_pool_hash
        or expected_packet_evidence_hash
    ):
        raise StudyValidationError("preliminary scoring cannot use pooled relevance adjudication")
    if selection_mode == "final" and (
        pooled_relevance_artifact is None
        or not expected_pooled_relevance_hash
        or blind_pool_artifact is None
        or not expected_blind_pool_hash
        or not expected_packet_evidence_hash
    ):
        raise StudyValidationError("final selection requires hash-bound blind-pool and pooled-relevance artifacts")
    if query_artifact is None or embedding_contract is None or embed is None or not callable(embed):
        raise StudyValidationError("frozen query and embedding evidence are required for semantic/RRF scoring")
    annotations = validate_frozen_annotations(
        guide, labels, expected_guide_hash=expected_guide_hash, expected_labels_hash=expected_labels_hash
    )
    if selection_mode == "final" and annotations.annotation_guide_version != 2:
        raise StudyValidationError("final selection requires annotation guide v2")
    frozen_queries = validate_frozen_query_artifact(query_artifact, expected_artifact_hash=expected_query_hash)
    base_query_hash = frozen_queries.artifact_hash
    base_frozen_queries = frozen_queries
    frozen_embedding = validate_frozen_embedding_contract(
        embedding_contract, expected_contract_hash=expected_embedding_contract_hash
    )
    stage_ids, target_ids, all_fixture_ids, target_strata = _stage_b_fixtures(fixtures)
    if set(annotations.labels) != set(target_ids):
        raise StudyValidationError("adjudicated labels must cover exactly the Stage B target fixture IDs")
    if not candidates or comparator not in candidates:
        raise StudyValidationError("study requires a comparator included in its candidates")
    if len({(candidate.model, candidate.effort) for candidate in candidates}) != len(candidates):
        raise StudyValidationError("report-compatible candidates cannot compare multiple prompts for one model tier")
    if not isinstance(manifest_hash, str) or not manifest_hash.startswith("sha256:"):
        raise StudyValidationError("study requires a frozen manifest SHA-256")
    blind_pool: FrozenBlindPool | None = None
    if selection_mode == "final":
        assert pooled_relevance_artifact is not None and blind_pool_artifact is not None
        blind_pool = validate_blind_pool_artifact(
            blind_pool_artifact,
            expected_artifact_hash=expected_blind_pool_hash,
            manifest_hash=manifest_hash,
            base_query_artifact=base_frozen_queries,
            expected_candidate_set_hash=candidate_set_hash(candidates),
            corpus_ids=set(stage_ids),
        )
        frozen_queries = validate_pooled_relevance_adjudication(
            pooled_relevance_artifact,
            expected_artifact_hash=expected_pooled_relevance_hash,
            manifest_hash=manifest_hash,
            base_query_artifact=base_frozen_queries,
            expected_candidate_set_hash=candidate_set_hash(candidates),
            blind_pool=blind_pool,
            corpus_ids=set(stage_ids),
            expected_packet_evidence_hash=expected_packet_evidence_hash,
        )
    parsed_queries, known_queries = _queries_by_target(
        frozen_queries.queries, target_ids=set(target_ids), corpus_ids=set(stage_ids)
    )
    query_embeddings = _embed_queries(
        parsed_queries, embed=embed, dimensions=frozen_embedding.dimensions
    )
    aggregates: list[dict[str, Any]] = []
    target_metrics: dict[CandidateSpec, dict[str, dict[str, list[float]]]] = {}
    candidate_rankings: dict[CandidateSpec, dict[str, tuple[str, ...]]] = {}
    for candidate in candidates:
        documents = _candidate_documents(
            fixtures=stage_ids,
            known_fixture_ids=all_fixture_ids,
            receipts=receipts,
            candidate=candidate,
            manifest_hash=manifest_hash,
        )
        document_embeddings = _embed_documents(
            documents,
            candidate=candidate,
            embed=embed,
            dimensions=frozen_embedding.dimensions,
        )
        aggregate, metrics, rankings = _candidate_scores(
            candidate=candidate,
            captions=documents,
            annotations=annotations,
            targets=target_ids,
            target_strata=target_strata,
            queries=parsed_queries,
            known_queries=known_queries,
            document_embeddings=document_embeddings,
            query_embeddings=query_embeddings,
        )
        aggregates.append(aggregate)
        target_metrics[candidate] = metrics
        candidate_rankings[candidate] = rankings
    actual_blind_pool = build_blind_pool_artifact(
        manifest_hash=manifest_hash,
        base_query_artifact=base_frozen_queries,
        candidates=candidates,
        candidate_rankings=candidate_rankings,
        corpus_ids=set(stage_ids),
    )
    actual_blind_pool_hash = artifact_sha256(actual_blind_pool)
    if selection_mode == "final" and actual_blind_pool_hash != expected_blind_pool_hash:
        raise StudyValidationError("blind pool artifact does not match the sealed candidates' actual top-ten rankings")
    comparator_metrics = target_metrics[comparator]
    powered_values: list[int | None] = []
    powered_by_candidate: dict[str, int | None] = {}
    for candidate, aggregate in zip(candidates, aggregates, strict=True):
        lower_bounds, powered, target_count = _lower_bounds_and_power(
            candidate=target_metrics[candidate],
            comparator=comparator_metrics,
            seed=seed,
            power_simulations=power_simulations,
            margins=(
                _V2_NONINFERIORITY_MARGINS
                if annotations.annotation_guide_version == 2
                else _V1_NONINFERIORITY_MARGINS
            ),
        )
        aggregate["lower_bounds"] = lower_bounds
        aggregate["lane_lower_bounds"] = _lane_lower_bounds(
            candidate=target_metrics[candidate], comparator=comparator_metrics, seed=seed + 1_000
        )
        aggregate["selection_mode"] = selection_mode
        aggregate["pooled_relevance_hash"] = expected_pooled_relevance_hash if selection_mode == "final" else None
        aggregate["powered_target_counts"] = powered
        aggregate["powered_hidden_target_count"] = target_count
        candidate_identity = f"{candidate.model}:{candidate.effort}:{candidate.prompt_version}"
        powered_by_candidate[candidate_identity] = target_count
        powered_values.append(target_count)
    return {
        "study_version": 1,
        "stage": "B",
        "guide_hash": annotations.guide_hash,
        "labels_hash": annotations.labels_hash,
        "annotation_guide_version": annotations.annotation_guide_version,
        "query_artifact_hash": base_query_hash,
        "selection_mode": selection_mode,
        "pooled_relevance_hash": expected_pooled_relevance_hash if selection_mode == "final" else None,
        "blind_pool": actual_blind_pool,
        "blind_pool_hash": actual_blind_pool_hash,
        "embedding_contract_hash": frozen_embedding.contract_hash,
        "embedding_model": frozen_embedding.model,
        "manifest_hash": manifest_hash,
        "bootstrap_resamples": 10_000,
        "powered_hidden_target_count": None if any(value is None for value in powered_values) else max(value for value in powered_values if value is not None),
        "powered_hidden_target_counts": powered_by_candidate,
        "semantic_rrf_status": {
            "state": "scored",
            "selection_instrument_version": _SELECTION_INSTRUMENT_VERSION,
        },
        "candidates": aggregates,
    }
