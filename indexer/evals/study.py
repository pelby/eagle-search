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
    ndcg_at_k,
    ocr_character_f1,
    recall_at_k,
    reciprocal_rank,
    required_target_count,
)


class StudyValidationError(ValueError):
    """A private study input is malformed, unsealed, or internally inconsistent."""


_GUIDE_KEYS = {
    "annotation_guide_version",
    "label_schema_version",
    "concept_fields",
    "ocr_legibilities",
    "type_metric",
    "critical_metric",
}
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
_NONINFERIORITY_MARGINS = {
    "ndcg_at_10": -0.03,
    "recall_at_10": -0.03,
    "concept_f1": -0.04,
    "ocr_character_f1": -0.02,
}
_EMBEDDING_CONTRACT_KEYS = {
    "embedding_contract_version",
    "model",
    "dimensions",
    "selection_instrument_version",
}
_QUERY_ARTIFACT_KEYS = {"query_artifact_version", "queries"}
_SELECTION_INSTRUMENT_VERSION = "selection-instrument-v1"


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
        aliases.append(normalised)
    if len(set(aliases)) != len(aliases):
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
    if set(guide) != _GUIDE_KEYS:
        raise StudyValidationError("annotation guide has unexpected fields")
    if guide["annotation_guide_version"] != 1 or guide["label_schema_version"] != 1:
        raise StudyValidationError("unsupported annotation guide version")
    concept_fields = _strings(guide["concept_fields"], "annotation guide concept_fields")
    if not concept_fields or set(concept_fields) - _CONCEPT_FIELDS:
        raise StudyValidationError("annotation guide concept_fields are unsupported")
    legibilities = frozenset(_strings(guide["ocr_legibilities"], "annotation guide ocr_legibilities"))
    if not legibilities or not legibilities.issubset({"high", "medium", "low"}):
        raise StudyValidationError("annotation guide ocr_legibilities are unsupported")
    if guide["type_metric"] != "normalised-alias" or guide["critical_metric"] != "normalised-substring":
        raise StudyValidationError("annotation guide names an unsupported metric")
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
    return FrozenAnnotations(tuple(concept_fields), legibilities, parsed, expected_guide_hash, expected_labels_hash)


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


def _stage_b_fixtures(fixtures: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, ...], tuple[str, ...], set[str]]:
    identifiers: set[str] = set()
    targets: list[str] = []
    all_stage_b: list[str] = []
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
    if not targets or not all_stage_b:
        raise StudyValidationError("Stage B requires target and corpus fixtures")
    return tuple(all_stage_b), tuple(targets), identifiers


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


def _contains_critical_term(result: CaptionResultV1, terms: Sequence[str]) -> bool:
    searchable = _normalise(result.search_text())
    return any(_normalise(term) in searchable for term in terms)


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
            if fixture_id not in corpus_ids or not isinstance(grade, int) or isinstance(grade, bool) or grade < 0:
                raise StudyValidationError("retrieval grades must reference Stage B corpus IDs with nonnegative integers")
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
    queries: Sequence[Mapping[str, Any]],
    known_queries: set[str],
    document_embeddings: Mapping[str, Sequence[float]],
    query_embeddings: Mapping[str, Sequence[float]],
) -> tuple[dict[str, Any], dict[str, dict[str, list[float]]]]:
    target_values: dict[str, dict[str, list[float]]] = {
        metric: {fixture_id: [] for fixture_id in targets}
        for metric in (*_NONINFERIORITY_MARGINS, "type_accuracy")
    }
    concept_precision: list[float] = []
    concept_f1: list[float] = []
    ocr_f1: list[float] = []
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
            ocr = ocr_character_f1(_predicted_ocr(result, annotations.ocr_legibilities), label.ocr_text)
            image_type = float(_normalise(result.image_type) in label.image_type_aliases)
            critical = _contains_critical_term(result, label.critical_absent_terms)
        concept_precision.append(concepts.precision)
        concept_f1.append(concepts.f1)
        ocr_f1.append(ocr)
        type_accuracy.append(image_type)
        critical_flags.append(critical)
        target_values["concept_f1"][fixture_id].append(concepts.f1)
        target_values["ocr_character_f1"][fixture_id].append(ocr)
        target_values["type_accuracy"][fixture_id].append(image_type)

    ranking = SelectionInstrument()
    documents = [captions[fixture_id].document for fixture_id in sorted(captions)]
    lane_values: dict[str, dict[str, list[float]]] = {
        lane: {"ndcg_at_10": [], "recall_at_10": [], "mrr": []}
        for lane in ("bm25", "semantic", "rrf")
    }
    known_passed = True
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
        for lane, identifiers in rankings.items():
            lane_values[lane]["ndcg_at_10"].append(ndcg_at_k(identifiers, query["grades"]))
            lane_values[lane]["recall_at_10"].append(recall_at_k(identifiers, query["grades"]))
            lane_values[lane]["mrr"].append(reciprocal_rank(identifiers, query["grades"]))
        target_values["ndcg_at_10"][query["target_fixture_id"]].append(lane_values["rrf"]["ndcg_at_10"][-1])
        target_values["recall_at_10"][query["target_fixture_id"]].append(lane_values["rrf"]["recall_at_10"][-1])
        if query_id in known_queries and lane_values["rrf"]["mrr"][-1] == 0.0:
            known_passed = False
    schema_valid = sum(caption.result is not None for caption in captions.values()) / len(captions)
    latencies = [caption.latency_seconds for caption in captions.values() if caption.latency_seconds is not None]
    aggregate = {
        "model": candidate.model,
        "effort": candidate.effort,
        "schema_valid_rate": schema_valid,
        "critical_hallucinations": critical_hallucination_count(critical_flags),
        "concept_precision": _mean(concept_precision),
        "concept_f1": _mean(concept_f1),
        "ocr_character_f1": _mean(ocr_f1),
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
        "latencies_seconds": latencies,
        "prompt_version": candidate.prompt_version,
        "per_target": target_values,
    }
    return aggregate, target_values


def _lower_bounds_and_power(
    *,
    candidate: Mapping[str, dict[str, list[float]]],
    comparator: Mapping[str, dict[str, list[float]]],
    seed: int,
    power_simulations: int,
) -> tuple[dict[str, float], dict[str, int | None], int | None]:
    lower_bounds: dict[str, float] = {}
    powered: dict[str, int | None] = {}
    for index, (metric, margin) in enumerate(_NONINFERIORITY_MARGINS.items()):
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
    if query_artifact is None or embedding_contract is None or embed is None or not callable(embed):
        raise StudyValidationError("frozen query and embedding evidence are required for semantic/RRF scoring")
    annotations = validate_frozen_annotations(
        guide, labels, expected_guide_hash=expected_guide_hash, expected_labels_hash=expected_labels_hash
    )
    frozen_queries = validate_frozen_query_artifact(query_artifact, expected_artifact_hash=expected_query_hash)
    frozen_embedding = validate_frozen_embedding_contract(
        embedding_contract, expected_contract_hash=expected_embedding_contract_hash
    )
    stage_ids, target_ids, all_fixture_ids = _stage_b_fixtures(fixtures)
    if set(annotations.labels) != set(target_ids):
        raise StudyValidationError("adjudicated labels must cover exactly the Stage B target fixture IDs")
    if not candidates or comparator not in candidates:
        raise StudyValidationError("study requires a comparator included in its candidates")
    if len({(candidate.model, candidate.effort) for candidate in candidates}) != len(candidates):
        raise StudyValidationError("report-compatible candidates cannot compare multiple prompts for one model tier")
    if not isinstance(manifest_hash, str) or not manifest_hash.startswith("sha256:"):
        raise StudyValidationError("study requires a frozen manifest SHA-256")
    parsed_queries, known_queries = _queries_by_target(
        frozen_queries.queries, target_ids=set(target_ids), corpus_ids=set(stage_ids)
    )
    query_embeddings = _embed_queries(
        parsed_queries, embed=embed, dimensions=frozen_embedding.dimensions
    )
    aggregates: list[dict[str, Any]] = []
    target_metrics: dict[CandidateSpec, dict[str, dict[str, list[float]]]] = {}
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
        aggregate, metrics = _candidate_scores(
            candidate=candidate,
            captions=documents,
            annotations=annotations,
            targets=target_ids,
            queries=parsed_queries,
            known_queries=known_queries,
            document_embeddings=document_embeddings,
            query_embeddings=query_embeddings,
        )
        aggregates.append(aggregate)
        target_metrics[candidate] = metrics
    comparator_metrics = target_metrics[comparator]
    powered_values: list[int | None] = []
    for candidate, aggregate in zip(candidates, aggregates, strict=True):
        lower_bounds, powered, target_count = _lower_bounds_and_power(
            candidate=target_metrics[candidate],
            comparator=comparator_metrics,
            seed=seed,
            power_simulations=power_simulations,
        )
        aggregate["lower_bounds"] = lower_bounds
        aggregate["powered_target_counts"] = powered
        powered_values.append(target_count)
    return {
        "study_version": 1,
        "stage": "B",
        "guide_hash": annotations.guide_hash,
        "labels_hash": annotations.labels_hash,
        "query_artifact_hash": frozen_queries.artifact_hash,
        "embedding_contract_hash": frozen_embedding.contract_hash,
        "embedding_model": frozen_embedding.model,
        "manifest_hash": manifest_hash,
        "bootstrap_resamples": 10_000,
        "powered_hidden_target_count": None if any(value is None for value in powered_values) else max(value for value in powered_values if value is not None),
        "semantic_rrf_status": {
            "state": "scored",
            "selection_instrument_version": _SELECTION_INSTRUMENT_VERSION,
        },
        "candidates": aggregates,
    }
