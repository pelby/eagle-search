"""Private, caption-blind labelling validation and adjudication preparation."""

from __future__ import annotations

from hashlib import sha256
import copy
import json
from pathlib import Path
import re
from typing import Any, Mapping


class BlindLabellingError(ValueError):
    """A blind evaluation label artifact is malformed or insufficiently independent."""


_GUIDE_PATH = Path(__file__).with_name("annotation-guide-v1.json")
_BATCH_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "eval-label-batch-v1.schema.json"
_SHA256_RE = re.compile(r"^sha256:[a-f0-9]{64}$")
_FIXTURE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_GUIDE_KEYS = {
    "annotation_guide_version",
    "label_schema_version",
    "concept_fields",
    "ocr_legibilities",
    "type_metric",
    "critical_metric",
}
_BATCH_KEYS = {"batch_version", "batch_id", "provider", "model", "caption_blind", "guide_hash", "labels"}
_LABEL_KEYS = {"concept_alias_groups", "image_type_aliases", "ocr_truth", "queries", "critical_absent_terms", "uncertainty"}


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise BlindLabellingError("blind labelling artifact must be canonical JSON") from error


def annotation_guide_hash(guide: Mapping[str, Any]) -> str:
    """Return the guide seal shared with the later private study."""

    return "sha256:" + sha256(_canonical_json(guide)).hexdigest()


def load_annotation_guide() -> dict[str, Any]:
    """Load the frozen guide whose fields are accepted by ``evals.study``."""

    try:
        guide = json.loads(_GUIDE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BlindLabellingError("annotation guide could not be loaded") from error
    _validate_guide(guide)
    return guide


def load_batch_schema() -> dict[str, Any]:
    """Expose the strict handoff schema without invoking a provider."""

    try:
        schema = json.loads(_BATCH_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BlindLabellingError("blind batch schema could not be loaded") from error
    if schema.get("additionalProperties") is not False or schema.get("title") != "Eagle Search blind evaluation label batch v1":
        raise BlindLabellingError("blind batch schema is not strict v1")
    return schema


def build_provider_batch_schema(fixture_ids: list[str]) -> dict[str, Any]:
    """Expand identifier keys for providers that require closed object schemas.

    The stored validation schema accepts arbitrary identifier-only keys. OpenAI's
    structured-output dialect deliberately rejects ``propertyNames`` and requires
    every object key to be declared. The frozen fixture list is therefore compiled
    into a closed, equivalent response schema before a paid labelling call.
    """

    if not isinstance(fixture_ids, list) or not fixture_ids:
        raise BlindLabellingError("provider label schema requires fixture IDs")
    validated = [_fixture_id(value, "provider schema fixture ID") for value in fixture_ids]
    if len(set(validated)) != len(validated):
        raise BlindLabellingError("provider label schema fixture IDs must be unique")
    schema = copy.deepcopy(load_batch_schema())
    labels = schema["properties"]["labels"]
    label_schema = labels.pop("additionalProperties")
    labels.pop("propertyNames", None)
    labels["additionalProperties"] = False
    labels["required"] = sorted(validated)
    labels["properties"] = {fixture_id: copy.deepcopy(label_schema) for fixture_id in sorted(validated)}
    return schema


def _strings(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
    permit_duplicates: bool = False,
) -> list[str]:
    if not isinstance(value, list) or len(value) < minimum or (maximum is not None and len(value) > maximum):
        raise BlindLabellingError(f"{name} has an invalid item count")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise BlindLabellingError(f"{name} must contain nonblank strings")
    if not permit_duplicates and len(set(value)) != len(value):
        raise BlindLabellingError(f"{name} must contain unique nonblank strings")
    return list(value)


def _fixture_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _FIXTURE_ID_RE.fullmatch(value):
        raise BlindLabellingError(f"{name} must be an identifier-only fixture ID")
    return value


def _validate_guide(guide: Any) -> None:
    if not isinstance(guide, Mapping) or set(guide) != _GUIDE_KEYS:
        raise BlindLabellingError("annotation guide has unexpected fields")
    if guide["annotation_guide_version"] != 1 or guide["label_schema_version"] != 1:
        raise BlindLabellingError("annotation guide version is unsupported")
    if guide["concept_fields"] != ["diagram_types", "subjects", "visual_style", "colours", "layout", "search_terms"]:
        raise BlindLabellingError("annotation guide concept fields are unsupported")
    if guide["ocr_legibilities"] != ["high", "medium"]:
        raise BlindLabellingError("annotation guide OCR legibilities are unsupported")
    if guide["type_metric"] != "normalised-alias" or guide["critical_metric"] != "normalised-substring":
        raise BlindLabellingError("annotation guide metrics are unsupported")


def _validate_label(value: Any, fixture_id: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _LABEL_KEYS:
        raise BlindLabellingError(f"blind label {fixture_id} has unexpected fields")
    groups = value["concept_alias_groups"]
    if not isinstance(groups, list):
        raise BlindLabellingError(f"blind label {fixture_id} concept alias groups are invalid")
    normalised_groups: list[list[str]] = []
    for index, group in enumerate(groups):
        normalised_groups.append(_strings(group, f"blind label {fixture_id} concept alias group {index}", minimum=1))
    if len({tuple(sorted(group)) for group in normalised_groups}) != len(normalised_groups):
        raise BlindLabellingError(f"blind label {fixture_id} duplicates a concept alias group")
    ocr = value["ocr_truth"]
    if not isinstance(ocr, Mapping) or set(ocr) != {"high", "medium"}:
        raise BlindLabellingError(f"blind label {fixture_id} OCR truth is invalid")
    return {
        "concept_alias_groups": normalised_groups,
        "image_type_aliases": _strings(value["image_type_aliases"], f"blind label {fixture_id} image types", minimum=1),
        "ocr_truth": {
            "high": _strings(
                ocr["high"],
                f"blind label {fixture_id} high OCR",
                permit_duplicates=True,
            ),
            "medium": _strings(
                ocr["medium"],
                f"blind label {fixture_id} medium OCR",
                permit_duplicates=True,
            ),
        },
        "queries": _strings(value["queries"], f"blind label {fixture_id} queries", minimum=2, maximum=4),
        "critical_absent_terms": _strings(value["critical_absent_terms"], f"blind label {fixture_id} critical absent terms"),
        "uncertainty": _uncertainty(value["uncertainty"], fixture_id),
    }


def _uncertainty(value: Any, fixture_id: str) -> str:
    if not isinstance(value, str) or len(value) > 1000:
        raise BlindLabellingError(f"blind label {fixture_id} uncertainty is invalid")
    return value


def _validate_image_hashes(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise BlindLabellingError("fixture source image hashes are required separately")
    hashes: dict[str, str] = {}
    for fixture_id, image_hash in value.items():
        fixture_id = _fixture_id(fixture_id, "source image hash key")
        if not isinstance(image_hash, str) or not _SHA256_RE.fullmatch(image_hash):
            raise BlindLabellingError("fixture source image hashes must be SHA-256 identifiers")
        hashes[fixture_id] = image_hash
    return hashes


def _validate_batch(batch: Any, *, guide_hash: str, fixture_ids: set[str]) -> dict[str, Any]:
    if not isinstance(batch, Mapping) or set(batch) != _BATCH_KEYS:
        raise BlindLabellingError("blind label batch has unexpected fields")
    if batch["batch_version"] != 1 or batch["caption_blind"] is not True:
        raise BlindLabellingError("blind label batch must be v1 and caption blind")
    batch_id = batch["batch_id"]
    if not isinstance(batch_id, str) or not _FIXTURE_ID_RE.fullmatch(batch_id):
        raise BlindLabellingError("blind label batch ID is invalid")
    provider, model = batch["provider"], batch["model"]
    if any(not isinstance(item, str) or not item.strip() or len(item) > 100 for item in (provider, model)):
        raise BlindLabellingError("blind label provider receipt metadata is invalid")
    if batch["guide_hash"] != guide_hash:
        raise BlindLabellingError("blind label batch guide hash does not match the frozen guide")
    raw_labels = batch["labels"]
    if not isinstance(raw_labels, Mapping) or set(raw_labels) != fixture_ids:
        raise BlindLabellingError("blind label batch fixture coverage is not exact")
    labels = {_fixture_id(fixture_id, "blind label key"): _validate_label(raw, fixture_id) for fixture_id, raw in raw_labels.items()}
    return {"batch_id": batch_id, "provider": provider, "model": model, "labels": labels}


def _different_fields(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[str]:
    return [field for field in sorted(_LABEL_KEYS) if left[field] != right[field]]


def prepare_blind_adjudication(
    first_batch: Mapping[str, Any],
    second_batch: Mapping[str, Any],
    *,
    guide: Mapping[str, Any],
    fixture_image_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Validate two independent blind batches and prepare private adjudication artifacts.

    Source hashes are supplied only to validate coverage and are deliberately not
    copied into either returned artifact.  This function has no candidate-caption
    input and performs no provider call.
    """

    _validate_guide(guide)
    guide_hash = annotation_guide_hash(guide)
    image_hashes = _validate_image_hashes(fixture_image_hashes)
    fixture_ids = set(image_hashes)
    batches = sorted(
        (
            _validate_batch(first_batch, guide_hash=guide_hash, fixture_ids=fixture_ids),
            _validate_batch(second_batch, guide_hash=guide_hash, fixture_ids=fixture_ids),
        ),
        key=lambda batch: (batch["provider"].casefold(), batch["model"].casefold(), batch["batch_id"]),
    )
    if batches[0]["batch_id"] == batches[1]["batch_id"]:
        raise BlindLabellingError("blind label batches must be independent")
    if batches[0]["provider"].casefold() == batches[1]["provider"].casefold():
        raise BlindLabellingError("blind label batches must be provider diverse")
    if all(batch["provider"].casefold() == "openai" for batch in batches):
        raise BlindLabellingError("one independent blind label batch must be non-OpenAI")

    packet_fixtures: list[dict[str, Any]] = []
    labels: dict[str, dict[str, Any]] = {}
    query_skeleton: list[dict[str, Any]] = []
    query_suggestion_count = 0
    for fixture_id in sorted(fixture_ids):
        left, right = batches[0]["labels"][fixture_id], batches[1]["labels"][fixture_id]
        differences = _different_fields(left, right)
        if differences:
            packet_fixtures.append(
                {
                    "fixture_id": fixture_id,
                    "disagreement_fields": differences,
                    "annotator_a": left,
                    "annotator_b": right,
                }
            )
        labels[fixture_id] = {
            "concept_alias_groups": [],
            "ocr_text": "",
            "image_type_aliases": [],
            "critical_absent_terms": [],
        }
        suggestions = {"annotator_a": left["queries"], "annotator_b": right["queries"]}
        query_suggestion_count += sum(len(items) for items in suggestions.values())
        query_skeleton.append(
            {
                "target_fixture_id": fixture_id,
                "required_queries": {"minimum": 2, "maximum": 4},
                "provider_suggestions": suggestions,
            }
        )
    summary = {
        "fixture_count": len(fixture_ids),
        "provider_count": 2,
        "disagreement_count": len(packet_fixtures),
        "query_suggestion_count": query_suggestion_count,
    }
    return {
        "summary": summary,
        "disagreement_packet": {"packet_version": 1, "guide_hash": guide_hash, "fixtures": packet_fixtures},
        "adjudication_skeleton": {
            "labels": {"labels_version": 1, "labels": labels},
            "queries": {"queries_version": 1, "fixtures": query_skeleton},
        },
    }


def aggregate_counts(artifacts: Mapping[str, Any]) -> dict[str, int]:
    """Return only safe aggregate counts, never labels, queries, paths, or hashes."""

    summary = artifacts.get("summary") if isinstance(artifacts, Mapping) else None
    expected = {"fixture_count", "provider_count", "disagreement_count", "query_suggestion_count"}
    if not isinstance(summary, Mapping) or set(summary) != expected or any(not isinstance(value, int) or value < 0 for value in summary.values()):
        raise BlindLabellingError("blind adjudication aggregate summary is invalid")
    return {key: int(summary[key]) for key in sorted(expected)}
