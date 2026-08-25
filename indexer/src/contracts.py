"""Versioned, dependency-free contracts shared across Eagle Search packages."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


class ContractError(ValueError):
    """Raised when a versioned contract is malformed or internally inconsistent."""


SHA256_ID_RE = re.compile(r"^sha256:[a-f0-9]{64}$")


def _strict_keys(payload: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ContractError(f"{name} keys differ; missing={missing}, extra={extra}")


def _string(value: Any, field_name: str, *, max_length: int = 1_000) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{field_name} must be a string")
    if len(value) > max_length:
        raise ContractError(f"{field_name} exceeds {max_length} characters")
    return value


def _strings(value: Any, field_name: str, *, maximum: int) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > maximum:
        raise ContractError(f"{field_name} must be an array of at most {maximum} strings")
    items = tuple(_string(item, f"{field_name}[]", max_length=300) for item in value)
    if any(not item.strip() for item in items):
        raise ContractError(f"{field_name} cannot contain blank strings")
    return items


def _sha256_id(value: Any, field_name: str) -> str:
    digest = _string(value, field_name, max_length=71)
    if not SHA256_ID_RE.fullmatch(digest):
        raise ContractError(f"{field_name} must be a sha256-prefixed lowercase digest")
    return digest


@dataclass(frozen=True)
class VisibleTextV1:
    text: str
    legibility: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VisibleTextV1":
        _strict_keys(payload, {"text", "legibility"}, "VisibleTextV1")
        text = _string(payload["text"], "visible_text.text", max_length=1_000)
        legibility = _string(payload["legibility"], "visible_text.legibility", max_length=20)
        if legibility not in {"high", "medium", "low"}:
            raise ContractError("visible_text.legibility must be high, medium or low")
        return cls(text=text, legibility=legibility)

    def to_dict(self) -> dict[str, str]:
        return {"text": self.text, "legibility": self.legibility}


@dataclass(frozen=True)
class CaptionResultV1:
    contract_version: int
    image_type: str
    diagram_types: tuple[str, ...]
    subjects: tuple[str, ...]
    visual_style: tuple[str, ...]
    colours: tuple[str, ...]
    layout: tuple[str, ...]
    visible_text: tuple[VisibleTextV1, ...]
    search_terms: tuple[str, ...]
    summary: str
    uncertainties: tuple[str, ...]

    _KEYS = {
        "contract_version",
        "image_type",
        "diagram_types",
        "subjects",
        "visual_style",
        "colours",
        "layout",
        "visible_text",
        "search_terms",
        "summary",
        "uncertainties",
    }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CaptionResultV1":
        if not isinstance(payload, Mapping):
            raise ContractError("CaptionResultV1 must be an object")
        _strict_keys(payload, cls._KEYS, "CaptionResultV1")
        if payload["contract_version"] != 1:
            raise ContractError("CaptionResultV1.contract_version must be 1")
        raw_visible = payload["visible_text"]
        if not isinstance(raw_visible, list) or len(raw_visible) > 40:
            raise ContractError("visible_text must contain at most 40 entries")
        visible = tuple(VisibleTextV1.from_dict(entry) for entry in raw_visible)
        image_type = _string(payload["image_type"], "image_type", max_length=100)
        # Model output is constrained more tightly by its response schema, while
        # the durable contract must also losslessly carry longer legacy captions.
        summary = _string(payload["summary"], "summary", max_length=20_000)
        if not image_type.strip() or not summary.strip():
            raise ContractError("image_type and summary cannot be blank")
        return cls(
            contract_version=1,
            image_type=image_type,
            diagram_types=_strings(payload["diagram_types"], "diagram_types", maximum=6),
            subjects=_strings(payload["subjects"], "subjects", maximum=12),
            visual_style=_strings(payload["visual_style"], "visual_style", maximum=8),
            colours=_strings(payload["colours"], "colours", maximum=10),
            layout=_strings(payload["layout"], "layout", maximum=10),
            visible_text=visible,
            search_terms=_strings(payload["search_terms"], "search_terms", maximum=12),
            summary=summary,
            uncertainties=_strings(payload["uncertainties"], "uncertainties", maximum=8),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "image_type": self.image_type,
            "diagram_types": list(self.diagram_types),
            "subjects": list(self.subjects),
            "visual_style": list(self.visual_style),
            "colours": list(self.colours),
            "layout": list(self.layout),
            "visible_text": [entry.to_dict() for entry in self.visible_text],
            "search_terms": list(self.search_terms),
            "summary": self.summary,
            "uncertainties": list(self.uncertainties),
        }

    def search_text(self) -> str:
        fields: Iterable[str] = (
            self.image_type,
            *self.diagram_types,
            *self.subjects,
            *self.visual_style,
            *self.colours,
            *self.layout,
            *(entry.text for entry in self.visible_text),
            *self.search_terms,
            self.summary,
        )
        return " ".join(value.strip() for value in fields if value.strip())


@dataclass(frozen=True)
class CaptionReceiptV1:
    receipt_version: int
    receipt_id: str
    image_hash: str
    caption_result: CaptionResultV1
    search_text: str
    provider: str
    model: str
    effort: str
    prompt_version: str
    schema_version: int
    created_at: str
    source: str = "vision"

    _KEYS = {
        "receipt_version",
        "receipt_id",
        "image_hash",
        "caption_result",
        "search_text",
        "provider",
        "model",
        "effort",
        "prompt_version",
        "schema_version",
        "created_at",
        "source",
    }

    @classmethod
    def create(
        cls,
        *,
        image_hash: str,
        caption_result: CaptionResultV1,
        provider: str,
        model: str,
        effort: str,
        prompt_version: str,
        created_at: str,
        source: str = "vision",
    ) -> "CaptionReceiptV1":
        image_hash = _sha256_id(image_hash, "image_hash")
        if source not in {"vision", "legacy"}:
            raise ContractError("source must be vision or legacy")
        values = {
            "receipt_version": 1,
            "image_hash": image_hash,
            "caption_result": caption_result.to_dict(),
            "search_text": caption_result.search_text(),
            "provider": provider,
            "model": model,
            "effort": effort,
            "prompt_version": prompt_version,
            "schema_version": 1,
            "source": source,
        }
        receipt_id = cls._calculate_id(values)
        return cls(
            receipt_id=receipt_id,
            created_at=created_at,
            caption_result=caption_result,
            **{key: value for key, value in values.items() if key != "caption_result"},
        )

    @staticmethod
    def _calculate_id(semantic_payload: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            semantic_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "receipt_version": self.receipt_version,
            "image_hash": self.image_hash,
            "caption_result": self.caption_result.to_dict(),
            "search_text": self.search_text,
            "provider": self.provider,
            "model": self.model,
            "effort": self.effort,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CaptionReceiptV1":
        if not isinstance(payload, Mapping):
            raise ContractError("CaptionReceiptV1 must be an object")
        _strict_keys(payload, cls._KEYS, "CaptionReceiptV1")
        if payload["receipt_version"] != 1 or payload["schema_version"] != 1:
            raise ContractError("unsupported caption receipt version")
        caption = CaptionResultV1.from_dict(payload["caption_result"])
        receipt = cls(
            receipt_version=1,
            receipt_id=_sha256_id(payload["receipt_id"], "receipt_id"),
            image_hash=_sha256_id(payload["image_hash"], "image_hash"),
            caption_result=caption,
            search_text=_string(payload["search_text"], "search_text", max_length=20_000),
            provider=_string(payload["provider"], "provider", max_length=100),
            model=_string(payload["model"], "model", max_length=100),
            effort=_string(payload["effort"], "effort", max_length=30),
            prompt_version=_string(payload["prompt_version"], "prompt_version", max_length=100),
            schema_version=1,
            created_at=_string(payload["created_at"], "created_at", max_length=100),
            source=_string(payload["source"], "source", max_length=30),
        )
        if receipt.search_text != caption.search_text():
            raise ContractError("receipt search_text does not match CaptionResultV1")
        if receipt.source not in {"vision", "legacy"}:
            raise ContractError("source must be vision or legacy")
        if receipt.receipt_id != cls._calculate_id(receipt.semantic_payload()):
            raise ContractError("receipt checksum does not match semantic payload")
        return receipt

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_version": self.receipt_version,
            "receipt_id": self.receipt_id,
            "image_hash": self.image_hash,
            "caption_result": self.caption_result.to_dict(),
            "search_text": self.search_text,
            "provider": self.provider,
            "model": self.model,
            "effort": self.effort,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "source": self.source,
        }


@dataclass(frozen=True)
class SearchResultV1:
    eagle_id: str
    name: str
    thumbnail_path: str
    image_path: str
    score: float
    matched_by: tuple[str, ...] = field(default_factory=tuple)
    tags: str = ""
    annotation: str = ""
    ai_description: str = ""
    folder_name: str = ""
    ext: str = ""
    width: int = 0
    height: int = 0
    created_at: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "eagle_id": self.eagle_id,
            "name": self.name,
            "thumbnail_path": self.thumbnail_path,
            "image_path": self.image_path,
            "score": self.score,
            "matched_by": list(self.matched_by),
            "tags": self.tags,
            "annotation": self.annotation,
            "ai_description": self.ai_description,
            "folder_name": self.folder_name,
            "ext": self.ext,
            "width": self.width,
            "height": self.height,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class SearchResponseV1:
    ok: bool
    query_text: str
    limit: int
    mode: str
    semantic_available: bool
    warnings: tuple[str, ...]
    results: tuple[SearchResultV1, ...]
    error: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contract_version": 1,
            "ok": self.ok,
            "query": {"text": self.query_text, "limit": self.limit},
            "retrieval": {
                "mode": self.mode,
                "semantic_available": self.semantic_available,
                "warnings": list(self.warnings),
            },
            "results": [result.to_dict() for result in self.results],
        }
        if self.error is not None:
            payload["error"] = self.error
        return payload


@dataclass(frozen=True)
class RunStatusV1:
    run_id: str
    state: str
    stage: str
    total: int
    completed: int
    pending: int
    failed: int
    provider: str = ""
    model: str = ""
    semantic_available: bool = False
    last_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"contract_version": 1, **self.__dict__}


@dataclass(frozen=True)
class NotesDiffV1:
    eagle_id: str
    original_hash: str
    proposed_hash: str
    changed_blocks: tuple[str, ...]
    refusal_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": 1,
            "eagle_id": self.eagle_id,
            "original_hash": self.original_hash,
            "proposed_hash": self.proposed_hash,
            "changed_blocks": list(self.changed_blocks),
            "refusal_reason": self.refusal_reason,
        }
