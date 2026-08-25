"""Versioned managed Notes blocks and explicit quiescent application."""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from ..contracts import CaptionReceiptV1, NotesDiffV1
from ..worker.lock import FileLock, LockUnavailable


CAPTION_START = "<!-- eagle-search:caption:start v=1 -->"
CAPTION_END = "<!-- eagle-search:caption:end -->"
GENERATION_START = "<!-- eagle-search:generation:start v=1 -->"
GENERATION_END = "<!-- eagle-search:generation:end -->"
_RESERVED_MARKER = "<!-- eagle-search:"


class ManagedNotesError(ValueError):
    """Raised when Notes markers are ambiguous or unsafe to edit."""


class NotesApi(Protocol):
    async def get_item(self, item_id: str) -> dict[str, Any]: ...

    async def update_item(self, item_id: str, *, annotation: str) -> None: ...


@dataclass(frozen=True)
class ManagedMerge:
    proposed: str
    changed_blocks: tuple[str, ...]


@dataclass(frozen=True)
class NotesApplyResult:
    status: str
    diff: NotesDiffV1
    before_annotation: str
    proposed_annotation: str
    observed_annotation: str = ""
    refusal_reason: str = ""


def _safe_machine_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ManagedNotesError(f"{field_name} must be text")
    if _RESERVED_MARKER in value:
        raise ManagedNotesError(f"{field_name} contains a reserved Eagle Search marker")
    return value


def import_intent_marker(intent_id: str) -> str:
    try:
        parsed = uuid.UUID(intent_id)
    except (ValueError, AttributeError) as exc:
        raise ManagedNotesError("import intent must be a UUID") from exc
    if str(parsed) != intent_id.lower():
        raise ManagedNotesError("import intent must use canonical UUID text")
    return f"<!-- eagle-search:import-intent id={str(parsed)} -->"


def render_caption_block(receipt: CaptionReceiptV1) -> str:
    caption = receipt.caption_result
    visible = " | ".join(entry.text for entry in caption.visible_text) or "(none)"
    aliases = ", ".join(caption.search_terms) or "(none)"
    summary = _safe_machine_text(caption.summary, "caption summary")
    visible = _safe_machine_text(visible, "visible text")
    aliases = _safe_machine_text(aliases, "search aliases")
    return "\n".join(
        (
            CAPTION_START,
            f"**AI visual description:** {summary}",
            f"**Visible text:** {visible}",
            f"**Search aliases:** {aliases}",
            f"**Receipt:** `{receipt.receipt_id}`",
            CAPTION_END,
        )
    )


def render_generation_block(
    *,
    prompt: str,
    source: str,
    tags: tuple[str, ...],
    intent_id: str,
) -> str:
    prompt = _safe_machine_text(prompt, "generation prompt")
    source = _safe_machine_text(source, "generation source")
    safe_tags = tuple(_safe_machine_text(tag, "generation tag") for tag in tags)
    return "\n".join(
        (
            GENERATION_START,
            import_intent_marker(intent_id),
            f"**Generation prompt:** {prompt}",
            f"**Source:** {source}",
            f"**Tags:** {', '.join(safe_tags)}",
            GENERATION_END,
        )
    )


def _block_span(text: str, role: str) -> tuple[int, int] | None:
    if role == "caption":
        start, end = CAPTION_START, CAPTION_END
    elif role == "generation":
        start, end = GENERATION_START, GENERATION_END
    else:
        raise ValueError(f"unknown managed role: {role}")
    marker_pattern = re.compile(rf"<!-- eagle-search:{role}:[^>]*-->")
    markers = marker_pattern.findall(text)
    if any(marker not in {start, end} for marker in markers):
        raise ManagedNotesError(f"unsupported or malformed {role} marker")
    if markers.count(start) > 1 or markers.count(end) > 1:
        raise ManagedNotesError(f"duplicate {role} managed block")
    if markers.count(start) != markers.count(end):
        raise ManagedNotesError(f"unmatched {role} managed block marker")
    if not markers:
        return None
    start_index = text.index(start)
    end_index = text.index(end)
    if end_index < start_index:
        raise ManagedNotesError(f"reversed {role} managed block markers")
    return start_index, end_index + len(end)


def _append_block(text: str, block: str) -> str:
    if not text:
        return block
    if text.endswith("\n\n"):
        separator = ""
    elif text.endswith("\n"):
        separator = "\n"
    else:
        separator = "\n\n"
    return text + separator + block


def _validate_replacement_block(block: str, role: str) -> None:
    if role == "caption":
        start, end = CAPTION_START, CAPTION_END
    elif role == "generation":
        start, end = GENERATION_START, GENERATION_END
    else:
        raise ValueError(f"unknown managed role: {role}")
    if not isinstance(block, str) or not block.startswith(start) or not block.endswith(end):
        raise ManagedNotesError(f"replacement {role} block is not well formed")
    inner = block[len(start) : -len(end)]
    if role == "generation":
        intent_pattern = re.compile(
            r"<!-- eagle-search:import-intent id=([0-9a-f]{8}-[0-9a-f]{4}-"
            r"[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}) -->"
        )
        matches = intent_pattern.findall(inner)
        if len(matches) != 1:
            raise ManagedNotesError("generation block requires one import-intent marker")
        marker = import_intent_marker(matches[0])
        inner = inner.replace(marker, "", 1)
    if _RESERVED_MARKER in inner:
        raise ManagedNotesError(f"replacement {role} block contains an unexpected marker")


def merge_managed_blocks(
    original: str,
    *,
    caption_block: str | None = None,
    generation_block: str | None = None,
) -> ManagedMerge:
    if not isinstance(original, str):
        raise ManagedNotesError("annotation must be text")
    # Validate the entire annotation before making any change, including a role
    # that is not part of this update.
    _block_span(original, "caption")
    _block_span(original, "generation")
    proposed = original
    changed: list[str] = []
    for role, block in (("caption", caption_block), ("generation", generation_block)):
        if block is None:
            continue
        _validate_replacement_block(block, role)
        span = _block_span(proposed, role)
        if span is None:
            updated = _append_block(proposed, block)
        else:
            updated = proposed[: span[0]] + block + proposed[span[1] :]
        if updated != proposed:
            changed.append(role)
            proposed = updated
    return ManagedMerge(proposed=proposed, changed_blocks=tuple(changed))


def _text_hash(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def build_notes_diff(
    eagle_id: str,
    original: str,
    proposed: str,
    changed_blocks: tuple[str, ...],
    refusal_reason: str = "",
) -> NotesDiffV1:
    return NotesDiffV1(
        eagle_id=eagle_id,
        original_hash=_text_hash(original),
        proposed_hash=_text_hash(proposed),
        changed_blocks=changed_blocks,
        refusal_reason=refusal_reason,
    )


def _item_snapshot(item: dict[str, Any], expected_id: str) -> tuple[str, int]:
    if not isinstance(item, dict) or item.get("id") != expected_id:
        raise ManagedNotesError("Eagle item response did not match the requested ID")
    annotation = item.get("annotation")
    last_modified = item.get("lastModified")
    if not isinstance(annotation, str):
        raise ManagedNotesError("Eagle item annotation is not text")
    if isinstance(last_modified, bool) or not isinstance(last_modified, int):
        raise ManagedNotesError("Eagle item lastModified is not an integer")
    return annotation, last_modified


class NotesApplier:
    """Dry-run or explicitly apply a managed block during a quiescent window."""

    def __init__(self, api: NotesApi, lock: FileLock) -> None:
        self.api = api
        self.lock = lock

    async def sync(
        self,
        eagle_id: str,
        *,
        caption_block: str | None = None,
        generation_block: str | None = None,
        apply: bool,
        quiescent: bool,
    ) -> NotesApplyResult:
        try:
            self.lock.acquire()
        except LockUnavailable as exc:
            empty = build_notes_diff(eagle_id, "", "", (), str(exc))
            return NotesApplyResult("refused", empty, "", "", refusal_reason=str(exc))
        try:
            before_item = await self.api.get_item(eagle_id)
            before, before_modified = _item_snapshot(before_item, eagle_id)
            try:
                merge = merge_managed_blocks(
                    before,
                    caption_block=caption_block,
                    generation_block=generation_block,
                )
            except ManagedNotesError as exc:
                diff = build_notes_diff(eagle_id, before, before, (), str(exc))
                return NotesApplyResult(
                    "refused",
                    diff,
                    before,
                    before,
                    observed_annotation=before,
                    refusal_reason=str(exc),
                )
            diff = build_notes_diff(eagle_id, before, merge.proposed, merge.changed_blocks)
            if not apply:
                return NotesApplyResult("dry-run", diff, before, merge.proposed, before)
            if not quiescent:
                reason = "apply requires an explicit quiescent Eagle window"
                refused = build_notes_diff(eagle_id, before, merge.proposed, merge.changed_blocks, reason)
                return NotesApplyResult("refused", refused, before, merge.proposed, before, reason)
            if not merge.changed_blocks:
                return NotesApplyResult("unchanged", diff, before, merge.proposed, before)

            guard_item = await self.api.get_item(eagle_id)
            guard_annotation, guard_modified = _item_snapshot(guard_item, eagle_id)
            if guard_annotation != before or guard_modified != before_modified:
                reason = "annotation or lastModified changed between reads"
                refused = build_notes_diff(eagle_id, before, merge.proposed, merge.changed_blocks, reason)
                return NotesApplyResult(
                    "refused",
                    refused,
                    before,
                    merge.proposed,
                    guard_annotation,
                    reason,
                )

            try:
                await self.api.update_item(eagle_id, annotation=merge.proposed)
            except Exception as exc:
                observed = ""
                try:
                    observed, _ = _item_snapshot(await self.api.get_item(eagle_id), eagle_id)
                except Exception:
                    pass
                reason = f"Eagle update outcome is ambiguous: {exc}"
                ambiguous = build_notes_diff(
                    eagle_id,
                    before,
                    merge.proposed,
                    merge.changed_blocks,
                    reason,
                )
                return NotesApplyResult(
                    "ambiguous",
                    ambiguous,
                    before,
                    merge.proposed,
                    observed,
                    reason,
                )

            observed, _ = _item_snapshot(await self.api.get_item(eagle_id), eagle_id)
            if observed != merge.proposed:
                reason = "post-write read-back did not match the proposed annotation"
                ambiguous = build_notes_diff(
                    eagle_id,
                    before,
                    merge.proposed,
                    merge.changed_blocks,
                    reason,
                )
                return NotesApplyResult(
                    "ambiguous",
                    ambiguous,
                    before,
                    merge.proposed,
                    observed,
                    reason,
                )
            return NotesApplyResult("applied", diff, before, merge.proposed, observed)
        finally:
            self.lock.release()
