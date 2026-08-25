"""R6 red-first tests for managed Notes and quiescent apply."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.contracts import CaptionReceiptV1, CaptionResultV1
from src.eagle.notes import (
    ManagedNotesError,
    NotesApplier,
    build_notes_diff,
    merge_managed_blocks,
    render_caption_block,
    render_generation_block,
)
from src.worker.lock import FileLock


def receipt() -> CaptionReceiptV1:
    caption = CaptionResultV1.from_dict(
        {
            "contract_version": 1,
            "image_type": "illustration",
            "diagram_types": [],
            "subjects": ["teacher", "easel"],
            "visual_style": ["3D"],
            "colours": ["navy"],
            "layout": ["subject on right"],
            "visible_text": [{"text": ">_", "legibility": "high"}],
            "search_terms": ["classroom", "teaching"],
            "summary": "A presenter points at an easel.",
            "uncertainties": [],
        }
    )
    return CaptionReceiptV1.create(
        image_hash=f"sha256:{'1' * 64}",
        caption_result=caption,
        provider="codex-cli",
        model="gpt-5.6-luna",
        effort="low",
        prompt_version="caption-v1",
        created_at="2026-08-25T15:00:00Z",
    )


class FakeNotesApi:
    def __init__(self, annotation: str, *, race_annotation: str | None = None, post_annotation: str | None = None) -> None:
        self.annotation = annotation
        self.last_modified = 10
        self.race_annotation = race_annotation
        self.post_annotation = post_annotation
        self.reads = 0
        self.updates: list[str] = []

    async def get_item(self, item_id: str) -> dict:
        self.reads += 1
        if self.reads == 2 and self.race_annotation is not None:
            self.annotation = self.race_annotation
            self.last_modified += 1
        return {
            "id": item_id,
            "annotation": self.annotation,
            "lastModified": self.last_modified,
        }

    async def update_item(self, item_id: str, *, annotation: str) -> None:
        self.updates.append(annotation)
        self.annotation = self.post_annotation if self.post_annotation is not None else annotation
        self.last_modified += 1


class ManagedNotesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.caption = render_caption_block(receipt())
        self.generation = render_generation_block(
            prompt="A robot teaching a class",
            source="codex-imagegen",
            tags=("ai-generated", "illustration"),
            intent_id="123e4567-e89b-12d3-a456-426614174000",
        )

    def test_empty_and_human_notes_append_deterministically(self) -> None:
        empty = merge_managed_blocks("", caption_block=self.caption)
        self.assertEqual(empty.proposed, self.caption)

        human = "Café notes — keep exactly\n第二行\n"
        merged = merge_managed_blocks(human, caption_block=self.caption)
        self.assertTrue(merged.proposed.startswith(human))
        self.assertEqual(merged.proposed[: len(human)].encode(), human.encode())
        self.assertEqual(merged.changed_blocks, ("caption",))

    def test_existing_blocks_update_idempotently_without_touching_human_bytes(self) -> None:
        human_before = "Human before\n"
        human_after = "\nHuman after"
        original = human_before + self.caption + human_after
        newer = self.caption.replace("presenter", "teacher")

        first = merge_managed_blocks(original, caption_block=newer)
        second = merge_managed_blocks(first.proposed, caption_block=newer)

        self.assertEqual(first.proposed, human_before + newer + human_after)
        self.assertEqual(second.proposed, first.proposed)
        self.assertEqual(second.changed_blocks, ())

    def test_caption_and_generation_roles_update_independently(self) -> None:
        original = self.caption + "\n\n" + self.generation
        changed_generation = self.generation.replace("robot", "android")

        merged = merge_managed_blocks(original, generation_block=changed_generation)

        self.assertTrue(merged.proposed.startswith(self.caption))
        self.assertIn("android", merged.proposed)
        self.assertEqual(merged.changed_blocks, ("generation",))

    def test_duplicate_malformed_and_unsupported_markers_refuse(self) -> None:
        duplicate = self.caption + "\n" + self.caption
        malformed = "<!-- eagle-search:caption:start v=1 -->\nbroken"
        unsupported = self.caption.replace("v=1", "v=2", 1)

        for value in (duplicate, malformed, unsupported):
            with self.subTest(value=value[:40]):
                with self.assertRaises(ManagedNotesError):
                    merge_managed_blocks(value, caption_block=self.caption)

    def test_diff_contains_exact_hashes_and_changed_roles(self) -> None:
        diff = build_notes_diff("item-1", "Human", "Human\n\n" + self.caption, ("caption",))

        self.assertEqual(diff.eagle_id, "item-1")
        self.assertNotEqual(diff.original_hash, diff.proposed_hash)
        self.assertEqual(diff.changed_blocks, ("caption",))


class NotesApplyTests(unittest.IsolatedAsyncioTestCase):
    async def test_dry_run_never_mutates(self) -> None:
        api = FakeNotesApi("Human")
        with tempfile.TemporaryDirectory() as directory:
            applier = NotesApplier(api, FileLock(Path(directory) / "notes.lock"))
            result = await applier.sync(
                "item-1",
                caption_block=render_caption_block(receipt()),
                apply=False,
                quiescent=False,
            )

        self.assertEqual(result.status, "dry-run")
        self.assertEqual(api.updates, [])
        self.assertEqual(result.before_annotation, "Human")

    async def test_apply_refuses_when_notes_change_between_reads(self) -> None:
        api = FakeNotesApi("Human", race_annotation="Human edited concurrently")
        with tempfile.TemporaryDirectory() as directory:
            applier = NotesApplier(api, FileLock(Path(directory) / "notes.lock"))
            result = await applier.sync(
                "item-1",
                caption_block=render_caption_block(receipt()),
                apply=True,
                quiescent=True,
            )

        self.assertEqual(result.status, "refused")
        self.assertIn("changed between reads", result.refusal_reason)
        self.assertEqual(api.updates, [])

    async def test_apply_verifies_read_back_and_retains_snapshot(self) -> None:
        api = FakeNotesApi("Human bytes")
        with tempfile.TemporaryDirectory() as directory:
            applier = NotesApplier(api, FileLock(Path(directory) / "notes.lock"))
            result = await applier.sync(
                "item-1",
                caption_block=render_caption_block(receipt()),
                apply=True,
                quiescent=True,
            )

        self.assertEqual(result.status, "applied")
        self.assertEqual(result.before_annotation, "Human bytes")
        self.assertEqual(result.observed_annotation, result.proposed_annotation)
        self.assertEqual(len(api.updates), 1)

    async def test_post_write_mismatch_is_ambiguous_and_never_auto_rolls_back(self) -> None:
        api = FakeNotesApi("Human", post_annotation="Human changed after write")
        with tempfile.TemporaryDirectory() as directory:
            applier = NotesApplier(api, FileLock(Path(directory) / "notes.lock"))
            result = await applier.sync(
                "item-1",
                caption_block=render_caption_block(receipt()),
                apply=True,
                quiescent=True,
            )

        self.assertEqual(result.status, "ambiguous")
        self.assertEqual(result.observed_annotation, "Human changed after write")
        self.assertEqual(len(api.updates), 1)


if __name__ == "__main__":
    unittest.main()
