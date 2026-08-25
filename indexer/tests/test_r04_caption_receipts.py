from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.contracts import CaptionReceiptV1, CaptionResultV1
from src.persistence.receipts import FileReceiptStore


CAPTION = {
    "contract_version": 1, "image_type": "diagram", "diagram_types": ["flowchart"],
    "subjects": ["classroom"], "visual_style": ["flat"], "colours": ["blue"],
    "layout": ["left to right"], "visible_text": [], "search_terms": ["teaching"],
    "summary": "A classroom flowchart.", "uncertainties": [],
}


def receipt(*, model: str = "luna", prompt: str = "v1") -> CaptionReceiptV1:
    return CaptionReceiptV1.create(
        image_hash="sha256:" + "a" * 64,
        caption_result=CaptionResultV1.from_dict(CAPTION), provider="test", model=model,
        effort="low", prompt_version=prompt, created_at="2026-08-25T00:00:00Z",
    )


class ReceiptStoreTests(unittest.TestCase):
    def test_receipts_are_immutable_and_active_pointer_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = FileReceiptStore(Path(directory))
            first = receipt()
            second = receipt(model="terra", prompt="v2")
            self.assertEqual(store.put_immutable(first), first.receipt_id)
            self.assertEqual(store.put_immutable(first), first.receipt_id)
            store.put_immutable(second)
            store.set_active(first.image_hash, first.receipt_id, "accepted")
            self.assertEqual(store.resolve_active(first.image_hash).receipt_id, first.receipt_id)
            (Path(directory) / first.image_hash / "active.json").unlink()
            self.assertEqual(store.resolve_active(first.image_hash).receipt_id, second.receipt_id)
