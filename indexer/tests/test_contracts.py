"""G0 contract tests for Eagle Search v2."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.contracts import CaptionReceiptV1, CaptionResultV1, ContractError


ROOT = Path(__file__).resolve().parents[1]


VALID_CAPTION = {
    "contract_version": 1,
    "image_type": "illustration",
    "diagram_types": [],
    "subjects": ["robot presenter", "presentation easel"],
    "visual_style": ["3D illustration"],
    "colours": ["navy", "coral", "lime"],
    "layout": ["subject on right", "copy space on left"],
    "visible_text": [{"text": ">_", "legibility": "high"}],
    "search_terms": ["teaching", "classroom", "presentation"],
    "summary": "A robot presenter points at an easel in a dark studio.",
    "uncertainties": [],
}


class CaptionResultContractTests(unittest.TestCase):
    def test_round_trip_and_deterministic_search_text(self) -> None:
        result = CaptionResultV1.from_dict(VALID_CAPTION)

        self.assertEqual(result.to_dict(), VALID_CAPTION)
        self.assertEqual(
            result.search_text(),
            "illustration robot presenter presentation easel 3D illustration "
            "navy coral lime subject on right copy space on left >_ teaching "
            "classroom presentation A robot presenter points at an easel in a dark studio.",
        )

    def test_extra_keys_are_rejected(self) -> None:
        payload = {**VALID_CAPTION, "filename": "secret.png"}

        with self.assertRaises(ContractError):
            CaptionResultV1.from_dict(payload)

    def test_unbounded_arrays_are_rejected(self) -> None:
        payload = {**VALID_CAPTION, "search_terms": [f"term-{i}" for i in range(13)]}

        with self.assertRaises(ContractError):
            CaptionResultV1.from_dict(payload)

    def test_visible_text_legibility_is_strict(self) -> None:
        payload = {
            **VALID_CAPTION,
            "visible_text": [{"text": "hello", "legibility": "probably"}],
        }

        with self.assertRaises(ContractError):
            CaptionResultV1.from_dict(payload)


class CaptionReceiptContractTests(unittest.TestCase):
    def _receipt(self, *, created_at: str, model: str = "gpt-5.6-luna") -> CaptionReceiptV1:
        return CaptionReceiptV1.create(
            image_hash=f"sha256:{'a' * 64}",
            caption_result=CaptionResultV1.from_dict(VALID_CAPTION),
            provider="codex-cli",
            model=model,
            effort="low",
            prompt_version="caption-v1",
            created_at=created_at,
        )

    def test_retry_timestamp_does_not_change_receipt_id(self) -> None:
        first = self._receipt(created_at="2026-08-25T20:00:00Z")
        retry = self._receipt(created_at="2026-08-25T20:01:00Z")

        self.assertEqual(first.receipt_id, retry.receipt_id)

    def test_model_change_creates_a_new_immutable_receipt(self) -> None:
        luna = self._receipt(created_at="2026-08-25T20:00:00Z")
        terra = self._receipt(
            created_at="2026-08-25T20:00:00Z",
            model="gpt-5.6-terra",
        )

        self.assertNotEqual(luna.receipt_id, terra.receipt_id)

    def test_receipt_round_trip_verifies_checksum(self) -> None:
        receipt = self._receipt(created_at="2026-08-25T20:00:00Z")

        loaded = CaptionReceiptV1.from_dict(receipt.to_dict())

        self.assertEqual(loaded.receipt_id, receipt.receipt_id)

    def test_tampered_receipt_is_rejected(self) -> None:
        receipt = self._receipt(created_at="2026-08-25T20:00:00Z")
        payload = receipt.to_dict()
        payload["model"] = "gpt-5.6-sol"

        with self.assertRaises(ContractError):
            CaptionReceiptV1.from_dict(payload)

    def test_invalid_image_hash_is_rejected(self) -> None:
        with self.assertRaises(ContractError):
            CaptionReceiptV1.create(
                image_hash="sha256:not-a-digest",
                caption_result=CaptionResultV1.from_dict(VALID_CAPTION),
                provider="codex-cli",
                model="gpt-5.6-luna",
                effort="low",
                prompt_version="caption-v1",
                created_at="2026-08-25T20:00:00Z",
            )


class JsonSchemaContractTests(unittest.TestCase):
    def test_caption_output_schema_has_explicit_types_for_constrained_scalars(self) -> None:
        schema_path = ROOT / "schemas" / "caption-result-v1.schema.json"
        with schema_path.open(encoding="utf-8") as schema_file:
            schema = json.load(schema_file)

        self.assertEqual(schema["properties"]["contract_version"]["type"], "integer")
        visible_properties = schema["properties"]["visible_text"]["items"]["properties"]
        self.assertEqual(visible_properties["legibility"]["type"], "string")


if __name__ == "__main__":
    unittest.main()
