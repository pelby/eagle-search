"""Private blind-labelling contracts without provider or caption calls."""

from __future__ import annotations

import copy
import json
import unittest

from evals.labelling import (
    BlindLabellingError,
    aggregate_counts,
    annotation_guide_hash,
    build_provider_batch_schema,
    load_annotation_guide,
    load_batch_schema,
    prepare_blind_adjudication,
)


def image_hash(number: int) -> str:
    return f"sha256:{number:064x}"


IMAGE_HASHES = {"f-1": image_hash(1), "f-2": image_hash(2)}


def label(*, concepts, image_types, high, medium, queries, absent, uncertainty):
    return {
        "concept_alias_groups": concepts,
        "image_type_aliases": image_types,
        "ocr_truth": {"high": high, "medium": medium},
        "queries": queries,
        "critical_absent_terms": absent,
        "uncertainty": uncertainty,
    }


def batch(*, provider: str, model: str, batch_id: str, guide_hash: str):
    return {
        "batch_version": 1,
        "batch_id": batch_id,
        "provider": provider,
        "model": model,
        "caption_blind": True,
        "guide_hash": guide_hash,
        "labels": {
            "f-1": label(
                concepts=[["teacher", "instructor"], ["whiteboard", "board"]],
                image_types=["photo", "photograph"],
                high=["WELCOME"],
                medium=[],
                queries=["teacher at a board", "classroom welcome sign"],
                absent=["unicorn"],
                uncertainty="Room type is uncertain.",
            ),
            "f-2": label(
                concepts=[["flowchart"], ["arrow"]],
                image_types=["diagram"],
                high=["START"],
                medium=["NEXT"],
                queries=["flowchart arrows", "process diagram", "start next diagram"],
                absent=["medical diagnosis"],
                uncertainty="Small text may be incomplete.",
            ),
        },
    }


class BlindLabellingTests(unittest.TestCase):
    def test_provider_schema_const_fields_are_explicitly_typed(self) -> None:
        schema = load_batch_schema()

        self.assertEqual(schema["properties"]["batch_version"]["type"], "integer")
        self.assertEqual(schema["properties"]["caption_blind"]["type"], "boolean")

    def test_provider_schema_expands_a_closed_fixture_key_set(self) -> None:
        schema = build_provider_batch_schema(["f-2", "f-1"])
        labels = schema["properties"]["labels"]

        self.assertNotIn("propertyNames", labels)
        self.assertFalse(labels["additionalProperties"])
        self.assertEqual(labels["required"], ["f-1", "f-2"])
        self.assertEqual(set(labels["properties"]), {"f-1", "f-2"})

        with self.assertRaises(BlindLabellingError):
            build_provider_batch_schema(["f-1", "f-1"])

    def setUp(self) -> None:
        self.guide = load_annotation_guide()
        self.guide_hash = annotation_guide_hash(self.guide)
        self.openai = batch(provider="openai", model="gpt-5.6-sol", batch_id="batch-a", guide_hash=self.guide_hash)
        self.non_openai = batch(provider="anthropic", model="claude", batch_id="batch-b", guide_hash=self.guide_hash)

    def test_handoff_schema_is_strict_and_keys_labels_by_fixture_identifier(self) -> None:
        schema = load_batch_schema()
        labels = schema["properties"]["labels"]
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(labels["propertyNames"]["pattern"], "^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

    def test_provider_diverse_exact_blind_batches_make_deterministic_private_adjudication_artifacts(self) -> None:
        self.non_openai["labels"]["f-2"]["queries"] = ["diagram arrows", "workflow chart"]
        first = prepare_blind_adjudication(
            self.openai,
            self.non_openai,
            guide=self.guide,
            fixture_image_hashes=IMAGE_HASHES,
        )
        second = prepare_blind_adjudication(
            self.non_openai,
            self.openai,
            guide=self.guide,
            fixture_image_hashes=IMAGE_HASHES,
        )

        self.assertEqual(first, second)
        self.assertEqual(aggregate_counts(first), {"fixture_count": 2, "provider_count": 2, "disagreement_count": 1, "query_suggestion_count": 9})
        self.assertEqual(set(first["adjudication_skeleton"]["labels"]["labels"]), {"f-1", "f-2"})
        self.assertEqual(first["adjudication_skeleton"]["labels"]["labels"]["f-1"].keys(), {"concept_alias_groups", "ocr_text", "image_type_aliases", "critical_absent_terms"})
        self.assertNotIn(IMAGE_HASHES["f-1"], json.dumps(first))
        self.assertNotIn("caption_result", json.dumps(first))

    def test_repeated_visible_text_is_valid_ocr_evidence(self) -> None:
        """The same literal can appear twice in an image and must retain both occurrences."""

        self.openai["labels"]["f-1"]["ocr_truth"]["high"] = ["VIDEO", "VIDEO"]
        artifacts = prepare_blind_adjudication(
            self.openai,
            self.non_openai,
            guide=self.guide,
            fixture_image_hashes=IMAGE_HASHES,
        )

        self.assertEqual(artifacts["summary"]["fixture_count"], 2)

    def test_refuses_nonblind_nondiverse_or_incomplete_batches_before_constructing_artifacts(self) -> None:
        malformed = copy.deepcopy(self.non_openai)
        malformed["caption_blind"] = False
        with self.assertRaises(BlindLabellingError):
            prepare_blind_adjudication(self.openai, malformed, guide=self.guide, fixture_image_hashes=IMAGE_HASHES)

        same_provider = copy.deepcopy(self.non_openai)
        same_provider["provider"] = "OpenAI"
        with self.assertRaises(BlindLabellingError):
            prepare_blind_adjudication(self.openai, same_provider, guide=self.guide, fixture_image_hashes=IMAGE_HASHES)

        incomplete = copy.deepcopy(self.non_openai)
        del incomplete["labels"]["f-2"]
        with self.assertRaises(BlindLabellingError):
            prepare_blind_adjudication(self.openai, incomplete, guide=self.guide, fixture_image_hashes=IMAGE_HASHES)

        leaked = copy.deepcopy(self.non_openai)
        leaked["labels"]["f-1"]["caption_result"] = {"summary": "not permitted"}
        with self.assertRaises(BlindLabellingError):
            prepare_blind_adjudication(self.openai, leaked, guide=self.guide, fixture_image_hashes=IMAGE_HASHES)

        tampered_guide = copy.deepcopy(self.non_openai)
        tampered_guide["guide_hash"] = image_hash(9)
        with self.assertRaises(BlindLabellingError):
            prepare_blind_adjudication(self.openai, tampered_guide, guide=self.guide, fixture_image_hashes=IMAGE_HASHES)


if __name__ == "__main__":
    unittest.main()
