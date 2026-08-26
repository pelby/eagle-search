"""Private study scoring contracts; all fixtures are synthetic and identifier-only."""

from __future__ import annotations

import copy
import unittest

from evals.report import CandidateAggregate
from evals.study import (
    CandidateSpec,
    StudyValidationError,
    artifact_sha256,
    run_stage_b_study,
    validate_frozen_annotations,
)


GUIDE = {
    "annotation_guide_version": 1,
    "label_schema_version": 1,
    "concept_fields": ["diagram_types", "subjects", "visual_style", "colours", "layout", "search_terms"],
    "ocr_legibilities": ["high", "medium"],
    "type_metric": "normalised-alias",
    "critical_metric": "normalised-substring",
}

LABELS = {
    "labels_version": 1,
    "labels": {
        "t-1": {
            "concept_alias_groups": [["teacher"], ["board"]],
            "ocr_text": "WELCOME",
            "image_type_aliases": ["photo"],
            "critical_absent_terms": ["unicorn"],
        },
        "t-2": {
            "concept_alias_groups": [["diagram"], ["arrow"]],
            "ocr_text": "START",
            "image_type_aliases": ["diagram"],
            "critical_absent_terms": ["medical diagnosis"],
        },
    },
}

FIXTURES = [
    {"fixture_id": "t-1", "stage": "B", "role": "target"},
    {"fixture_id": "t-2", "stage": "B", "role": "target"},
    {"fixture_id": "d-1", "stage": "B", "role": "distractor"},
    {"fixture_id": "d-2", "stage": "B", "role": "distractor"},
]

QUERIES = [
    {"query_id": "q-1", "target_fixture_id": "t-1", "query": "teacher board", "grades": {"t-1": 3}, "known_query": True},
    {"query_id": "q-2", "target_fixture_id": "t-2", "query": "diagram arrow", "grades": {"t-2": 3}},
]

QUERY_ARTIFACT = {"query_artifact_version": 1, "queries": QUERIES}
EMBEDDING_CONTRACT = {
    "embedding_contract_version": 1,
    "model": "deterministic-test-embedder",
    "dimensions": 2,
    "selection_instrument_version": "selection-instrument-v1",
}


def caption(*, image_type: str, terms: list[str], text: str, subject: str = "") -> dict:
    return {
        "contract_version": 1,
        "image_type": image_type,
        "diagram_types": [term for term in terms if term in {"diagram", "arrow"}],
        "subjects": [subject] if subject else [term for term in terms if term == "teacher"],
        "visual_style": [],
        "colours": [],
        "layout": [],
        "visible_text": [{"text": text, "legibility": "high"}],
        "search_terms": terms,
        "summary": "synthetic caption",
        "uncertainties": [],
    }


def receipt(candidate: CandidateSpec, fixture_id: str, result: dict, *, manifest_hash: str = "sha256:" + "a" * 64) -> dict:
    return {
        "fixture_id": fixture_id,
        "model": candidate.model,
        "effort": candidate.effort,
        "prompt_version": candidate.prompt_version,
        "manifest_hash": manifest_hash,
        "caption_result": result,
    }


def deterministic_embedder(calls: list[str]):
    def embed(text: str) -> list[float]:
        calls.append(text)
        lowered = text.casefold()
        if "teacher" in lowered or "board" in lowered:
            return [1.0, 0.0]
        if "diagram" in lowered or "arrow" in lowered:
            return [0.0, 1.0]
        return [1.0, 1.0]
    return embed


class EvalStudyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.luna = CandidateSpec("gpt-5.6-luna", "low", "caption-v1")
        self.terra = CandidateSpec("gpt-5.6-terra", "low", "caption-v1")

    def _receipts(self) -> list[dict]:
        output: list[dict] = []
        good = {
            "t-1": caption(image_type="photo", terms=["teacher", "board"], text="WELCOME"),
            "t-2": caption(image_type="diagram", terms=["diagram", "arrow"], text="START"),
            "d-1": caption(image_type="photo", terms=["garden"], text=""),
            "d-2": caption(image_type="photo", terms=["city"], text=""),
        }
        weaker = {
            "t-1": caption(image_type="diagram", terms=["board"], text="WELCOM"),
            "t-2": caption(image_type="diagram", terms=["diagram"], text="START"),
            "d-1": caption(image_type="photo", terms=["teacher", "board"], text=""),
            "d-2": caption(image_type="photo", terms=["diagram", "arrow"], text=""),
        }
        for candidate, results in ((self.luna, good), (self.terra, weaker)):
            output.extend(receipt(candidate, fixture_id, result) for fixture_id, result in results.items())
        return output

    def test_hashes_and_identifier_only_adjudicated_labels_are_verified_before_scoring(self) -> None:
        annotations = validate_frozen_annotations(
            GUIDE,
            LABELS,
            expected_guide_hash=artifact_sha256(GUIDE),
            expected_labels_hash=artifact_sha256(LABELS),
        )
        self.assertEqual(set(annotations.labels), {"t-1", "t-2"})

        tampered = copy.deepcopy(LABELS)
        tampered["labels"]["t-1"]["ocr_text"] = "altered"
        with self.assertRaises(StudyValidationError):
            validate_frozen_annotations(
                GUIDE,
                tampered,
                expected_guide_hash=artifact_sha256(GUIDE),
                expected_labels_hash=artifact_sha256(LABELS),
            )

        leaked = copy.deepcopy(LABELS)
        leaked["labels"]["t-1"]["image_path"] = "/private/not-permitted.png"
        with self.assertRaises(StudyValidationError):
            validate_frozen_annotations(
                GUIDE,
                leaked,
                expected_guide_hash=artifact_sha256(GUIDE),
                expected_labels_hash=artifact_sha256(leaked),
            )

    def test_stage_b_scores_candidate_isolated_documents_and_emits_report_compatible_payload(self) -> None:
        calls: list[str] = []
        result = run_stage_b_study(
            guide=GUIDE,
            labels=LABELS,
            expected_guide_hash=artifact_sha256(GUIDE),
            expected_labels_hash=artifact_sha256(LABELS),
            fixtures=FIXTURES,
            receipts=self._receipts(),
            query_artifact=QUERY_ARTIFACT,
            expected_query_hash=artifact_sha256(QUERY_ARTIFACT),
            embedding_contract=EMBEDDING_CONTRACT,
            expected_embedding_contract_hash=artifact_sha256(EMBEDDING_CONTRACT),
            embed=deterministic_embedder(calls),
            candidates=[self.luna, self.terra],
            comparator=self.terra,
            manifest_hash="sha256:" + "a" * 64,
            seed=11,
            power_simulations=20,
        )

        luna = next(item for item in result["candidates"] if item["model"] == self.luna.model)
        self.assertEqual(luna["schema_valid_rate"], 1.0)
        self.assertEqual(luna["critical_hallucinations"], 0)
        self.assertEqual(luna["type_accuracy"], 1.0)
        self.assertEqual(luna["lanes"]["bm25"]["recall_at_10"], 1.0)
        self.assertEqual(luna["lanes"]["semantic"]["recall_at_10"], 1.0)
        self.assertEqual(luna["lanes"]["rrf"]["recall_at_10"], 1.0)
        self.assertEqual(result["semantic_rrf_status"]["state"], "scored")
        self.assertEqual(len(calls), 10)  # two frozen queries plus four candidate-owned documents per tier
        self.assertEqual(result["bootstrap_resamples"], 10_000)

        report_fields = {
            "model", "effort", "schema_valid_rate", "critical_hallucinations", "concept_precision", "concept_f1",
            "ocr_character_f1", "ndcg_at_10", "recall_at_10", "known_query_passed",
            "background_throughput_accepted", "lower_bounds", "lanes", "latencies_seconds",
        }
        aggregates = [CandidateAggregate(**{key: candidate[key] for key in report_fields}) for candidate in result["candidates"]]
        self.assertEqual(len(aggregates), 2)

    def test_invalid_schema_and_critical_terms_reduce_private_target_metrics(self) -> None:
        receipts = self._receipts()
        for item in receipts:
            if item["model"] == self.luna.model and item["fixture_id"] == "t-1":
                item["caption_result"] = caption(
                    image_type="photo", terms=["teacher", "board", "unicorn"], text="WELCOME"
                )
            if item["model"] == self.luna.model and item["fixture_id"] == "t-2":
                item["caption_result"] = {"not": "a caption contract"}
        result = run_stage_b_study(
            guide=GUIDE,
            labels=LABELS,
            expected_guide_hash=artifact_sha256(GUIDE),
            expected_labels_hash=artifact_sha256(LABELS),
            fixtures=FIXTURES,
            receipts=receipts,
            query_artifact=QUERY_ARTIFACT,
            expected_query_hash=artifact_sha256(QUERY_ARTIFACT),
            embedding_contract=EMBEDDING_CONTRACT,
            expected_embedding_contract_hash=artifact_sha256(EMBEDDING_CONTRACT),
            embed=deterministic_embedder([]),
            candidates=[self.luna, self.terra],
            comparator=self.terra,
            manifest_hash="sha256:" + "a" * 64,
            seed=3,
            power_simulations=20,
        )
        luna = next(item for item in result["candidates"] if item["model"] == self.luna.model)
        self.assertEqual(luna["schema_valid_rate"], 0.75)
        self.assertEqual(luna["critical_hallucinations"], 1)
        self.assertLess(luna["ocr_character_f1"], 1.0)

    def test_stage_b_rejects_cross_candidate_or_incomplete_receipt_coverage(self) -> None:
        receipts = self._receipts()
        receipts.pop()
        with self.assertRaises(StudyValidationError):
            run_stage_b_study(
                guide=GUIDE,
                labels=LABELS,
                expected_guide_hash=artifact_sha256(GUIDE),
                expected_labels_hash=artifact_sha256(LABELS),
                fixtures=FIXTURES,
                receipts=receipts,
                query_artifact=QUERY_ARTIFACT,
                expected_query_hash=artifact_sha256(QUERY_ARTIFACT),
                embedding_contract=EMBEDDING_CONTRACT,
                expected_embedding_contract_hash=artifact_sha256(EMBEDDING_CONTRACT),
                embed=deterministic_embedder([]),
                candidates=[self.luna, self.terra],
                comparator=self.terra,
                manifest_hash="sha256:" + "a" * 64,
                seed=3,
                power_simulations=20,
            )

    def test_missing_or_tampered_embedding_evidence_refuses_before_scoring(self) -> None:
        common = {
            "guide": GUIDE,
            "labels": LABELS,
            "expected_guide_hash": artifact_sha256(GUIDE),
            "expected_labels_hash": artifact_sha256(LABELS),
            "fixtures": FIXTURES,
            "receipts": self._receipts(),
            "query_artifact": QUERY_ARTIFACT,
            "expected_query_hash": artifact_sha256(QUERY_ARTIFACT),
            "candidates": [self.luna, self.terra],
            "comparator": self.terra,
            "manifest_hash": "sha256:" + "a" * 64,
            "seed": 3,
            "power_simulations": 20,
        }
        with self.assertRaises(StudyValidationError):
            run_stage_b_study(**common)
        tampered = {**EMBEDDING_CONTRACT, "dimensions": 3}
        with self.assertRaises(StudyValidationError):
            run_stage_b_study(
                **common,
                embedding_contract=tampered,
                expected_embedding_contract_hash=artifact_sha256(EMBEDDING_CONTRACT),
                embed=deterministic_embedder([]),
            )
        with self.assertRaises(StudyValidationError):
            run_stage_b_study(
                **common,
                embedding_contract=EMBEDDING_CONTRACT,
                expected_embedding_contract_hash=artifact_sha256(EMBEDDING_CONTRACT),
                embed=lambda _text: [1.0],
            )
        tampered_queries = copy.deepcopy(QUERY_ARTIFACT)
        tampered_queries["queries"][0]["query"] = "changed frozen query"
        with self.assertRaises(StudyValidationError):
            run_stage_b_study(
                **{**common, "query_artifact": tampered_queries},
                embedding_contract=EMBEDDING_CONTRACT,
                expected_embedding_contract_hash=artifact_sha256(EMBEDDING_CONTRACT),
                embed=deterministic_embedder([]),
            )
        with self.assertRaises(StudyValidationError):
            run_stage_b_study(
                **common,
                embedding_contract=EMBEDDING_CONTRACT,
                expected_embedding_contract_hash=artifact_sha256(EMBEDDING_CONTRACT),
                embed=lambda _text: [float("nan"), 0.0],
            )


if __name__ == "__main__":
    unittest.main()
