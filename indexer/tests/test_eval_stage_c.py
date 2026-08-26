"""Hidden Stage C contracts use synthetic identifiers and never open images."""

from __future__ import annotations

import copy
import unittest

from evals.fixture_builder import build_private_manifest
from evals.report import FROZEN_EVALUATION_THRESHOLDS, FROZEN_FINAL_SELECTION_THRESHOLDS
from evals.stage_c import StageCValidationError, _stage_c_corpus, build_stage_c_effects, run_stage_c_study
from evals.study import CandidateSpec, artifact_sha256
from src.captioning.codex_cli import CAPTION_INPUT_PREPARATION_VERSION
from src.captioning.prompt import CAPTION_PROMPT_VERSION


GUIDE = {
    "annotation_guide_version": 2,
    "label_schema_version": 1,
    "concept_fields": ["subjects", "search_terms"],
    "ocr_legibilities": ["high", "medium"],
    "type_metric": "normalised-alias",
    "critical_metric": "token-phrase",
    "concept_metric": "token-aware-alias-recall",
    "semantic_query_policy": "separate-frozen-query-artifact",
}
EMBEDDING_CONTRACT = {
    "embedding_contract_version": 1,
    "model": "deterministic-test-embedder",
    "dimensions": 2,
    "selection_instrument_version": "selection-instrument-v1",
}


def _caption(*, terms: list[str]) -> dict:
    return {
        "contract_version": 1,
        "image_type": "photo",
        "diagram_types": [],
        "subjects": [terms[0]],
        "visual_style": [],
        "colours": [],
        "layout": [],
        "visible_text": [],
        "search_terms": terms,
        "summary": "synthetic caption",
        "uncertainties": [],
    }


class StageCHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        candidates = [
            {
                "image_hash": f"sha256:{index:064x}",
                "image_path": f"/private/synthetic-{index}.png",
                "stratum": f"stratum-{index % 5}",
            }
            for index in range(400)
        ]
        cls.snapshot = build_private_manifest(
            candidates,
            snapshot_id="synthetic-stage-c",
            seed=71,
            sealing_inputs={
                "labels_hash": "sha256:" + "1" * 64,
                "gates_hash": "sha256:" + "2" * 64,
                "amendment_hash": "sha256:" + "3" * 64,
                "instrument_version": "selection-instrument-v1",
                "target_count": 60,
            },
        )
        cls.luna = CandidateSpec("gpt-5.6-luna", "low", "caption-v3")
        cls.sol = CandidateSpec("gpt-5.6-sol", "low", "caption-v3")
        cls.effects = build_stage_c_effects(
            {
                "study_version": 1,
                "candidates": [
                    {
                        "model": spec.model,
                        "effort": spec.effort,
                        "prompt_version": spec.prompt_version,
                        "powered_hidden_target_count": 60,
                    }
                    for spec in (cls.luna, cls.sol)
                ],
            },
            stage_c_candidates=[
                "gpt-5.6-luna:low:caption-v3",
                "gpt-5.6-sol:low:caption-v3",
            ],
        )
        cls.gates = {
            "stage_c_gates_version": 2,
            "snapshot_hash": artifact_sha256(cls.snapshot),
            "stage_b_effects_hash": artifact_sha256(cls.effects),
            "guide_hash": artifact_sha256(GUIDE),
            "labels_hash": "pending",
            "query_artifact_hash": "pending",
            "embedding_contract_hash": artifact_sha256(EMBEDDING_CONTRACT),
            "candidates": ["gpt-5.6-luna:low:caption-v3", "gpt-5.6-sol:low:caption-v3"],
            "comparator": "gpt-5.6-sol:low:caption-v3",
            "selection_algorithm": "stratified-shuffle-v1",
            "selection_seed": 991,
            "seed": 992,
            "prompt_version": CAPTION_PROMPT_VERSION,
            "input_preparation_version": CAPTION_INPUT_PREPARATION_VERSION,
            "thresholds": FROZEN_EVALUATION_THRESHOLDS,
            "final_selection_thresholds": FROZEN_FINAL_SELECTION_THRESHOLDS,
        }
        corpus, selected_ids = _stage_c_corpus(
            snapshot=cls.snapshot,
            gates=cls.gates,
            stage_b_effects=cls.effects,
        )
        cls.corpus = corpus
        cls.selected_ids = selected_ids
        cls.labels = {
            "labels_version": 1,
            "labels": {
                fixture_id: {
                    "concept_alias_groups": [["teacher"]],
                    "ocr_text": "",
                    "image_type_aliases": ["photo"],
                    "critical_absent_terms": ["unicorn"],
                }
                for fixture_id in sorted(selected_ids)
            },
        }
        cls.queries = {
            "query_artifact_version": 1,
            "queries": [
                {
                    "query_id": f"q-{fixture_id}",
                    "target_fixture_id": fixture_id,
                    "query": f"teacher {fixture_id}",
                    "grades": {fixture_id: 3},
                    "known_query": index == 0,
                }
                for index, fixture_id in enumerate(sorted(selected_ids))
            ],
        }
        cls.gates["labels_hash"] = artifact_sha256(cls.labels)
        cls.gates["query_artifact_hash"] = artifact_sha256(cls.queries)
        cls.approval = {
            "approval_version": 1,
            "approved": True,
            "gates_hash": artifact_sha256(cls.gates),
        }
        cls.receipts = []
        corpus_ids = {fixture["fixture_id"] for fixture in corpus}
        for spec in (cls.luna, cls.sol):
            for fixture in corpus:
                fixture_id = fixture["fixture_id"]
                terms = ["teacher", fixture_id] if fixture_id in selected_ids else ["garden", fixture_id]
                cls.receipts.append(
                    {
                        "fixture_id": fixture_id,
                        "model": spec.model,
                        "effort": spec.effort,
                        "prompt_version": spec.prompt_version,
                        "manifest_hash": artifact_sha256(cls.snapshot),
                        "caption_result": _caption(terms=terms),
                        "latency_seconds": 1.0,
                    }
                )
        stage_b_id = next(
            fixture["fixture_id"]
            for fixture in cls.snapshot["fixtures"]
            if fixture["stage"] == "B"
        )
        assert stage_b_id not in corpus_ids
        cls.receipts.append(
            {
                "fixture_id": stage_b_id,
                "model": cls.luna.model,
                "effort": cls.luna.effort,
                "prompt_version": cls.luna.prompt_version,
                "manifest_hash": artifact_sha256(cls.snapshot),
                "caption_result": {"invalid": "must be filtered before scoring"},
            }
        )

    def _arguments(self) -> dict:
        return {
            "snapshot": self.snapshot,
            "gates": self.gates,
            "approval": self.approval,
            "stage_b_effects": self.effects,
            "guide": GUIDE,
            "labels": self.labels,
            "queries": self.queries,
            "embedding_contract": EMBEDDING_CONTRACT,
            "receipts": self.receipts,
            "embed": lambda _text: [1.0, 1.0],
            "power_simulations": 20,
        }

    def test_scores_only_deterministically_selected_hidden_corpus(self) -> None:
        result = run_stage_c_study(**self._arguments())
        self.assertEqual(result["stage"], "C")
        self.assertEqual(result["selected_target_count"], 60)
        self.assertEqual(result["completed"], len(self.corpus) * 2)
        self.assertEqual(result["stage_c_gates_version"], 2)
        self.assertEqual(result["final_selection_thresholds"], FROZEN_FINAL_SELECTION_THRESHOLDS)
        self.assertEqual(result["approval_hash"], artifact_sha256(self.approval))
        selected_again = _stage_c_corpus(
            snapshot=self.snapshot,
            gates=self.gates,
            stage_b_effects=self.effects,
        )[1]
        self.assertEqual(selected_again, self.selected_ids)

    def test_refuses_tampered_approval_or_stage_b_effects(self) -> None:
        arguments = self._arguments()
        tampered_approval = {**self.approval, "approved": False}
        with self.assertRaises(StageCValidationError):
            run_stage_c_study(**{**arguments, "approval": tampered_approval})
        tampered_effects = {**self.effects, "powered_hidden_target_count": 61}
        with self.assertRaises(StageCValidationError):
            run_stage_c_study(**{**arguments, "stage_b_effects": tampered_effects})

    def test_eliminated_infeasible_candidate_does_not_block_powered_finalists(self) -> None:
        results = {
            "study_version": 1,
            "candidates": [
                {"model": "gpt-5.6-luna", "effort": "low", "prompt_version": "caption-v3", "powered_hidden_target_count": None},
                {"model": "gpt-5.6-terra", "effort": "low", "prompt_version": "caption-v3", "powered_hidden_target_count": 60},
                {"model": "gpt-5.6-sol", "effort": "low", "prompt_version": "caption-v3", "powered_hidden_target_count": 72},
            ],
        }

        effects = build_stage_c_effects(
            results,
            stage_c_candidates=[
                "gpt-5.6-terra:low:caption-v3",
                "gpt-5.6-sol:low:caption-v3",
            ],
        )

        self.assertEqual(effects["powered_hidden_target_count"], 72)
        self.assertNotIn("gpt-5.6-luna:low:caption-v3", effects["candidate_powered_target_counts"])

    def test_refuses_missing_or_tampered_final_selection_thresholds(self) -> None:
        arguments = self._arguments()
        missing = {key: value for key, value in self.gates.items() if key != "final_selection_thresholds"}
        missing_approval = {**self.approval, "gates_hash": artifact_sha256(missing)}
        with self.assertRaises(StageCValidationError):
            run_stage_c_study(**{**arguments, "gates": missing, "approval": missing_approval})

        tampered = copy.deepcopy(self.gates)
        tampered["final_selection_thresholds"]["absolute"]["concept_recall"] = 0.79
        tampered_approval = {**self.approval, "gates_hash": artifact_sha256(tampered)}
        with self.assertRaises(StageCValidationError):
            run_stage_c_study(**{**arguments, "gates": tampered, "approval": tampered_approval})

    def test_refuses_duplicate_snapshot_ids_and_unselected_label_leakage(self) -> None:
        arguments = self._arguments()
        duplicate = copy.deepcopy(self.snapshot)
        duplicate["fixtures"][1]["fixture_id"] = duplicate["fixtures"][0]["fixture_id"]
        duplicate_gates = {**self.gates, "snapshot_hash": artifact_sha256(duplicate)}
        duplicate_approval = {**self.approval, "gates_hash": artifact_sha256(duplicate_gates)}
        with self.assertRaises(StageCValidationError):
            run_stage_c_study(
                **{
                    **arguments,
                    "snapshot": duplicate,
                    "gates": duplicate_gates,
                    "approval": duplicate_approval,
                }
            )

        leaked_labels = copy.deepcopy(self.labels)
        unselected = next(
            fixture["fixture_id"]
            for fixture in self.snapshot["fixtures"]
            if fixture["stage"] == "C"
            and fixture["role"] == "target"
            and fixture["fixture_id"] not in self.selected_ids
        )
        leaked_labels["labels"][unselected] = copy.deepcopy(next(iter(self.labels["labels"].values())))
        leaked_gates = {**self.gates, "labels_hash": artifact_sha256(leaked_labels)}
        leaked_approval = {**self.approval, "gates_hash": artifact_sha256(leaked_gates)}
        with self.assertRaises(StageCValidationError):
            run_stage_c_study(
                **{
                    **arguments,
                    "labels": leaked_labels,
                    "gates": leaked_gates,
                    "approval": leaked_approval,
                }
            )


if __name__ == "__main__":
    unittest.main()
