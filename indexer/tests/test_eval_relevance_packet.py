"""Blind pooled-relevance packets use only synthetic pixels and opaque IDs."""

from __future__ import annotations

import copy
import hashlib
import json
import stat
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from evals.fixture_builder import build_private_manifest
from evals.relevance_packet import RelevancePacketError, build_relevance_packet
from evals.report import FROZEN_EVALUATION_THRESHOLDS, FROZEN_FINAL_SELECTION_THRESHOLDS
from evals.stage_c import _stage_c_corpus
from evals.study import CandidateSpec, artifact_sha256, candidate_set_hash
from src.captioning.codex_cli import CAPTION_INPUT_PREPARATION_VERSION
from src.captioning.prompt import CAPTION_PROMPT_VERSION


class RelevancePacketTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        candidates = []
        for index in range(400):
            image = self.root / f"synthetic-{index:03d}.png"
            Image.new("RGB", (16, 12), (index % 256, index // 256, 64)).save(image)
            image_hash = "sha256:" + hashlib.sha256(image.read_bytes()).hexdigest()
            candidates.append(
                {
                    "image_hash": image_hash,
                    "image_path": str(image),
                    "stratum": f"stratum-{index % 5}",
                }
            )
        self.manifest = build_private_manifest(
            candidates,
            snapshot_id="synthetic-relevance-packet",
            seed=71,
            sealing_inputs={
                "labels_hash": "sha256:" + "1" * 64,
                "gates_hash": "sha256:" + "2" * 64,
                "amendment_hash": "sha256:" + "3" * 64,
                "instrument_version": "selection-instrument-v1",
                "target_count": 60,
            },
        )
        stage_b = [fixture for fixture in self.manifest["fixtures"] if fixture["stage"] == "B"]
        target = next(fixture for fixture in stage_b if fixture["role"] == "target")
        distractors = [fixture for fixture in stage_b if fixture["role"] == "distractor"][:2]
        self.corpus_ids = {fixture["fixture_id"] for fixture in stage_b}
        self.query = {
            "query_artifact_version": 1,
            "queries": [
                {
                    "query_id": "q-classroom",
                    "target_fixture_id": target["fixture_id"],
                    "query": "a teacher presenting in a classroom",
                    "grades": {target["fixture_id"]: 3},
                    "known_query": True,
                }
            ],
        }
        self.candidate_set_hash = "sha256:" + "4" * 64
        self.corpus = {
            "relevance_corpus_version": 1,
            "manifest_hash": artifact_sha256(self.manifest),
            "stage": "B",
            "fixture_ids": sorted(self.corpus_ids),
        }
        self.blind_pool = {
            "blind_pool_version": 1,
            "manifest_hash": artifact_sha256(self.manifest),
            "base_query_artifact_hash": artifact_sha256(self.query),
            "candidate_set_hash": self.candidate_set_hash,
            "ranking_lane": "rrf",
            "query_items": {
                "q-classroom": sorted(fixture["fixture_id"] for fixture in distractors),
            },
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _build(self, packet_name: str, **overrides):
        arguments = {
            "manifest": self.manifest,
            "expected_manifest_hash": artifact_sha256(self.manifest),
            "blind_pool": self.blind_pool,
            "expected_blind_pool_hash": artifact_sha256(self.blind_pool),
            "query_artifact": self.query,
            "expected_query_artifact_hash": artifact_sha256(self.query),
            "expected_candidate_set_hash": self.candidate_set_hash,
            "corpus_artifact": self.corpus,
            "expected_corpus_artifact_hash": artifact_sha256(self.corpus),
            "private_root": self.root / "private",
            "allowed_root": self.root,
            "packet_name": packet_name,
        }
        return build_relevance_packet(**{**arguments, **overrides})

    def test_builds_deterministic_owner_only_blind_packet(self) -> None:
        first = self._build("packet-a")
        second = self._build("packet-b")
        first_root = Path(first["packet_directory"])
        second_root = Path(second["packet_directory"])

        self.assertEqual(first["packet_hash"], second["packet_hash"])
        self.assertEqual(
            (first_root / "contact-001.png").read_bytes(),
            (second_root / "contact-001.png").read_bytes(),
        )
        self.assertEqual(stat.S_IMODE((self.root / "private").stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(first_root.stat().st_mode), 0o700)
        for path in first_root.iterdir():
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

        packet = json.loads((first_root / "packet.json").read_text(encoding="utf-8"))
        schema = json.loads((first_root / "grading-schema.json").read_text(encoding="utf-8"))
        expected_ids = sorted(
            set(self.blind_pool["query_items"]["q-classroom"])
            | set(self.query["queries"][0]["grades"])
        )
        self.assertEqual(packet["queries"][0]["fixture_ids"], expected_ids)
        self.assertNotIn("target_fixture_id", json.dumps(packet))
        self.assertNotIn("ranking_lane", json.dumps(packet))
        self.assertNotIn('"grades": {', json.dumps(packet))
        self.assertEqual(packet["corpus_artifact_hash"], artifact_sha256(self.corpus))
        self.assertEqual(packet["relevance_packet_version"], 2)
        self.assertEqual(first["packet_evidence_hash"], packet["packet_evidence_hash"])
        self.assertEqual(
            schema["properties"]["packet_evidence_hash"]["const"],
            first["packet_evidence_hash"],
        )
        self.assertEqual(schema["properties"]["pooled_relevance_version"]["const"], 2)
        grade_shape = schema["properties"]["query_grades"]["properties"]["q-classroom"]
        self.assertEqual(grade_shape["required"], expected_ids)
        self.assertFalse(grade_shape["additionalProperties"])

    def test_refuses_tampered_missing_or_misbound_pool_before_writing(self) -> None:
        tampered = copy.deepcopy(self.blind_pool)
        tampered["query_items"]["q-classroom"].append("not-in-corpus")
        with self.assertRaisesRegex(RelevancePacketError, "SHA-256"):
            self._build("tampered", blind_pool=tampered)

        missing = copy.deepcopy(self.blind_pool)
        missing["query_items"] = {}
        with self.assertRaisesRegex(RelevancePacketError, "cover every frozen query"):
            self._build(
                "missing",
                blind_pool=missing,
                expected_blind_pool_hash=artifact_sha256(missing),
            )

        misbound = copy.deepcopy(self.blind_pool)
        misbound["base_query_artifact_hash"] = "sha256:" + "9" * 64
        with self.assertRaisesRegex(RelevancePacketError, "another query artifact"):
            self._build(
                "misbound",
                blind_pool=misbound,
                expected_blind_pool_hash=artifact_sha256(misbound),
            )

        self.assertFalse((self.root / "private" / "tampered").exists())
        self.assertFalse((self.root / "private" / "missing").exists())
        self.assertFalse((self.root / "private" / "misbound").exists())

    def test_refuses_recomputed_pool_that_leaks_hidden_stage_c(self) -> None:
        stage_c_id = next(
            fixture["fixture_id"]
            for fixture in self.manifest["fixtures"]
            if fixture["stage"] == "C" and fixture["role"] == "target"
        )
        leaked = copy.deepcopy(self.blind_pool)
        leaked["query_items"]["q-classroom"] = [stage_c_id]
        with self.assertRaisesRegex(RelevancePacketError, "corpus fixture IDs"):
            self._build(
                "hidden-c-leak",
                blind_pool=leaked,
                expected_blind_pool_hash=artifact_sha256(leaked),
            )
        self.assertFalse((self.root / "private" / "hidden-c-leak").exists())

    def test_stage_c_requires_hash_bound_selected_corpus_not_all_manifest_ids(self) -> None:
        candidate_identity = "gpt-5.6-sol:low:caption-v3"
        effects = {
            "stage_b_effects_version": 2,
            "stage_b_results_hash": "sha256:" + "a" * 64,
            "stage_c_candidates": [candidate_identity],
            "candidate_powered_target_counts": {candidate_identity: 60},
            "powered_hidden_target_count": 60,
        }
        stage_c_candidate_hash = candidate_set_hash([
            CandidateSpec("gpt-5.6-sol", "low", "caption-v3")
        ])
        gates = {
            "stage_c_gates_version": 2,
            "snapshot_hash": artifact_sha256(self.manifest),
            "stage_b_effects_hash": artifact_sha256(effects),
            "guide_hash": "sha256:" + "5" * 64,
            "labels_hash": "sha256:" + "6" * 64,
            "query_artifact_hash": "pending",
            "embedding_contract_hash": "sha256:" + "8" * 64,
            "candidates": [candidate_identity],
            "comparator": candidate_identity,
            "selection_algorithm": "stratified-shuffle-v1",
            "selection_seed": 991,
            "seed": 992,
            "prompt_version": CAPTION_PROMPT_VERSION,
            "input_preparation_version": CAPTION_INPUT_PREPARATION_VERSION,
            "thresholds": FROZEN_EVALUATION_THRESHOLDS,
            "final_selection_thresholds": FROZEN_FINAL_SELECTION_THRESHOLDS,
        }
        selected_corpus, selected_targets = _stage_c_corpus(
            snapshot=self.manifest,
            gates=gates,
            stage_b_effects=effects,
        )
        allowed = sorted(fixture["fixture_id"] for fixture in selected_corpus)
        c_targets = [fixture for fixture in selected_corpus if fixture["fixture_id"] in selected_targets]
        c_distractors = [fixture for fixture in selected_corpus if fixture["role"] == "distractor"]
        corpus = {
            "relevance_corpus_version": 1,
            "manifest_hash": artifact_sha256(self.manifest),
            "stage": "C",
            "fixture_ids": allowed,
        }
        query = {
            "query_artifact_version": 1,
            "queries": [{
                "query_id": "q-hidden",
                "target_fixture_id": c_targets[0]["fixture_id"],
                "query": "synthetic hidden query",
                "grades": {c_targets[0]["fixture_id"]: 3},
                "known_query": True,
            }],
        }
        gates["query_artifact_hash"] = artifact_sha256(query)
        approval = {
            "approval_version": 1,
            "approved": True,
            "gates_hash": artifact_sha256(gates),
        }
        pool = {
            "blind_pool_version": 1,
            "manifest_hash": artifact_sha256(self.manifest),
            "base_query_artifact_hash": artifact_sha256(query),
            "candidate_set_hash": stage_c_candidate_hash,
            "ranking_lane": "rrf",
            "query_items": {"q-hidden": [c_distractors[0]["fixture_id"]]},
        }
        result = self._build(
            "stage-c",
            corpus_artifact=corpus,
            expected_corpus_artifact_hash=artifact_sha256(corpus),
            query_artifact=query,
            expected_query_artifact_hash=artifact_sha256(query),
            blind_pool=pool,
            expected_blind_pool_hash=artifact_sha256(pool),
            expected_candidate_set_hash=stage_c_candidate_hash,
            stage_c_gates=gates,
            expected_stage_c_gates_hash=artifact_sha256(gates),
            stage_b_effects=effects,
            expected_stage_b_effects_hash=artifact_sha256(effects),
            stage_c_approval=approval,
            expected_stage_c_approval_hash=artifact_sha256(approval),
        )
        self.assertTrue((Path(result["packet_directory"]) / "contact-001.png").is_file())

        swapped = copy.deepcopy(corpus)
        removed = c_targets[0]["fixture_id"]
        replacement = next(
            fixture["fixture_id"]
            for fixture in self.manifest["fixtures"]
            if fixture["stage"] == "C"
            and fixture["role"] == "target"
            and fixture["fixture_id"] not in set(allowed)
        )
        swapped["fixture_ids"] = sorted(
            replacement if fixture_id == removed else fixture_id
            for fixture_id in swapped["fixture_ids"]
        )
        with self.assertRaisesRegex(RelevancePacketError, "exact sealed Stage C selection"):
            self._build(
                "stage-c-swapped",
                corpus_artifact=swapped,
                expected_corpus_artifact_hash=artifact_sha256(swapped),
                query_artifact=query,
                expected_query_artifact_hash=artifact_sha256(query),
                blind_pool=pool,
                expected_blind_pool_hash=artifact_sha256(pool),
                expected_candidate_set_hash=stage_c_candidate_hash,
                stage_c_gates=gates,
                expected_stage_c_gates_hash=artifact_sha256(gates),
                stage_b_effects=effects,
                expected_stage_b_effects_hash=artifact_sha256(effects),
                stage_c_approval=approval,
                expected_stage_c_approval_hash=artifact_sha256(approval),
            )

        changed_query = copy.deepcopy(query)
        changed_query["queries"][0]["query"] = "a recomputed but ungated query"
        changed_pool = copy.deepcopy(pool)
        changed_pool["base_query_artifact_hash"] = artifact_sha256(changed_query)
        with self.assertRaisesRegex(RelevancePacketError, "query artifact"):
            self._build(
                "stage-c-query-swap",
                corpus_artifact=corpus,
                expected_corpus_artifact_hash=artifact_sha256(corpus),
                query_artifact=changed_query,
                expected_query_artifact_hash=artifact_sha256(changed_query),
                blind_pool=changed_pool,
                expected_blind_pool_hash=artifact_sha256(changed_pool),
                expected_candidate_set_hash=stage_c_candidate_hash,
                stage_c_gates=gates,
                expected_stage_c_gates_hash=artifact_sha256(gates),
                stage_b_effects=effects,
                expected_stage_b_effects_hash=artifact_sha256(effects),
                stage_c_approval=approval,
                expected_stage_c_approval_hash=artifact_sha256(approval),
            )

        changed_candidate_hash = candidate_set_hash([
            CandidateSpec("gpt-5.6-terra", "low", "caption-v3")
        ])
        changed_candidate_pool = copy.deepcopy(pool)
        changed_candidate_pool["candidate_set_hash"] = changed_candidate_hash
        with self.assertRaisesRegex(RelevancePacketError, "sealed Stage C candidates"):
            self._build(
                "stage-c-candidate-swap",
                corpus_artifact=corpus,
                expected_corpus_artifact_hash=artifact_sha256(corpus),
                query_artifact=query,
                expected_query_artifact_hash=artifact_sha256(query),
                blind_pool=changed_candidate_pool,
                expected_blind_pool_hash=artifact_sha256(changed_candidate_pool),
                expected_candidate_set_hash=changed_candidate_hash,
                stage_c_gates=gates,
                expected_stage_c_gates_hash=artifact_sha256(gates),
                stage_b_effects=effects,
                expected_stage_b_effects_hash=artifact_sha256(effects),
                stage_c_approval=approval,
                expected_stage_c_approval_hash=artifact_sha256(approval),
            )

    def test_refuses_output_outside_allowed_private_root_before_mutation(self) -> None:
        allowed = self.root / "allowed"
        allowed.mkdir(mode=0o755)
        allowed.chmod(0o755)
        outside = self.root / "outside" / "packet-root"
        with self.assertRaisesRegex(RelevancePacketError, "under"):
            self._build(
                "outside",
                private_root=outside,
                allowed_root=allowed,
            )
        self.assertFalse(outside.exists())
        self.assertEqual(stat.S_IMODE(allowed.stat().st_mode), 0o755)

    def test_refuses_image_bytes_changed_after_manifest_seal_before_writing(self) -> None:
        required_id = self.query["queries"][0]["target_fixture_id"]
        fixture = next(
            fixture for fixture in self.manifest["fixtures"]
            if fixture["fixture_id"] == required_id
        )
        Image.new("RGB", (16, 12), "red").save(fixture["image_path"])
        with self.assertRaisesRegex(RelevancePacketError, "bytes do not match"):
            self._build("changed-pixels")
        self.assertFalse((self.root / "private").exists())


if __name__ == "__main__":
    unittest.main()
