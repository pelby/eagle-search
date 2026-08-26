"""R2/R12 tests for private, statistically valid caption-model evaluation."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from evals.metrics import (
    cluster_bootstrap_lower_bound,
    concept_scores,
    exact_ocr_character_f1,
    ndcg_at_k,
    normalised_character_error_rate,
    ocr_character_f1,
    recall_at_k,
    reciprocal_rank,
    required_target_count,
)
from evals.runner import (
    EvalFixture,
    FixtureManifest,
    PrivateReceiptJournal,
    completed_run_keys,
    evaluate_candidate_retrieval,
    load_fixture_manifest,
    run_caption_stage,
    reseal_authenticated_receipt_journal,
    seal_hidden_manifest,
)
from src.contracts import CaptionReceiptV1, CaptionResultV1
from src.captioning.codex_cli import CaptionGlobalFailure, CaptionItemFailure
from src.captioning.prompt import CAPTION_PROMPT_VERSION
from src.selection_instrument import SelectionDocument


class EvalMetricTests(unittest.TestCase):
    def test_rank_metrics_match_hand_calculation(self) -> None:
        grades = {"a": 3, "b": 2, "c": 1}
        ranking = ["b", "a", "x", "c"]

        self.assertAlmostEqual(ndcg_at_k(ranking, grades, 3), 0.7896, places=3)
        self.assertEqual(recall_at_k(ranking, grades, 2), 2 / 3)
        self.assertEqual(reciprocal_rank(ranking, grades), 1.0)

    def test_concept_and_ocr_metrics_are_deterministic(self) -> None:
        concepts = concept_scores(
            ["teacher", "easel"],
            [{"teacher", "instructor"}, {"board", "easel"}, {"robot"}],
        )

        self.assertEqual(concepts.precision, 1.0)
        self.assertEqual(concepts.recall, 2 / 3)
        self.assertAlmostEqual(ocr_character_f1("Hello, world", "hello world"), 1.0)
        self.assertGreater(normalised_character_error_rate("hello", "hallo"), 0)

    def test_v2_ocr_preserves_search_significant_characters(self) -> None:
        for predicted, expected in (("50", "50%"), ("15", "1.5"), ("AB", "A-B"), ("abc", "a b c")):
            with self.subTest(predicted=predicted, expected=expected):
                self.assertLess(exact_ocr_character_f1(predicted, expected), 1.0)

        self.assertEqual(exact_ocr_character_f1(" Caf\u00e9  50% ", "CAF\u00c9 50%"), 1.0)

    def test_paired_cluster_bootstrap_is_seeded_and_respects_noninferiority(self) -> None:
        candidate = {"one": [0.90, 0.91], "two": [0.88, 0.89], "three": [0.92]}
        comparator = {"one": [0.90, 0.90], "two": [0.89, 0.90], "three": [0.92]}

        first = cluster_bootstrap_lower_bound(candidate, comparator, seed=7, resamples=1000)
        second = cluster_bootstrap_lower_bound(candidate, comparator, seed=7, resamples=1000)

        self.assertEqual(first, second)
        self.assertGreaterEqual(first, -0.03)

    def test_power_target_is_bounded_or_declared_infeasible(self) -> None:
        target = required_target_count(
            [0.01, -0.01, 0.00, 0.01, -0.01],
            margin=0.03,
            seed=4,
            simulations=200,
            minimum=60,
            maximum=120,
        )
        self.assertTrue(target is None or 60 <= target <= 120)

    def test_hidden_manifest_hash_changes_when_gate_changes(self) -> None:
        base = {"seed": 9, "targets": ["t1"], "gates": {"ndcg": 0.85}}
        changed = {"seed": 9, "targets": ["t1"], "gates": {"ndcg": 0.86}}

        self.assertNotEqual(seal_hidden_manifest(base), seal_hidden_manifest(changed))

    def test_candidate_retrieval_isolated_and_resume_keys_are_effort_specific(self) -> None:
        documents = {
            "luna": [SelectionDocument("target", "classroom teacher")],
            "terra": [SelectionDocument("target", "abstract geometry")],
        }
        scored = evaluate_candidate_retrieval(
            documents,
            [{"query": "classroom", "grades": {"target": 2}}],
        )

        self.assertEqual(scored["luna"].recall_at_10, 1.0)
        self.assertEqual(scored["terra"].recall_at_10, 0.0)
        self.assertEqual(
            completed_run_keys(
                [
                    {"fixture_id": "one", "model": "luna", "effort": "low", "prompt_version": "caption-v1", "image_hash": "sha256:a"},
                    {"fixture_id": "one", "model": "luna", "effort": "medium", "prompt_version": "caption-v1", "image_hash": "sha256:a"},
                ]
            ),
            {("one", "luna", "low", "caption-v1", "sha256:a"), ("one", "luna", "medium", "caption-v1", "sha256:a")},
        )

    def test_private_manifest_stage_runner_is_resumable_without_cross_model_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "private"
            snapshot.mkdir()
            (snapshot / "manifest.json").write_text(
                '{"snapshot_version":1,"snapshot_id":"safe","fixtures":['
                '{"fixture_id":"smoke","image_hash":"sha256:a","image_path":"/private/a.png","stage":"A"},'
                '{"fixture_id":"hidden","image_hash":"sha256:b","image_path":"/private/b.png","stage":"C"}'
                ']}',
                encoding="utf-8",
            )
            journal = PrivateReceiptJournal(snapshot, enforce_private_root=False)
            manifest = load_fixture_manifest(snapshot, enforce_private_root=False)
            calls = []

            def caption(fixture, model):
                calls.append((fixture.fixture_id, model))
                return {"caption_result": {"summary": "private"}}

            first = run_caption_stage(manifest, stage="A", models=["gpt-5.6-luna"], effort="low", journal=journal, caption=caption)
            second = run_caption_stage(manifest, stage="A", models=["gpt-5.6-luna"], effort="low", journal=journal, caption=caption)
            medium = run_caption_stage(manifest, stage="A", models=["gpt-5.6-luna"], effort="medium", journal=journal, caption=caption)

            self.assertEqual(len(first.created), 1)
            self.assertEqual(second.created, ())
            self.assertEqual(len(medium.created), 1)
            self.assertEqual(calls, [("smoke", "gpt-5.6-luna"), ("smoke", "gpt-5.6-luna")])
            self.assertEqual(len(journal.load()), 2)

    def test_item_failure_is_bounded_retryable_and_global_failure_stops_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "private"
            snapshot.mkdir()
            (snapshot / "manifest.json").write_text(
                '{"snapshot_version":1,"snapshot_id":"safe","fixtures":['
                '{"fixture_id":"first","image_hash":"sha256:a","image_path":"/private/a.png","stage":"A"},'
                '{"fixture_id":"later","image_hash":"sha256:b","image_path":"/private/b.png","stage":"A"}'
                ']}',
                encoding="utf-8",
            )
            journal = PrivateReceiptJournal(snapshot, enforce_private_root=False)
            manifest = load_fixture_manifest(snapshot, enforce_private_root=False)
            calls = []

            def flaky_caption(fixture, model):
                calls.append(fixture.fixture_id)
                if fixture.fixture_id == "first" and calls.count("first") == 1:
                    raise CaptionItemFailure("transient fixture failure")
                return {"caption_result": {"summary": "private"}}

            first = run_caption_stage(manifest, stage="A", models=["gpt-5.6-luna"], effort="low", journal=journal, caption=flaky_caption)
            self.assertEqual(len(first.created), 1)
            self.assertEqual(len(first.failed), 1)
            self.assertEqual(calls, ["first", "later"])
            self.assertEqual(len(completed_run_keys(journal.load())), 1)

            retried = run_caption_stage(manifest, stage="A", models=["gpt-5.6-luna"], effort="low", journal=journal, caption=flaky_caption)
            self.assertEqual(len(retried.created), 1)
            self.assertEqual(retried.failed, ())
            self.assertEqual(calls, ["first", "later", "first"])
            self.assertEqual(len(completed_run_keys(journal.load())), 2)
            self.assertFalse(any(entry.get("state") == "failed" for entry in journal.load()))

            global_calls = []

            def globally_unavailable(fixture, model):
                global_calls.append(fixture.fixture_id)
                raise CaptionGlobalFailure("quota exhausted")

            with self.assertRaises(CaptionGlobalFailure):
                run_caption_stage(manifest, stage="A", models=["gpt-5.6-terra"], effort="low", journal=journal, caption=globally_unavailable)
            self.assertEqual(global_calls, ["first"])

    def test_legacy_valid_receipt_without_effort_remains_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "private"
            snapshot.mkdir()
            (snapshot / "manifest.json").write_text(
                '{"snapshot_version":1,"snapshot_id":"safe","fixtures":['
                '{"fixture_id":"smoke","image_hash":"sha256:a","image_path":"/private/a.png","stage":"A"}'
                ']}',
                encoding="utf-8",
            )
            receipt_directory = snapshot / "caption-receipts"
            receipt_directory.mkdir()
            (receipt_directory / "legacy.json").write_text(
                '{"fixture_id":"smoke","model":"gpt-5.6-luna","prompt_version":"'
                + CAPTION_PROMPT_VERSION
                + '","image_hash":"sha256:a"}',
                encoding="utf-8",
            )
            journal = PrivateReceiptJournal(snapshot, enforce_private_root=False)
            manifest = load_fixture_manifest(snapshot, enforce_private_root=False)
            calls = []

            low = run_caption_stage(
                manifest,
                stage="A",
                models=["gpt-5.6-luna"],
                effort="low",
                journal=journal,
                caption=lambda fixture, model: calls.append((fixture, model)),
            )
            medium = run_caption_stage(
                manifest,
                stage="A",
                models=["gpt-5.6-luna"],
                effort="medium",
                journal=journal,
                caption=lambda fixture, model: calls.append((fixture, model)) or {"caption_result": {"summary": "private"}},
            )

            self.assertEqual(low.created, ())
            self.assertEqual(low.failed, ())
            self.assertEqual(len(medium.created), 1)
            self.assertEqual(calls, [(manifest.fixtures[0], "gpt-5.6-luna")])

    def test_receipt_reseal_copies_only_exact_authenticated_evidence_without_captioning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, destination = Path(temporary) / "source", Path(temporary) / "destination"
            source.mkdir()
            fixture = EvalFixture("f-1", "sha256:" + "a" * 64, "/private/f-1.png", "B")
            source_manifest = FixtureManifest("source", (fixture,), "sha256:" + "1" * 64)
            destination_manifest = FixtureManifest("destination", (fixture,), "sha256:" + "2" * 64)
            caption = CaptionResultV1.from_dict({
                "contract_version": 1, "image_type": "photo", "diagram_types": [], "subjects": [],
                "visual_style": [], "colours": [], "layout": [], "visible_text": [], "search_terms": ["shape"],
                "summary": "A shape.", "uncertainties": [],
            })
            immutable = CaptionReceiptV1.create(
                image_hash=fixture.image_hash, caption_result=caption, provider="codex-cli", model="gpt-5.6-luna",
                effort="low", prompt_version="caption-v3", created_at="2026-08-26T00:00:00Z",
            )
            immutable_path = source / "model-receipts" / fixture.image_hash
            immutable_path.mkdir(parents=True)
            (immutable_path / f"{immutable.receipt_id}.json").write_text(__import__("json").dumps(immutable.to_dict()), encoding="utf-8")
            PrivateReceiptJournal(source, enforce_private_root=False).put({
                "fixture_id": fixture.fixture_id, "image_hash": fixture.image_hash, "model": "gpt-5.6-luna",
                "effort": "low", "prompt_version": "caption-v3", "manifest_hash": source_manifest.manifest_hash,
                "receipt_id": immutable.receipt_id, "caption_result": caption.to_dict(), "search_text": immutable.search_text,
            })
            self.assertEqual(
                reseal_authenticated_receipt_journal(
                    source_snapshot=source, destination_snapshot=destination, source_manifest=source_manifest,
                    destination_manifest=destination_manifest, candidate=("gpt-5.6-luna", "low", "caption-v3"),
                    enforce_private_root=False,
                ),
                1,
            )
            copied = PrivateReceiptJournal(destination, enforce_private_root=False).load()
            self.assertEqual(copied[0]["manifest_hash"], destination_manifest.manifest_hash)
            self.assertEqual(copied[0]["receipt_id"], immutable.receipt_id)
            self.assertTrue((destination / "model-receipts" / fixture.image_hash / f"{immutable.receipt_id}.json").is_file())

            for index, changed in enumerate(
                (
                    EvalFixture(fixture.fixture_id, fixture.image_hash, "/another/path.png", "C"),
                    EvalFixture(fixture.fixture_id, fixture.image_hash, "/another/path.png", "B", "other-stratum"),
                    EvalFixture(fixture.fixture_id, fixture.image_hash, "/another/path.png", "B", "", "distractor"),
                )
            ):
                with self.subTest(changed=changed):
                    with self.assertRaisesRegex(ValueError, "stages, roles and strata"):
                        reseal_authenticated_receipt_journal(
                            source_snapshot=source,
                            destination_snapshot=Path(temporary) / f"mutated-{index}",
                            source_manifest=source_manifest,
                            destination_manifest=FixtureManifest(
                                f"mutated-{index}",
                                (changed,),
                                f"sha256:{index + 3:064x}",
                            ),
                            candidate=("gpt-5.6-luna", "low", "caption-v3"),
                            enforce_private_root=False,
                        )


if __name__ == "__main__":
    unittest.main()
