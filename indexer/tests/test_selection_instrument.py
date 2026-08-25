"""Tests for the frozen, untuned caption selection instrument."""

from __future__ import annotations

import unittest

from src.selection_instrument import (
    SelectionDocument,
    SelectionInstrument,
    cosine_similarity,
    reciprocal_rank_fusion,
)


class SelectionInstrumentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.docs = [
            SelectionDocument("classroom", "A robot teaches beside a classroom easel."),
            SelectionDocument("office", "A team meeting around a boardroom table."),
            SelectionDocument("garden", "A bright garden with flowers and trees."),
        ]

    def test_bm25_lane_ranks_literal_caption_evidence(self) -> None:
        result = SelectionInstrument().rank("classroom teaching", self.docs)

        self.assertEqual(result.bm25_ids[0], "classroom")
        self.assertEqual(result.fused_ids[0], "classroom")

    def test_semantic_lane_is_reported_separately_and_fused(self) -> None:
        embeddings = {
            "classroom": [1.0, 0.0],
            "office": [0.7, 0.3],
            "garden": [0.0, 1.0],
        }
        result = SelectionInstrument().rank(
            "learning space",
            self.docs,
            document_embeddings=embeddings,
            query_embedding=[1.0, 0.0],
        )

        self.assertEqual(result.semantic_ids[0], "classroom")
        self.assertEqual(result.fused_ids[0], "classroom")

    def test_candidate_corpora_do_not_share_text(self) -> None:
        candidate_a = [SelectionDocument("target", "classroom teacher")]
        candidate_b = [SelectionDocument("target", "abstract shapes")]

        result_a = SelectionInstrument().rank("classroom", candidate_a)
        result_b = SelectionInstrument().rank("classroom", candidate_b)

        self.assertEqual(result_a.bm25_ids, ["target"])
        self.assertEqual(result_b.bm25_ids, [])

    def test_cosine_rejects_invalid_vectors(self) -> None:
        with self.assertRaises(ValueError):
            cosine_similarity([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            cosine_similarity([0.0, 0.0], [1.0, 0.0])

    def test_rrf_is_deterministic_for_ties(self) -> None:
        fused = reciprocal_rank_fusion([["b", "a"], ["a", "b"]])

        self.assertEqual([item_id for item_id, _ in fused], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
