"""The seven section-11.3 mutants must fail only inside disposable copies."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from harness.mutations import MutationHarness, named_mutants


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class MutationHarnessTests(unittest.TestCase):
    def test_all_named_mutants_are_killed_without_touching_live_worktree(self) -> None:
        harness = MutationHarness(REPOSITORY_ROOT, temporary_parent=Path(tempfile.gettempdir()))
        before = harness.live_hashes()

        results = harness.run_all()

        self.assertEqual([result.name for result in results], [mutant.name for mutant in named_mutants()])
        self.assertTrue(all(result.killed for result in results), results)
        self.assertTrue(all(result.return_code != 0 for result in results), results)
        self.assertTrue(all(not Path(result.temporary_root).exists() for result in results), results)
        self.assertEqual(harness.live_hashes(), before)

    def test_missing_mutation_anchor_refuses_without_running_a_test(self) -> None:
        harness = MutationHarness(REPOSITORY_ROOT)
        mutant = named_mutants()[0].with_anchor("text that cannot exist in the target")

        with self.assertRaisesRegex(RuntimeError, "exactly once"):
            harness.run(mutant)


if __name__ == "__main__":
    unittest.main()
