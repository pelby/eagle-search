"""Composed fake-service journeys from plan section 11.2."""

from __future__ import annotations

import asyncio
import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from harness.composed import ComposedJourneyHarness, DeterministicEmbedder
from src import db
from src.cli import CliRuntime, main
from src.retrieval.hybrid import hybrid_search


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class ComposedJourneyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.harness = ComposedJourneyHarness(self.root, repository_root=REPOSITORY_ROOT)

    async def asyncTearDown(self) -> None:
        self.temporary.cleanup()

    async def test_generated_image_prompt_import_caption_embedding_classroom_search(self) -> None:
        result = await self.harness.generated_image_to_classroom_search()

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.evidence["search_ids"], ["eagle-1"])
        self.assertEqual(result.evidence["caption_counts"], {"complete": 1})
        self.assertEqual(result.evidence["embedded"], 1)
        self.assertEqual(result.evidence["add_calls"], 1)
        self.assertIn("known classroom prompt", result.evidence["initial_annotation"])

    async def test_human_notes_dry_run_then_apply_preserves_exact_bytes(self) -> None:
        result = await self.harness.notes_dry_run_and_apply()

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.evidence["human_before_bytes"], result.evidence["human_after_bytes"])
        self.assertEqual(result.evidence["managed_caption_blocks"], 1)
        self.assertEqual(result.evidence["writes_after_dry_run"], 0)
        self.assertEqual(result.evidence["writes_after_apply"], 1)
        self.assertTrue(result.evidence["readback_verified"])

    async def test_caption_failure_remains_retryable_then_completes(self) -> None:
        result = self.harness.caption_failure_retry_completion()

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.evidence["first"], {"completed": 0, "failed": 1})
        self.assertEqual(result.evidence["second"], {"completed": 1, "failed": 0})
        self.assertEqual(result.evidence["final_counts"], {"complete": 1})

    async def test_offline_created_file_is_discovered_at_startup(self) -> None:
        result = self.harness.startup_scan_persists_offline_file()

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.evidence["queued"], 1)
        self.assertEqual(result.evidence["state"], "pending")

    async def test_offline_created_file_recovers_to_one_import(self) -> None:
        result = await self.harness.offline_import_recovery()

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.evidence["state_after_outage"], "pending")
        self.assertEqual(result.evidence["state_after_recovery"], "complete")
        self.assertEqual(result.evidence["add_calls_after_recovery"], 1)
        self.assertEqual(result.evidence["recovery_outcome"], "imported")

    async def test_receipt_rebuild_uses_production_path_without_model_call(self) -> None:
        result = self.harness.receipt_rebuild_and_notes_fallback()

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.evidence["search_ids_after_rebuild"], ["receipt-item"])
        self.assertEqual(result.evidence["caption_model_calls"], 0)
        self.assertTrue(result.evidence["production_notes_fallback_available"])
        self.assertEqual(result.evidence["receipt_rows"], 1)

    async def test_raycast_trigger_maps_to_cli_status_and_searchable_result(self) -> None:
        result = await self.harness.raycast_trigger_worker_status_search()

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.evidence["raycast_subcommand"], "index")
        self.assertIn(result.evidence["raycast_subcommand"], result.evidence["cli_subcommands"])
        self.assertTrue(result.evidence["detached_and_unref"])
        self.assertEqual(result.evidence["worker_state"], "complete")
        self.assertEqual(result.evidence["search_ids"], ["trigger-item"])


class SemanticFloorGuardTests(unittest.TestCase):
    def test_semantic_only_nonsense_below_floor_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connection = db.init_db(Path(directory) / "db.sqlite")
            db.upsert_image(
                connection,
                {
                    "eagle_id": "semantic-only",
                    "name": "Unrelated object",
                    "visual_caption": "A metal fastener on a plain table.",
                    "visual_search_text": "metal fastener plain table",
                },
            )
            db.store_embedding(
                connection,
                "semantic-only",
                DeterministicEmbedder.model,
                "visual",
                db.image_content_hash(connection, "semantic-only"),
                [0.1, 0.995] + [0.0] * 766,
            )

            response = hybrid_search(connection, "classroom", DeterministicEmbedder())

            self.assertEqual(response.results, ())
            connection.close()


class RaycastClientJourneyTests(unittest.TestCase):
    def test_raycast_search_client_invokes_canonical_cli(self) -> None:
        source = REPOSITORY_ROOT / "raycast-extension" / "src" / "lib" / "indexer.ts"
        script = (
            'import { buildSearchArgs } from '
            + json.dumps(source.as_uri())
            + '; console.log(JSON.stringify(buildSearchArgs("classroom", 5)));'
        )
        completed = subprocess.run(
            ["node", "--no-warnings", "--input-type=module", "--eval", script],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        argv = json.loads(completed.stdout)
        self.assertEqual(argv[:4], ["run", "python", "-m", "src"])

        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            connection = db.init_db(home / "db.sqlite")
            db.upsert_image(connection, {"eagle_id": "fixture", "name": "Classroom", "ai_description": "teacher"})
            connection.close()
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main(argv[4:] + ["--no-log"], runtime=CliRuntime(home=home, embedder=DeterministicEmbedder()))

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(payload["results"][0]["eagle_id"], "fixture")


if __name__ == "__main__":
    unittest.main()
