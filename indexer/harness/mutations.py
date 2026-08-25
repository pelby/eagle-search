"""Run named safety mutants only in auto-destroyed repository copies."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Mutant:
    name: str
    target: str
    anchor: str
    replacement: str
    test_id: str

    def with_anchor(self, anchor: str) -> "Mutant":
        return replace(self, anchor=anchor)


@dataclass(frozen=True)
class MutationResult:
    name: str
    killed: bool
    return_code: int
    test_id: str
    temporary_root: str
    output: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "killed": self.killed,
            "return_code": self.return_code,
            "test_id": self.test_id,
            "temporary_root": self.temporary_root,
            "output": self.output,
        }


def named_mutants() -> tuple[Mutant, ...]:
    return (
        Mutant(
            "replace-rather-than-merge-notes",
            "indexer/src/eagle/notes.py",
            "    return text + separator + block\n",
            "    return block  # MUTANT: overwrite all human Notes\n",
            "tests.test_composed_journeys.ComposedJourneyTests.test_human_notes_dry_run_then_apply_preserves_exact_bytes",
        ),
        Mutant(
            "mark-blank-caption-complete",
            "indexer/src/worker/caption_state.py",
            "                if not isinstance(receipt_id, str) or not SHA256_ID_RE.fullmatch(receipt_id):\n",
            "                if not isinstance(receipt_id, str):  # MUTANT: blank is accepted\n",
            "tests.test_r05_caption_state.CaptionStateTests.test_blank_receipt_is_failed_not_complete_and_can_be_retried",
        ),
        Mutant(
            "remove-worker-lock",
            "indexer/src/worker/lock.py",
            "            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)\n",
            "            handle.flush()  # MUTANT: no inter-process exclusion\n",
            "tests.test_r09_worker.WorkerLockTests.test_second_process_cannot_enter_same_worker_lock",
        ),
        Mutant(
            "remove-semantic-relevance-floor",
            "indexer/src/retrieval/hybrid.py",
            "        if not lexical and semantic_score[eagle_id] < semantic_floor:\n",
            "        if False and not lexical and semantic_score[eagle_id] < semantic_floor:  # MUTANT\n",
            "tests.test_composed_journeys.SemanticFloorGuardTests.test_semantic_only_nonsense_below_floor_is_rejected",
        ),
        Mutant(
            "remove-watcher-startup-scan",
            "indexer/src/eagle/importer.py",
            "        for path in sorted(Path(folder).iterdir()):\n",
            "        for path in ():  # MUTANT: filesystem events only\n",
            "tests.test_composed_journeys.ComposedJourneyTests.test_offline_created_file_is_discovered_at_startup",
        ),
        Mutant(
            "acknowledge-timeout-without-marker-reconciliation",
            "indexer/src/eagle/importer.py",
            "            return ImportOutcome(\"ambiguous\", intent.intent_id, error=str(exc))\n",
            "            return self._acknowledge(intent.intent_id, \"unverified-timeout\", reconciled=False)  # MUTANT\n",
            "tests.test_r10_importer.ImporterTests.test_ambiguous_timeout_never_retries_add_until_marker_appears",
        ),
        Mutant(
            "disconnect-raycast-from-canonical-cli",
            "raycast-extension/src/lib/indexer.ts",
            '  return ["run", "python", "-m", "src", "search", query, "--mode", mode, "--limit", String(limit), "--json"];\n',
            '  return ["run", "python", "-m", "src", "legacy-search", query, "--mode", mode, "--limit", String(limit), "--json"]; // MUTANT\n',
            "tests.test_composed_journeys.RaycastClientJourneyTests.test_raycast_search_client_invokes_canonical_cli",
        ),
    )


class MutationHarness:
    def __init__(
        self,
        repository_root: Path,
        *,
        temporary_parent: Path | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.repository_root = Path(repository_root).resolve()
        self.temporary_parent = Path(temporary_parent) if temporary_parent else None
        self.timeout_seconds = timeout_seconds

    def live_hashes(self) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for target in sorted({mutant.target for mutant in named_mutants()}):
            data = (self.repository_root / target).read_bytes()
            hashes[target] = hashlib.sha256(data).hexdigest()
        return hashes

    @staticmethod
    def _ignore(_directory: str, names: list[str]) -> set[str]:
        ignored = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv"}
        return {name for name in names if name in ignored or name.endswith(".pyc")}

    def _copy_candidate(self, temporary_root: Path) -> None:
        shutil.copytree(
            self.repository_root / "indexer",
            temporary_root / "indexer",
            ignore=self._ignore,
        )
        raycast_target = temporary_root / "raycast-extension" / "src" / "lib"
        raycast_target.mkdir(parents=True)
        shutil.copy2(
            self.repository_root / "raycast-extension" / "src" / "lib" / "indexer.ts",
            raycast_target / "indexer.ts",
        )

    def run(self, mutant: Mutant) -> MutationResult:
        before = self.live_hashes()
        temporary_path = ""
        with tempfile.TemporaryDirectory(
            prefix="eagle-search-mutant-",
            dir=self.temporary_parent,
        ) as directory:
            temporary_root = Path(directory)
            temporary_path = str(temporary_root)
            self._copy_candidate(temporary_root)
            target = temporary_root / mutant.target
            source = target.read_text(encoding="utf-8")
            if source.count(mutant.anchor) != 1:
                raise RuntimeError(
                    f"mutation anchor must occur exactly once for {mutant.name}; "
                    f"found {source.count(mutant.anchor)}"
                )
            target.write_text(source.replace(mutant.anchor, mutant.replacement, 1), encoding="utf-8")
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(temporary_root / "indexer")
            completed = subprocess.run(
                [sys.executable, "-m", "unittest", mutant.test_id, "-v"],
                cwd=temporary_root / "indexer",
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            output = completed.stdout + completed.stderr
            killed = completed.returncode != 0 and (
                "FAILED" in output or "ERROR" in output or "FAIL:" in output
            )
            result = MutationResult(
                name=mutant.name,
                killed=killed,
                return_code=completed.returncode,
                test_id=mutant.test_id,
                temporary_root=temporary_path,
                output=output[-4_000:],
            )
        if self.live_hashes() != before:
            raise RuntimeError("live worktree changed while running an isolated mutant")
        return result

    def run_all(self) -> tuple[MutationResult, ...]:
        return tuple(self.run(mutant) for mutant in named_mutants())

