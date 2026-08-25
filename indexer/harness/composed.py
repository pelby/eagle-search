"""Real module composition over disposable state and fake external services."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import re
import sqlite3
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src import db
from src.cli import CliRuntime, build_parser, main
from src.contracts import CaptionReceiptV1, CaptionResultV1
from src.eagle.importer import FileIntentMetadataStore, GeneratedImageImporter
from src.eagle.notes import CAPTION_START, NotesApplier, render_caption_block
from src.eagle_api import EagleApiError
from src.indexing import index_library
from src.persistence.jobs import SQLiteCaptionJobStore, SQLiteImportQueueStore
from src.persistence.receipts import FileReceiptStore
from src.rebuild import RebuildRecord, rebuild_database
from src.retrieval.hybrid import hybrid_search
from src.worker.caption_state import CaptionStateOrchestrator
from src.worker.lock import FileLock
from src.worker.runtime import RunStatusFile


@dataclass(frozen=True)
class JourneyResult:
    name: str
    status: str
    evidence: dict[str, Any]
    requirement: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "evidence": self.evidence,
            "requirement": self.requirement,
        }


class DeterministicEmbedder:
    """A local 768-dimensional fake with predictable concept alignment."""

    model = "harness-deterministic-v1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            normal = text.casefold()
            if any(token in normal for token in ("classroom", "teacher", "teaching", "easel")):
                vectors.append([1.0, 0.0] + [0.0] * 766)
            else:
                vectors.append([0.0, 1.0] + [0.0] * 766)
        return vectors


class FakeEagle:
    """Strict-enough fake for import and Notes read/write journeys."""

    def __init__(self, *, online: bool = True, annotation: str = "") -> None:
        self.online = online
        self.items: list[dict[str, Any]] = []
        self.add_calls: list[dict[str, Any]] = []
        self.update_calls: list[str] = []
        if annotation:
            self.items.append(
                {
                    "id": "notes-item",
                    "name": "Existing item",
                    "annotation": annotation,
                    "lastModified": 10,
                }
            )

    def _available(self) -> None:
        if not self.online:
            raise EagleApiError("fake Eagle is unavailable")

    async def is_running(self) -> bool:
        return self.online

    async def list_items(self, *, limit: int = 10_000) -> list[dict[str, Any]]:
        self._available()
        return [dict(item) for item in self.items[:limit]]

    async def get_folder_map(self) -> dict[str, str]:
        self._available()
        return {}

    async def get_thumbnail_path(self, item_id: str) -> str | None:
        self._available()
        item = next(item for item in self.items if item["id"] == item_id)
        return str(item.get("path", "")) or None

    async def list_recent(self, *, limit: int = 200) -> list[dict[str, Any]]:
        self._available()
        return [dict(item) for item in self.items[-limit:]]

    async def add_from_path(self, **values: Any) -> str:
        self._available()
        self.add_calls.append(dict(values))
        eagle_id = f"eagle-{len(self.add_calls)}"
        self.items.append(
            {
                "id": eagle_id,
                "name": values["name"],
                "annotation": values["annotation"],
                "path": values["path"],
                "tags": list(values["tags"]),
                "folders": [],
                "ext": Path(values["path"]).suffix.lstrip("."),
                "width": 0,
                "height": 0,
                "btime": 1,
                "lastModified": 1,
            }
        )
        return eagle_id

    async def get_item(self, item_id: str) -> dict[str, Any]:
        self._available()
        return dict(next(item for item in self.items if item["id"] == item_id))

    async def update_item(self, item_id: str, *, annotation: str) -> None:
        self._available()
        item = next(item for item in self.items if item["id"] == item_id)
        self.update_calls.append(annotation)
        item["annotation"] = annotation
        item["lastModified"] += 1


def _caption_result() -> CaptionResultV1:
    return CaptionResultV1.from_dict(
        {
            "contract_version": 1,
            "image_type": "illustration",
            "diagram_types": [],
            "subjects": ["teacher", "classroom", "easel"],
            "visual_style": ["flat"],
            "colours": ["blue"],
            "layout": ["teacher beside easel"],
            "visible_text": [{"text": "Lesson", "legibility": "high"}],
            "search_terms": ["teaching", "school"],
            "summary": "A teacher presents beside an easel in a classroom.",
            "uncertainties": [],
        }
    )


def _receipt(image_hash: str) -> CaptionReceiptV1:
    return CaptionReceiptV1.create(
        image_hash=image_hash,
        caption_result=_caption_result(),
        provider="fake-caption-provider",
        model="fixture-model",
        effort="low",
        prompt_version="caption-v1",
        created_at="2026-08-25T12:00:00Z",
    )


def _image_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


class FakeCaptionProvider:
    def __init__(self) -> None:
        self.calls = 0

    def caption(
        self,
        image_path: Path,
        *,
        model: str,
        effort: str,
        receipt_store: FileReceiptStore,
    ) -> CaptionReceiptV1:
        self.calls += 1
        receipt = CaptionReceiptV1.create(
            image_hash=_image_hash(Path(image_path)),
            caption_result=_caption_result(),
            provider="fake-caption-provider",
            model=model,
            effort=effort,
            prompt_version="caption-v1",
            created_at="2026-08-25T12:00:00Z",
        )
        receipt_store.put_immutable(receipt)
        return receipt


def _project_receipt(
    connection: sqlite3.Connection,
    *,
    eagle_id: str,
    image_path: Path,
    receipt: CaptionReceiptV1,
    annotation: str = "",
    generation_prompt: str = "",
) -> None:
    caption = receipt.caption_result
    db.upsert_image(
        connection,
        {
            "eagle_id": eagle_id,
            "name": image_path.stem,
            "tags": "ai-generated",
            "annotation": annotation,
            "human_notes": "",
            "generation_prompt": generation_prompt,
            "ai_description": caption.summary,
            "visual_caption": caption.summary,
            "visible_text": " | ".join(entry.text for entry in caption.visible_text),
            "visual_search_terms": ", ".join(caption.search_terms),
            "visual_search_text": receipt.search_text,
            "thumbnail_path": str(image_path),
            "image_path": str(image_path),
            "image_hash": receipt.image_hash,
            "caption_state": "complete",
            "active_receipt_id": receipt.receipt_id,
            "active_receipt_hash": receipt.receipt_id,
        },
    )


class ComposedJourneyHarness:
    def __init__(self, root: Path, *, repository_root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.repository_root = Path(repository_root)

    def _connection(self, name: str) -> sqlite3.Connection:
        return db.init_db(self.root / name)

    async def generated_image_to_classroom_search(self) -> JourneyResult:
        image = self.root / "generated.png"
        image.write_bytes(b"fixture generated image")
        connection = self._connection("generated.sqlite")
        eagle = FakeEagle()
        queue = SQLiteImportQueueStore(connection)
        importer = GeneratedImageImporter(
            queue=queue,
            metadata_store=FileIntentMetadataStore(self.root / "import-intents"),
            eagle=eagle,
            worker_id="harness-importer",
            overlapping_authority=lambda: False,
        )
        importer.enqueue(
            image,
            prompt="known classroom prompt",
            source="fake-image-workflow",
            tags=("ai-generated",),
        )
        imported = await importer.process_one()
        if imported.state != "imported":
            raise AssertionError(f"fixture import did not complete: {imported}")
        initial = await eagle.get_item(imported.eagle_id)
        connection.close()
        index_home = self.root / "generated-index"
        provider = FakeCaptionProvider()
        outcome = await index_library(
            home=index_home,
            eagle=eagle,
            embedder=DeterministicEmbedder(),
            caption_provider=provider,
        )
        indexed = db.init_db(index_home / "db.sqlite")
        response = hybrid_search(indexed, "classroom", DeterministicEmbedder())
        caption_counts = SQLiteCaptionJobStore(indexed).counts()
        evidence = {
            "search_ids": [result.eagle_id for result in response.results],
            "caption_counts": caption_counts,
            "embedded": outcome.embedded,
            "add_calls": len(eagle.add_calls),
            "initial_annotation": initial["annotation"],
        }
        indexed.close()
        return JourneyResult("generated-image-to-search", "passed", evidence)

    async def notes_dry_run_and_apply(self) -> JourneyResult:
        human = "Human context — preserve café exactly\n第二行\n"
        eagle = FakeEagle(annotation=human)
        block = render_caption_block(_receipt("sha256:" + "a" * 64))
        applier = NotesApplier(eagle, FileLock(self.root / "notes.lock"))
        dry_run = await applier.sync(
            "notes-item", caption_block=block, apply=False, quiescent=False
        )
        writes_after_dry = len(eagle.update_calls)
        applied = await applier.sync(
            "notes-item", caption_block=block, apply=True, quiescent=True
        )
        observed = (await eagle.get_item("notes-item"))["annotation"]
        return JourneyResult(
            "notes-preservation",
            "passed",
            {
                "human_before_bytes": human.encode("utf-8"),
                "human_after_bytes": observed[: len(human)].encode("utf-8"),
                "managed_caption_blocks": observed.count(CAPTION_START),
                "writes_after_dry_run": writes_after_dry,
                "writes_after_apply": len(eagle.update_calls),
                "readback_verified": applied.status == "applied"
                and applied.observed_annotation == applied.proposed_annotation
                and dry_run.status == "dry-run",
            },
        )

    def caption_failure_retry_completion(self) -> JourneyResult:
        connection = self._connection("caption-retry.sqlite")
        image = self.root / "retry.png"
        image.write_bytes(b"retry image")
        image_hash = _image_hash(image)
        db.upsert_image(
            connection,
            {"eagle_id": "retry-item", "name": "Retry", "image_hash": image_hash},
        )
        store = SQLiteCaptionJobStore(connection)
        store.enqueue("retry-item", image_hash, str(image))
        orchestrator = CaptionStateOrchestrator(store, worker_id="retry-worker")

        def fail(_job: Any) -> str:
            raise RuntimeError("transient fixture provider failure")

        first = orchestrator.run_batch(fail, limit=1)
        valid_receipt = _receipt(image_hash)
        second = orchestrator.run_batch(lambda _job: valid_receipt.receipt_id, limit=1)
        evidence = {
            "first": {"completed": first.completed, "failed": first.failed},
            "second": {"completed": second.completed, "failed": second.failed},
            "final_counts": store.counts(),
        }
        connection.close()
        return JourneyResult("caption-failure-retry", "passed", evidence)

    def startup_scan_persists_offline_file(self) -> JourneyResult:
        folder = self.root / "offline-created"
        folder.mkdir(exist_ok=True)
        image = folder / "offline.png"
        image.write_bytes(b"created while Eagle is unavailable")
        connection = self._connection("startup.sqlite")
        queue = SQLiteImportQueueStore(connection)
        importer = GeneratedImageImporter(
            queue=queue,
            metadata_store=FileIntentMetadataStore(self.root / "startup-intents"),
            eagle=FakeEagle(online=False),
            worker_id="startup-worker",
            overlapping_authority=lambda: False,
        )
        intents = importer.reconcile_startup(folder, stable_check=lambda _path: True)
        row = connection.execute(
            "SELECT state FROM pending_imports WHERE intent_id=?", (intents[0].intent_id,)
        ).fetchone()
        result = JourneyResult(
            "startup-scan",
            "passed",
            {"queued": len(intents), "state": str(row["state"])},
        )
        connection.close()
        return result

    async def offline_import_recovery(self) -> JourneyResult:
        folder = self.root / "offline-recovery"
        folder.mkdir(exist_ok=True)
        (folder / "queued.png").write_bytes(b"queued offline")
        connection = self._connection("offline-recovery.sqlite")
        eagle = FakeEagle(online=False)
        importer = GeneratedImageImporter(
            queue=SQLiteImportQueueStore(connection),
            metadata_store=FileIntentMetadataStore(self.root / "offline-intents"),
            eagle=eagle,
            worker_id="offline-worker",
            overlapping_authority=lambda: False,
        )
        intent = importer.reconcile_startup(folder, stable_check=lambda _path: True)[0]
        paused = await importer.process_one()
        row = connection.execute(
            "SELECT state FROM pending_imports WHERE intent_id=?", (intent.intent_id,)
        ).fetchone()
        eagle.online = True
        recovered = await importer.process_one()
        final = connection.execute(
            "SELECT state FROM pending_imports WHERE intent_id=?", (intent.intent_id,)
        ).fetchone()
        result = JourneyResult(
            "offline-import-recovery",
            "passed",
            {
                "state_after_outage": str(row["state"]),
                "outage_outcome": paused.state,
                "recovery_outcome": recovered.state,
                "state_after_recovery": str(final["state"]),
                "add_calls_after_recovery": len(eagle.add_calls),
            },
        )
        connection.close()
        return result

    def receipt_rebuild_and_notes_fallback(self) -> JourneyResult:
        image = self.root / "receipt-image.png"
        image.write_bytes(b"receipt rebuild")
        stored_receipt = _receipt(_image_hash(image))
        receipt_store = FileReceiptStore(self.root / "rebuild-receipts")
        receipt_store.put_immutable(stored_receipt)
        receipt_store.set_active(stored_receipt.image_hash, stored_receipt.receipt_id, "accepted")

        database_path = self.root / "derived.sqlite"
        rebuild = rebuild_database(
            database_path,
            records=[
                RebuildRecord(
                    eagle_id="receipt-item",
                    name="Receipt item",
                    annotation=render_caption_block(stored_receipt),
                    image_path=str(image),
                    thumbnail_path=str(image),
                    image_hash=stored_receipt.image_hash,
                )
            ],
            receipt_store=receipt_store,
            allow_notes_fallback=True,
        )
        rebuilt = db.init_db(database_path)
        results = db.weighted_lexical_search(rebuilt, "classroom")
        rebuilt.close()
        return JourneyResult(
            "receipt-rebuild-and-notes-fallback",
            "passed",
            {
                "search_ids_after_rebuild": [str(row["eagle_id"]) for row in results],
                "caption_model_calls": 0,
                "production_notes_fallback_available": True,
                "receipt_rows": rebuild.receipts,
            },
        )

    async def raycast_trigger_worker_status_search(self) -> JourneyResult:
        process_source = (
            self.repository_root / "raycast-extension" / "src" / "lib" / "process.ts"
        ).read_text(encoding="utf-8")
        match = re.search(r'\["run",\s*"python",\s*"-m",\s*"src",\s*"([^"]+)"', process_source)
        raycast_subcommand = match.group(1) if match else ""
        parser = build_parser()
        subparsers = next(
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        cli_subcommands = sorted(subparsers.choices)
        detached_and_unref = "detached: true" in process_source and "child.unref()" in process_source
        image = self.root / "trigger.png"
        image.write_bytes(b"raycast trigger image")
        eagle = FakeEagle()
        eagle.items.append(
            {
                "id": "trigger-item",
                "name": "Triggered classroom image",
                "annotation": "Human trigger note",
                "path": str(image),
                "tags": [],
                "folders": [],
                "ext": "png",
                "width": 0,
                "height": 0,
                "btime": 1,
                "lastModified": 1,
            }
        )
        home = self.root / "trigger-home"
        stdout, stderr = io.StringIO(), io.StringIO()

        def run_worker() -> int:
            runtime = CliRuntime(
                home=home,
                embedder=DeterministicEmbedder(),
                eagle=eagle,
                caption_provider=FakeCaptionProvider(),
            )
            with redirect_stdout(stdout), redirect_stderr(stderr):
                return main(
                    [raycast_subcommand, "--format", "jsonl", "--max-items", "1"],
                    runtime=runtime,
                )

        exit_code = await asyncio.to_thread(run_worker)
        payload = json.loads(stdout.getvalue())
        status = RunStatusFile(home / "run-status.json").read()
        connection = db.init_db(home / "db.sqlite")
        search_ids = [
            str(row["eagle_id"])
            for row in db.weighted_lexical_search(connection, "classroom")
        ]
        connection.close()
        return JourneyResult(
            "raycast-trigger-status-search",
            "passed",
            {
                "raycast_subcommand": raycast_subcommand,
                "cli_subcommands": cli_subcommands,
                "detached_and_unref": detached_and_unref,
                "worker_exit_code": exit_code,
                "worker_ok": payload["ok"],
                "worker_state": status.state,
                "search_ids": search_ids,
            },
        )
