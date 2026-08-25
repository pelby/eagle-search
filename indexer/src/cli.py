"""Consumer-neutral command surface for Eagle Search.

JSON commands write exactly one envelope to stdout. Operational diagnostics belong
on stderr so Raycast and future adapters can share the same parser.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from . import db
from .eagle.importer import FileIntentMetadataStore, GeneratedImageImporter
from .eagle.notes import NotesApplier, render_caption_block
from .eagle_api import EagleApiClient
from .indexing import index_library
from .persistence.jobs import SQLiteImportQueueStore
from .persistence.feedback import SearchFeedbackStore
from .persistence.receipts import FileReceiptStore
from .retrieval.backfill import backfill_embeddings
from .retrieval.embeddings import Embedder, OllamaEmbedder
from .retrieval.hybrid import hybrid_search
from .rebuild import load_legacy_manifest, rebuild_database, snapshot_eagle_records
from .watcher.host import WatchGeneratedHost
from .worker.runtime import RunStatusFile
from .worker.lock import FileLock
from evals.cli import aggregate_report_json, create_manifest_json
from evals.runner import PrivateReceiptJournal, load_fixture_manifest, run_caption_stage


def _default_home() -> Path:
    configured = os.environ.get("EAGLE_SEARCH_HOME")
    return Path(configured).expanduser() if configured else Path.home() / ".eagle-search"


@dataclass(frozen=True)
class CliRuntime:
    """Injectable local state and adapters for tests and host integrations."""

    home: Path
    embedder: Embedder
    eagle: Any | None
    caption_provider: Any | None

    def __init__(
        self,
        *,
        home: Path | None = None,
        embedder: Embedder | None = None,
        eagle: Any | None = None,
        caption_provider: Any | None = None,
    ) -> None:
        object.__setattr__(self, "home", Path(home or _default_home()))
        object.__setattr__(self, "embedder", embedder or OllamaEmbedder())
        object.__setattr__(self, "eagle", eagle)
        object.__setattr__(self, "caption_provider", caption_provider)

    @property
    def database_path(self) -> Path:
        return self.home / "db.sqlite"

    @property
    def status_path(self) -> Path:
        return self.home / "run-status.json"

    @property
    def receipts_path(self) -> Path:
        return self.home / "captions"

    @property
    def evals_path(self) -> Path:
        return self.home / "evals"


def _runtime_eagle(runtime: CliRuntime) -> Any:
    return runtime.eagle or EagleApiClient()


class CliRequestError(ValueError):
    """A stable invalid-request error, suitable for a versioned JSON envelope."""


def _error(code: str, message: str) -> dict[str, Any]:
    return {
        "contract_version": 1,
        "ok": False,
        "error": {"code": code, "message": " ".join(str(message).split())[:500]},
    }


def _emit(payload: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return
    if payload.get("ok") is False:
        print(payload["error"]["message"], file=sys.stderr)
        return
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _connection(runtime: CliRuntime, *, require_existing: bool = False) -> sqlite3.Connection:
    if require_existing and not runtime.database_path.is_file():
        raise FileNotFoundError(f"search database does not exist: {runtime.database_path}")
    return db.init_db(runtime.database_path)


def _command_search(arguments: argparse.Namespace, runtime: CliRuntime) -> dict[str, Any]:
    if arguments.limit < 1 or arguments.limit > 100:
        raise CliRequestError("limit must be between 1 and 100")
    connection = _connection(runtime, require_existing=True)
    try:
        response = hybrid_search(
            connection,
            arguments.query,
            runtime.embedder,
            limit=arguments.limit,
            mode=arguments.mode,
        )
        if not arguments.no_log:
            SearchFeedbackStore(connection).record(
                arguments.query,
                [result.eagle_id for result in response.results],
                no_result=not response.results,
                retrieval_mode=response.mode,
            )
        return response.to_dict()
    finally:
        connection.close()


def _command_status(_arguments: argparse.Namespace, runtime: CliRuntime) -> dict[str, Any]:
    if not runtime.status_path.is_file():
        return {
            "contract_version": 1,
            "run_id": "",
            "state": "idle",
            "stage": "",
            "total": 0,
            "completed": 0,
            "pending": 0,
            "failed": 0,
            "provider": "",
            "model": "",
            "semantic_available": False,
            "last_error": "",
        }
    return RunStatusFile(runtime.status_path).read().to_dict()


def _command_embed(arguments: argparse.Namespace, runtime: CliRuntime) -> dict[str, Any]:
    connection = _connection(runtime, require_existing=True)
    try:
        result = backfill_embeddings(
            connection,
            runtime.embedder,
            batch_size=arguments.batch_size,
            limit=arguments.limit,
        )
        return {"contract_version": 1, "ok": result["error"] is None, **result}
    finally:
        connection.close()


def _command_stats(_arguments: argparse.Namespace, runtime: CliRuntime) -> dict[str, Any]:
    connection = _connection(runtime, require_existing=True)
    try:
        return {"contract_version": 1, "ok": True, **db.stats(connection)}
    finally:
        connection.close()


def _command_index(arguments: argparse.Namespace, runtime: CliRuntime) -> dict[str, Any]:
    outcome = asyncio.run(
        index_library(
            home=runtime.home,
            eagle=_runtime_eagle(runtime),
            embedder=runtime.embedder,
            caption_provider=runtime.caption_provider,
            model=arguments.model,
            effort=arguments.effort,
            max_items=arguments.max_items,
        )
    )
    return outcome.to_dict()


def _command_retry_failed(_arguments: argparse.Namespace, runtime: CliRuntime) -> dict[str, Any]:
    connection = _connection(runtime, require_existing=True)
    try:
        with connection:
            cursor = connection.execute(
                "UPDATE caption_jobs SET state='pending',next_attempt_at='',claim_token='',"
                "claimed_at='',updated_at=datetime('now') WHERE state='failed'"
            )
        return {"contract_version": 1, "ok": True, "retried": int(cursor.rowcount)}
    finally:
        connection.close()


def _command_notes_sync(arguments: argparse.Namespace, runtime: CliRuntime) -> dict[str, Any]:
    if arguments.apply and not arguments.quiescent:
        raise CliRequestError("--apply requires --quiescent after closing active Eagle editors")
    connection = _connection(runtime, require_existing=True)
    try:
        if arguments.id:
            rows = connection.execute(
                "SELECT eagle_id,image_hash FROM images WHERE eagle_id=?",
                (arguments.id,),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT eagle_id,image_hash FROM images WHERE active_receipt_id<>'' ORDER BY eagle_id"
            ).fetchall()
        if arguments.id and not rows:
            raise CliRequestError(f"unknown Eagle item: {arguments.id}")
        receipt_store = FileReceiptStore(runtime.receipts_path)
        applier = NotesApplier(
            _runtime_eagle(runtime),
            FileLock(runtime.home / "notes.lock"),
        )

        async def run_sync() -> list[dict[str, Any]]:
            results: list[dict[str, Any]] = []
            for row in rows:
                eagle_id = str(row["eagle_id"])
                receipt = receipt_store.resolve_active(str(row["image_hash"]))
                if receipt is None:
                    results.append(
                        {
                            "eagle_id": eagle_id,
                            "status": "refused",
                            "refusal_reason": "active immutable caption receipt is missing",
                        }
                    )
                    continue
                result = await applier.sync(
                    eagle_id,
                    caption_block=render_caption_block(receipt),
                    apply=arguments.apply,
                    quiescent=arguments.quiescent,
                )
                results.append(
                    {
                        "eagle_id": eagle_id,
                        "status": result.status,
                        "diff": result.diff.to_dict(),
                        "before_annotation": result.before_annotation,
                        "proposed_annotation": result.proposed_annotation,
                        "observed_annotation": result.observed_annotation,
                        "refusal_reason": result.refusal_reason,
                    }
                )
            return results

        results = asyncio.run(run_sync())
        unsafe = [entry for entry in results if entry["status"] in {"refused", "ambiguous"}]
        return {
            "contract_version": 1,
            "ok": not unsafe,
            "mode": "apply" if arguments.apply else "dry-run",
            "results": results,
        }
    finally:
        connection.close()


def _auto_import_overlaps(info: dict[str, Any], path: Path) -> bool:
    preferences = info.get("preferences", {})
    auto = preferences.get("autoImport", {}) if isinstance(preferences, dict) else {}
    if not isinstance(auto, dict) or str(auto.get("enable", "false")).casefold() != "true":
        return False
    configured = auto.get("path")
    if not isinstance(configured, str) or not configured:
        return True
    try:
        path.resolve().relative_to(Path(configured).expanduser().resolve())
        return True
    except ValueError:
        return False


def _command_import_generated(arguments: argparse.Namespace, runtime: CliRuntime) -> dict[str, Any]:
    image_path = Path(arguments.path).expanduser().resolve()
    prompt_path = Path(arguments.prompt_file).expanduser().resolve()
    prompt = prompt_path.read_text(encoding="utf-8")
    connection = _connection(runtime)
    eagle = _runtime_eagle(runtime)

    async def run_import() -> dict[str, Any]:
        info = await eagle.application_info()
        importer = GeneratedImageImporter(
            queue=SQLiteImportQueueStore(connection),
            metadata_store=FileIntentMetadataStore(runtime.home / "import-intents"),
            eagle=eagle,
            worker_id="import-cli",
            overlapping_authority=lambda: _auto_import_overlaps(info, image_path),
        )
        importer.queue.release_stale(older_than_seconds=300)
        intent = importer.enqueue(
            image_path,
            prompt=prompt,
            source=arguments.source,
            tags=tuple(arguments.tag),
        )
        outcome = await importer.process_one()
        return {
            "contract_version": 1,
            "ok": outcome.state in {"imported", "reconciled"},
            "intent_id": intent.intent_id,
            **outcome.__dict__,
        }

    try:
        return asyncio.run(run_import())
    finally:
        connection.close()


def _command_rebuild(arguments: argparse.Namespace, runtime: CliRuntime) -> dict[str, Any]:
    destination = Path(arguments.output).expanduser().resolve()
    if destination == runtime.database_path.expanduser().resolve():
        raise CliRequestError("rebuild output must not be the active database; verify it before swapping")

    async def run_rebuild() -> dict[str, Any]:
        eagle = _runtime_eagle(runtime)
        records = await snapshot_eagle_records(
            eagle,
            thumbnail_root=runtime.home / "rebuild-thumbnails",
            limit=arguments.limit,
        )
        manifest = load_legacy_manifest(runtime.receipts_path / "legacy-manifest-v1.json")
        outcome = rebuild_database(
            destination,
            records=records,
            receipt_store=FileReceiptStore(runtime.receipts_path),
            legacy_manifest=manifest,
            allow_notes_fallback=arguments.allow_notes_fallback,
        )
        connection = db.init_db(destination)
        try:
            semantic = backfill_embeddings(connection, runtime.embedder, batch_size=32, limit=10_000)
        finally:
            connection.close()
        return {**outcome.to_dict(), "semantic": semantic}

    return asyncio.run(run_rebuild())


def _command_watch_generated(arguments: argparse.Namespace, runtime: CliRuntime) -> dict[str, Any]:
    folder = Path(arguments.folder or (Path.home() / "Pictures" / "generated")).expanduser().resolve()
    host = WatchGeneratedHost(
        home=runtime.home,
        folder=folder,
        eagle=_runtime_eagle(runtime),
        poll_interval_seconds=arguments.interval,
    )
    if arguments.once:
        return asyncio.run(host.once(maximum=arguments.maximum))

    async def run_forever() -> dict[str, Any]:
        await host.run(stop=lambda: False, maximum=arguments.maximum)
        return {"contract_version": 1, "ok": True, "state": "stopped"}

    return asyncio.run(run_forever())


def _eval_path(runtime: CliRuntime, raw: str) -> Path:
    resolved = Path(raw).expanduser().resolve()
    root = runtime.evals_path.expanduser().resolve()
    if not resolved.is_relative_to(root):
        raise CliRequestError(f"private evaluation files must be under {root}")
    return resolved


def _command_eval_manifest(arguments: argparse.Namespace, runtime: CliRuntime) -> dict[str, Any]:
    return {
        "contract_version": 1,
        "ok": True,
        **create_manifest_json(
            _eval_path(runtime, arguments.candidates),
            _eval_path(runtime, arguments.output),
            snapshot_id=arguments.snapshot_id,
            seed=arguments.seed,
            sealing_inputs={
                "labels_hash": arguments.labels_hash,
                "gates_hash": arguments.gates_hash,
                "amendment_hash": arguments.amendment_hash,
                "instrument_version": arguments.instrument_version,
                "target_count": arguments.target_count,
            },
            allowed_root=runtime.evals_path,
        ),
    }


def _command_eval_captions(arguments: argparse.Namespace, runtime: CliRuntime) -> dict[str, Any]:
    snapshot = _eval_path(runtime, arguments.snapshot)
    manifest = load_fixture_manifest(snapshot, enforce_private_root=False)
    journal = PrivateReceiptJournal(snapshot, enforce_private_root=False)
    receipt_store = FileReceiptStore(snapshot / "model-receipts")
    provider = runtime.caption_provider
    if provider is None:
        from .captioning.codex_cli import CodexCliCaptionProvider

        provider = CodexCliCaptionProvider()

    def caption(fixture, model):
        started = time.monotonic()
        receipt = provider.caption(
            Path(fixture.image_path),
            model=model,
            effort=arguments.effort,
            receipt_store=receipt_store,
        )
        if receipt.image_hash != fixture.image_hash:
            raise RuntimeError(f"fixture content changed before captioning: {fixture.fixture_id}")
        return {
            "effort": arguments.effort,
            "receipt_id": receipt.receipt_id,
            "caption_result": receipt.caption_result.to_dict(),
            "search_text": receipt.search_text,
            "latency_seconds": round(time.monotonic() - started, 6),
        }

    created = run_caption_stage(
        manifest,
        stage=arguments.stage,
        models=arguments.model,
        journal=journal,
        caption=caption,
    )
    return {
        "contract_version": 1,
        "ok": True,
        "stage": arguments.stage,
        "models": list(arguments.model),
        "created": len(created),
        "completed_receipts": len(journal.load()),
    }


def _command_eval_report(arguments: argparse.Namespace, runtime: CliRuntime) -> dict[str, Any]:
    report = aggregate_report_json(
        _eval_path(runtime, arguments.results),
        _eval_path(runtime, arguments.output),
        comparator=arguments.comparator,
        allowed_root=runtime.evals_path,
    )
    return {"contract_version": 1, "ok": report["status"] != "failed", **report}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eagle-search")
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser("index", help="Incrementally index Eagle and process missing captions")
    index.add_argument("--format", choices=("json", "jsonl"), default="json")
    index.add_argument("--max-items", type=int)
    index.add_argument("--model", default="gpt-5.6-luna")
    index.add_argument("--effort", choices=("low", "medium"), default="low")
    index.add_argument("--json", action="store_true")
    index.set_defaults(handler=_command_index)

    search = commands.add_parser("search", help="Search the local Eagle index")
    search.add_argument("query")
    search.add_argument("--mode", choices=("automatic", "exact", "best"), default="automatic")
    search.add_argument("--limit", type=int, default=30)
    search.add_argument("--no-log", action="store_true")
    search.add_argument("--json", action="store_true")
    search.set_defaults(handler=_command_search)

    status = commands.add_parser("status", help="Read the latest background-worker status")
    status.add_argument("--json", action="store_true")
    status.set_defaults(handler=_command_status)

    embed = commands.add_parser("embed", help="Backfill missing local text embeddings")
    embed.add_argument("--missing", action="store_true", default=True)
    embed.add_argument("--batch-size", type=int, default=32)
    embed.add_argument("--limit", type=int, default=1000)
    embed.add_argument("--json", action="store_true")
    embed.set_defaults(handler=_command_embed)

    stats = commands.add_parser("stats", help="Show local index statistics")
    stats.add_argument("--json", action="store_true")
    stats.set_defaults(handler=_command_stats)

    retry = commands.add_parser("retry-failed", help="Return failed caption jobs to the pending queue")
    retry.add_argument("--json", action="store_true")
    retry.set_defaults(handler=_command_retry_failed)

    notes = commands.add_parser("notes-sync", help="Dry-run or apply managed caption Notes blocks")
    notes_mode = notes.add_mutually_exclusive_group(required=True)
    notes_mode.add_argument("--dry-run", action="store_true")
    notes_mode.add_argument("--apply", action="store_true")
    notes.add_argument("--quiescent", action="store_true")
    notes.add_argument("--id")
    notes.add_argument("--json", action="store_true")
    notes.set_defaults(handler=_command_notes_sync)

    generated = commands.add_parser("import-generated", help="Import one generated image exactly once")
    generated.add_argument("--path", required=True)
    generated.add_argument("--prompt-file", required=True)
    generated.add_argument("--source", required=True)
    generated.add_argument("--tag", action="append", default=[])
    generated.add_argument("--json", action="store_true")
    generated.set_defaults(handler=_command_import_generated)

    watcher = commands.add_parser("watch-generated", help="Watch and exactly-once import generated images")
    watcher.add_argument("--folder")
    watcher.add_argument("--once", action="store_true")
    watcher.add_argument("--maximum", type=int, default=1)
    watcher.add_argument("--interval", type=float, default=1.0)
    watcher.add_argument("--json", action="store_true")
    watcher.set_defaults(handler=_command_watch_generated)

    rebuild = commands.add_parser("rebuild", help="Create a new derived database from receipts and Eagle metadata")
    rebuild.add_argument("--output", required=True)
    rebuild.add_argument("--limit", type=int, default=10_000)
    rebuild.add_argument("--allow-notes-fallback", action="store_true")
    rebuild.add_argument("--json", action="store_true")
    rebuild.set_defaults(handler=_command_rebuild)

    eval_manifest = commands.add_parser("eval-manifest", help="Create a sealed private evaluation fixture")
    eval_manifest.add_argument("--candidates", required=True)
    eval_manifest.add_argument("--output", required=True)
    eval_manifest.add_argument("--snapshot-id", required=True)
    eval_manifest.add_argument("--seed", type=int, required=True)
    eval_manifest.add_argument("--labels-hash", required=True)
    eval_manifest.add_argument("--gates-hash", required=True)
    eval_manifest.add_argument("--amendment-hash", required=True)
    eval_manifest.add_argument("--instrument-version", required=True)
    eval_manifest.add_argument("--target-count", type=int, default=120)
    eval_manifest.add_argument("--json", action="store_true")
    eval_manifest.set_defaults(handler=_command_eval_manifest)

    eval_captions = commands.add_parser("eval-captions", help="Run or resume a private caption-model stage")
    eval_captions.add_argument("--snapshot", required=True)
    eval_captions.add_argument("--stage", choices=("A", "B", "C"), required=True)
    eval_captions.add_argument("--model", action="append", required=True)
    eval_captions.add_argument("--effort", choices=("low", "medium"), default="low")
    eval_captions.add_argument("--json", action="store_true")
    eval_captions.set_defaults(handler=_command_eval_captions)

    eval_report = commands.add_parser("eval-report", help="Aggregate private metrics into an anonymous decision report")
    eval_report.add_argument("--results", required=True)
    eval_report.add_argument("--output", required=True)
    eval_report.add_argument("--comparator", required=True)
    eval_report.add_argument("--json", action="store_true")
    eval_report.set_defaults(handler=_command_eval_report)
    return parser


def main(argv: Sequence[str] | None = None, *, runtime: CliRuntime | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    json_output = bool(
        getattr(arguments, "json", False)
        or getattr(arguments, "format", "") in {"json", "jsonl"}
    )
    try:
        payload = arguments.handler(arguments, runtime or CliRuntime())
    except CliRequestError as error:
        _emit(_error("invalid_request", str(error)), json_output=json_output)
        return 2
    except FileNotFoundError as error:
        _emit(_error("missing_database", str(error)), json_output=json_output)
        return 3
    except (sqlite3.DatabaseError, ValueError) as error:
        _emit(_error("invalid_state", str(error)), json_output=json_output)
        return 4
    except Exception as error:
        _emit(_error("operation_failed", f"{type(error).__name__}: {error}"), json_output=json_output)
        return 1
    _emit(payload, json_output=json_output)
    return 0 if payload.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
