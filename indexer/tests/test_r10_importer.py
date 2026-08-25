"""R10 red-first tests for exact-once generated-image import."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.eagle.importer import (
    FileIntentMetadataStore,
    GeneratedImageImporter,
    ImportAuthorityError,
    ImportOutcome,
    file_is_stable,
)
from src.eagle.notes import import_intent_marker
from src.eagle_api import EagleApiClient, EagleAmbiguousCommitError, EagleApiError, EagleProtocolError
from src.ports import ImportIntent


class FakeHttpResponse:
    def __init__(self, payload: object, *, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> object:
        return self.payload


class FakeHttpClient:
    def __init__(self, handler) -> None:
        self.handler = handler

    async def request(self, method: str, url: str, **kwargs):
        return await self.handler(method, url, kwargs)


class FakeImportQueue:
    def __init__(self) -> None:
        self.intents: dict[str, ImportIntent] = {}
        self.claim_order: list[str] = []
        self.reconciled: list[tuple[str, str]] = []
        self.acknowledged: list[tuple[str, str]] = []
        self.released: list[tuple[str, str]] = []

    def enqueue(self, path: str, intent_id: str) -> ImportIntent:
        intent = self.intents.get(intent_id)
        if intent is None:
            intent = ImportIntent(intent_id=intent_id, path=path, state="pending")
            self.intents[intent_id] = intent
            self.claim_order.append(intent_id)
        return intent

    def claim(self, *, worker_id: str) -> ImportIntent | None:
        if not self.claim_order:
            return None
        intent_id = self.claim_order[0]
        intent = self.intents[intent_id]
        return replace(intent, state="claimed", attempts=intent.attempts + 1)

    def release(self, intent_id: str, error: str = "") -> None:
        self.released.append((intent_id, error))

    def release_stale(self, *, older_than_seconds: int) -> int:
        return 0

    def reconcile(self, intent_id: str, eagle_id: str) -> None:
        self.reconciled.append((intent_id, eagle_id))

    def acknowledge(self, intent_id: str, eagle_id: str) -> None:
        self.acknowledged.append((intent_id, eagle_id))
        if self.claim_order and self.claim_order[0] == intent_id:
            self.claim_order.pop(0)
        self.intents[intent_id] = replace(self.intents[intent_id], state="complete", eagle_id=eagle_id)


class FakeEagle:
    def __init__(self) -> None:
        self.items: list[dict] = []
        self.add_calls: list[dict] = []
        self.timeout_after_commit = False
        self.hide_marker_scans = 0
        self.corrupt_readback = False
        self.unavailable_scans = 0

    async def list_recent(self, *, limit: int = 200) -> list[dict]:
        if self.unavailable_scans > 0:
            self.unavailable_scans -= 1
            raise EagleApiError("Eagle unavailable")
        if self.hide_marker_scans > 0:
            self.hide_marker_scans -= 1
            return []
        return list(self.items[-limit:])

    async def add_from_path(self, **kwargs) -> str:
        self.add_calls.append(kwargs)
        eagle_id = f"eagle-{len(self.add_calls)}"
        self.items.append(
            {
                "id": eagle_id,
                "annotation": kwargs["annotation"],
                "lastModified": 20,
                "name": kwargs["name"],
            }
        )
        if self.timeout_after_commit:
            raise EagleAmbiguousCommitError("timed out after Eagle may have committed")
        return eagle_id

    async def get_item(self, item_id: str) -> dict:
        item = next(entry for entry in self.items if entry["id"] == item_id)
        if self.corrupt_readback:
            return {**item, "annotation": "marker missing"}
        return dict(item)


class ImporterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.image = self.root / "generated.png"
        self.image.write_bytes(b"stable-image")
        self.queue = FakeImportQueue()
        self.metadata = FileIntentMetadataStore(self.root / "intents")
        self.eagle = FakeEagle()

    async def asyncTearDown(self) -> None:
        self.temp.cleanup()

    def importer(self, *, overlapping_authority: bool = False) -> GeneratedImageImporter:
        return GeneratedImageImporter(
            queue=self.queue,
            metadata_store=self.metadata,
            eagle=self.eagle,
            worker_id="importer-1",
            overlapping_authority=lambda: overlapping_authority,
        )

    async def test_prompt_source_tags_and_intent_marker_are_atomic_on_initial_add(self) -> None:
        importer = self.importer()
        intent = importer.enqueue(
            self.image,
            prompt="A robot teaching a class",
            source="codex-imagegen",
            tags=("ai-generated", "illustration"),
        )

        outcome = await importer.process_one()

        self.assertEqual(outcome.state, "imported")
        self.assertEqual(len(self.eagle.add_calls), 1)
        sent = self.eagle.add_calls[0]
        self.assertIn(import_intent_marker(intent.intent_id), sent["annotation"])
        self.assertIn("A robot teaching a class", sent["annotation"])
        self.assertEqual(sent["source"], "codex-imagegen")
        self.assertEqual(sent["tags"], ("ai-generated", "illustration"))
        self.assertEqual(self.queue.acknowledged, [(intent.intent_id, "eagle-1")])

    async def test_commit_then_timeout_reconciles_marker_without_second_add(self) -> None:
        importer = self.importer()
        intent = importer.enqueue(self.image, prompt="Prompt", source="generator", tags=("ai-generated",))
        self.eagle.timeout_after_commit = True
        self.eagle.hide_marker_scans = 1  # pre-add scan is empty; post-timeout sees committed item

        outcome = await importer.process_one()

        self.assertEqual(outcome.state, "reconciled")
        self.assertEqual(len(self.eagle.add_calls), 1)
        self.assertEqual(self.queue.reconciled, [(intent.intent_id, "eagle-1")])
        self.assertEqual(self.queue.acknowledged, [(intent.intent_id, "eagle-1")])

    async def test_ambiguous_timeout_never_retries_add_until_marker_appears(self) -> None:
        importer = self.importer()
        intent = importer.enqueue(self.image, prompt="Prompt", source="generator", tags=())
        self.eagle.timeout_after_commit = True
        self.eagle.hide_marker_scans = 2  # pre-add and immediate post-timeout scans are empty

        first = await importer.process_one()
        self.eagle.timeout_after_commit = False
        second = await importer.process_one()

        self.assertEqual(first.state, "ambiguous")
        self.assertEqual(second.state, "reconciled")
        self.assertEqual(len(self.eagle.add_calls), 1)
        self.assertEqual(self.queue.acknowledged, [(intent.intent_id, "eagle-1")])

    async def test_returned_id_is_not_acknowledged_without_marker_readback(self) -> None:
        importer = self.importer()
        importer.enqueue(self.image, prompt="Prompt", source="generator", tags=())
        self.eagle.corrupt_readback = True

        outcome = await importer.process_one()

        self.assertEqual(outcome.state, "ambiguous")
        self.assertEqual(self.queue.acknowledged, [])

    async def test_existing_marker_is_deduped_without_add(self) -> None:
        importer = self.importer()
        intent = importer.enqueue(self.image, prompt="Prompt", source="generator", tags=())
        self.eagle.items.append(
            {
                "id": "already-there",
                "annotation": import_intent_marker(intent.intent_id),
                "lastModified": 1,
                "name": "generated",
            }
        )

        outcome = await importer.process_one()

        self.assertEqual(outcome.state, "reconciled")
        self.assertEqual(self.eagle.add_calls, [])
        self.assertEqual(self.queue.acknowledged, [(intent.intent_id, "already-there")])

    async def test_eagle_unavailable_releases_claim_then_recovers_once(self) -> None:
        importer = self.importer()
        intent = importer.enqueue(self.image, prompt="Prompt", source="generator", tags=())
        self.eagle.unavailable_scans = 1

        paused = await importer.process_one()
        recovered = await importer.process_one()

        self.assertEqual(paused.state, "paused")
        self.assertEqual(self.queue.released[0][0], intent.intent_id)
        self.assertEqual(recovered.state, "imported")
        self.assertEqual(len(self.eagle.add_calls), 1)

    async def test_overlapping_import_authority_refuses_before_queue_or_eagle(self) -> None:
        importer = self.importer(overlapping_authority=True)

        with self.assertRaises(ImportAuthorityError):
            importer.enqueue(self.image, prompt="Prompt", source="generator", tags=())
        with self.assertRaises(ImportAuthorityError):
            await importer.process_one()

        self.assertEqual(self.queue.intents, {})
        self.assertEqual(self.eagle.add_calls, [])

    def test_startup_reconciliation_is_stable_and_idempotent(self) -> None:
        importer = self.importer()

        first = importer.reconcile_startup(self.root, stable_check=lambda _path: True)
        second = importer.reconcile_startup(self.root, stable_check=lambda _path: True)

        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].intent_id, second[0].intent_id)
        self.assertEqual(len(self.queue.intents), 1)

    def test_file_stability_requires_two_matching_regular_file_samples(self) -> None:
        samples = iter(
            [
                (10, 100, True),
                (11, 101, True),
            ]
        )

        self.assertFalse(
            file_is_stable(
                self.image,
                stat_sample=lambda _path: next(samples),
                sleep=lambda _seconds: None,
            )
        )
        self.assertTrue(file_is_stable(self.image, sleep=lambda _seconds: None))


class EagleApiContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_application_info_validates_versioned_object(self) -> None:
        async def handler(_method: str, _url: str, _kwargs) -> FakeHttpResponse:
            return FakeHttpResponse(
                {"status": "success", "data": {"version": "4.0.0", "preferences": {}}}
            )

        info = await EagleApiClient(http_client=FakeHttpClient(handler)).application_info()

        self.assertEqual(info["version"], "4.0.0")

    async def test_get_item_and_add_from_path_validate_success_shapes(self) -> None:
        async def handler(_method: str, url: str, _kwargs) -> FakeHttpResponse:
            if url.endswith("/api/item/info"):
                return FakeHttpResponse(
                    {"status": "success", "data": {"id": "item-1", "annotation": "", "lastModified": 1}},
                )
            if url.endswith("/api/item/addFromPath"):
                return FakeHttpResponse({"status": "success", "data": {"id": "item-1"}})
            raise AssertionError(url)

        api = EagleApiClient(http_client=FakeHttpClient(handler))
        item = await api.get_item("item-1")
        eagle_id = await api.add_from_path(
            path="/tmp/image.png",
            name="image",
            annotation="marker",
            source="generator",
            tags=("ai-generated",),
        )

        self.assertEqual(item["id"], "item-1")
        self.assertEqual(eagle_id, "item-1")

    async def test_malformed_success_is_rejected(self) -> None:
        async def handler(_method: str, _url: str, _kwargs) -> FakeHttpResponse:
            return FakeHttpResponse({"status": "success", "data": []})

        api = EagleApiClient(http_client=FakeHttpClient(handler))
        with self.assertRaises(EagleProtocolError):
            await api.get_item("item-1")

    async def test_add_timeout_is_classified_as_ambiguous(self) -> None:
        async def handler(_method: str, _url: str, _kwargs) -> FakeHttpResponse:
            raise TimeoutError("late")

        api = EagleApiClient(http_client=FakeHttpClient(handler))
        with self.assertRaises(EagleAmbiguousCommitError):
            await api.add_from_path(
                path="/tmp/image.png",
                name="image",
                annotation="marker",
                source="generator",
                tags=(),
            )


if __name__ == "__main__":
    unittest.main()
