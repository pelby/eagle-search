"""Strict, bounded client for Eagle's local REST API."""

from __future__ import annotations

from typing import Any
from urllib.parse import unquote

try:  # Keep dependency-free fake-service tests runnable with plain python3.
    import httpx  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - environment dependent.
    httpx = None  # type: ignore


EAGLE_API = "http://localhost:41595"


class EagleApiError(RuntimeError):
    """A definite transport, HTTP or application-level Eagle failure."""


class EagleProtocolError(EagleApiError):
    """Eagle returned a success envelope with an unsafe shape."""


class EagleAmbiguousCommitError(EagleApiError):
    """A mutating request timed out after Eagle may have committed it."""


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    return httpx is not None and isinstance(exc, httpx.TimeoutException)


class EagleApiClient:
    """Validate every Eagle response before exposing it to safety workflows."""

    def __init__(
        self,
        *,
        http_client: Any | None = None,
        base_url: str = EAGLE_API,
        read_timeout: float = 10.0,
        write_timeout: float = 20.0,
    ) -> None:
        if read_timeout <= 0 or write_timeout <= 0:
            raise ValueError("Eagle timeouts must be positive")
        self.http_client = http_client
        self.base_url = base_url.rstrip("/")
        self.read_timeout = min(read_timeout, 30.0)
        self.write_timeout = min(write_timeout, 30.0)

    async def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        ambiguous_on_timeout: bool = False,
    ) -> dict[str, Any]:
        timeout = self.write_timeout if method == "POST" else self.read_timeout
        try:
            if self.http_client is not None:
                response = await self.http_client.request(
                    method,
                    f"{self.base_url}{endpoint}",
                    params=params,
                    json=json_body,
                    timeout=timeout,
                )
            else:
                if httpx is None:
                    raise EagleApiError("httpx is required for live Eagle API calls")
                async with httpx.AsyncClient() as client:
                    response = await client.request(
                        method,
                        f"{self.base_url}{endpoint}",
                        params=params,
                        json=json_body,
                        timeout=timeout,
                    )
            response.raise_for_status()
        except Exception as exc:
            if _is_timeout(exc) and ambiguous_on_timeout:
                raise EagleAmbiguousCommitError(
                    f"{endpoint} timed out after Eagle may have committed"
                ) from exc
            if isinstance(exc, EagleApiError):
                raise
            raise EagleApiError(f"Eagle request failed for {endpoint}: {exc}") from exc
        try:
            payload = response.json()
        except Exception as exc:
            raise EagleProtocolError(f"Eagle returned non-JSON for {endpoint}") from exc
        if not isinstance(payload, dict):
            raise EagleProtocolError(f"Eagle response for {endpoint} is not an object")
        if payload.get("status") != "success":
            raise EagleApiError(f"Eagle application error for {endpoint}: {payload}")
        return payload

    async def is_running(self) -> bool:
        try:
            await self._request("GET", "/api/application/info")
            return True
        except EagleApiError:
            return False

    async def get_item(self, item_id: str) -> dict[str, Any]:
        if not item_id:
            raise ValueError("item ID cannot be blank")
        payload = await self._request("GET", "/api/item/info", params={"id": item_id})
        data = payload.get("data")
        if not isinstance(data, dict) or data.get("id") != item_id:
            raise EagleProtocolError("Eagle item/info data does not match the requested item")
        annotation = data.get("annotation")
        last_modified = data.get("lastModified")
        if not isinstance(annotation, str):
            raise EagleProtocolError("Eagle item annotation is not text")
        if isinstance(last_modified, bool) or not isinstance(last_modified, int):
            raise EagleProtocolError("Eagle item lastModified is not an integer")
        return dict(data)

    async def update_item(
        self,
        item_id: str,
        *,
        annotation: str | None = None,
        tags: tuple[str, ...] | None = None,
        source: str | None = None,
    ) -> None:
        if not item_id:
            raise ValueError("item ID cannot be blank")
        body: dict[str, Any] = {"id": item_id}
        if annotation is not None:
            if not isinstance(annotation, str):
                raise TypeError("annotation must be text")
            body["annotation"] = annotation
        if tags is not None:
            if any(not isinstance(tag, str) for tag in tags):
                raise TypeError("tags must be text")
            body["tags"] = list(tags)
        if source is not None:
            if not isinstance(source, str):
                raise TypeError("source must be text")
            body["url"] = source
        if len(body) == 1:
            raise ValueError("update_item requires at least one changed field")
        await self._request(
            "POST",
            "/api/item/update",
            json_body=body,
            ambiguous_on_timeout=True,
        )

    async def add_from_path(
        self,
        *,
        path: str,
        name: str,
        annotation: str,
        source: str,
        tags: tuple[str, ...],
    ) -> str:
        if not path or not name:
            raise ValueError("path and name cannot be blank")
        if any(not isinstance(value, str) for value in (annotation, source, *tags)):
            raise TypeError("annotation, source and tags must be text")
        payload = await self._request(
            "POST",
            "/api/item/addFromPath",
            json_body={
                "path": path,
                "name": name,
                "annotation": annotation,
                "website": source,
                "tags": list(tags),
            },
            ambiguous_on_timeout=True,
        )
        data = payload.get("data")
        if isinstance(data, str):
            eagle_id = data
        elif isinstance(data, dict):
            eagle_id = data.get("id")
        else:
            eagle_id = None
        if not isinstance(eagle_id, str) or not eagle_id:
            raise EagleProtocolError("Eagle addFromPath did not return an item ID")
        return eagle_id

    async def list_items(self, *, limit: int = 10_000) -> list[dict[str, Any]]:
        if not 1 <= limit <= 10_000:
            raise ValueError("item list limit must be between 1 and 10000")
        payload = await self._request("GET", "/api/item/list", params={"limit": limit})
        data = payload.get("data")
        if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
            raise EagleProtocolError("Eagle item/list data is not an array of objects")
        if any(not isinstance(item.get("id"), str) or not item["id"] for item in data):
            raise EagleProtocolError("Eagle item/list contains an invalid item ID")
        return [dict(item) for item in data]

    async def list_recent(self, *, limit: int = 200) -> list[dict[str, Any]]:
        if not 1 <= limit <= 1_000:
            raise ValueError("recent item limit must be between 1 and 1000")
        payload = await self._request(
            "GET",
            "/api/item/list",
            params={"limit": limit, "orderBy": "-CREATEDATE"},
        )
        data = payload.get("data")
        if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
            raise EagleProtocolError("Eagle recent-item data is not an array of objects")
        items: list[dict[str, Any]] = []
        for item in data:
            if not isinstance(item.get("id"), str) or not isinstance(item.get("annotation", ""), str):
                raise EagleProtocolError("Eagle recent item has invalid ID or annotation")
            items.append(dict(item))
        return items

    async def get_thumbnail_path(self, item_id: str) -> str | None:
        payload = await self._request("GET", "/api/item/thumbnail", params={"id": item_id})
        data = payload.get("data")
        if data is None:
            return None
        if not isinstance(data, str):
            raise EagleProtocolError("Eagle thumbnail path is not text")
        return unquote(data)

    async def get_folder_map(self) -> dict[str, str]:
        payload = await self._request("GET", "/api/folder/list")
        data = payload.get("data")
        if not isinstance(data, list):
            raise EagleProtocolError("Eagle folder/list data is not an array")
        folder_map: dict[str, str] = {}

        def flatten(folders: list[Any]) -> None:
            for folder in folders:
                if not isinstance(folder, dict):
                    raise EagleProtocolError("Eagle folder is not an object")
                folder_id, name = folder.get("id"), folder.get("name")
                if not isinstance(folder_id, str) or not isinstance(name, str):
                    raise EagleProtocolError("Eagle folder has invalid ID or name")
                folder_map[folder_id] = name
                children = folder.get("children", [])
                if not isinstance(children, list):
                    raise EagleProtocolError("Eagle folder children is not an array")
                flatten(children)

        flatten(data)
        return folder_map


async def is_eagle_running() -> bool:
    return await EagleApiClient().is_running()


async def list_all_items() -> list[dict[str, Any]]:
    return await EagleApiClient().list_items(limit=10_000)


async def get_thumbnail_path(item_id: str) -> str | None:
    return await EagleApiClient().get_thumbnail_path(item_id)


async def get_folder_map() -> dict[str, str]:
    return await EagleApiClient().get_folder_map()
