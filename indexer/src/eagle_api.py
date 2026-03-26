"""Eagle API client — communicates with Eagle's local REST API (port 41595)."""

from urllib.parse import unquote

import httpx

EAGLE_API = "http://localhost:41595"


async def is_eagle_running() -> bool:
    """Check if Eagle API is accessible."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{EAGLE_API}/api/application/info", timeout=3)
            data = resp.json()
            return data.get("status") == "success"
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


async def list_all_items() -> list[dict]:
    """Fetch all items from Eagle library."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{EAGLE_API}/api/item/list",
            params={"limit": 10000},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "success":
            raise RuntimeError(f"Eagle API error: {data}")
        return data["data"]


async def get_thumbnail_path(item_id: str) -> str | None:
    """Get decoded filesystem path to item's thumbnail."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{EAGLE_API}/api/item/thumbnail",
            params={"id": item_id},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "success":
            return None
        return unquote(data["data"])


async def get_folder_map() -> dict[str, str]:
    """Get folder_id → folder_name mapping (flattened from nested tree)."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{EAGLE_API}/api/folder/list", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "success":
            return {}

    folder_map: dict[str, str] = {}

    def flatten(folders: list[dict]) -> None:
        for f in folders:
            folder_map[f["id"]] = f["name"]
            if f.get("children"):
                flatten(f["children"])

    flatten(data["data"])
    return folder_map
