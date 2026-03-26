"""CLI entry point for eagle-search indexer."""

import asyncio
import shutil
import sys
from pathlib import Path

from . import db, describe, eagle_api

THUMBNAILS_DIR = Path.home() / ".eagle-search" / "thumbnails"


async def _process_item(
    item: dict,
    folder_map: dict[str, str],
    sem: asyncio.Semaphore,
    index: int,
    total: int,
) -> dict | None:
    """Process a single Eagle item: copy thumbnail, describe, return db record."""
    async with sem:
        eagle_id = item["id"]
        name = item.get("name", "")
        print(f"  [{index}/{total}] {name}...", flush=True)

        # Get thumbnail path from Eagle API
        thumb_src = await eagle_api.get_thumbnail_path(eagle_id)
        if not thumb_src:
            print(f"    ⚠ No thumbnail for {name}", flush=True)
            thumb_src = ""

        # Copy thumbnail to local cache
        thumb_dest = ""
        image_path = ""
        if thumb_src and Path(thumb_src).exists():
            thumb_dest = str(THUMBNAILS_DIR / f"{eagle_id}.png")
            try:
                shutil.copy2(thumb_src, thumb_dest)
            except OSError as e:
                print(f"    ⚠ Failed to copy thumbnail: {e}", flush=True)
                thumb_dest = thumb_src  # fall back to original path

            # Derive original image path (remove _thumbnail suffix)
            image_path = thumb_src.replace("_thumbnail.png", f".{item.get('ext', 'png')}")

        # Get AI description
        ai_desc = ""
        describe_path = thumb_dest or thumb_src
        if describe_path and Path(describe_path).exists():
            try:
                ai_desc = await describe.describe_image(describe_path)
            except Exception as e:
                print(f"    ⚠ Description failed: {e}", flush=True)
        else:
            print(f"    ⚠ No image to describe", flush=True)

        # Resolve folder names
        folder_ids = item.get("folders", [])
        folder_names = [folder_map.get(fid, "") for fid in folder_ids]
        folder_name = ", ".join(f for f in folder_names if f)

        # Build tags string
        tags = ", ".join(item.get("tags", []))

        return {
            "eagle_id": eagle_id,
            "name": name,
            "tags": tags,
            "annotation": item.get("annotation", ""),
            "ai_description": ai_desc,
            "thumbnail_path": thumb_dest or thumb_src or "",
            "image_path": image_path,
            "folder_name": folder_name,
            "ext": item.get("ext", ""),
            "width": item.get("width", 0),
            "height": item.get("height", 0),
            "created_at": item.get("btime", 0),
        }


async def cmd_index(force: bool = False) -> None:
    """Index Eagle library. Skip already-indexed items unless force=True."""
    # Check Eagle is running
    if not await eagle_api.is_eagle_running():
        print("✘ Eagle is not running.  Open Eagle and try again.")
        sys.exit(1)

    print("Connecting to Eagle API...")
    THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)

    # Get all items and folder map
    items = await eagle_api.list_all_items()
    folder_map = await eagle_api.get_folder_map()
    print(f"Found {len(items)} items in Eagle library")

    # Filter to unindexed items
    conn = db.init_db()
    if not force:
        existing = db.get_indexed_ids(conn)
        new_items = [i for i in items if i["id"] not in existing]
        print(f"Already indexed: {len(existing)}, new: {len(new_items)}")
    else:
        new_items = items
        print(f"Force re-indexing all {len(new_items)} items")

    if not new_items:
        print("Nothing to index.  All items are up to date.")
        conn.close()
        return

    # Process items with concurrency limit
    sem = asyncio.Semaphore(5)
    tasks = [
        _process_item(item, folder_map, sem, i + 1, len(new_items))
        for i, item in enumerate(new_items)
    ]
    results = await asyncio.gather(*tasks)

    # Write to database
    success = 0
    for record in results:
        if record:
            db.upsert_image(conn, record)
            success += 1

    conn.close()
    print(f"\n✔ Indexed {success} new images ({success + len(items) - len(new_items)} total)")


def cmd_stats() -> None:
    """Show index statistics."""
    conn = db.init_db()
    s = db.stats(conn)
    conn.close()
    print(f"Total indexed:  {s['total']}")
    print(f"With AI desc:   {s['described']}")
    print(f"Last indexed:   {s['last_indexed']}")


def cmd_search(query: str) -> None:
    """Test search from CLI."""
    conn = db.init_db()
    results = db.search(conn, query)
    conn.close()

    if not results:
        print(f"No results for '{query}'")
        return

    print(f"Results for '{query}' ({len(results)} found):\n")
    for r in results:
        desc_preview = (r["ai_description"] or "")[:80]
        print(f"  {r['name']}")
        print(f"    Folder: {r['folder_name']}  Tags: {r['tags']}")
        if desc_preview:
            print(f"    Desc: {desc_preview}...")
        print()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m src.cli <command> [args]")
        print("Commands: index, reindex, stats, search <query>")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "index":
        asyncio.run(cmd_index(force=False))
    elif cmd == "reindex":
        asyncio.run(cmd_index(force=True))
    elif cmd == "stats":
        cmd_stats()
    elif cmd == "search":
        if len(sys.argv) < 3:
            print("Usage: python -m src.cli search <query>")
            sys.exit(1)
        cmd_search(" ".join(sys.argv[2:]))
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
