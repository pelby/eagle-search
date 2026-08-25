"""The versioned, vision-only caption prompt and its frozen response schema."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


CAPTION_PROMPT_VERSION = "caption-v1"
_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "caption-result-v1.schema.json"


def caption_prompt() -> str:
    """Return the sole prompt supplied with a thumbnail.

    The provider deliberately receives no library metadata.  That keeps captions
    visual evidence rather than a restatement of potentially unrelated context.
    """

    return (
        "Describe this image for a local visual search index using only visible evidence. "
        "Record image and diagram type, concrete subjects, style, colours, layout, "
        "legible visible text, and compact search aliases. Transcribe text only when "
        "it is visibly legible. Do not infer unseen details, identities, brands, intent, "
        "or relationships. Put ambiguity in uncertainties. Return only the required JSON."
    )


@lru_cache(maxsize=1)
def output_schema() -> dict[str, Any]:
    """Load the G0-frozen JSON Schema without mutating it."""

    with _SCHEMA_PATH.open(encoding="utf-8") as schema_file:
        return json.load(schema_file)


def output_schema_path() -> Path:
    """The absolute schema path passed directly to ``codex exec``."""

    return _SCHEMA_PATH
