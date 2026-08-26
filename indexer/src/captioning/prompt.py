"""The versioned, vision-only caption prompt and its frozen response schema."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


# v3 includes the visual-input contract: SVG payloads stored under misleading
# Eagle cache suffixes are rasterised onto white before the model sees them.
# Bumping the receipt identity prevents reuse of captions made from absent or
# incorrectly composited pixels under the earlier provider boundary.
CAPTION_PROMPT_VERSION = "caption-v3"
_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "caption-result-v1.schema.json"


def caption_prompt() -> str:
    """Return the sole prompt supplied with a thumbnail.

    The provider deliberately receives no library metadata.  That keeps captions
    visual evidence rather than a restatement of potentially unrelated context.
    """

    return (
        "Describe this image for a local visual search index using only visible evidence. "
        "Record image and diagram type, concrete subjects, style, colours, layout, "
        "legible visible text, and compact search aliases. Keep the summary strictly literal. "
        "For search_terms, include six to twelve concise aliases covering visible objects, "
        "actions, composition, and broader activity, setting, or function when the visual "
        "arrangement directly suggests it. For example, a figure pointing at a board or easel "
        "can support presentation, teaching, classroom, or demonstration aliases even when the "
        "literal room type is uncertain. These are retrieval aliases, not factual claims. "
        "Transcribe text only when it is visibly legible. Do not infer unseen details, identities, "
        "brands, intent, or relationships in factual fields. Put ambiguity in uncertainties. "
        "Return only the required JSON."
    )


@lru_cache(maxsize=1)
def output_schema() -> dict[str, Any]:
    """Load the G0-frozen JSON Schema without mutating it."""

    with _SCHEMA_PATH.open(encoding="utf-8") as schema_file:
        return json.load(schema_file)


def output_schema_path() -> Path:
    """The absolute schema path passed directly to ``codex exec``."""

    return _SCHEMA_PATH
