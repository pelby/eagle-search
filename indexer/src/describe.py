"""Generate rich image descriptions via OpenRouter (Gemini Flash)."""

import base64
import os

import httpx

OPENROUTER_API = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.5-flash"

DESCRIPTION_PROMPT = """Describe this image for a search index. Be comprehensive but concise (100-200 words).

Include ALL of the following:
- Image type: photo, diagram, illustration, screenshot, logo, icon, infographic, chart, wireframe, mockup, etc.
- Specific diagram type if applicable: Venn diagram, flowchart, org chart, mind map, timeline, staircase, matrix, etc.
- Visual style: hand-drawn, sketchy, minimal, detailed, flat, 3D, realistic, cartoon, Excalidraw-style, etc.
- Color palette: black and white, monochrome, colorful, orange, teal, muted, vibrant, etc.
- ALL text visible in the image: transcribe labels, titles, headings, captions word-for-word
- Subject matter and concepts depicted
- Layout: grid, centered, overlapping circles, stacked, side-by-side, horizontal, vertical, etc.
- 3-5 alternative search terms someone might use to find this image

Format as flowing text, not bullet points. Pack in keywords naturally."""


def _get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set.  "
            "Add it to ~/.secrets or export it in your shell."
        )
    return key


def _encode_image(path: str) -> str:
    """Read image file and return base64-encoded string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _mime_type(path: str) -> str:
    ext = path.rsplit(".", 1)[-1].lower()
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
        "svg": "image/svg+xml",
    }.get(ext, "image/png")


async def describe_image(image_path: str) -> str:
    """Send image to Gemini Flash via OpenRouter, return description."""
    api_key = _get_api_key()
    img_b64 = _encode_image(image_path)
    mime = _mime_type(image_path)

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": DESCRIPTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{img_b64}",
                        },
                    },
                ],
            }
        ],
        "max_tokens": 500,
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            OPENROUTER_API,
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

    choices = data.get("choices", [])
    if not choices:
        return ""

    return choices[0].get("message", {}).get("content", "").strip()
