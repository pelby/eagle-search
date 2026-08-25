"""Caption providers and fixed prompt material.

This package intentionally contains no credential handling and no network client.
The default provider delegates to the already-authenticated Codex CLI.
"""

from .codex_cli import CodexCliCaptionProvider

__all__ = ["CodexCliCaptionProvider"]
