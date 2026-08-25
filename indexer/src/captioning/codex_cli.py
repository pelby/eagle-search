"""Safe, bounded Codex CLI caption provider.

The provider is intentionally process-only: it neither reads keys nor falls back to
another provider.  Auth/quota errors stop a batch; per-image errors are bounded and
remain retryable at the job-store layer.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from ..contracts import CaptionReceiptV1, CaptionResultV1, ContractError
from ..ports import ReceiptStore
from .prompt import CAPTION_PROMPT_VERSION, caption_prompt, output_schema_path


ALLOWED_MODELS = frozenset({"gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"})
ALLOWED_EFFORTS = frozenset({"low", "medium"})
_GLOBAL_ERROR_MARKERS = (
    "login",
    "log in",
    "authentication",
    "unauthorized",
    "forbidden",
    "quota",
    "subscription",
    "billing",
)


class CaptionFailure(RuntimeError):
    """A provider result unsuitable for a caption receipt."""


class CaptionGlobalFailure(CaptionFailure):
    """Auth/quota failures: stop the run rather than repeat the same call."""


class CaptionItemFailure(CaptionFailure):
    """One image failed and can be retried later by the job store."""


class SubprocessRun(Protocol):
    def __call__(self, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]: ...


def image_sha256(path: Path) -> str:
    """Return the content hash used in immutable receipt identity."""

    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for block in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_error(value: str) -> str:
    """Keep operational errors actionable without retaining unbounded subprocess text."""

    return " ".join(value.split())[:500] or "Codex CLI returned no error detail"


def _is_global_error(message: str) -> bool:
    text = message.casefold()
    return any(marker in text for marker in _GLOBAL_ERROR_MARKERS)


class CodexCliCaptionProvider:
    """One-shot structured vision captioning through an authenticated CLI session."""

    provider_name = "codex-cli"

    def __init__(
        self,
        *,
        command: str = "codex",
        timeout_seconds: int = 90,
        retries: int = 1,
        run: SubprocessRun = subprocess.run,
        now: Callable[[], str] = _utc_now,
    ) -> None:
        if timeout_seconds <= 0 or retries < 0:
            raise ValueError("timeout_seconds must be positive and retries cannot be negative")
        self.command = command
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self._run = run
        self._now = now

    def argv(self, image_path: Path, *, model: str, effort: str) -> list[str]:
        """Build argv directly; never delegate parsing to a shell."""

        if model not in ALLOWED_MODELS:
            raise ValueError(f"unsupported Codex caption model: {model}")
        if effort not in ALLOWED_EFFORTS:
            raise ValueError(f"unsupported Codex caption effort: {effort}")
        return [
            self.command,
            "exec",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "-m",
            model,
            "-c",
            f'model_reasoning_effort="{effort}"',
            "-i",
            str(image_path),
            "--output-schema",
            str(output_schema_path()),
            "--color",
            "never",
            caption_prompt(),
        ]

    def caption(
        self,
        image_path: Path,
        *,
        model: str,
        effort: str,
        receipt_store: ReceiptStore,
    ) -> CaptionReceiptV1:
        """Caption one local image, then hand its immutable receipt to the store port."""

        image_path = Path(image_path)
        if not image_path.is_file():
            raise CaptionItemFailure("thumbnail is missing or not a regular file")
        argv = self.argv(image_path, model=model, effort=effort)
        last_error: CaptionItemFailure | None = None
        for attempt in range(self.retries + 1):
            try:
                completed = self._run(
                    argv,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    shell=False,
                )
            except subprocess.TimeoutExpired as error:
                last_error = CaptionItemFailure(f"Codex CLI timed out after {self.timeout_seconds}s")
            except OSError as error:
                message = _safe_error(str(error))
                if _is_global_error(message):
                    raise CaptionGlobalFailure(message) from error
                last_error = CaptionItemFailure(message)
            else:
                if completed.returncode != 0:
                    message = _safe_error(f"{completed.stderr}\n{completed.stdout}")
                    if _is_global_error(message):
                        raise CaptionGlobalFailure(message)
                    last_error = CaptionItemFailure(message)
                else:
                    try:
                        payload = json.loads(completed.stdout.strip())
                        caption = CaptionResultV1.from_dict(payload)
                    except (json.JSONDecodeError, ContractError) as error:
                        last_error = CaptionItemFailure(f"invalid structured caption: {_safe_error(str(error))}")
                    else:
                        receipt = CaptionReceiptV1.create(
                            image_hash=image_sha256(image_path),
                            caption_result=caption,
                            provider=self.provider_name,
                            model=model,
                            effort=effort,
                            prompt_version=CAPTION_PROMPT_VERSION,
                            created_at=self._now(),
                        )
                        stored_id = receipt_store.put_immutable(receipt)
                        if stored_id != receipt.receipt_id:
                            raise CaptionItemFailure("receipt store returned an unexpected receipt id")
                        return receipt
            if attempt == self.retries:
                break
        assert last_error is not None
        raise last_error
