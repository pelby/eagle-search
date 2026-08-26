"""Safe, bounded Codex CLI caption provider.

The provider is intentionally process-only: it neither reads keys nor falls back to
another provider.  Auth/quota errors stop a batch; per-image errors are bounded and
remain retryable at the job-store layer.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Protocol

from ..contracts import CaptionReceiptV1, CaptionResultV1, ContractError
from ..ports import ReceiptStore
from .prompt import CAPTION_PROMPT_VERSION, caption_prompt, output_schema_path


ALLOWED_MODELS = frozenset({"gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"})
ALLOWED_EFFORTS = frozenset({"low", "medium"})
CAPTION_PROVIDER_NAME = "codex-cli"
CAPTION_INPUT_PREPARATION_VERSION = "svg-white-raster-v1"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_SVG_TAG = re.compile(br"<(?:[A-Za-z_][A-Za-z0-9_.-]*:)?svg(?:[\s>/])", re.IGNORECASE)
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


def _contains_svg_content(path: Path) -> bool:
    """Sniff bounded source bytes because Eagle cache suffixes are not authoritative."""

    with path.open("rb") as source:
        prefix = source.read(64 * 1024)
    if prefix.startswith(
        (
            _PNG_SIGNATURE,
            b"\xff\xd8\xff",
            b"GIF87a",
            b"GIF89a",
            b"II*\x00",
            b"MM\x00*",
            b"BM",
            b"\x00\x00\x01\x00",
        )
    ):
        return False
    if prefix.startswith(b"RIFF") and prefix[8:12] == b"WEBP":
        return False
    if len(prefix) >= 12 and prefix[4:8] == b"ftyp":
        return False
    return _SVG_TAG.search(prefix) is not None


class CodexCliCaptionProvider:
    """One-shot structured vision captioning through an authenticated CLI session."""

    provider_name = CAPTION_PROVIDER_NAME

    def __init__(
        self,
        *,
        command: str = "codex",
        timeout_seconds: int = 90,
        retries: int = 1,
        run: SubprocessRun = subprocess.run,
        convert_run: SubprocessRun = subprocess.run,
        conversion_timeout_seconds: int = 30,
        now: Callable[[], str] = _utc_now,
    ) -> None:
        if timeout_seconds <= 0 or conversion_timeout_seconds <= 0 or retries < 0:
            raise ValueError("timeouts must be positive and retries cannot be negative")
        self.command = command
        self.timeout_seconds = timeout_seconds
        self.conversion_timeout_seconds = conversion_timeout_seconds
        self.retries = retries
        self._run = run
        self._convert_run = convert_run
        self._now = now

    @contextmanager
    def _prepared_image(self, image_path: Path) -> Iterator[Path]:
        try:
            is_svg = _contains_svg_content(image_path)
        except OSError as error:
            raise CaptionItemFailure(f"thumbnail could not be read: {_safe_error(str(error))}") from error
        if not is_svg:
            yield image_path
            return

        try:
            temporary_context = tempfile.TemporaryDirectory(prefix="eagle-search-svg-")
        except OSError as error:
            raise CaptionItemFailure(f"SVG conversion could not create private storage: {_safe_error(str(error))}") from error
        with temporary_context as temporary:
            temporary_root = Path(temporary)
            try:
                os.chmod(temporary_root, 0o700)
            except OSError as error:
                raise CaptionItemFailure("SVG conversion could not secure private storage") from error
            prepared = temporary_root / "prepared.png"
            argv = [
                "rsvg-convert",
                "--format",
                "png",
                "--background-color",
                "white",
                "--output",
                str(prepared),
                str(image_path),
            ]
            try:
                completed = self._convert_run(
                    argv,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.conversion_timeout_seconds,
                    shell=False,
                )
            except subprocess.TimeoutExpired as error:
                raise CaptionItemFailure(
                    f"SVG conversion timed out after {self.conversion_timeout_seconds}s"
                ) from error
            except OSError as error:
                raise CaptionItemFailure(f"SVG conversion failed: {_safe_error(str(error))}") from error
            if completed.returncode != 0:
                detail = " ".join(f"{completed.stderr}\n{completed.stdout}".split())[:500] or "no error detail"
                raise CaptionItemFailure(f"SVG conversion failed: {detail}")
            try:
                with prepared.open("rb") as converted:
                    signature = converted.read(len(_PNG_SIGNATURE))
            except OSError as error:
                raise CaptionItemFailure("SVG conversion did not produce a valid PNG") from error
            if signature != _PNG_SIGNATURE:
                raise CaptionItemFailure("SVG conversion did not produce a valid PNG")
            try:
                os.chmod(prepared, 0o600)
            except OSError as error:
                raise CaptionItemFailure("SVG conversion could not secure the prepared PNG") from error
            yield prepared

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
            "--ignore-user-config",
            "--ignore-rules",
            "--disable",
            "plugins",
            "--disable",
            "apps",
            "--disable",
            "memories",
            "--disable",
            "skill_search",
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
        original_argv = self.argv(image_path, model=model, effort=effort)
        try:
            original_hash = image_sha256(image_path)
        except OSError as error:
            raise CaptionItemFailure(f"thumbnail could not be read: {_safe_error(str(error))}") from error
        with self._prepared_image(image_path) as prepared_image:
            argv = (
                original_argv
                if prepared_image == image_path
                else self.argv(prepared_image, model=model, effort=effort)
            )
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
                                image_hash=original_hash,
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
