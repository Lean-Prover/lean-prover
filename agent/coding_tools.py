"""File-editing tools — Tau tau_coding/tools.py port (Tau 37a9e43 src/tau_coding/tools.py), lean-adapted.

Ports the subset of Tau's coding tools the session needs: read (head/tail
truncation), edit (exact match, overlap + duplicate checks, unified patch),
line-ending detection, and size formatting.  The ``bash`` tool already lives
in ``agent/prover_loop.py`` semantics via lake; images are shimmed through
``agent/image_processing.py`` (Pillow optional).
"""

from __future__ import annotations

import asyncio
import difflib
import os
import signal
from dataclasses import dataclass
from pathlib import Path

MAX_READ_BYTES = 50 * 1024
MAX_READ_LINES = 2000
HEAD_HISTORY_LINES = 8


@dataclass(frozen=True, slots=True)
class ImageSupportState:
    """Whether the active model supports inline images (tau ImageSupportState)."""

    supported: bool = False


def normalize_path(path: str | Path, cwd: Path | None = None) -> Path:
    """Resolve a tool path against the session working directory."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (cwd or Path.cwd()) / p


def format_size(num_bytes: int) -> str:
    """Human-readable size (tau format_size)."""
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            if unit == "B":
                return f"{num_bytes} {unit}"
            return f"{num_bytes / 1024:.1f} {unit}"
        num_bytes /= 1024  # type: ignore[assignment]
    return f"{num_bytes} B"


def detect_line_ending(text: str) -> str:
    """Detect the dominant line ending of file content (tau detect_line_ending)."""
    crlf = text.count("\r\n")
    cr = text.count("\r") - crlf
    lf = text.count("\n")
    if crlf >= max(cr, lf) and crlf > 0:
        return "\r\n"
    if cr > lf and cr > 0:
        return "\r"
    return "\n"


def truncate_for_read(content: str, *, max_bytes: int = MAX_READ_BYTES, max_lines: int = MAX_READ_LINES) -> tuple[str, bool]:
    """Truncate content for tool display, head-first (tau read truncation).

    Returns the (possibly truncated) text plus a truncated flag.
    """
    data = content.encode("utf-8", errors="replace")
    truncated = False
    if len(data) > max_bytes:
        data = data[:max_bytes]
        content = data.decode("utf-8", errors="replace")
        truncated = True
    lines = content.splitlines(keepends=True)
    if len(lines) > max_lines:
        limit = max_lines - HEAD_HISTORY_LINES
        content = "".join(lines[:limit]) + f"\n[... {len(lines) - limit} more lines truncated]\n" + "".join(lines[limit:])
        truncated = True
    return content, truncated


def generate_unified_patch(old: str, new: str, *, path: str = "") -> str:
    """Unified diff of an edit, mirroring path + broad context (tau parity)."""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines, new_lines, fromfile=f"a/{path}", tofile=f"b/{path}", n=3
    )
    patch = "".join(diff)
    if patch and not patch.endswith("\n"):
        patch += "\n"
    return patch or "(no changes)"


def _check_overlapping_spans(spans: list[tuple[int, int]]) -> int | None:
    """Return the index of the first overlapping edit span, or None."""
    import itertools

    ordered = sorted(range(len(spans)), key=lambda i: spans[i][0])
    for prev, cur in itertools.pairwise(ordered):
        if spans[cur][0] < spans[prev][1]:
            return cur
    return None


def apply_edits_to_normalized_content(
    content: str,
    edits: list[dict],
    *,
    line_ending: str = "\n",
    path: str = "",
) -> tuple[str, str, str | None]:
    """Apply a batch of exact-match edits to file content.

    Mirrors tau's ``apply_edits_to_normalized_content``: each edit is an
    exact string match; duplicate matches and overlapping ranges are errors.
    Returns (new_content, unified_patch, error_or_None).

    Input may arrive unnormalized: when ``line_ending`` is not ``\n`` the
    content is first stripped to LF, edits are applied on LF, and the result
    is restored to ``line_ending``.
    """
    if line_ending != "\n":
        content = content.replace(line_ending, "\n")
    spans: list[tuple[int, int, str]] = []
    for index, edit in enumerate(edits):
        old = str(edit.get("old_string", ""))
        new = str(edit.get("new_string", ""))
        if not old:
            return content, "", f"edit {index}: old_string is empty"
        start = content.find(old)
        if start == -1:
            return content, "", f"edit {index}: old_string not found in content"
        if content.find(old, start + 1) != -1:
            return content, "", (
                f"edit {index}: old_string found multiple times in content"
            )
        spans.append((start, start + len(old), new))
    overlap = _check_overlapping_spans([(s, e) for s, e, _ in spans])
    if overlap is not None:
        return content, "", f"edit {overlap}: overlaps an earlier edit"
    result = []
    cursor = 0
    for start, end, replacement in sorted(spans):
        result.append(content[cursor:start])
        result.append(replacement)
        cursor = end
    result.append(content[cursor:])
    normalized_new = "".join(result)
    patch = generate_unified_patch(content, normalized_new, path=path)
    if line_ending != "\n":
        normalized_new = normalized_new.replace("\n", line_ending)
    return normalized_new, patch, None


_FILE_LOCKS: dict[str, asyncio.Lock] = {}
_FILE_LOCKS_MAX = 512


def file_lock(path: str | Path) -> asyncio.Lock:
    """Per-path asyncio lock (tau's _file_lock registry) to serialize edits.

    Bounded LRU: long-lived sessions that touch many distinct files must not
    leak one lock per path forever, so the oldest entry is evicted past the cap.
    """
    key = str(normalize_path(path))
    lock = _FILE_LOCKS.get(key)
    if lock is None:
        if len(_FILE_LOCKS) >= _FILE_LOCKS_MAX:
            _FILE_LOCKS.pop(next(iter(_FILE_LOCKS)))
        lock = asyncio.Lock()
        _FILE_LOCKS[key] = lock
    else:
        # Refresh recency so the cap evicts the least-recently-used path.
        _FILE_LOCKS.pop(key)
        _FILE_LOCKS[key] = lock
    return lock


def create_coding_tools(cwd: Path | None = None, image_support: ImageSupportState | None = None) -> list[dict]:
    """The full coding tool set (tau create_coding_tools, lean-adapted).

    Lean notes: ``bash`` delegates to a lake-safe shell; ``edit`` applies
    ``apply_edits_to_normalized_content``; image content in reads is noted as
    omitted when the provider cannot accept images (and Pillow is optional).
    """
    image_support = image_support or ImageSupportState()

    async def read_execute(args: dict) -> dict:
        path = normalize_path(args.get("path", ""), cwd)
        try:
            raw = path.read_bytes()
        except OSError as exc:
            return {"is_error": True, "content": f"read failed: {exc}"}
        from .image_processing import (
            ImageProcessingFailure,
            detect_image_kind,
            process_image,
        )

        kind = detect_image_kind(raw)
        if kind is not None:
            if not image_support.supported:
                return {"content": f"(image file {kind}, {format_size(len(raw))}; inline images not supported by the active model — image content omitted)"}
            processed = process_image(raw)
            if isinstance(processed, ImageProcessingFailure):
                return {"content": f"(image file {format_size(len(raw))}; could not inline: {processed.reason})"}
            import base64

            return {"content": "", "image": {"data": base64.b64encode(processed.data).decode("ascii"), "mime": f"image/{processed.kind}"}}
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            return {"content": f"(binary file, {format_size(len(raw))})"}
        offset = int(args.get("offset", 0) or 0)
        if offset:
            content = "\n".join(content.splitlines()[offset:])
        text, truncated = truncate_for_read(content)
        result: dict = {"content": text}
        if truncated:
            result["truncated"] = True
        return result

    async def edit_execute(args: dict) -> dict:
        path = normalize_path(args.get("path", ""), cwd)
        edits_raw = args.get("edits") or (
            [{"old_string": args.get("old_string", ""), "new_string": args.get("new_string", "")}]
            if args.get("old_string") is not None
            else []
        )
        if not edits_raw:
            return {"is_error": True, "content": "edit requires edits list or old_string"}
        async with file_lock(path):
            try:
                original = path.read_text(encoding="utf-8")
            except OSError as exc:
                return {"is_error": True, "content": f"read failed: {exc}"}
            line_ending = detect_line_ending(original)
            # apply_edits_to_normalized_content normalizes content to LF and
            # restores line_ending itself, so pass the original content and
            # only normalize the edit strings (which must match LF content).
            edits = [{"old_string": e.get("old_string", "").replace(line_ending, "\n") if line_ending != "\n" else e.get("old_string", ""), "new_string": e.get("new_string", "").replace(line_ending, "\n") if line_ending != "\n" else e.get("new_string", "")} for e in edits_raw]
            new_content, patch, error = apply_edits_to_normalized_content(original, edits, line_ending=line_ending, path=str(path))
            if error:
                return {"is_error": True, "content": error}
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(new_content, encoding="utf-8")
            except OSError as exc:
                return {"is_error": True, "content": f"write failed: {exc}"}
            return {"content": patch}

    async def write_execute(args: dict) -> dict:
        path = normalize_path(args.get("path", ""), cwd)
        content = str(args.get("content", ""))
        async with file_lock(path):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            except OSError as exc:
                return {"is_error": True, "content": f"write failed: {exc}"}
            return {"content": f"wrote {format_size(len(content.encode('utf-8')))} to {path}"}

    async def bash_execute(args: dict) -> dict:
        command = str(args.get("command", "")).strip()
        if not command:
            return {"is_error": True, "content": "bash requires a command"}

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(cwd or Path.cwd()),
                start_new_session=True,
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=int(args.get("timeout", 120) or 120))
            except asyncio.TimeoutError:
                # Kill the whole process group — killing just the shell would
                # orphan the command's child processes.
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    proc.kill()
                proc.returncode = -9
                return {"is_error": True, "content": "command timed out"}
            output = stdout.decode("utf-8", errors="replace")
            text, truncated = truncate_for_read(output)
            result = {"content": text or f"(exit {proc.returncode})"}
            if proc.returncode not in (0, None):
                result["exit_code"] = proc.returncode
            if truncated:
                result["truncated"] = True
            return result
        except OSError as exc:
            return {"is_error": True, "content": f"spawn failed: {exc}"}

    return [
        {
            "name": "read",
            "description": "Read a file's content with bounded output",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "description": "skip this many lines first"},
                },
                "required": ["path"],
            },
            "execute": read_execute,
        },
        {
            "name": "edit",
            "description": "Apply exact-match string edits to a file (batch supported)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "edits": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["path"],
            },
            "execute": edit_execute,
        },
        {
            "name": "write",
            "description": "Write a file's full content (creates directories as needed)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            "execute": write_execute,
        },
        {
            "name": "bash",
            "description": "Run a shell command in the session working directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "description": "seconds, default 120"},
                },
                "required": ["command"],
            },
            "execute": bash_execute,
        },
    ]
