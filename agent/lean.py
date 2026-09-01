"""Lean 4 toolchain interface: build + diagnostic parsing."""

from __future__ import annotations

import os
import re
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path

DIAG_RE = re.compile(
    r"^(?P<file>[^:\s]+\.lean):(?P<line>\d+):(?P<col>\d+):\s*(?P<sev>\w+):\s*(?P<msg>.*)$"
)

# Lean quotes the marker differently across versions ('sorry' vs `sorry`).
SORRY_WARNING_RE = re.compile(r"declaration uses [`']sorry[`']")


def has_sorry_warning(output: str) -> bool:
    """True when the output reports an unproven `sorry` declaration."""
    return bool(SORRY_WARNING_RE.search(output))


def split_signature(statement: str) -> str:
    """Return the theorem signature up to and including the LAST `:= by`.

    Accepts statements with or without a trailing proof. Benchmarks with
    scaffolding definitions ahead of the target theorem (Putnam `_solution`
    scaffolds, FormalQualBench namespaces) carry multiple `:= by`s — the
    target theorem is always the LAST declaration, so we cut at the last one.
    """
    s = statement.strip()
    last_end = None
    for m in re.finditer(r":=\s*by\b", s):
        last_end = m.end()
    if last_end is not None:
        return s[:last_end]
    return s + " := by"


# ---- sandboxing (leanprover/comparator + lean-workbench pattern) ----
# LLM-generated Lean is arbitrary code the harness executes. PROVER_SANDBOX
# wraps every compile in a restricted profile: filesystem read-only except
# the project .lake dir, no network (bwrap), private /tmp. Opt-in because it
# needs bubblewrap or landrun installed; a missing tool degrades to direct
# execution with a one-time warning rather than breaking the loop.

_warned_missing: set[str] = set()


def _warn_once(tool: str) -> None:
    if tool not in _warned_missing:
        _warned_missing.add(tool)
        import sys

        print(f"[sandbox warn] PROVER_SANDBOX={tool} requested but `{tool}` "
              "is not installed — running unsandboxed.", file=sys.stderr)


def sandbox_cmd(cmd: list[str], cwd: Path, mode: str | None = None) -> list[str]:
    """Prefix cmd with the configured sandbox wrapper (plain list transform).

    mode: PROVER_SANDBOX value — 'bwrap', 'landrun', or unset/'off'.
    """
    mode = (mode if mode is not None else os.environ.get("PROVER_SANDBOX", "")).strip().lower()
    if not mode or mode in ("off", "0", "none"):
        return cmd
    import shutil

    lake_dir = cwd / ".lake"
    if mode == "bwrap" or (mode == "auto" and shutil.which("bwrap")):
        if not shutil.which("bwrap"):
            _warn_once("bwrap")
            return cmd
        return [
            "bwrap",
            "--dev", "/dev",
            "--proc", "/proc",
            "--ro-bind", "/", "/",
            # writes: build artifacts + our tmp probe files only
            "--bind", str(lake_dir), str(lake_dir),
            "--bind", str(cwd / "tmp"), str(cwd / "tmp"),
            "--tmpfs", "/tmp",
            "--unshare-net",
            "--die-with-parent",
            *cmd,
        ]
    if mode == "landrun" or (mode == "auto" and shutil.which("landrun")):
        if not shutil.which("landrun"):
            _warn_once("landrun")
            return cmd
        return [
            "landrun",
            "--allow-read", "/",
            "--allow-write", str(lake_dir),
            "--allow-write", str(cwd / "tmp"),
            "--",
            *cmd,
        ]
    else:
        # auto requested but neither bwrap nor landrun is installed: degrade
        # to direct execution, but say so — never silently run unsandboxed.
        import sys

        if "auto" not in _warned_missing:
            _warned_missing.add("auto")
            print("[sandbox warn] PROVER_SANDBOX=auto but neither `bwrap` nor "
                  "`landrun` is installed — running unsandboxed.",
                  file=sys.stderr)
    return cmd


@dataclass
class Diagnostic:
    file: str
    line: int
    col: int
    severity: str
    message: str


def _run_group(cmd: list[str], cwd: Path, timeout: int) -> tuple[int, str]:
    """Run cmd in a new process group; on timeout KILL the whole group.

    subprocess.run(timeout=...) only kills the direct child — grandchildren
    (lake spawns lean) keep stdout open, so its internal drain hangs forever.
    Running in its own session and using killpg() avoids that.
    """
    try:
        proc = subprocess.Popen(
            sandbox_cmd(cmd, cwd),
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        # Missing `lake`/sandbox wrapper (or the sandbox refused to launch):
        # degrade to a failure result instead of killing the whole run.
        return -1, f"(failed to launch: {exc})"
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return proc.returncode, (stdout or b"").decode(errors="replace") + (stderr or b"").decode(errors="replace")
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            proc.kill()
        proc.wait()
        return -1, f"(process group killed: timeout after {timeout}s)"


def build(lean_dir: Path, timeout: int = 120) -> tuple[bool, str]:
    """Run `lake build` in lean_dir.

    Returns (proved, raw_output). proved means: exit code 0 AND no
    declaration uses 'sorry' (lake treats sorry as a warning, not an error,
    so exit code alone is not enough).
    """
    returncode, output = _run_group(["lake", "build"], lean_dir, timeout)
    proved = returncode == 0 and not has_sorry_warning(output)
    return proved, output


def check_file(lean_file: Path, lean_dir: Path, timeout: int = 60) -> tuple[bool, str]:
    """Run `lake env lean` on a single file for fast, isolated verification.

    Uses `lake env` to set up the correct package environment (Mathlib, etc.).
    Returns (proved, raw_output). proved means: exit code 0 AND no
    declaration uses 'sorry'.
    """
    returncode, output = _run_group(["lake", "env", "lean", str(lean_file)], lean_dir, timeout)
    proved = returncode == 0 and not has_sorry_warning(output)
    return proved, output


def compile_only(lean_file: Path, lean_dir: Path, timeout: int = 60) -> tuple[int, str]:
    """Check a file is well-formed without requiring the proof to close.

    Unlike check_file, a trailing 'sorry' is tolerated (exit code 0): we only
    care that the file parses and the declaration type-checks.
    Returns (returncode, output).
    """
    return _run_group(["lake", "env", "lean", str(lean_file)], lean_dir, timeout)


def parse_diagnostics(output: str) -> list[Diagnostic]:
    diags = []
    for line in output.splitlines():
        m = DIAG_RE.match(line.strip())
        if m:
            diags.append(
                Diagnostic(
                    file=m.group("file"),
                    line=int(m.group("line")),
                    col=int(m.group("col")),
                    severity=m.group("sev"),
                    message=m.group("msg").strip(),
                )
            )
    return diags


def _diag_priority(d: Diagnostic) -> int:
    """Fix-order rank (leanprover/skills lean-proof convention):
    0 syntax, 1 other errors, 2 unsolved goals, 3 warnings."""
    if d.severity != "error":
        return 3
    msg = d.message.lower()
    if "unsolved goals" in msg:
        return 2
    if "syntax" in msg or "expected" in msg:
        return 0
    return 1


def error_report(lean_dir: Path, output: str, max_diags: int = 8, context: int = 4) -> str:
    """Human/LLM-readable error report with surrounding source context.

    Diagnostics are shown in fix-priority order (syntax first, warnings last)
    so the model sees the root cause before the cascade it produced.
    """
    diags = [d for d in parse_diagnostics(output) if d.severity in ("error", "warning")]
    if not diags:
        return output.strip()[-4000:]  # fallback: raw tail

    diags.sort(key=_diag_priority)  # stable: original order kept within a rank
    lines_out = []
    for d in diags[:max_diags]:
        lines_out.append(f"{d.file}:{d.line}:{d.col}: {d.severity}: {d.message}")
        src = lean_dir / d.file
        if src.exists():
            src_lines = src.read_text().splitlines()
            lo = max(0, d.line - 1 - context)
            hi = min(len(src_lines), d.line + context)
            for i in range(lo, hi):
                marker = ">" if i == d.line - 1 else " "
                lines_out.append(f"{marker} {i + 1:4d} | {src_lines[i]}")
        lines_out.append("")
    return "\n".join(lines_out)
