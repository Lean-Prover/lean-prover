"""Session manager — durable index + resume/branch primitives
(ported from huggingface/tau session_manager.py, adapted to prover's layout).

Tau keeps a per-project `index.jsonl` of session records beside the
append-only transcripts and replays the root→leaf path on resume. Prover's
sessions are flat JSONL event streams, so the port is flattened to match:

- `~/.prover/sessions/index.jsonl` — one record per session (upsert: rewrite
  without same-id line + append, like tau's `_upsert`).
- resume = seed the repair loop's LLM history from a recorded session and
  continue the loop (tau: reload messages, re-bind harness).
- branch = same seed but truncated at step k, then a fresh step budget
  (tau: append a LeafEntry pointing at an earlier entry; prover has no
  entry tree so the truncated seed is the equivalent pointer).
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from . import session as sess


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """Index entry for one recorded session (tau's CodingSessionRecord)."""

    id: str
    path: str
    problem_id: str | None
    model: str
    status: str | None = None  # "running" | "proved" | "failed" | "stopped"
    proved: bool | None = None
    steps: int | None = None
    title: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> SessionRecord:
        data = json.loads(line)
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in allowed})


class SessionManager:
    """Index maintenance + lookup for recorded sessions (tau's SessionManager)."""

    def __init__(self, sessions_dir: Path | None = None) -> None:
        self.dir = sessions_dir or sess.sessions_dir()

    @property
    def index_path(self) -> Path:
        return self.dir / "index.jsonl"

    def _write_index_atomic(self, lines: list[str]) -> None:
        """Atomically rewrite the index (temp file + fsync + rename)."""
        payload = "\n".join(lines) + ("\n" if lines else "")
        fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=".index-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.index_path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def upsert(self, record: SessionRecord) -> None:
        """Insert or replace one session's index record (tau's _upsert).

        Written atomically (temp file + rename) so a crash or concurrent
        reader never observes a truncated/partial index.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        if self.index_path.exists():
            for line in self.index_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    if json.loads(line).get("id") != record.id:
                        lines.append(line)
                except json.JSONDecodeError:
                    continue
        lines.append(record.to_json())
        self._write_index_atomic(lines)

    def touch(self, session_id: str, **fields: object) -> SessionRecord | None:
        """Update mutable fields of an existing record and bump updated_at.

        Only a whitelist of non-identity fields may change, so a caller can
        never corrupt the index by rewriting ``id``/``path``/``created_at``
        (which would break the upsert-by-id contract).
        """
        record = self.get(session_id)
        if record is None:
            return None
        mutable = ("title", "status", "proved", "steps")
        data = asdict(record)
        data.update({k: v for k, v in fields.items() if k in mutable})
        data["updated_at"] = time.time()
        updated = SessionRecord(**data)
        self.upsert(updated)
        return updated

    def list_sessions(self) -> list[SessionRecord]:
        """All records, most recently updated first (tau's ordering),
        falling back to on-disk transcripts for unindexed sessions."""
        seen: dict[str, SessionRecord] = {}
        if self.index_path.exists():
            for line in self.index_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = SessionRecord.from_json(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                prev = seen.get(rec.id)
                if prev is None or rec.updated_at >= prev.updated_at:
                    seen[rec.id] = rec
        # Pick up transcripts that predate the index.
        for sp in sorted(self.dir.glob("*.jsonl"), reverse=True):
            if sp.name == "index.jsonl":
                continue
            if sp.stem not in seen:
                mtime = sp.stat().st_mtime
                seen[sp.stem] = SessionRecord(
                    id=sp.stem, path=str(sp), problem_id=self._problem_id_of(sp),
                    model="?", status=None, created_at=mtime, updated_at=mtime,
                )
        return sorted(seen.values(), key=lambda r: r.updated_at, reverse=True)

    def get(self, session_id: str) -> SessionRecord | None:
        for rec in self.list_sessions():
            if rec.id == session_id:
                return rec
        return None

    def rename(self, session_id: str, title: str) -> SessionRecord | None:
        """Rename a session record (tau's touch_session title update)."""
        updated = self.touch(session_id, title=title)
        if updated is not None:
            return updated
        # Not indexed yet: create a record from the transcript so the rename sticks.
        path = self.dir / f"{session_id}.jsonl"
        if not path.exists():
            return None
        record = SessionRecord(
            id=session_id, path=str(path), problem_id=self._problem_id_of(path),
            model="?", title=title, created_at=path.stat().st_mtime,
            updated_at=time.time(),
        )
        self.upsert(record)
        return record

    def delete(self, session_id: str) -> bool:
        """Remove a session's transcript and its index record. Returns True if
        anything was removed (missing sessions return False, so callers can
        report "not found" without clobbering real state)."""
        removed = False
        path = self.dir / f"{session_id}.jsonl"
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass
            removed = True
        if self.index_path.exists():
            lines: list[str] = []
            dropped = False
            for line in self.index_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    keep = json.loads(line).get("id") != session_id
                except json.JSONDecodeError:
                    keep = True
                if not keep:
                    dropped = True
                    continue
                lines.append(line)
            if dropped:
                self._write_index_atomic(lines)
                removed = True
        return removed

    def recover_interrupted(self) -> int:
        """Mark sessions stuck in 'running' as failed (interrupted runs).

        A killed process leaves a 'running' index record whose transcript has
        a 'start' but no terminal 'result' event. Called at session-list time
        so stale 'running' entries surface as failed instead of hanging
        forever. Returns the number of sessions recovered.
        """
        recovered = 0
        for rec in self.list_sessions():
            if rec.status != "running":
                continue
            path = Path(rec.path)
            has_result = False
            if path.exists():
                has_result = any(
                    r.get("event") == "result" for r in sess.read_session(path)
                )
            if has_result:
                continue
            self.touch(rec.id, status="failed")
            recovered += 1
        return recovered

    @staticmethod
    def _problem_id_of(path: Path) -> str | None:
        for rec in sess.read_session(path):
            if rec.get("event") == "start":
                return rec.get("problem_id")
        return None


def history_from_records(records: list[dict]) -> list[dict]:
    """Rebuild the LLM message history from a session's event records.

    Returns chat-format messages (user/assistant) reconstructed from the
    build/goals/llm_response triples. The user turn mirrors what prove()
    sent: signature + diagnostics report + open goals.
    """
    history: list[dict] = []
    signature: str | None = None
    pending_user = None  # last build report + goals, awaiting an LLM reply
    for rec in records:
        ev = rec.get("event")
        if ev == "start":
            stmt = str(rec.get("statement", "")).strip()
            if ":=" in stmt:
                signature = stmt[: stmt.index(":=")].rstrip() + " :="
            else:
                signature = stmt
        elif ev == "build":
            if rec.get("ok"):
                pending_user = None
                continue
            pending_user = {
                "signature": signature or "",
                "report": str(rec.get("report") or ""),
                "goals": None,
            }
        elif ev == "goals" and pending_user is not None:
            pending_user["goals"] = str(rec.get("goals", ""))
        elif ev == "llm_request":
            continue
        elif ev == "llm_response":
            body = str(rec.get("body", "") or "")
            if not body.strip():
                continue
            if pending_user is not None:
                user_msg = (
                    f"Theorem signature:\n{pending_user['signature']}\n\n"
                    f"Compiler diagnostics:\n{pending_user['report']}\n\n"
                    + (f"Open goals at the end of your last proof attempt:\n"
                       f"{pending_user['goals']}\n\n" if pending_user["goals"] else "")
                    + "Write ONLY the tactic proof body."
                )
                history.append({"role": "user", "content": user_msg})
                pending_user = None
            history.append({"role": "assistant", "content": body})
    return history


def last_body(records: list[dict]) -> str | None:
    """The last non-empty llm_response body in a session (the resume point)."""
    body = None
    for rec in records:
        if rec.get("event") == "llm_response":
            b = str(rec.get("body", "") or "")
            if b.strip():
                body = b
    return body
