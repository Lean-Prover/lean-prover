"""TUI proof-run handlers (extracted from the former ``agent/tui.py`` monolith).

``ProverRunMixin`` carries the proof-run orchestration, live event rendering,
proof-diff, terminal-title / clipboard / selection plumbing that the dashboard
app needs. It is mixed into ``ProverApp`` (``agent.tui.app``) so the app class
stays focused on composition, bindings, and command dispatch while this module
owns the "proving" behaviour.

All widget access goes through ``call_from_thread`` (these methods run on
``thread=True`` Textual workers); the parallel pool is a bounded
``ThreadPoolExecutor`` running inside a single Textual worker.
"""

from __future__ import annotations

import os
import re
import threading
from contextlib import suppress
from pathlib import Path

from rich.markup import escape
from textual import events, on, work
from textual.widgets import RichLog

from agent import session as sess
from agent.loop import prove
from agent.tui.screens import ProofDiffScreen, ReplayScreen
from agent.tui_render import ProblemRow, render_event


class ProverRunMixin:
    """Proof-run + live-rendering methods for the TUI dashboard app."""

    # ---------------------------------------------------------------- proving

    def _start_run(self, problems: list[dict]) -> None:
        self._stop_flag = False
        self._run_active = True
        self._sync_text_selection_state()
        self._refresh_status()
        self._sync_terminal_title()
        self._run_problems(problems, self.n_workers)

    @work(group="prove")
    async def _run_problems(self, problems: list[dict], workers: int) -> None:
        """Run problems — one Textual thread-worker per problem, bounded by a
        semaphore so at most ``workers`` prove concurrently."""
        sem = threading.BoundedSemaphore(max(1, workers))
        pending = [self._pool_prove(problem, sem) for problem in problems]
        for worker in pending:
            await worker.wait()
        if self._stop_flag:
            self._mark_pending_skipped(problems)
        self._run_done()

    @work(thread=True, group="prove", exit_on_error=False)
    def _pool_prove(self, problem: dict, sem: threading.BoundedSemaphore) -> None:
        """Run one problem on a Textual thread worker (semaphore-bounded)."""
        if self._stop_flag:
            return
        try:
            with sem:
                self._prove_one(problem)
        except Exception as exc:  # noqa: BLE001 — a single failure must not kill the TUI
            self.call_from_thread(self._log, f"[red]prove {problem['id']} failed: {exc}[/red]")

    def _mark_pending_skipped(self, problems: list[dict]) -> None:
        """Mark rows that were still pending when a run stopped as skipped."""
        for problem in problems:
            row = self._row_by_id(problem["id"])
            if row is not None and row.status == "pending":
                row.set_status("stopped")
                self.counts["stopped"] = self.counts.get("stopped", 0) + 1
        self._refresh_status()

    def _prove_one(self, problem: dict) -> None:
        """Prove a single problem (called from a worker thread)."""
        if self._stop_flag:
            return
        pid = problem["id"]
        self.call_from_thread(self._set_row_status, pid, "running")
        result = prove(
            problem["statement"],
            max_steps=20,
            verbose=False,
            problem_id=pid,
            goal_feedback=True,
            on_event=lambda ev, p=pid: self._probe_event(p, ev),
            should_stop=lambda: self._stop_flag,
        )
        status = "proved" if result.proved else ("stopped" if result.stopped else "failed")
        self.call_from_thread(self._finish_row, pid, status, result)

    def _probe_event(self, problem_id: str, ev: dict) -> None:
        """Forward a prove() event to the UI thread."""
        self.call_from_thread(self._render_event, problem_id, ev)
        # Show diff panel on file write events (full-file mode)
        if ev.get("event") == "llm_response" and ev.get("accepted"):
            body = ev.get("body")
            if body and ("\ntheorem" in body or "\nlemma" in body):
                self.call_from_thread(self._show_proof_diff, problem_id, body)

    def _render_event(self, problem_id: str, ev: dict) -> None:
        """Render a prove() event record into the main panels."""
        tag = f"{problem_id[:16]}: " if self.n_workers > 1 else ""
        render_event(ev, problem_id,
                     self.query_one("#log", RichLog),
                     self.query_one("#goals", RichLog),
                     self.query_one("#errors", RichLog),
                     self.query_one("#proof", RichLog),
                     tag=tag)
        # Track live token usage for cost meter
        if ev.get("event") == "llm_response":
            tokens = ev.get("tokens", 0)
            if isinstance(tokens, (int, float)):
                self._live_tokens += int(tokens)
        elif ev.get("event") == "result":
            cost = ev.get("cost", 0.0)
            if isinstance(cost, (int, float)):
                self._live_cost += float(cost)
        # Auto-compact warning: check context window usage
        elif ev.get("event") == "compaction":
            dropped = ev.get("dropped", 0)
            self._log(f"[dim]auto-compacted: dropped {dropped} turns[/dim]")
            if problem_id not in self._compact_warning_shown:
                self._compact_warning_shown.add(problem_id)
                self.notify(
                    f"Session {problem_id} auto-compacted "
                    f"(dropped {dropped} turns to stay within context window)",
                    severity="warning",
                )
        self._refresh_status()

    # ---------------------------------------------------------------- rows/status

    def _row_by_id(self, problem_id: str) -> ProblemRow | None:
        for row in self.query(ProblemRow):
            if row.problem["id"] == problem_id:
                return row
        return None

    def _set_row_status(self, problem_id: str, status: str) -> None:
        row = self._row_by_id(problem_id)
        if row:
            row.set_status(status)

    def _finish_row(self, problem_id: str, status: str, result) -> None:
        self._set_row_status(problem_id, status)
        self.counts[status] = self.counts.get(status, 0) + 1
        proof = result.proof if result.proved else ""
        if proof and self.n_workers <= 1:
            panel = self.query_one("#proof", RichLog)
            panel.clear()
            panel.write(escape(proof))
        self._refresh_status()

    def _finish_custom_row(self, problem_id: str, status: str, result) -> None:
        """Finish handler for runs not backed by a ProblemRow (branch/custom)."""
        self.counts[status] = self.counts.get(status, 0) + 1
        proof = result.proof if result.proved else ""
        if proof:
            panel = self.query_one("#proof", RichLog)
            panel.clear()
            panel.write(escape(proof))
        self._run_active = False
        self._sync_text_selection_state()
        self._refresh_status()
        self._terminal_notification.notify_turn_finished()
        self._drain_queued_prompts()

    def _run_done(self) -> None:
        self._run_active = False
        self._sync_text_selection_state()
        self._refresh_status()
        self._sync_terminal_title()
        self._log(f"[bold]run finished[/bold] — proved {self.counts['proved']} so far")
        self._terminal_notification.notify_turn_finished()
        self._drain_queued_prompts()

    def _drain_queued_prompts(self) -> None:
        """Prove prompts queued while a run was active (tau's queued follow-ups).

        Starts exactly one: when it finishes, _run_done re-enters here, so
        queued prompts prove sequentially instead of racing on the shared
        #proof/#goals panels.
        """
        if not self._queued_prompts:
            return
        text = self._queued_prompts.pop(0)
        self._log(f"[dim]applying queued:[/dim] {text[:60]}")
        self._prove_statement(text)

    def _prove_statement(self, statement: str) -> None:
        """Prove an inline theorem from /prove <statement>."""
        m = re.search(r"\b(?:theorem|lemma|example)\s+(\w+)", statement)
        base = m.group(1) if m else "custom"
        self._custom_seq += 1
        problem = {
            "id": f"{base}-{self._custom_seq}" if self._custom_seq > 1 else base,
            "statement": statement,
            "difficulty": "custom",
        }
        self._log(f"\n[bold magenta]custom: {problem['id']}[/bold magenta]")
        self._start_run([problem])

    def _replay_by_id(self, session_id: str) -> None:
        for sp in sess.list_sessions():
            if sp.stem == session_id:
                self.push_screen(ReplayScreen(sp))
                return

    def _branch_run(self, session_id: str, branch_at: int | None) -> None:
        """Re-run a recorded session's theorem, seeded from its history
        (tau: branch_to_entry repoints the leaf; here prove(resume_from=...))."""
        from agent.session_manager import SessionManager

        if self._run_active:
            self.notify("Already running — press s to stop first.", severity="warning")
            return
        rec = SessionManager().get(session_id)
        if rec is None:
            self.notify(f"Session not found: {session_id}", severity="error")
            return
        records = sess.read_session(Path(rec.path))
        start = next((r for r in records if r.get("event") == "start"), {})
        statement = start.get("statement")
        if not statement:
            self.notify("Session has no recorded statement.", severity="error")
            return
        pid = start.get("problem_id") or "branch"

        self._stop_flag = False
        self._run_active = True
        self._sync_text_selection_state()
        self._refresh_status()
        self._log(f"\n[bold]── branch {session_id}"
                  f"{' @ turn ' + str(branch_at) if branch_at is not None else ''}[/bold]")
        self._branch_worker(session_id, branch_at, statement, pid, records)

    @work(thread=True, group="prove", exit_on_error=False)
    def _branch_worker(
        self, session_id: str, branch_at: int | None, statement: str, pid: str, records: list
    ) -> None:
        """Prove a resumed/branched session on a Textual thread worker."""
        summary = None
        if branch_at is not None and os.environ.get("PROVER_BRANCH_SUMMARY", "1") != "0":
            from agent.branch_summary import summarize_branch_with_model

            summary = summarize_branch_with_model(records)
            if summary:
                self.call_from_thread(
                    self._log, "[dim]branch summary generated[/dim]")
            else:
                self.call_from_thread(
                    self._log,
                    "[dim]branch summary unavailable — continuing with "
                    "recorded history only[/dim]")
        result = prove(
            statement, max_steps=20, verbose=False,
            problem_id=pid, goal_feedback=True,
            on_event=lambda ev: self._probe_event(pid, ev),
            should_stop=lambda: self._stop_flag,
            resume_from=session_id, branch_at=branch_at,
            branch_summary=summary,
        )
        status = ("proved" if result.proved
                  else "stopped" if result.stopped else "failed")
        self.call_from_thread(self._finish_custom_row, pid, status, result)

    # ---------------------------------------------------------------- diff

    def _show_proof_diff(self, problem_id: str, body: str) -> None:
        """Show a diff view of the proof when full-file mode writes a new file."""
        try:
            import agent.tui as _tui

            target = _tui.REPO / "lean" / "tmp" / f"Prover_{problem_id}.lean"
            if not target.exists():
                return
            current = target.read_text(encoding="utf-8")
            old = self._get_last_proof_body(problem_id)
            if old is None:
                old = current
            diff = self._compute_diff(old, body)
            if diff:
                self.push_screen(ProofDiffScreen(problem_id, diff))
                self._log(f"[dim]diff for {problem_id}: {len(diff.splitlines())} lines[/dim]")
        except Exception as exc:  # noqa: BLE001 — diff is best-effort
            self._log(f"[dim]diff failed: {exc}[/dim]")

    def _get_last_proof_body(self, problem_id: str) -> str | None:
        """Get the last proof body for a problem from its session."""
        try:
            from agent import session as sess_mod

            sessions = list(sess_mod.list_sessions())
            for sp in sessions:
                if sp.stem == problem_id:
                    records = sess_mod.read_session(sp)
                    for rec in reversed(records):
                        if rec.get("event") == "llm_response" and rec.get("body"):
                            return rec["body"]
        except Exception as exc:  # noqa: BLE001
            self._log(f"[dim]failed to get last proof body: {exc}[/dim]")
        return None

    @staticmethod
    def _compute_diff(old: str, new: str) -> str:
        """Compute a simple unified diff between two strings."""
        import difflib

        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)
        return "".join(difflib.unified_diff(old_lines, new_lines,
                                            fromfile="old", tofile="new", n=3))

    # ---------------------------------------------------------------- clipboard

    def _sync_terminal_title(self) -> None:
        """Reflect run state in the terminal tab title (tau parity)."""
        title = f"{len(self.problems)} problems · proved {self.counts.get('proved', 0)}"
        self._terminal_title.update(title, running=self._run_active)

    def _sync_text_selection_state(self) -> None:
        """Disable native text selection while the transcript is mutating (tau parity)."""
        type(self).ALLOW_SELECT = not self._run_active
        if self._run_active and self.screen_stack:
            with suppress(Exception):
                self.screen.clear_selection()

    def copy_to_clipboard(self, text: str) -> None:
        """Copy text using pyperclip when available, then Textual's fallback (tau parity)."""
        if self._supports_pyperclip is None:
            try:
                import pyperclip  # type: ignore[import-untyped]
            except ImportError:
                self._supports_pyperclip = False
            else:
                if pyperclip is not None:
                    self._supports_pyperclip = True
        if self._supports_pyperclip:
            import pyperclip  # type: ignore[import-untyped]

            with suppress(Exception):
                pyperclip.copy(text)
        super().copy_to_clipboard(text)

    @on(events.TextSelected)
    async def on_text_selected(self) -> None:
        """Optionally copy selected text automatically (tau parity)."""
        active_screen = self.screen
        if not (
            self.tui_settings.auto_copy_selection
            or getattr(active_screen, "auto_copy_selection", False)
        ):
            return
        selection = active_screen.get_selected_text()
        if selection:
            self.copy_to_clipboard(selection)
            self.notify("Copied selection to clipboard.")
