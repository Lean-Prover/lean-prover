"""Slash command registry for prover (ported from huggingface/tau commands.py).

Same architecture as tau: a `CommandRegistry` of `SlashCommand` records whose
handlers are *pure parsers* — they inspect a `CommandContext` and return a
frozen `CommandResult` of flags. They never mutate app state. The frontend
(TUI, or CLI one day) inspects the flags and performs the side effects.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class CommandSession(Protocol):
    """Session attributes available to slash-command handlers."""

    @property
    def model(self) -> str: ...

    @property
    def session_dir(self) -> Path: ...

    @property
    def session_ids(self) -> Sequence[str]: ...

    @property
    def current_session_id(self) -> str | None: ...

    @property
    def problems_total(self) -> int: ...

    @property
    def counts(self) -> dict[str, int]: ...

    @property
    def n_workers(self) -> int: ...

    @property
    def max_workers(self) -> int: ...

    @property
    def is_running(self) -> bool: ...

    @property
    def thinking_level(self) -> str: ...


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Result of handling a slash command. The frontend acts on flags."""

    handled: bool
    exit_requested: bool = False
    clear_requested: bool = False
    stop_requested: bool = False
    run_requested: bool = False
    prove_requested: bool = False
    prove_statement: str | None = None
    workers_requested: int | None = None
    sessions_picker_requested: bool = False
    replay_session_id: str | None = None
    branch_requested: bool = False
    branch_at: int | None = None
    leaderboard_requested: bool = False
    prompts_requested: bool = False
    reload_requested: bool = False
    export_requested: bool = False
    export_destination: Path | None = None
    usage_requested: bool = False
    usage_session_id: str | None = None         # /usage [id|all] — None = current
    theme: str | None = None
    thinking_level: str | None = None
    new_session_requested: bool = False
    compact_summary: str | None = None
    rename_requested: bool = False
    rename_session_id: str | None = None
    rename_title: str | None = None
    models_requested: bool = False
    message: str | None = None
    tree_picker_requested: bool = False          # /tree — session entry tree picker
    fork_requested: bool = False                 # /fork — branch at an entry
    fork_path: str | None = None                 # entry path ("3", "1.2")
    switch_session_id: str | None = None         #   target session id (live switch)
    set_model_choice: str | None = None          # /model <name> — model/provider choice
    set_model_provider: str | None = None        #   provider part of the choice
    set_thinking: str | None = None              # /thinking <level>
    login_requested: bool = False                # /login <provider>
    login_provider: str | None = None
    logout_requested: bool = False               # /logout <provider>
    skills_requested: bool = False               # /skills
    contexts_requested: bool = False             # /contexts
    tools_requested: bool = False                # /tools
    update_check_requested: bool = False         # /update
    permissions_requested: bool = False          # /permissions — ACL manager
    permissions_action: str | None = None        #   list | remember | revoke | mode
    permissions_tool: str | None = None          #   tool name (remember)
    permissions_pattern: str = ""                #   argument pattern (remember)
    permissions_allow: bool | None = None        #   allow (True) or deny (False) (remember)
    permissions_mode: str | None = None          #   ask|auto|yolo (mode)
    settings_requested: bool = False             # /settings — open the TUI settings screen


@dataclass(frozen=True, slots=True)
class CommandContext:
    """Runtime context passed to slash-command handlers."""

    session: CommandSession
    registry: CommandRegistry
    text: str
    name: str
    args: str


CommandHandler = Callable[[CommandContext], CommandResult]


@dataclass(frozen=True, slots=True)
class SlashCommand:
    """One slash command: identity + help text + pure handler."""

    name: str
    description: str
    usage: str
    handler: CommandHandler
    aliases: tuple[str, ...] = ()
    search_terms: tuple[str, ...] = ()


class CommandRegistry:
    """Name/alias dispatch table for slash commands (tau parity)."""

    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}
        self._aliases: dict[str, str] = {}

    def register(self, command: SlashCommand) -> None:
        if command.name in self._commands:
            raise ValueError(f"duplicate command: {command.name}")
        self._commands[command.name] = command
        for alias in command.aliases:
            if alias in self._aliases or alias in self._commands:
                raise ValueError(f"duplicate command alias: {alias}")
            self._aliases[alias] = command.name

    def get(self, name: str) -> SlashCommand | None:
        return self._commands.get(self._aliases.get(name, name))

    def list_commands(self) -> list[SlashCommand]:
        return sorted(self._commands.values(), key=lambda c: c.name)

    def execute(self, session: CommandSession, text: str) -> CommandResult:
        """Parse and run one input line.

        Returns CommandResult(handled=False) for anything that is not a known
        slash command so the frontend can fall through to its default path.
        """
        stripped = text.strip()
        if not stripped.startswith("/"):
            return CommandResult(handled=False)
        name, args = self._parse_command(stripped)
        command = self.get(name)
        if command is None:
            return CommandResult(
                handled=False,
                message=f"Unknown command: /{name}. Try /help.",
            )
        context = CommandContext(
            session=session,
            registry=self,
            text=stripped,
            name=name,
            args=args,
        )
        return command.handler(context)

    @staticmethod
    def _parse_command(text: str) -> tuple[str, str]:
        body = text.lstrip("/").strip()
        if not body:
            return "", ""
        parts = body.split(None, 1)
        return parts[0], (parts[1].strip() if len(parts) > 1 else "")


def create_default_command_registry() -> CommandRegistry:
    """Register the built-in command set."""
    registry = CommandRegistry()
    registry.register(SlashCommand(
        name="quit", description="Exit the TUI.",
        usage="/quit", handler=_quit_command,
        aliases=("exit", "q"), search_terms=("close",),
    ))
    registry.register(SlashCommand(
        name="help", description="List available commands.",
        usage="/help", handler=_help_command,
        search_terms=("commands",),
    ))
    registry.register(SlashCommand(
        name="hotkeys", description="Show common keyboard shortcuts.",
        usage="/hotkeys", handler=_hotkeys_command,
    ))
    registry.register(SlashCommand(
        name="status", description="Show run/model/worker status.",
        usage="/status", handler=_status_command,
        search_terms=("info", "session"),
    ))
    registry.register(SlashCommand(
        name="clear", description="Clear the log panel.",
        usage="/clear", handler=_clear_command,
    ))
    registry.register(SlashCommand(
        name="prove", description="Prove a custom theorem (opens editor, or inline with args).",
        usage="/prove [statement]", handler=_prove_command,
        search_terms=("custom", "run theorem"),
    ))
    registry.register(SlashCommand(
        name="run", description="Run remaining pending problems.",
        usage="/run", handler=_run_command,
        search_terms=("rest", "remaining", "benchmark"),
    ))
    registry.register(SlashCommand(
        name="stop", description="Stop after the current step.",
        usage="/stop", handler=_stop_command,
        search_terms=("abort", "cancel"),
    ))
    registry.register(SlashCommand(
        name="workers", description="Set the number of parallel proof workers (1–N).",
        usage="/workers [n]", handler=_workers_command,
        search_terms=("parallel",),
    ))
    registry.register(SlashCommand(
        name="resume", description="Replay a recorded session (picker with no args).",
        usage="/resume [session-id]", handler=_resume_command,
        search_terms=("sessions", "replay", "history"),
    ))
    registry.register(SlashCommand(
        name="branch", description="Re-run a theorem from an earlier point of a session.",
        usage="/branch <session-id> [turn]", handler=_branch_command,
        search_terms=("retry", "fork", "continue"),
    ))
    registry.register(SlashCommand(
        name="export", description="Export the log panel to a file.",
        usage="/export <path>", handler=_export_command,
        search_terms=("save",),
    ))
    registry.register(SlashCommand(
        name="leaderboard", description="Show the local leaderboard.",
        usage="/leaderboard", handler=_leaderboard_command,
        aliases=("board",), search_terms=("scores",),
    ))
    registry.register(SlashCommand(
        name="prompts", description="Pick a markdown prompt template to apply.",
        usage="/prompts", handler=_prompts_command,
        search_terms=("template", "brief", "expand"),
    ))
    registry.register(SlashCommand(
        name="reload", description="Reload problems, leaderboard and themes from disk.",
        usage="/reload", handler=_reload_command,
        search_terms=("refresh", "re-read", "restart"),
    ))
    registry.register(SlashCommand(
        name="usage", description="Show the token/cost dashboard for a session.",
        usage="/usage [session-id|all]", handler=_usage_command,
        aliases=("cost",), search_terms=("tokens", "billing"),
    ))
    registry.register(SlashCommand(
        name="model", description="Show the active model.",
        usage="/model", handler=_model_command,
    ))
    registry.register(SlashCommand(
        name="models", description="Browse, add and select model profiles.",
        usage="/models", handler=_models_command,
        search_terms=("endpoint", "api", "profile", "models", "manage"),
    ))
    registry.register(SlashCommand(
        name="theme", description="Show or set the TUI theme.",
        usage="/theme [name]", handler=_theme_command,
        search_terms=("colors", "appearance"),
    ))
    registry.register(SlashCommand(
        name="thinking", description="Show or set the model thinking level.",
        usage="/thinking [off|minimal|low|medium|high|xhigh]", handler=_thinking_command,
        aliases=("think",), search_terms=("reasoning", "reasoning_effort"),
    ))
    registry.register(SlashCommand(
        name="system", description="Show the proof loop's system prompt.",
        usage="/system", handler=_system_command,
    ))
    registry.register(SlashCommand(
        name="new", description="Start a fresh session: clear statuses, counts and log.",
        usage="/new", handler=_new_command,
        aliases=("fresh",), search_terms=("clear", "reset"),
    ))
    registry.register(SlashCommand(
        name="compact", description="Request a manual history compaction summary.",
        usage="/compact [instructions]", handler=_compact_command,
        search_terms=("summarize",),
    ))
    registry.register(SlashCommand(
        name="name", description="Rename the current (most recent) session.",
        usage="/name <new name>", handler=_name_command,
        search_terms=("title",),
    ))
    registry.register(SlashCommand(
        name="tree", description="Show the session entry tree (branch picker).",
        usage="/tree", handler=_tree_command,
        search_terms=("branches", "entries", "history"),
    ))
    registry.register(SlashCommand(
        name="fork", description="Fork the session at an entry (branch summary + fresh tail).",
        usage="/fork [path]", handler=_fork_command,
        search_terms=("branch", "split"),
    ))
    registry.register(SlashCommand(
        name="login", description="Sign in to a provider (openai-codex|anthropic|github-copilot).",
        usage="/login [provider]", handler=_login_command,
        search_terms=("auth", "oauth", "signin", "credentials"),
    ))
    registry.register(SlashCommand(
        name="logout", description="Remove stored credentials for a provider.",
        usage="/logout <provider>", handler=_logout_command,
        search_terms=("signout", "credentials"),
    ))
    registry.register(SlashCommand(
        name="skills", description="List discovered skills.",
        usage="/skills", handler=_skills_command,
        search_terms=("injections", "prompts"),
    ))
    registry.register(SlashCommand(
        name="contexts", description="List project context files.",
        usage="/contexts", handler=_contexts_command,
        search_terms=("AGENTS.md", ".prover.md", "project"),
    ))
    registry.register(SlashCommand(
        name="tools", description="List the tools bound to the session.",
        usage="/tools", handler=_tools_command,
        search_terms=("capabilities",),
    ))
    registry.register(SlashCommand(
        name="stats", description="Show lifetime totals for the active session.",
        usage="/stats", handler=_stats_command,
        search_terms=("totals", "lifetime"),
    ))
    registry.register(SlashCommand(
        name="update", description="Check for a newer lean-prover release.",
        usage="/update", handler=_update_command,
        search_terms=("upgrade", "version"),
    ))
    registry.register(SlashCommand(
        name="permissions", description="Manage per-tool permission rules (ACL).",
        usage="/permissions [list|remember|revoke|mode] ...",
        handler=_permissions_command,
        aliases=("perm",), search_terms=("acl", "allow", "deny", "approve"),
    ))
    registry.register(SlashCommand(
        name="settings", description="Open the TUI settings screen.",
        usage="/settings", handler=_settings_command,
        aliases=("prefs",), search_terms=("options", "preferences", "config"),
    ))
    return registry


# --------------------------------------------------------------------------- handlers


def _quit_command(context: CommandContext) -> CommandResult:
    return CommandResult(handled=True, exit_requested=True)


def _new_command(context: CommandContext) -> CommandResult:
    if context.session.is_running:
        return CommandResult(handled=True, message="Already running — use /stop first.")
    return CommandResult(handled=True, new_session_requested=True)


def _compact_command(context: CommandContext) -> CommandResult:
    return CommandResult(handled=True, compact_summary=context.args.strip() or None)


def _name_command(context: CommandContext) -> CommandResult:
    args = context.args.strip()
    session_id = context.session.current_session_id
    if session_id is None:
        return CommandResult(handled=True,
                             message="No recorded session to rename yet.")
    if not args:
        return CommandResult(
            handled=True,
            message=f"Current session: {session_id}\nUsage: /name <new name>",
        )
    try:
        name = _validated_session_name(args)
    except ValueError as exc:
        return CommandResult(handled=True, message=str(exc))
    return CommandResult(handled=True, rename_requested=True,
                         rename_session_id=session_id, rename_title=name,
                         message=f"Session renamed to {name}.")


def _validated_session_name(value: str) -> str:
    name = value.strip()
    if not name:
        raise ValueError("Usage: /name <new name>")
    if any(char in name for char in "\r\n\t"):
        raise ValueError("Session name must be a single line.")
    return name


def _clear_command(context: CommandContext) -> CommandResult:
    return CommandResult(handled=True, clear_requested=True)


def _stop_command(context: CommandContext) -> CommandResult:
    if not context.session.is_running:
        return CommandResult(handled=True, message="Nothing is running.")
    return CommandResult(handled=True, stop_requested=True,
                         message="Stopping after the current step…")


def _run_command(context: CommandContext) -> CommandResult:
    if context.session.is_running:
        return CommandResult(handled=True, message="Already running — use /stop first.")
    return CommandResult(handled=True, run_requested=True)


def _prove_command(context: CommandContext) -> CommandResult:
    if context.session.is_running:
        return CommandResult(handled=True, message="Already running — use /stop first.")
    if context.args:
        return CommandResult(handled=True, prove_requested=True,
                             prove_statement=context.args)
    return CommandResult(handled=True, prove_requested=True)


def _workers_command(context: CommandContext) -> CommandResult:
    session = context.session
    if not context.args:
        return CommandResult(
            handled=True,
            message=f"Parallel workers: {session.n_workers} (max {session.max_workers}). "
                    f"Usage: /workers <n>",
        )
    try:
        n = int(context.args)
    except ValueError:
        return CommandResult(handled=True, message="Usage: /workers <n>")
    if not 1 <= n <= session.max_workers:
        return CommandResult(
            handled=True,
            message=f"Workers must be between 1 and {session.max_workers}.",
        )
    return CommandResult(handled=True, workers_requested=n,
                         message=f"Workers set to {n}.")


def _resume_command(context: CommandContext) -> CommandResult:
    if context.args:
        if context.args not in set(context.session.session_ids):
            return CommandResult(handled=True,
                                 message=f"Session not found: {context.args}")
        return CommandResult(handled=True, replay_session_id=context.args)
    return CommandResult(handled=True, sessions_picker_requested=True)


def _branch_command(context: CommandContext) -> CommandResult:
    if not context.args:
        return CommandResult(handled=True,
                             message="Usage: /branch <session-id> [turn]")
    parts = context.args.split(None, 1)
    session_id = parts[0]
    if session_id not in set(context.session.session_ids):
        return CommandResult(handled=True,
                             message=f"Session not found: {session_id}")
    turn = None
    if len(parts) > 1:
        try:
            turn = max(0, int(parts[1]))
        except ValueError:
            return CommandResult(handled=True,
                                 message="Usage: /branch <session-id> [turn]")
    return CommandResult(handled=True, replay_session_id=session_id,
                         branch_requested=True, branch_at=turn,
                         message=(f"Branching {session_id} from turn {turn}."
                                  if turn is not None
                                  else f"Resuming {session_id} from where it failed."))


def _export_command(context: CommandContext) -> CommandResult:
    if not context.args:
        return CommandResult(handled=True, message="Usage: /export <path>")
    return CommandResult(handled=True, export_requested=True,
                         export_destination=Path(context.args))


def _tree_command(context: CommandContext) -> CommandResult:
    return CommandResult(handled=True, tree_picker_requested=True)


def _fork_command(context: CommandContext) -> CommandResult:
    path = context.args.strip() or None
    return CommandResult(handled=True, fork_requested=True, fork_path=path,
                         message=("Opening fork picker…" if path is None
                                  else f"Forking session at entry {path}."))


_VALID_LOGIN_PROVIDERS = ("openai-codex", "anthropic", "github-copilot")


def _login_command(context: CommandContext) -> CommandResult:
    provider = context.args.strip().lower()
    if not provider:
        return CommandResult(handled=True,
                             message="Usage: /login <provider> — "
                                     + ", ".join(_VALID_LOGIN_PROVIDERS))
    if provider not in _VALID_LOGIN_PROVIDERS:
        return CommandResult(handled=True,
                             message=f"Unknown provider: {provider}. "
                                     f"Available: {', '.join(_VALID_LOGIN_PROVIDERS)}")
    return CommandResult(handled=True, login_requested=True, login_provider=provider,
                         message=f"Starting {provider} sign-in…")


def _logout_command(context: CommandContext) -> CommandResult:
    provider = context.args.strip().lower()
    if not provider or provider not in _VALID_LOGIN_PROVIDERS:
        return CommandResult(handled=True,
                             message="Usage: /logout <provider> — "
                                     + ", ".join(_VALID_LOGIN_PROVIDERS))
    return CommandResult(handled=True, logout_requested=True,
                         message=f"Removing {provider} credentials.",
                         login_provider=provider)


def _skills_command(context: CommandContext) -> CommandResult:
    return CommandResult(handled=True, skills_requested=True)


def _contexts_command(context: CommandContext) -> CommandResult:
    return CommandResult(handled=True, contexts_requested=True)


def _tools_command(context: CommandContext) -> CommandResult:
    return CommandResult(handled=True, tools_requested=True)


def _stats_command(context: CommandContext) -> CommandResult:
    return CommandResult(handled=True, usage_requested=True,
                         usage_session_id=context.session.current_session_id,
                         message=f"Stats for {context.session.current_session_id or 'active session'}")


def _update_command(context: CommandContext) -> CommandResult:
    return CommandResult(handled=True, update_check_requested=True)


_PERMISSION_MODES = ("ask", "auto", "yolo")


def _permissions_command(context: CommandContext) -> CommandResult:
    args = context.args.strip()
    if not args:
        return CommandResult(
            handled=True, permissions_requested=True,
            message="Usage: /permissions list | remember <allow|deny> <tool> [pattern] "
                    "| revoke <id> | mode <ask|auto|yolo>",
        )
    verb, rest = (args.split(None, 1) + ["", ""])[:2]
    verb = verb.lower()
    if verb == "list":
        return CommandResult(handled=True, permissions_requested=True,
                             permissions_action="list")
    if verb == "mode":
        mode = rest.strip().lower()
        if mode not in _PERMISSION_MODES:
            return CommandResult(handled=True, permissions_requested=True,
                                 message=f"Invalid mode: {mode}. "
                                         f"Available: {', '.join(_PERMISSION_MODES)}")
        return CommandResult(handled=True, permissions_requested=True,
                             permissions_action="mode", permissions_mode=mode)
    if verb == "revoke":
        rid = rest.strip()
        if not rid:
            return CommandResult(handled=True, permissions_requested=True,
                                 message="Usage: /permissions revoke <rule-id>")
        return CommandResult(handled=True, permissions_requested=True,
                             permissions_action="revoke", permissions_tool=rid)
    if verb == "remember":
        # remember <allow|deny> <tool> [pattern...]
        allow_txt, tail = (rest.split(None, 1) + ["", ""])[:2]
        if allow_txt.lower() not in ("allow", "deny"):
            return CommandResult(handled=True, permissions_requested=True,
                                 message="Usage: /permissions remember <allow|deny> <tool> [pattern]")
        tool, pattern = (tail.split(None, 1) + ["", ""])[:2]
        if not tool:
            return CommandResult(handled=True, permissions_requested=True,
                                 message="Usage: /permissions remember <allow|deny> <tool> [pattern]")
        return CommandResult(
            handled=True, permissions_requested=True, permissions_action="remember",
            permissions_tool=tool, permissions_pattern=pattern or "",
            permissions_allow=(allow_txt.lower() == "allow"),
            message=f"Remembered {allow_txt.lower()} for tool '{tool}'.",
        )
    return CommandResult(handled=True, permissions_requested=True,
                         message=f"Unknown /permissions verb: {verb}")


def _leaderboard_command(context: CommandContext) -> CommandResult:
    return CommandResult(handled=True, leaderboard_requested=True)


def _prompts_command(context: CommandContext) -> CommandResult:
    return CommandResult(handled=True, prompts_requested=True)


def _settings_command(context: CommandContext) -> CommandResult:
    return CommandResult(handled=True, settings_requested=True)


def _reload_command(context: CommandContext) -> CommandResult:
    return CommandResult(handled=True, reload_requested=True, message="Reloading…")


def _usage_command(context: CommandContext) -> CommandResult:
    if not context.args:
        current = context.session.current_session_id
        return CommandResult(
            handled=True, usage_requested=True, usage_session_id=current,
            message=("Usage: /usage [session-id|all]" if not current
                     else f"Usage for {current}"),
        )
    arg = context.args.strip()
    return CommandResult(handled=True, usage_requested=True,
                         usage_session_id=arg,
                         message=f"Usage for {arg}")


def _status_command(context: CommandContext) -> CommandResult:
    s = context.session
    counts = s.counts
    done = counts.get("proved", 0) + counts.get("failed", 0) + counts.get("stopped", 0)
    state = "running" if s.is_running else "idle"
    lines = [
        f"model:   {s.model}",
        f"state:   {state} · workers={s.n_workers}",
        f"thinking {getattr(s, 'thinking_level', None) or 'off'}",
        (f"proved:  {counts.get('proved', 0)} · failed {counts.get('failed', 0)} · "
         f"stopped {counts.get('stopped', 0)}"),
        f"progress {done}/{s.problems_total}",
        f"sessions {s.session_dir} ({len(s.session_ids)} recorded)",
    ]
    return CommandResult(handled=True, message="\n".join(lines))


def _model_command(context: CommandContext) -> CommandResult:
    args = context.args.strip()
    if not args:
        return CommandResult(handled=True, message=f"Model: {context.session.model}")
    if "@" in args:
        model, provider = args.split("@", 1)
        model, provider = model.strip(), provider.strip()
    else:
        model, provider = args, ""
    return CommandResult(
        handled=True,
        set_model_choice=model,
        set_model_provider=provider or None,
        message=f"Model set to {model}" + (f" ({provider})" if provider else "") + ".",
    )


def _models_command(context: CommandContext) -> CommandResult:
    return CommandResult(handled=True, models_requested=True)


def _thinking_command(context: CommandContext) -> CommandResult:
    from .thinking import (
        THINKING_LEVEL_DESCRIPTIONS,
        normalize_thinking_level,
    )

    current = getattr(context.session, "thinking_level", None) or "off"
    if not context.args:
        levels = ", ".join(
            f"{level}" + (" (active)" if level == current else "")
            for level in THINKING_LEVEL_DESCRIPTIONS
        )
        return CommandResult(handled=True,
                             message=f"Thinking: {current}. Available: {levels}")
    raw = context.args.strip()
    try:
        level = normalize_thinking_level(raw)
    except ValueError as exc:
        return CommandResult(handled=True, message=str(exc))
    return CommandResult(handled=True, thinking_level=level, set_thinking=level,
                         message=f"Thinking level set to {level} "
                                 f"({THINKING_LEVEL_DESCRIPTIONS[level]}).")


def _system_command(context: CommandContext) -> CommandResult:
    from .loop import SYSTEM

    return CommandResult(handled=True, message=SYSTEM)


def _theme_command(context: CommandContext) -> CommandResult:
    from .themes import available_tui_theme_names, get_tui_theme

    current = getattr(context.session, "theme", None) or "prover-dark"
    if not context.args:
        names = ", ".join(available_tui_theme_names())
        return CommandResult(handled=True,
                             message=f"Theme: {current}. Available: {names}")
    name = context.args.strip()
    try:
        get_tui_theme(name)
    except KeyError:
        return CommandResult(handled=True,
                             message=f"Unknown theme: {name}")
    return CommandResult(handled=True, theme=name,
                         message=f"Theme set to {name}.")


def _hotkeys_command(context: CommandContext) -> CommandResult:
    lines = [
        "p  prove selected problem",
        "c  prove a custom theorem",
        "r  run remaining problems",
        "w  set parallel workers",
        "s  stop after the current step",
        "v  browse/replay recorded sessions",
        "l  show leaderboard",
        "q  quit",
        "",
        "ctrl+k  command palette    ctrl+e  edit last queued prompt",
        "",
        "slash commands work in the prompt: /help lists them",
    ]
    return CommandResult(handled=True, message="\n".join(lines))


def _help_command(context: CommandContext) -> CommandResult:
    commands = context.registry.list_commands()
    width = max(len(c.usage) for c in commands)
    lines = [f"{c.usage:<{width}}  {c.description}" for c in commands]
    return CommandResult(handled=True, message="\n".join(lines))
