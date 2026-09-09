"""``prover local`` — configure/discover/doctor the llama.cpp local backend.

Tau /local parity (lean-adapted): a built-in llama.cpp backend over its
OpenAI-compatible HTTP API. lean-prover never installs/start/stops the server;
it probes one configured endpoint, lists discovered models, and records a safe
snapshot. ``use`` registers a model profile so ``prove`` targets the local
server.
"""

from __future__ import annotations

import argparse

from .common import _lazy_main

_m = _lazy_main()


def _backend(args: argparse.Namespace):
    from ..local_backend import LlamaCppBackend

    return LlamaCppBackend(config_dir=getattr(args, "config_dir", None))


def cmd_local(args: argparse.Namespace) -> int:
    action = getattr(args, "action", "status")
    backend = _backend(args)
    if action == "status":
        return _status(backend)
    if action == "configure":
        return _configure(backend, args)
    if action == "doctor":
        return _doctor(backend)
    if action == "reset":
        return _reset(backend)
    if action == "use":
        return _use(backend, args)
    print(f"unknown action: {action}", file=__import__("sys").stderr)
    return 2


def _status(backend) -> int:
    snap = backend.snapshot()
    print(f"backend: {backend.id}")
    print(f"endpoint: {backend.effective_endpoint()}")
    print(f"configured model: {snap.get('model') or '(none)'}")
    try:
        ids = backend.probe()
        print(f"reachable: yes · models: {', '.join(ids) if ids else '(none discovered)'}")
        return 0
    except Exception as exc:  # noqa: BLE001 — status is diagnostic
        print(f"reachable: no ({exc})")
        return 1


def _configure(backend, args) -> int:
    url = getattr(args, "url", None)
    key = getattr(args, "key", None)
    model = getattr(args, "model", None)
    if not url:
        print("configure requires --url <endpoint>")
        return 2
    body = backend.configure(endpoint=url, model=model, api_key=key)
    print(f"configured {backend.id} → {body['endpoint']}"
          + (f" (model: {model})" if model else ""))
    print("run `prover local` to probe, or `prover local use --model <id>` to target it.")
    return 0


def _doctor(backend) -> int:
    report = backend.doctor()
    print(f"endpoint: {report['endpoint']}")
    print(f"reachable: {'yes' if report['reachable'] else 'no'}")
    if report["models"]:
        print(f"models: {', '.join(report['models'])}")
    if report["streaming"]:
        print("streaming: yes")
    if report["warning"]:
        print(f"warning: {report['warning']}")
    return 0 if report["reachable"] else 1


def _reset(backend) -> int:
    backend.reset()
    print(f"reset {backend.id} integration settings (server untouched).")
    return 0


def _use(backend, args) -> int:
    model = getattr(args, "model", None)
    if not model:
        print("use requires --model <id> (from `prover local`)")
        return 2
    from ..models import ModelProfile, save_store

    profile = ModelProfile(
        name=model,
        base_url=f"{backend.effective_endpoint()}/v1",
        api_key=backend.effective_api_key() or "",
    )
    save_store(active=model, profiles=[profile])
    backend.configure(model=model)
    print(f"targeting {backend.id} model {model} → {backend.effective_endpoint()}/v1")
    print("set PROVER_MODEL to route prove to it, or pick it in the TUI.")
    return 0
