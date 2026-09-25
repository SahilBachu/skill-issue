"""skill-issue command line."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__, paths
from .config import MODES, Config


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def _route_via_daemon(prompt: str, cwd: str, wait: float) -> dict[str, Any] | None:
    from . import daemon

    h = daemon.ensure_running(wait_ready=wait)
    if not h or h.get("status") != "ready":
        return None
    res = daemon.request("POST", "/route", {"prompt": prompt, "cwd": cwd, "candidates": True}, timeout=30)
    return res if "selected" in res else None


def cmd_route(a: argparse.Namespace) -> int:
    cfg = Config.load()
    if a.gate:
        cfg.set("gate.name", a.gate)
    if a.mode:
        cfg.set("mode", a.mode)
    cwd = str(Path(a.cwd or os.getcwd()).resolve())
    res: dict[str, Any] | None = None
    if not a.no_daemon and not a.gate and not a.mode:
        res = _route_via_daemon(a.prompt, cwd, wait=a.wait)
    if res is None:
        from .inject import render_context
        from .router import Router

        r = Router(cfg).route(a.prompt, Path(cwd))
        res = r.to_dict()
        res["context"] = render_context(r)
    if a.json:
        _print_json(res)
        return 0
    sel = res["selected"]
    t = res.get("timings_ms", {})
    print(
        f"mode={res['mode']} gate={res['gate']} catalog={res['catalog_size']} threshold={res['threshold']:.2f} "
        f"time={t.get('total', 0):.0f}ms"
    )
    if not sel:
        print("-> no skill (nothing injected)")
    for s in sel:
        flag = "" if s["installed"] else "  [not installed]"
        print(f"-> {s['name']}  p={s['prob']:.2f}{flag}")
    if a.verbose:
        print("\ncandidates scored by the gate:")
        for c in res.get("candidates", []):
            print(f"   {c['prob'] if c['prob'] is not None else 0:5.2f}  #{c['retrieval_rank']:<3} {c['name']}")
    return 0


def cmd_index(a: argparse.Namespace) -> int:
    from .discover import discover, skill_roots

    cfg = Config.load()
    cwd = Path(a.cwd or os.getcwd())
    roots = skill_roots(cfg, cwd)
    skills = discover(cfg, cwd, roots)
    if a.json:
        _print_json(
            [{"id": s.id, "name": s.name, "source": s.source, "path": s.path, "sha256": s.sha256} for s in skills]
        )
        return 0
    print(f"{len(skills)} installed skills")
    by_src: dict[str, int] = {}
    for s in skills:
        by_src[s.source] = by_src.get(s.source, 0) + 1
    for src, n in sorted(by_src.items()):
        print(f"  {n:4d}  {src}")
    if a.verbose:
        for s in sorted(skills, key=lambda s: s.id):
            print(f"  {s.id:40s} {s.path}")
    if a.warm:
        from .router import Router

        t = time.perf_counter()
        Router(cfg).catalog(cwd)
        print(f"index warmed in {time.perf_counter() - t:.1f}s")
    return 0


def cmd_mode(a: argparse.Namespace) -> int:
    cfg = Config.load()
    if a.mode is None:
        print(cfg.mode)
        return 0
    cfg.set("mode", a.mode)
    path = cfg.save()
    print(f"mode set to {a.mode} ({path})")
    _reload_daemon()
    return 0


def _reload_daemon() -> None:
    from . import daemon

    if daemon.health() is not None:
        daemon.request("POST", "/reload", {})
        print("daemon reloading")


def cmd_config(a: argparse.Namespace) -> int:
    cfg = Config.load()
    if a.action == "path":
        print(paths.config_file())
    elif a.action == "show":
        from .config import dumps_toml

        print(dumps_toml(cfg.data), end="")
    elif a.action == "set":
        if a.key is None or a.value is None:
            print("usage: skill-issue config set <key> <value>", file=sys.stderr)
            return 2
        cfg.set(a.key, _parse_value(a.value))
        print(f"saved {cfg.save()}")
        _reload_daemon()
    return 0


def _parse_value(v: str) -> Any:
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def cmd_daemon(a: argparse.Namespace) -> int:
    from . import daemon

    if a.action == "run":
        daemon.run(port=a.port)
        return 0
    if a.action == "start":
        h = daemon.ensure_running(wait_ready=a.wait)
        print(json.dumps(h) if h else "failed to start; see " + str(paths.daemon_log()))
        return 0 if h else 1
    if a.action == "stop":
        print("stopped" if daemon.stop() else "not running")
        return 0
    if a.action == "restart":
        daemon.stop()
        h = daemon.ensure_running(wait_ready=a.wait)
        print(json.dumps(h) if h else "failed")
        return 0 if h else 1
    h = daemon.health()
    print(json.dumps(h, indent=2) if h else "not running")
    return 0 if h else 1


def cmd_doctor(a: argparse.Namespace) -> int:
    from . import daemon, models
    from .discover import discover

    ok = True

    def line(status: str, what: str, detail: str = "") -> None:
        nonlocal ok
        mark = {"ok": "[ok]  ", "warn": "[warn]", "fail": "[FAIL]"}[status]
        ok = ok and status != "fail"
        print(f"{mark} {what}" + (f": {detail}" if detail else ""))

    cfg = Config.load()
    print(f"skill-issue {__version__} on {platform.system()} {platform.release()}, Python {platform.python_version()}")
    line("ok" if sys.version_info >= (3, 10) else "fail", "python >= 3.10", sys.executable)
    (paths.home() / "python.txt").parent.mkdir(parents=True, exist_ok=True)
    (paths.home() / "python.txt").write_text(sys.executable, encoding="utf-8")
    try:
        import torch

        dev = (
            "cuda " + torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu only")
        )
        line(
            "ok" if dev != "cpu only" else "warn",
            "torch " + torch.__version__,
            dev
            + (
                ""
                if dev != "cpu only"
                else " (the Laya gate is slow on CPU; consider `skill-issue config set gate.name cross-encoder`)"
            ),
        )
    except Exception as e:
        line("fail", "torch", str(e))
    line(
        "ok",
        "config",
        f"{paths.config_file()} ({'exists' if paths.config_file().is_file() else 'defaults'}), mode={cfg.mode}",
    )
    from .gates import resolve

    gname, ref, gdev = resolve(cfg)
    line("ok", "gate", f"{gname} on {gdev}")
    if ref:
        line(
            "ok" if models.is_downloaded(ref) else "warn",
            f"gate model ({gname})",
            ref + ("" if models.is_downloaded(ref) else " not downloaded yet; run `skill-issue models download`"),
        )
    emb = cfg.get("retrieval.embed_model")
    line("ok", "embedding model", emb)
    skills = discover(cfg, Path.cwd())
    line(
        "ok" if skills else "warn",
        "installed skills",
        f"{len(skills)} found" + ("" if skills else " (nothing to route to yet)"),
    )
    h = daemon.health()
    if h is None:
        line("warn", "daemon", "not running (it starts on first prompt, or `skill-issue daemon start`)")
    else:
        st = h.get("status")
        line(
            "ok" if st == "ready" else ("fail" if st == "error" else "warn"),
            "daemon",
            f"{st}, pid {h.get('pid')}, gate {h.get('gate')}"
            + (f", error: {h.get('error')}" if h.get("error") else ""),
        )
        if st == "ready":
            times = []
            for _ in range(5):
                t = time.perf_counter()
                daemon.request("POST", "/route", {"prompt": "write unit tests for the parser", "cwd": os.getcwd()})
                times.append((time.perf_counter() - t) * 1000)
            times.sort()
            p50 = times[2]
            line("ok" if p50 < 150 else "warn", "route latency (daemon, p50 of 5)", f"{p50:.0f} ms")
    plugin_hint = paths.claude_home() / "plugins" / "installed_plugins.json"
    try:
        installed = "skill-issue@" in plugin_hint.read_text(encoding="utf-8")
    except OSError:
        installed = False
    line(
        "ok" if installed else "warn",
        "Claude Code plugin",
        "installed" if installed else "not installed (/plugin marketplace add SahilBachu/skill-issue)",
    )
    return 0 if ok else 1


def cmd_models(a: argparse.Namespace) -> int:
    from . import models
    from .gates import resolve

    cfg = Config.load()
    gname, ref, _ = resolve(cfg, **({"name": a.gate} if a.gate else {}))
    if ref:
        print(f"gate ({gname}): {ref}")
        models.ensure_model(ref, allow_patterns=models.LAYA_FILES if gname == "laya" else None)
    emb = cfg.get("retrieval.embed_model")
    print(f"embedder: {emb}")
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(emb)
    print("done")
    return 0


def _approved_index() -> dict[str, Any]:
    from .approved.registry import load_index_file

    cfg = Config.load()
    ref = cfg.get("approved.index") or ""
    return load_index_file(Path(ref).expanduser() if ref else None)


def cmd_install(a: argparse.Namespace) -> int:
    from .approved import install as inst
    from .approved.registry import find

    matches = find(_approved_index(), a.skill)
    if not matches:
        print(f"no approved skill named {a.skill!r}", file=sys.stderr)
        return 1
    if len(matches) > 1:
        print(f"{len(matches)} approved skills are named {a.skill!r}; pass the id instead:", file=sys.stderr)
        for m in matches:
            print(f"  {m['id']}  (tier {m['tier']})", file=sys.stderr)
        return 1
    try:
        dest = inst.install(matches[0], a.confirm, scope=a.scope)
    except inst.InstallError as e:
        print(str(e), file=sys.stderr)
        return 2
    print(f"installed to {dest}")
    return 0


def cmd_scan(a: argparse.Namespace) -> int:
    from dataclasses import asdict

    from .approved.scan import scan_dir, verdict

    findings = scan_dir(Path(a.path))
    if a.json:
        _print_json({"verdict": verdict(findings), "findings": [asdict(f) for f in findings]})
    else:
        print(f"verdict: {verdict(findings)}  ({len(findings)} findings)")
        for f in findings:
            print(f"  [{f.severity:6s}] {f.rule:26s} {f.file}:{f.line}  {f.excerpt}")
    return 1 if verdict(findings) == "fail" else 0


def cmd_approved(a: argparse.Namespace) -> int:
    idx = _approved_index()
    skills = idx["skills"]
    if a.query:
        q = a.query.lower()
        skills = [s for s in skills if q in s["name"].lower() or q in s["description"].lower()]
    by_tier: dict[int, int] = {}
    for s in idx["skills"]:
        by_tier[s["tier"]] = by_tier.get(s["tier"], 0) + 1
    print(f"approved index: {idx['count']} skills  " + "  ".join(f"tier {k}: {v}" for k, v in sorted(by_tier.items())))
    for s in skills[: a.limit]:
        print(f"  t{s['tier']}  {s['name']:32s} {s['repo']}  {s['description'][:70]}")
    return 0


def cmd_hook(a: argparse.Namespace) -> int:
    from . import hook

    return hook.main([a.event])


def cmd_mcp(a: argparse.Namespace) -> int:
    from .mcp_server import main as mcp_main

    mcp_main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="skill-issue", description="A fast local skill router for coding agents.")
    p.add_argument("--version", action="version", version=f"skill-issue {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("route", help="route one prompt and show which skills would be injected")
    r.add_argument("prompt")
    r.add_argument("--json", action="store_true")
    r.add_argument("-v", "--verbose", action="store_true", help="also show every gated candidate")
    r.add_argument("--cwd")
    r.add_argument("--gate", help="override the gate for this call (runs in-process)")
    r.add_argument("--mode", choices=MODES, help="override the mode for this call (runs in-process)")
    r.add_argument("--no-daemon", action="store_true", help="load models in this process instead of the daemon")
    r.add_argument("--wait", type=float, default=180, help="seconds to wait for the daemon to load models")
    r.set_defaults(fn=cmd_route)

    i = sub.add_parser("index", help="list installed skills the router can see")
    i.add_argument("--cwd")
    i.add_argument("--json", action="store_true")
    i.add_argument("-v", "--verbose", action="store_true")
    i.add_argument("--warm", action="store_true", help="also build the embedding cache now")
    i.set_defaults(fn=cmd_index)

    m = sub.add_parser("mode", help="show or set the mode")
    m.add_argument("mode", nargs="?", choices=MODES)
    m.set_defaults(fn=cmd_mode)

    c = sub.add_parser("config", help="show, locate, or edit the config file")
    c.add_argument("action", choices=["show", "path", "set"])
    c.add_argument("key", nargs="?")
    c.add_argument("value", nargs="?")
    c.set_defaults(fn=cmd_config)

    d = sub.add_parser("daemon", help="manage the background daemon")
    d.add_argument("action", choices=["start", "stop", "restart", "status", "run"])
    d.add_argument("--port", type=int)
    d.add_argument("--wait", type=float, default=0, help="seconds to wait for models (start/restart)")
    d.set_defaults(fn=cmd_daemon)

    sub.add_parser("doctor", help="check the setup").set_defaults(fn=cmd_doctor)

    md = sub.add_parser("models", help="download model weights now")
    md.add_argument("action", choices=["download"])
    md.add_argument("--gate")
    md.set_defaults(fn=cmd_models)

    ins = sub.add_parser("install", help="install an approved skill (asks for confirmation)")
    ins.add_argument("skill", help="approved skill name or id")
    ins.add_argument("--confirm", help="first 12 characters of the skill's sha256, to confirm non-interactively")
    ins.add_argument("--scope", choices=["user", "project"], default="user")
    ins.set_defaults(fn=cmd_install)

    sc = sub.add_parser("scan", help="statically scan a skill folder (never runs anything)")
    sc.add_argument("path")
    sc.add_argument("--json", action="store_true")
    sc.set_defaults(fn=cmd_scan)

    ap = sub.add_parser("approved", help="browse the approved index")
    ap.add_argument("query", nargs="?")
    ap.add_argument("--limit", type=int, default=30)
    ap.set_defaults(fn=cmd_approved)

    hk = sub.add_parser("hook", help="run a Claude Code hook event (used by the plugin)")
    hk.add_argument("event", choices=["prompt", "session-start"])
    hk.set_defaults(fn=cmd_hook)

    sub.add_parser("mcp", help="run the MCP server on stdio").set_defaults(fn=cmd_mcp)
    return p


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = build_parser().parse_args(argv)
    rc: int = args.fn(args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
