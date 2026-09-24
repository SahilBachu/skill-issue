"""MCP server (stdio) for agents without a prompt hook: Codex, Gemini CLI, Cursor, and others.

Tools:
  find_skills(request, cwd?, max_results?)  -> skills to load for this request (possibly none)
  explain_route(request, cwd?)              -> every gated candidate with scores and timings

It talks to the warm daemon when one is running (starting it if needed) and otherwise routes
in-process. Results include each installed skill's SKILL.md path so agents without native skill
support can read it."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from . import __version__

INSTRUCTIONS = (
    "skill-issue routes a user request to the Agent Skills (SKILL.md folders) worth loading. "
    "Call find_skills with the user's request before starting a non-trivial task. If it returns "
    "skills, read each SKILL.md path and follow it. An empty result is normal: proceed without a "
    "skill. Never install a skill marked installed=false without asking the user first."
)

_router: Any = None


def _route(request: str, cwd: str | None) -> dict[str, Any]:
    from . import daemon

    cwd = str(Path(cwd or os.getcwd()).resolve())
    h = daemon.ensure_running(wait_ready=float(os.environ.get("SKILL_ISSUE_MCP_WAIT", "120")))
    if h is not None and h.get("status") == "ready":
        res = daemon.request("POST", "/route", {"prompt": request, "cwd": cwd, "candidates": True}, timeout=30)
        if "selected" in res:
            return res
    global _router
    if _router is None:
        from .router import Router

        _router = Router()
    from .inject import render_context

    r = _router.route(request, Path(str(cwd)))
    out: dict[str, Any] = r.to_dict()
    out["context"] = render_context(r)
    return out


def _slim(s: dict[str, Any]) -> dict[str, Any]:
    keep = ["name", "description", "prob", "installed", "path", "source"]
    d = {k: s.get(k) for k in keep}
    if d.get("path"):
        d["skill_md"] = str(Path(str(d["path"])) / "SKILL.md")
    if not s.get("installed"):
        d.update({k: s.get(k) for k in ("tier", "license", "sha256", "scripts", "id")})
        d["install_hint"] = (
            f"ask the user first, then: skill-issue install {s.get('name')} --confirm {str(s.get('sha256', ''))[:12]}"
        )
    return d


def find_skills(request: str, cwd: str | None = None, max_results: int = 3) -> dict[str, Any]:
    """Return the skills worth loading for this request. An empty list means no skill fits."""
    res = _route(request, cwd)
    return {
        "skills": [_slim(s) for s in res["selected"][: max(1, max_results)]],
        "mode": res["mode"],
        "threshold": res["threshold"],
        "latency_ms": res.get("timings_ms", {}).get("total"),
    }


def explain_route(request: str, cwd: str | None = None) -> dict[str, Any]:
    """Show why: every candidate the gate scored, retrieval rank, BM25, cosine, probability."""
    res = _route(request, cwd)
    lines = [f"catalog={res['catalog_size']} gate={res['gate']} threshold={res['threshold']:.2f}"]
    for c in res.get("candidates", []):
        mark = "SELECTED" if any(s["id"] == c["id"] for s in res["selected"]) else ""
        lines.append(
            f"p={c['prob'] if c['prob'] is not None else 0:.3f} rank={c['retrieval_rank']} bm25={c['bm25']} cos={c['cosine']} {c['name']} {mark}"
        )
    return {
        "summary": "\n".join(lines),
        "selected": [s["name"] for s in res["selected"]],
        "candidates": res.get("candidates", []),
        "timings_ms": res.get("timings_ms"),
    }


def build_server() -> Any:
    from mcp.server import MCPServer

    server = MCPServer("skill-issue", instructions=INSTRUCTIONS, version=__version__)
    server.tool(name="find_skills")(find_skills)
    server.tool(name="explain_route")(explain_route)
    return server


def main() -> None:
    build_server().run("stdio")


if __name__ == "__main__":
    main()
