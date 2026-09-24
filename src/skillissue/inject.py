"""Turn a RouteResult into the context block the agent sees."""

from __future__ import annotations

from .retrieval import Candidate
from .router import RouteResult

TIER_LABEL = {1: "official registry", 2: "listed in multiple curated lists", 3: "passed static scan"}


def _short(text: str, n: int = 220) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3].rstrip() + "..."


def _line(c: Candidate) -> str:
    s = c.skill
    p = f"{c.prob:.2f}" if c.prob is not None else "?"
    if s.installed:
        return f"- {s.name} [{p}]: {_short(s.description)}"
    repo = s.meta.get("repo", s.source)
    commit = str(s.meta.get("commit") or "")[:12]
    scripts = ", ".join(s.scripts[:6]) + (" ..." if len(s.scripts) > 6 else "") if s.scripts else "none"
    tier = TIER_LABEL.get(s.tier or 0, "unvetted")
    return (
        f"- {s.name} [{p}] NOT INSTALLED: {_short(s.description, 160)}\n"
        f"  source: {repo}/{s.meta.get('path', '')} @ {commit} | trust: tier {s.tier} ({tier}) | "
        f"license: {s.license} | sha256: {s.sha256[:16]}... | scripts: {scripts}"
    )


def render_context(result: RouteResult) -> str:
    """Empty string when nothing was selected: the hook then injects nothing."""
    if not result.selected:
        return ""
    installed = [c for c in result.selected if c.skill.installed]
    missing = [c for c in result.selected if not c.skill.installed]
    lines = ["<skill-issue>"]
    if installed:
        lines.append("Installed skills that match this request (local skill router, confidence in brackets):")
        lines += [_line(c) for c in installed]
        lines.append("If one fits, load it with the Skill tool before you start. Ignore any that do not fit.")
    if missing:
        if installed:
            lines.append("")
        lines.append("Approved skills that may help but are NOT installed:")
        lines += [_line(c) for c in missing]
        lines.append(
            "Never install a skill yourself. If one would clearly help, show the user its name, source, "
            "trust tier, hash and scripts, and ask. Install only after they say yes, with: "
            "skill-issue install <name> --confirm <first 12 chars of sha256>"
        )
    lines.append("</skill-issue>")
    return "\n".join(lines)
