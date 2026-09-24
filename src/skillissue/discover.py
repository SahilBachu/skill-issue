"""Find installed skills: personal, project, plugin, and the cross-agent .agents dirs."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

from . import paths
from .config import Config
from .skill import Skill, SkillParseError, load_skill_dir

log = logging.getLogger(__name__)


def _skill_dirs(root: Path, max_depth: int = 3) -> Iterator[Path]:
    """Directories under root that contain a SKILL.md, shallowest first, no nesting."""
    if not root.is_dir():
        return
    stack = [(root, 0)]
    while stack:
        d, depth = stack.pop()
        if (d / "SKILL.md").is_file() and d != root:
            yield d
            continue  # a skill shadows anything nested below it
        if depth >= max_depth:
            continue
        try:
            children = sorted((c for c in d.iterdir() if c.is_dir()), reverse=True)
        except OSError:
            continue
        for c in children:
            if not c.name.startswith("."):
                stack.append((c, depth + 1))


def _project_roots(cwd: Path) -> list[Path]:
    """cwd and its ancestors up to (and including) the git root, or just cwd."""
    cwd = cwd.resolve()
    chain = [cwd, *cwd.parents]
    for i, d in enumerate(chain):
        if (d / ".git").exists():
            return chain[: i + 1]
    return [cwd]


def _enabled_plugins(claude_home: Path) -> list[tuple[str, Path]]:
    """(plugin@marketplace, installPath) for installed plugins that are enabled in settings."""
    installed = claude_home / "plugins" / "installed_plugins.json"
    settings = claude_home / "settings.json"
    try:
        data = json.loads(installed.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    enabled: dict[str, bool] = {}
    try:
        enabled = json.loads(settings.read_text(encoding="utf-8")).get("enabledPlugins", {}) or {}
    except (OSError, ValueError):
        pass
    out: list[tuple[str, Path]] = []
    for key, entries in (data.get("plugins") or {}).items():
        if enabled and not enabled.get(key, False):
            continue
        for e in entries if isinstance(entries, list) else [entries]:
            p = e.get("installPath") if isinstance(e, dict) else None
            if p:
                out.append((key, Path(p)))
    return out


def skill_roots(cfg: Config, cwd: Path | None = None) -> list[tuple[str, str, Path]]:
    """(source label, id prefix, root dir) in precedence order."""
    cwd = cwd or Path.cwd()
    ch = paths.claude_home()
    roots: list[tuple[str, str, Path]] = []
    if cfg.get("sources.claude_project", True):
        for d in _project_roots(cwd):
            roots.append(("project", "", d / ".claude" / "skills"))
    if cfg.get("sources.claude_user", True):
        roots.append(("user", "", ch / "skills"))
    if cfg.get("sources.agents", True):
        for d in _project_roots(cwd):
            roots.append(("project", "", d / ".agents" / "skills"))
        roots.append(("user", "", Path.home() / ".agents" / "skills"))
    for extra in cfg.get("sources.extra_dirs", []) or []:
        roots.append(("extra", "", Path(extra).expanduser()))
    if cfg.get("sources.claude_plugins", True):
        for key, install in _enabled_plugins(ch):
            plugin = key.split("@", 1)[0]
            roots.append((f"plugin:{key}", f"{plugin}:", install / "skills"))
    return roots


def discover(cfg: Config, cwd: Path | None = None, roots: Iterable[tuple[str, str, Path]] | None = None) -> list[Skill]:
    """All installed skills, deduplicated by id (earlier roots win)."""
    seen: dict[str, Skill] = {}
    for source, prefix, root in roots if roots is not None else skill_roots(cfg, cwd):
        for d in _skill_dirs(root):
            try:
                s = load_skill_dir(d, source=source, id_prefix=prefix)
            except (SkillParseError, OSError) as e:
                log.debug("skip %s: %s", d, e)
                continue
            if s.meta.get("disable_model_invocation"):
                continue  # only the user can invoke these; the router must not suggest them
            seen.setdefault(s.id, s)
    return list(seen.values())
