"""Skill model and SKILL.md parsing.

A skill is a directory with a SKILL.md file: YAML frontmatter (name, description, ...) followed
by a markdown body. We treat every skill file as untrusted data. Nothing here executes anything.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_FRONTMATTER = re.compile(r"\A﻿?---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_WS = re.compile(r"[ \t]+")
_BLANKS = re.compile(r"\n{3,}")

# Files larger than this are hashed but never read as text.
MAX_TEXT_BYTES = 512 * 1024
SCRIPT_SUFFIXES = {
    ".py", ".sh", ".bash", ".zsh", ".js", ".mjs", ".cjs", ".ts", ".ps1", ".bat", ".cmd",
    ".rb", ".pl", ".php", ".go", ".rs", ".lua", ".r", ".swift", ".kt", ".java",
}


@dataclass
class Skill:
    """One Agent Skill, installed or from an index."""

    id: str
    name: str
    description: str
    body: str = ""
    path: str | None = None
    source: str = "local"
    sha256: str = ""
    installed: bool = True
    license: str | None = None
    tier: int | None = None
    scripts: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def routing_text(self, body_chars: int = 1500) -> str:
        """Text used for retrieval: name, description, and the start of the body."""
        return f"{self.name}\n{self.description}\n{body_excerpt(self.body, body_chars)}".strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "body": self.body,
            "path": self.path,
            "source": self.source,
            "sha256": self.sha256,
            "installed": self.installed,
            "license": self.license,
            "tier": self.tier,
            "scripts": list(self.scripts),
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Skill:
        return cls(
            id=d["id"],
            name=d["name"],
            description=d.get("description", ""),
            body=d.get("body", ""),
            path=d.get("path"),
            source=d.get("source", "local"),
            sha256=d.get("sha256", ""),
            installed=bool(d.get("installed", True)),
            license=d.get("license"),
            tier=d.get("tier"),
            scripts=list(d.get("scripts", [])),
            meta=dict(d.get("meta", {})),
        )


class SkillParseError(ValueError):
    pass


def parse_skill_md(text: str) -> tuple[dict[str, Any], str]:
    """Split SKILL.md into (frontmatter dict, body). Tolerates missing or malformed YAML."""
    m = _FRONTMATTER.match(text)
    if not m:
        return {}, text.strip()
    raw = m.group(1)
    body = text[m.end():].strip()
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError:
        data = _loose_frontmatter(raw)
    if not isinstance(data, dict):
        data = {}
    return data, body


def _loose_frontmatter(raw: str) -> dict[str, Any]:
    """Fallback for frontmatter that is not valid YAML (common: unquoted colons)."""
    out: dict[str, Any] = {}
    for line in raw.splitlines():
        if ":" in line and not line.startswith((" ", "\t", "-")):
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip().strip("'\"")
    return out


def body_excerpt(body: str, limit: int = 1500) -> str:
    """Start of the body with code blocks dropped and whitespace squeezed."""
    if not body or limit <= 0:
        return ""
    text = _CODE_FENCE.sub(" ", body)
    text = _WS.sub(" ", text)
    text = _BLANKS.sub("\n\n", text).strip()
    return text[:limit]


def _as_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return " ".join(str(x) for x in v)
    return str(v).strip()


def hash_skill_dir(skill_dir: Path) -> str:
    """SHA-256 over every file in the skill directory (sorted relative paths + bytes).

    This is the pin for approved-mode installs: any change to any file changes the hash.
    """
    h = hashlib.sha256()
    files = sorted(p for p in skill_dir.rglob("*") if p.is_file() and ".git" not in p.parts)
    for p in files:
        rel = p.relative_to(skill_dir).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def list_scripts(skill_dir: Path) -> list[str]:
    return sorted(
        p.relative_to(skill_dir).as_posix()
        for p in skill_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SCRIPT_SUFFIXES
    )


def load_skill_dir(skill_dir: Path, source: str = "local", id_prefix: str = "") -> Skill:
    """Load one installed skill from its directory."""
    md = skill_dir / "SKILL.md"
    if not md.is_file():
        raise SkillParseError(f"no SKILL.md in {skill_dir}")
    raw = md.read_bytes()[:MAX_TEXT_BYTES].decode("utf-8", errors="replace")
    fm, body = parse_skill_md(raw)
    name = _as_text(fm.get("name")) or skill_dir.name
    desc = _as_text(fm.get("description"))
    when = _as_text(fm.get("when_to_use"))
    if when:
        desc = f"{desc} {when}".strip()
    meta: dict[str, Any] = {}
    if fm.get("disable-model-invocation") in (True, "true"):
        meta["disable_model_invocation"] = True
    return Skill(
        id=f"{id_prefix}{name}",
        name=name,
        description=desc,
        body=body,
        path=str(skill_dir),
        source=source,
        sha256=hash_skill_dir(skill_dir),
        installed=True,
        scripts=list_scripts(skill_dir),
        meta=meta,
    )
