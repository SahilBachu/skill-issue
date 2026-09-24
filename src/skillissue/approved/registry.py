"""The approved index: which public skills approved mode may suggest, pinned by content hash.

Trust tiers
  1  official registries (anthropics/skills, openai/skills, huggingface/skills,
     anthropics/claude-plugins-official)
  2  the skill's repo is referenced by at least two well-known curated lists
  3  neither, but the skill passes our static scan with no high or medium findings

Every tier must also: have a permissive license we can redistribute metadata under, and have no
`high` scan findings. The index stores metadata, the hash, and a short excerpt used for routing.
It never stores or ships skill scripts."""

from __future__ import annotations

import json
import logging
from importlib import resources
from pathlib import Path
from typing import Any

from ..config import Config
from ..skill import Skill, body_excerpt
from .scan import scan_files, summarize

log = logging.getLogger(__name__)

OFFICIAL_REPOS = {"anthropics/skills", "openai/skills", "huggingface/skills", "anthropics/claude-plugins-official"}
PERMISSIVE = {"Apache-2.0", "MIT", "BSD-3-Clause", "BSD-2-Clause", "ISC", "0BSD", "Unlicense", "CC-BY-4.0"}
INDEX_VERSION = 1


def assign_tier(repo: str, list_mentions: int, scan_verdict: str) -> int | None:
    if scan_verdict == "fail":
        return None
    if repo in OFFICIAL_REPOS:
        return 1
    if list_mentions >= 2:
        return 2
    if scan_verdict == "pass":
        return 3
    return None


def build_index(
    records: list[dict[str, Any]],
    files_for: Any,
    list_mentions: dict[str, int],
    excerpt_chars: int = 1500,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the index. `files_for(record)` yields (relpath, text) for the skill's files.

    Returns (index, rejected) where rejected lists every skill left out and why."""
    skills: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_names: dict[str, int] = {}
    for r in records:
        lic = r.get("license")
        if lic not in PERMISSIVE:
            rejected.append({"id": r["id"], "reason": f"license {lic!r} not permissive or unknown"})
            continue
        findings = scan_files(files_for(r))
        summary = summarize(findings)
        mentions = int(list_mentions.get(r["repo"].lower(), 0))
        tier = assign_tier(r["repo"], mentions, summary["verdict"])
        if tier is None:
            reason = (
                "static scan found high-severity issues"
                if summary["verdict"] == "fail"
                else "tier 3 requires a clean scan"
            )
            rejected.append({"id": r["id"], "reason": reason, "scan": summary})
            continue
        skills.append(
            {
                "id": r["id"],
                "name": r["name"],
                "description": r["description"],
                "excerpt": body_excerpt(r.get("body", ""), excerpt_chars),
                "repo": r["repo"],
                "commit": r["commit"],
                "path": r["path"],
                "license": lic,
                "license_source": r.get("license_source"),
                "tier": tier,
                "list_mentions": mentions,
                "sha256": r["sha256"],
                "scripts": r.get("scripts", []),
                "scan": summary,
            }
        )
        seen_names[r["name"]] = seen_names.get(r["name"], 0) + 1
    # Prefer the most trusted copy when names collide; keep ids unique regardless.
    skills.sort(key=lambda s: (s["tier"], s["repo"], s["path"]))
    index = {
        "version": INDEX_VERSION,
        "tiers": {"1": "official registry", "2": "referenced by 2+ curated lists", "3": "clean static scan"},
        "count": len(skills),
        "skills": skills,
    }
    return index, rejected


def _bundled_index_path() -> Path:
    return Path(str(resources.files("skillissue") / "data" / "approved-index.json"))


def load_index_file(path: Path | None = None) -> dict[str, Any]:
    p = path or _bundled_index_path()
    data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    return data


def index_to_skills(index: dict[str, Any]) -> list[Skill]:
    out = []
    for s in index.get("skills", []):
        out.append(
            Skill(
                id=f"approved:{s['id']}",
                name=s["name"],
                description=s["description"],
                body=s.get("excerpt", ""),
                path=None,
                source=f"approved:{s['repo']}",
                sha256=s["sha256"],
                installed=False,
                license=s.get("license"),
                tier=s.get("tier"),
                scripts=list(s.get("scripts", [])),
                meta={
                    "repo": s["repo"],
                    "commit": s["commit"],
                    "path": s["path"],
                    "scan": s.get("scan", {}),
                    "index_id": s["id"],
                },
            )
        )
    return out


def load_approved_index(cfg: Config) -> list[Skill]:
    ref = cfg.get("approved.index") or ""
    try:
        idx = load_index_file(Path(ref).expanduser() if ref else None)
    except (OSError, ValueError) as e:
        log.warning("approved index unavailable: %s", e)
        return []
    return index_to_skills(idx)


def find(index: dict[str, Any], query: str) -> list[dict[str, Any]]:
    """Look up by exact id, then exact name (all matches), for `skill-issue install`."""
    q = query.removeprefix("approved:")
    by_id = [s for s in index["skills"] if s["id"] == q]
    if by_id:
        return by_id
    return [s for s in index["skills"] if s["name"] == q]
