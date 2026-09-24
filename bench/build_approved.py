"""Build src/skillissue/data/approved-index.json from the pinned corpus.

    python -m bench.corpus fetch
    python -m bench.build_approved

Also writes data/approved/rejected.jsonl (every skill left out and why) and
src/skillissue/data/APPROVED_NOTICES.md (attribution for each source repo)."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from bench.corpus import CACHE, CORPUS, LOCK, ROOT, SOURCES
from skillissue.approved.registry import build_index
from skillissue.skill import MAX_TEXT_BYTES

OUT = ROOT / "src" / "skillissue" / "data" / "approved-index.json"
NOTICES = ROOT / "src" / "skillissue" / "data" / "APPROVED_NOTICES.md"
REJECTED = ROOT / "data" / "approved" / "rejected.jsonl"
_GH_LINK = re.compile(r"github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", re.I)


def list_mentions(locks: dict[str, Any]) -> dict[str, int]:
    """For each repo, how many curated lists link to it (README links or hosted copies)."""
    counts: dict[str, set[str]] = {}
    for repo, _tier, role in SOURCES:
        if role != "list":
            continue
        root = CORPUS / repo.replace("/", "__") / locks[repo]["commit"][:12]
        text = ""
        for p in root.rglob("*"):
            if p.is_file() and p.name.lower().startswith("readme"):
                text += p.read_text(encoding="utf-8", errors="replace") + "\n"
        for m in _GH_LINK.finditer(text):
            target = m.group(1).lower().removesuffix(".git").rstrip(".")
            counts.setdefault(target, set()).add(repo)
    return {k: len(v) for k, v in counts.items()}


def files_for(rec: dict[str, Any]) -> Iterator[tuple[str, str]]:
    root = CORPUS / rec["repo"].replace("/", "__") / rec["commit"][:12] / rec["path"]
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.stat().st_size <= MAX_TEXT_BYTES:
            try:
                yield p.relative_to(root).as_posix(), p.read_bytes().decode("utf-8")
            except UnicodeDecodeError:
                continue


def main() -> None:
    locks = json.loads(LOCK.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (CACHE / "github_skills.jsonl").open(encoding="utf-8")]
    mentions = list_mentions(locks)
    index, rejected = build_index(records, files_for, mentions)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(index, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    REJECTED.parent.mkdir(parents=True, exist_ok=True)
    with REJECTED.open("w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tiers: dict[int, int] = {}
    for s in index["skills"]:
        tiers[s["tier"]] = tiers.get(s["tier"], 0) + 1
    repos = sorted({(s["repo"], locks[s["repo"]]["commit"]) for s in index["skills"]})
    lics: dict[str, set[str]] = {}
    for s in index["skills"]:
        lics.setdefault(s["repo"], set()).add(s["license"])
    lines = [
        "# Approved index: sources and licenses",
        "",
        "The approved index stores each skill's name, description, a short excerpt of its SKILL.md,",
        "and metadata. That text comes from the repositories below, under their licenses.",
        "Skill scripts are never included; installs fetch the pinned commit from GitHub.",
        "",
        "| Repository | Commit | License(s) |",
        "|---|---|---|",
    ]
    for repo, commit in repos:
        lines.append(f"| [{repo}](https://github.com/{repo}) | `{commit[:12]}` | {', '.join(sorted(lics[repo]))} |")
    # MIT and Apache-2.0 require the license notice to travel with copied text: include each
    # source repo's own LICENSE file verbatim.
    lines += ["", "## License texts", ""]
    for repo, commit in repos:
        root = CORPUS / repo.replace("/", "__") / commit[:12]
        lic_files = sorted(p for p in root.iterdir() if p.is_file() and p.name.upper().startswith(("LICENSE", "LICENCE")))
        lines += [f"### {repo}", ""]
        if lic_files:
            lines += ["```text", lic_files[0].read_text(encoding="utf-8", errors="replace").strip(), "```", ""]
        else:
            lines += ["No top-level LICENSE file. Each included skill carries its own license file in the source repo.", ""]
    NOTICES.write_text("\n".join(lines) + "\n", encoding="utf-8")
    reasons: dict[str, int] = {}
    for r in rejected:
        key = r["reason"].split(" not permissive")[0] if "license" in r["reason"] else r["reason"]
        reasons[key] = reasons.get(key, 0) + 1
    print(f"approved: {index['count']} skills, tiers {dict(sorted(tiers.items()))}", file=sys.stderr)
    print(f"rejected: {len(rejected)} {reasons}", file=sys.stderr)


if __name__ == "__main__":
    main()
