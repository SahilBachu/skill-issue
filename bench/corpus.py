"""Fetch public skill repos at pinned commits and build the skill pool used by the benchmark.

Nothing fetched here is executed. Tarballs are streamed; only small text files are kept, in
.cache/corpus (git-ignored). The repo only commits metadata: repo, commit, path, hash, license.

    python -m bench.corpus fetch      # resolve + pin commits, download, parse
    python -m bench.corpus pool       # merge with SkillRet into .cache/pool.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

from skillissue.skill import SCRIPT_SUFFIXES, parse_skill_md

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
CORPUS = CACHE / "corpus"
LOCK = ROOT / "data" / "corpus" / "sources.lock.json"
MANIFEST = ROOT / "data" / "corpus" / "manifest.jsonl"
POOL = CACHE / "pool.jsonl"

# (repo, tier, role). Tier 1 = official registries. Role "skills" = parse SKILL.md files;
# role "list" = curated link list used only to count cross-list mentions (trust tier 2).
SOURCES: list[tuple[str, int | None, str]] = [
    ("anthropics/skills", 1, "skills"),
    ("openai/skills", 1, "skills"),
    ("huggingface/skills", 1, "skills"),
    ("anthropics/claude-plugins-official", 1, "skills"),
    ("microsoft/skills", None, "skills"),
    ("obra/superpowers", None, "skills"),
    ("alirezarezvani/claude-skills", None, "skills"),
    ("davepoon/buildwithclaude", None, "skills"),
    ("VoltAgent/awesome-agent-skills", None, "list"),
    ("travisvn/awesome-claude-skills", None, "list"),
    ("ComposioHQ/awesome-claude-skills", None, "list"),
    ("sickn33/agentic-awesome-skills", None, "list"),
]

TEXT_SUFFIXES = SCRIPT_SUFFIXES | {
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".cfg",
    ".ini",
    ".xml",
    ".html",
    ".css",
    "",
}
MAX_KEEP = 256 * 1024


def _gh_json(url: str) -> Any:
    req = urllib.request.Request(
        url, headers={"Accept": "application/vnd.github+json", "User-Agent": "skill-issue-bench"}
    )
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def resolve(repo: str) -> dict[str, Any]:
    info = _gh_json(f"https://api.github.com/repos/{repo}")
    branch = info["default_branch"]
    sha = _gh_json(f"https://api.github.com/repos/{repo}/commits/{branch}")["sha"]
    return {"repo": repo, "commit": sha, "branch": branch, "license": (info.get("license") or {}).get("spdx_id")}


def download(repo: str, sha: str, only_readme: bool = False) -> Path:
    """Stream the tarball, keep small text files. Returns the extracted root.

    only_readme: keep just README* and SKILL.md files (for curated lists).
    """
    dest = CORPUS / repo.replace("/", "__") / sha[:12]
    done = dest / ".complete"
    if done.is_file():
        return dest
    url = f"https://codeload.github.com/{repo}/tar.gz/{sha}"
    req = urllib.request.Request(url, headers={"User-Agent": "skill-issue-bench"})
    binaries: list[dict[str, Any]] = []
    with urllib.request.urlopen(req, timeout=600) as r, tarfile.open(fileobj=r, mode="r|gz") as tar:
        for m in tar:
            if not m.isfile():
                continue
            rel = m.name.split("/", 1)[1] if "/" in m.name else m.name
            if not rel or ".." in Path(rel).parts:
                continue
            name = Path(rel).name
            if only_readme and not (name.lower().startswith("readme") or name == "SKILL.md"):
                continue
            suffix = Path(rel).suffix.lower()
            is_license = name.upper().startswith(("LICENSE", "LICENCE", "COPYING", "NOTICE"))
            f = tar.extractfile(m)
            if f is None:
                continue
            data = f.read()
            if (suffix in TEXT_SUFFIXES or is_license) and len(data) <= MAX_KEEP:
                out = dest / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(data)
            else:
                binaries.append({"path": rel, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    dest.mkdir(parents=True, exist_ok=True)
    (dest / ".binaries.json").write_text(json.dumps(binaries), encoding="utf-8")
    done.write_text("ok", encoding="utf-8")
    return dest


_LICENSE_PATTERNS = [
    ("Apache-2.0", re.compile(r"Apache License,?\s+Version 2\.0", re.I)),
    ("MIT", re.compile(r"Permission is hereby granted, free of charge", re.I)),
    ("BSD-3-Clause", re.compile(r"Redistribution and use in source and binary forms.*Neither the name", re.I | re.S)),
    ("BSD-2-Clause", re.compile(r"Redistribution and use in source and binary forms", re.I)),
    ("CC-BY-4.0", re.compile(r"Creative Commons Attribution 4\.0", re.I)),
    ("proprietary", re.compile(r"all rights reserved|proprietary|not (?:be )?(?:re)?distribut", re.I)),
]
PERMISSIVE = {"Apache-2.0", "MIT", "BSD-3-Clause", "BSD-2-Clause", "CC-BY-4.0", "ISC", "0BSD", "Unlicense"}


def detect_license(text: str) -> str | None:
    for spdx, pat in _LICENSE_PATTERNS:
        if pat.search(text[:20000]):
            return spdx
    return None


def _license_for(skill_dir: Path, root: Path, repo_spdx: str | None) -> tuple[str | None, str]:
    """Nearest LICENSE file from the skill dir up to the repo root; else the repo's SPDX id."""
    d = skill_dir
    while True:
        for f in sorted(d.iterdir()) if d.is_dir() else []:
            if f.is_file() and f.name.upper().startswith(("LICENSE", "LICENCE")):
                lic = detect_license(f.read_text(encoding="utf-8", errors="replace"))
                if lic:
                    return lic, f.relative_to(root).as_posix()
        if d == root:
            break
        d = d.parent
    if repo_spdx and repo_spdx != "NOASSERTION":
        return repo_spdx, "github-api"
    return None, "none"


def _dir_hash(skill_dir: Path, root: Path, binaries: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Same scheme as skillissue.skill.hash_skill_dir, including binaries we did not keep."""
    rel_dir = skill_dir.relative_to(root).as_posix()
    entries: dict[str, bytes] = {}
    for p in skill_dir.rglob("*"):
        if p.is_file():
            entries[p.relative_to(skill_dir).as_posix()] = hashlib.sha256(p.read_bytes()).digest()
    prefix = rel_dir + "/"
    for b in binaries:
        if b["path"].startswith(prefix):
            entries[b["path"][len(prefix) :]] = bytes.fromhex(b["sha256"])
    h = hashlib.sha256()
    for rel in sorted(entries):
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(entries[rel])
    scripts = sorted(r for r in entries if Path(r).suffix.lower() in SCRIPT_SUFFIXES)
    return h.hexdigest(), scripts


def parse_repo(lock: dict[str, Any], root: Path, tier: int | None) -> list[dict[str, Any]]:
    binaries = json.loads((root / ".binaries.json").read_text(encoding="utf-8"))
    out = []
    seen_dirs: list[Path] = []
    for md in sorted(root.rglob("SKILL.md"), key=lambda p: len(p.parts)):
        d = md.parent
        if d == root or any(parent in seen_dirs for parent in d.parents):
            continue  # shallower skill shadows nested ones
        seen_dirs.append(d)
        text = md.read_text(encoding="utf-8", errors="replace")
        fm, body = parse_skill_md(text)
        name = str(fm.get("name") or d.name).strip()
        desc = str(fm.get("description") or "").strip()
        if not desc:
            continue
        lic, lic_src = _license_for(d, root, lock.get("license"))
        sha, scripts = _dir_hash(d, root, binaries)
        rel = d.relative_to(root).as_posix()
        out.append(
            {
                "id": f"{lock['repo']}:{rel}",
                "name": name,
                "description": desc,
                "body": body,
                "repo": lock["repo"],
                "commit": lock["commit"],
                "path": rel,
                "license": lic,
                "license_source": lic_src,
                "tier": tier,
                "sha256": sha,
                "scripts": scripts,
                "origin": "github",
            }
        )
    return out


def cmd_fetch(args: argparse.Namespace) -> None:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    locks: dict[str, Any] = json.loads(LOCK.read_text(encoding="utf-8")) if LOCK.is_file() and not args.update else {}
    records: list[dict[str, Any]] = []
    for repo, tier, role in SOURCES:
        if repo not in locks:
            locks[repo] = resolve(repo)
            print(f"pinned {repo} @ {locks[repo]['commit'][:12]}", file=sys.stderr)
        lock = locks[repo]
        root = download(repo, lock["commit"], only_readme=(role == "list"))
        if role == "skills":
            recs = parse_repo(lock, root, tier)
            print(f"{repo}: {len(recs)} skills", file=sys.stderr)
            records.extend(recs)
    LOCK.write_text(json.dumps(locks, indent=2) + "\n", encoding="utf-8")
    with (CACHE / "github_skills.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # Committed manifest: metadata only, no skill content.
    with MANIFEST.open("w", encoding="utf-8") as f:
        for r in records:
            meta = {
                k: r[k]
                for k in (
                    "id",
                    "name",
                    "repo",
                    "commit",
                    "path",
                    "license",
                    "license_source",
                    "tier",
                    "sha256",
                    "scripts",
                )
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} skills", file=sys.stderr)


def load_skillret() -> list[dict[str, Any]]:
    src = CACHE / "skillret" / "data" / "skills.jsonl"
    out = []
    with src.open(encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            _, body = parse_skill_md(s["skill_md"])
            out.append(
                {
                    "id": f"skillret:{s['id']}",
                    "name": s["name"],
                    "description": s["description"],
                    "body": body,
                    "repo": s["repo"],
                    "commit": None,
                    "path": s["source_url"],
                    "license": {"mit": "MIT", "apache-2.0": "Apache-2.0"}.get(str(s["license"]).lower(), s["license"]),
                    "license_source": "skillret",
                    "tier": None,
                    "sha256": hashlib.sha256(s["skill_md"].encode("utf-8")).hexdigest(),
                    "scripts": [],
                    "origin": "skillret",
                    "skillret_id": s["id"],
                    "category": s.get("sub"),
                }
            )
    return out


def cmd_pool(args: argparse.Namespace) -> None:
    gh = [json.loads(line) for line in (CACHE / "github_skills.jsonl").open(encoding="utf-8")]
    sr = load_skillret()
    # Drop SkillRet entries that duplicate a pinned GitHub skill (same repo + name).
    gh_keys = {(r["repo"].lower(), r["name"].lower()) for r in gh}
    sr = [r for r in sr if (r["repo"].lower(), r["name"].lower()) not in gh_keys]
    with POOL.open("w", encoding="utf-8") as f:
        for r in gh + sr:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"pool: {len(gh)} github + {len(sr)} skillret = {len(gh) + len(sr)}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m bench.corpus")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--update", action="store_true", help="re-resolve commits instead of using the lock")
    sub.add_parser("pool")
    args = ap.parse_args(argv)
    {"fetch": cmd_fetch, "pool": cmd_pool}[args.cmd](args)


if __name__ == "__main__":
    main()
