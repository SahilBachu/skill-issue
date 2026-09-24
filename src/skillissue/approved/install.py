"""Install an approved skill. Only ever runs on an explicit user command, and only after the user
confirms by retyping the start of the pinned hash (or answering yes on a terminal).

Steps: show details -> confirm -> download the pinned commit -> extract only the skill folder
-> recompute the hash -> refuse on mismatch -> copy into the skills dir. Nothing is executed."""

from __future__ import annotations

import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

from .. import paths
from ..skill import hash_skill_dir

CONFIRM_CHARS = 12


class InstallError(RuntimeError):
    pass


def describe(s: dict[str, Any]) -> str:
    scan = s.get("scan", {})
    counts = ", ".join(f"{k} x{v}" for k, v in sorted(scan.get("counts", {}).items())) or "no findings"
    scripts = "\n".join(f"    {x}" for x in s.get("scripts", [])) or "    (none)"
    return (
        f"name:     {s['name']}\n"
        f"source:   https://github.com/{s['repo']}/tree/{s['commit']}/{s['path']}\n"
        f"tier:     {s['tier']}\n"
        f"license:  {s['license']}\n"
        f"sha256:   {s['sha256']}\n"
        f"scan:     {scan.get('verdict', '?')} ({counts})\n"
        f"scripts:\n{scripts}\n"
        f"about:    {s['description'][:300]}"
    )


def target_dir(name: str, scope: str, cwd: Path | None = None) -> Path:
    safe = PurePosixPath(name).name.replace("\\", "_")
    if not safe or safe in (".", ".."):
        raise InstallError(f"bad skill name {name!r}")
    base = (cwd or Path.cwd()) / ".claude" / "skills" if scope == "project" else paths.claude_home() / "skills"
    return base / safe


def fetch_skill(s: dict[str, Any], dest: Path) -> None:
    """Download the pinned commit tarball and extract only the skill's folder into dest."""
    url = f"https://codeload.github.com/{s['repo']}/tar.gz/{s['commit']}"
    prefix = s["path"].strip("/") + "/"
    req = urllib.request.Request(url, headers={"User-Agent": "skill-issue"})
    found = False
    with urllib.request.urlopen(req, timeout=300) as r, tarfile.open(fileobj=r, mode="r|gz") as tar:
        for m in tar:
            if not m.isfile():
                continue  # no symlinks, devices, or hardlinks
            rel = m.name.split("/", 1)[1] if "/" in m.name else ""
            if not rel.startswith(prefix):
                continue
            sub = PurePosixPath(rel[len(prefix) :])
            if sub.is_absolute() or ".." in sub.parts:
                raise InstallError(f"unsafe path in archive: {rel}")
            f = tar.extractfile(m)
            if f is None:
                continue
            out = dest.joinpath(*sub.parts)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(f.read())
            found = True
    if not found:
        raise InstallError("skill folder not found in the pinned commit")


def install(
    s: dict[str, Any],
    confirm: str | None,
    scope: str = "user",
    cwd: Path | None = None,
    interactive: bool | None = None,
) -> Path:
    print(describe(s))
    print()
    want = s["sha256"][:CONFIRM_CHARS]
    if confirm is not None:
        if confirm.strip().lower() != want:
            raise InstallError(f"confirmation does not match: expected the first {CONFIRM_CHARS} chars of the sha256")
    else:
        tty = sys.stdin.isatty() if interactive is None else interactive
        if not tty:
            raise InstallError(
                f"confirmation required. Review the details above, then run:\n  skill-issue install {s['name']} --confirm {want}"
            )
        ans = input(f"Install {s['name']} into {scope} skills? Type 'yes' to confirm: ").strip().lower()
        if ans != "yes":
            raise InstallError("not installed")
    dest = target_dir(s["name"], scope, cwd)
    if dest.exists():
        raise InstallError(f"{dest} already exists; remove it first if you want to replace it")
    with tempfile.TemporaryDirectory(prefix="skill-issue-") as tmp:
        stage = Path(tmp) / s["name"]
        stage.mkdir()
        fetch_skill(s, stage)
        got = hash_skill_dir(stage)
        if got != s["sha256"]:
            raise InstallError(f"hash mismatch: pinned {s['sha256']}, downloaded {got}. Not installed.")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(stage, dest)
    return dest
