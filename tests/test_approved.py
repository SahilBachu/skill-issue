import io
import tarfile
from pathlib import Path

import pytest

from skillissue.approved import install as inst
from skillissue.approved.registry import assign_tier, build_index, find, index_to_skills
from skillissue.skill import hash_skill_dir

from .conftest import write_skill


def rec(repo: str, name: str, lic: str = "MIT") -> dict:
    return {"id": f"{repo}:skills/{name}", "name": name, "description": f"{name} does things", "body": "body",
            "repo": repo, "commit": "c" * 40, "path": f"skills/{name}", "license": lic, "sha256": "f" * 64, "scripts": []}


def test_assign_tier():
    assert assign_tier("anthropics/skills", 0, "warn") == 1
    assert assign_tier("x/y", 2, "warn") == 2
    assert assign_tier("x/y", 0, "pass") == 3
    assert assign_tier("x/y", 0, "warn") is None
    assert assign_tier("anthropics/skills", 5, "fail") is None


def test_build_index_filters_license_and_scan():
    files = {
        "a": [("SKILL.md", "fine")],
        "b": [("SKILL.md", "fine")],
        "c": [("run.py", "exec(base64.b64decode(x))")],
        "d": [("SKILL.md", "fine")],
    }
    records = [rec("anthropics/skills", "a"), rec("x/y", "b", lic="proprietary"), rec("x/y", "c"), rec("x/y", "d")]
    index, rejected = build_index(records, lambda r: files[r["name"]], {})
    assert [s["name"] for s in index["skills"]] == ["a", "d"]
    assert {s["name"]: s["tier"] for s in index["skills"]} == {"a": 1, "d": 3}
    reasons = {r["id"].split("/")[-1]: r["reason"] for r in rejected}
    assert "license" in reasons["b"]
    assert "high-severity" in reasons["c"]


def test_index_to_skills_marks_uninstalled():
    index, _ = build_index([rec("anthropics/skills", "a")], lambda r: [], {})
    s = index_to_skills(index)[0]
    assert not s.installed and s.tier == 1 and s.id == "approved:anthropics/skills:skills/a"
    assert find(index, "a")[0]["name"] == "a"
    assert find(index, "approved:anthropics/skills:skills/a")


def _tarball(skill_src: Path, repo_prefix: str = "repo-abc", evil: bool = False) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in sorted(skill_src.rglob("*")):
            if p.is_file():
                tar.add(p, arcname=f"{repo_prefix}/skills/{skill_src.name}/{p.relative_to(skill_src).as_posix()}")
        tar.add(skill_src / "SKILL.md", arcname=f"{repo_prefix}/README.md")
        if evil:
            info = tarfile.TarInfo(f"{repo_prefix}/skills/{skill_src.name}/../../escape.txt")
            data = b"nope"
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


@pytest.fixture
def pinned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env):
    src = write_skill(tmp_path / "src", "xlsx-helper", "Edit spreadsheets", extra={"scripts/recalc.py": "print('hi')"})
    entry = {"name": "xlsx-helper", "description": "Edit spreadsheets", "repo": "o/r", "commit": "c" * 40,
             "path": "skills/xlsx-helper", "tier": 1, "license": "MIT", "sha256": hash_skill_dir(src),
             "scripts": ["scripts/recalc.py"], "scan": {"verdict": "pass", "counts": {}}}
    payload = {"data": _tarball(src)}
    monkeypatch.setattr(inst.urllib.request, "urlopen", lambda req, timeout=0: io.BytesIO(payload["data"]))
    return entry, payload, src


def test_install_requires_confirmation_when_not_interactive(pinned):
    entry, _, _ = pinned
    with pytest.raises(inst.InstallError, match="confirmation required"):
        inst.install(entry, None, interactive=False)


def test_install_rejects_wrong_confirmation(pinned):
    entry, _, _ = pinned
    with pytest.raises(inst.InstallError, match="does not match"):
        inst.install(entry, "0" * 12)


def test_install_verifies_hash_and_copies(pinned, env):
    entry, _, _ = pinned
    dest = inst.install(entry, entry["sha256"][:12])
    assert dest == env["claude"] / "skills" / "xlsx-helper"
    assert (dest / "scripts" / "recalc.py").read_text() == "print('hi')"
    assert hash_skill_dir(dest) == entry["sha256"]


def test_install_refuses_hash_mismatch(pinned, env):
    entry, payload, src = pinned
    (src / "scripts" / "recalc.py").write_text("print('tampered')", encoding="utf-8")
    payload["data"] = _tarball(src)
    with pytest.raises(inst.InstallError, match="hash mismatch"):
        inst.install(entry, entry["sha256"][:12])
    assert not (env["claude"] / "skills" / "xlsx-helper").exists()


def test_install_refuses_path_traversal(pinned):
    entry, payload, src = pinned
    payload["data"] = _tarball(src, evil=True)
    with pytest.raises(inst.InstallError, match="unsafe path"):
        inst.install(entry, entry["sha256"][:12])


def test_install_refuses_existing(pinned, env):
    entry, _, _ = pinned
    (env["claude"] / "skills" / "xlsx-helper").mkdir(parents=True)
    with pytest.raises(inst.InstallError, match="already exists"):
        inst.install(entry, entry["sha256"][:12])
