import json
from pathlib import Path

from skillissue.config import Config, dumps_toml
from skillissue.discover import discover

from .conftest import write_skill


def test_discovers_user_and_project_skills(env, tmp_path: Path):
    proj = tmp_path / "proj"
    (proj / ".git").mkdir(parents=True)
    write_skill(proj / ".claude" / "skills", "project-only", "A project skill")
    names = {s.name for s in discover(Config(), cwd=proj)}
    assert {"pdf-tools", "sql-tuning", "project-only"} <= names


def test_project_skill_shadows_user_skill(env, tmp_path: Path):
    proj = tmp_path / "proj"
    (proj / ".git").mkdir(parents=True)
    write_skill(proj / ".claude" / "skills", "pdf-tools", "PROJECT VERSION")
    s = {x.name: x for x in discover(Config(), cwd=proj)}
    assert s["pdf-tools"].description == "PROJECT VERSION"


def test_disable_model_invocation_is_skipped(env):
    d = env["claude"] / "skills" / "manual-only"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        "---\nname: manual-only\ndescription: x\ndisable-model-invocation: true\n---\n", encoding="utf-8"
    )
    assert "manual-only" not in {s.name for s in discover(Config(), cwd=env["tmp"])}


def test_plugin_skills_are_namespaced_and_respect_enabled(env):
    plug = env["tmp"] / "plugcache" / "myplug"
    write_skill(plug / "skills", "helper", "Plugin helper")
    other = env["tmp"] / "plugcache" / "offplug"
    write_skill(other / "skills", "hidden", "Disabled plugin skill")
    (env["claude"] / "plugins").mkdir()
    (env["claude"] / "plugins" / "installed_plugins.json").write_text(
        json.dumps(
            {
                "version": 2,
                "plugins": {
                    "myplug@mkt": [{"installPath": str(plug)}],
                    "offplug@mkt": [{"installPath": str(other)}],
                },
            }
        ),
        encoding="utf-8",
    )
    (env["claude"] / "settings.json").write_text(
        json.dumps({"enabledPlugins": {"myplug@mkt": True, "offplug@mkt": False}}), encoding="utf-8"
    )
    ids = {s.id for s in discover(Config(), cwd=env["tmp"])}
    assert "myplug:helper" in ids
    assert "offplug:hidden" not in ids


def test_sources_can_be_disabled(env):
    cfg = Config({"sources": {"claude_user": False}})
    assert discover(cfg, cwd=env["tmp"]) == []


def test_config_roundtrip(env):
    cfg = Config()
    cfg.set("mode", "approved")
    cfg.set("gate.threshold", 0.42)
    cfg.set("sources.extra_dirs", ["/a", "/b"])
    path = cfg.save()
    back = Config.load(path)
    assert back.mode == "approved"
    assert back.get("gate.threshold") == 0.42
    assert back.get("sources.extra_dirs") == ["/a", "/b"]
    assert back.get("retrieval.top_k") == 40  # defaults survive


def test_bad_mode_falls_back():
    assert Config({"mode": "yolo"}).mode == "installed"


def test_dumps_toml_escapes():
    assert 'x = "a\\"b"' in dumps_toml({"x": 'a"b'})
