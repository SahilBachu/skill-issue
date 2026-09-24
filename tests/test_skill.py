from pathlib import Path

from skillissue.skill import body_excerpt, hash_skill_dir, list_scripts, load_skill_dir, parse_skill_md

from .conftest import write_skill


def test_parse_frontmatter_and_body():
    fm, body = parse_skill_md("---\nname: a\ndescription: does a\n---\n# Title\nhello")
    assert fm == {"name": "a", "description": "does a"}
    assert body == "# Title\nhello"


def test_parse_crlf_and_bom():
    fm, body = parse_skill_md("﻿---\r\nname: a\r\ndescription: x\r\n---\r\nbody")
    assert fm["name"] == "a"
    assert body == "body"


def test_parse_no_frontmatter():
    fm, body = parse_skill_md("# Just markdown")
    assert fm == {}
    assert body == "# Just markdown"


def test_parse_invalid_yaml_falls_back():
    fm, _ = parse_skill_md("---\nname: a\ndescription: Use when: things: happen\n---\nx")
    assert fm["name"] == "a"
    assert "Use when" in fm["description"]


def test_body_excerpt_drops_code_and_limits():
    text = "intro\n```py\nsecret_code()\n```\n" + "word " * 1000
    ex = body_excerpt(text, 50)
    assert "secret_code" not in ex
    assert len(ex) <= 50


def test_hash_changes_with_any_file(tmp_path: Path):
    d = write_skill(tmp_path, "s", "desc", extra={"scripts/run.py": "print(1)"})
    h1 = hash_skill_dir(d)
    assert h1 == hash_skill_dir(d)
    (d / "scripts" / "run.py").write_text("print(2)", encoding="utf-8")
    assert hash_skill_dir(d) != h1


def test_hash_depends_on_paths(tmp_path: Path):
    a = write_skill(tmp_path / "a", "s", "desc", extra={"x.txt": "1"})
    b = write_skill(tmp_path / "b", "s", "desc", extra={"y.txt": "1"})
    assert hash_skill_dir(a) != hash_skill_dir(b)


def test_load_skill_dir(tmp_path: Path):
    d = write_skill(tmp_path, "pdf", "Work with PDFs", "body text", extra={"scripts/x.sh": "echo hi"})
    s = load_skill_dir(d, source="user")
    assert s.name == "pdf"
    assert s.description == "Work with PDFs"
    assert s.scripts == ["scripts/x.sh"]
    assert len(s.sha256) == 64
    assert list_scripts(d) == ["scripts/x.sh"]


def test_when_to_use_appended(tmp_path: Path):
    d = tmp_path / "w"
    d.mkdir()
    (d / "SKILL.md").write_text("---\nname: w\ndescription: base\nwhen_to_use: on tuesdays\n---\n", encoding="utf-8")
    assert load_skill_dir(d).description == "base on tuesdays"
