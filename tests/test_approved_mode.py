"""Approved mode end to end with the real bundled index (model-free retrieval gate)."""

from pathlib import Path

from skillissue.approved.registry import load_index_file
from skillissue.config import Config
from skillissue.gates import Calibration, RetrievalGate
from skillissue.inject import render_context
from skillissue.router import Router


def _router(mode: str) -> Router:
    cfg = Config(
        {
            "mode": mode,
            "retrieval": {"embed_model": "none"},
            "sources": {"agents": False, "claude_plugins": False},
        }
    )
    gate = RetrievalGate({"log_bm25": 1.0}, Calibration(2.0, -5.0, 0.5))
    return Router(cfg, gate=gate, load_models=False)


def test_bundled_index_is_sane():
    idx = load_index_file()
    assert idx["count"] == len(idx["skills"]) > 900
    for s in idx["skills"]:
        assert s["tier"] in (1, 2, 3)
        assert len(s["sha256"]) == 64 and len(s["commit"]) == 40
        assert s["license"] in {
            "Apache-2.0",
            "MIT",
            "BSD-3-Clause",
            "BSD-2-Clause",
            "ISC",
            "0BSD",
            "Unlicense",
            "CC-BY-4.0",
        }
        assert s["scan"]["verdict"] != "fail"


def test_installed_mode_never_suggests_uninstalled(env, tmp_path: Path):
    res = _router("installed").route("deploy this app to vercel and give me the preview link", tmp_path)
    assert all(c.skill.installed for c in res.candidates)


def test_approved_mode_suggests_uninstalled_with_provenance(env, tmp_path: Path):
    res = _router("approved").route(
        "fine-tune a small LLM with TRL SFT on Hugging Face Jobs and push it to the Hub", tmp_path
    )
    assert res.catalog_size > 900
    uninstalled = [c for c in res.selected if not c.skill.installed]
    assert uninstalled, [c.skill.name for c in res.candidates]
    s = uninstalled[0].skill
    assert s.meta["repo"] and len(s.meta["commit"]) == 40 and s.tier in (1, 2, 3)
    ctx = render_context(res)
    assert "NOT INSTALLED" in ctx and "Never install a skill yourself" in ctx and "--confirm" in ctx


def test_approved_mode_prefers_the_installed_copy(env, tmp_path: Path):
    # env installs a skill named vercel-deploy; the approved copy with the same name is hidden.
    res = _router("approved").route("deploy this app to vercel", tmp_path)
    names = [(c.skill.name, c.skill.installed) for c in res.candidates]
    assert ("vercel-deploy", True) in names
    assert ("vercel-deploy", False) not in names
