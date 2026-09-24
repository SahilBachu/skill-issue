import numpy as np
import pytest

from skillissue.config import Config
from skillissue.gates import Calibration, RetrievalGate, fit_platt
from skillissue.inject import render_context
from skillissue.retrieval import BM25Index, HybridRetriever, normalize_text
from skillissue.router import Router
from skillissue.skill import Skill

from .conftest import SKILLS


@pytest.fixture
def skills() -> list[Skill]:
    return [Skill(id=n, name=n, description=d, body=b) for n, (d, b) in SKILLS.items()]


def test_normalize_splits_identifiers():
    assert normalize_text("pdf-processing") == "pdf processing"
    assert normalize_text("parseConfig") == "parse Config"


def test_bm25_finds_by_name_parts(skills):
    idx = BM25Index(skills)
    scores = idx.scores("merge two pdf files")
    assert skills[int(np.argmax(scores))].name == "pdf-tools"


def test_bm25_empty_query(skills):
    assert BM25Index(skills).scores("the a of").sum() == 0


def test_hybrid_ranks_and_is_deterministic(skills, hash_embedder):
    r = HybridRetriever(skills, hash_embedder)
    a = [c.skill.name for c in r.retrieve("slow sql query needs an index", 5)]
    b = [c.skill.name for c in r.retrieve("slow sql query needs an index", 5)]
    assert a == b
    assert a[0] == "sql-tuning"


def test_bm25_only_mode(skills):
    r = HybridRetriever(skills, None)
    assert r.retrieve("deploy to vercel", 3)[0].skill.name == "vercel-deploy"


def test_fit_platt_recovers_direction():
    rng = np.random.default_rng(0)
    x = rng.normal(size=4000)
    p = 1 / (1 + np.exp(-(2.0 * x - 1.0)))
    y = (rng.random(4000) < p).astype(float)
    a, b = fit_platt(x, y)
    assert 1.6 < a < 2.4
    assert -1.4 < b < -0.6


def test_calibration_apply_bounds():
    c = Calibration(1.0, 0.0)
    out = c.apply(np.array([-1000.0, 0.0, 1000.0]))
    assert out[0] < 1e-10 and abs(out[1] - 0.5) < 1e-9 and out[2] > 1 - 1e-10


def _router(skills, hash_embedder, threshold):
    gate = RetrievalGate({"cosine": 10.0, "log_bm25": 1.0, "rrf60": 0.0}, Calibration(1.0, -4.0, threshold))
    return Router(Config(), embedder=hash_embedder, gate=gate, skills=skills, load_models=False)


def test_router_selects_relevant_skill(skills, hash_embedder):
    res = _router(skills, hash_embedder, 0.5).route("extract the tables from this pdf file")
    assert [c.skill.name for c in res.selected] == ["pdf-tools"]
    assert res.catalog_size == len(skills)
    assert set(res.timings) == {"catalog", "retrieve", "gate", "total"}


def test_router_abstains_below_threshold(skills, hash_embedder):
    res = _router(skills, hash_embedder, 0.999).route("what's the weather like")
    assert res.selected == []
    assert render_context(res) == ""


def test_render_context_installed(skills, hash_embedder):
    res = _router(skills, hash_embedder, 0.5).route("extract the tables from this pdf file")
    ctx = render_context(res)
    assert ctx.startswith("<skill-issue>") and ctx.endswith("</skill-issue>")
    assert "pdf-tools" in ctx and "Skill tool" in ctx


def test_render_context_uninstalled_requires_confirmation(hash_embedder):
    s = Skill(id="approved:x", name="xlsx-helper", description="Edit Excel spreadsheets and formulas", installed=False,
              tier=1, license="MIT", sha256="a" * 64, scripts=["scripts/recalc.py"],
              meta={"repo": "o/r", "commit": "c" * 40, "path": "skills/xlsx"})
    gate = RetrievalGate({"cosine": 10.0}, Calibration(1.0, -2.0, 0.5))
    res = Router(Config(), embedder=hash_embedder, gate=gate, skills=[s], load_models=False).route("edit excel spreadsheet formulas")
    ctx = render_context(res)
    assert "NOT INSTALLED" in ctx
    assert "Never install a skill yourself" in ctx
    assert "--confirm" in ctx and "scripts/recalc.py" in ctx


def test_route_result_to_dict(skills, hash_embedder):
    d = _router(skills, hash_embedder, 0.5).route("pdf tables").to_dict()
    assert {"mode", "gate", "threshold", "selected", "candidates", "timings_ms"} <= set(d)
    assert all(0 <= c["prob"] <= 1 for c in d["candidates"])
