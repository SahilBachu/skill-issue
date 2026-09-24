from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

SKILLS = {
    "pdf-tools": (
        "Extract text and tables from PDF files, merge and split PDFs, fill PDF forms.",
        "Use pypdf to read pages.",
    ),
    "sql-tuning": ("Analyze slow SQL queries, read EXPLAIN plans, and suggest indexes.", "Run EXPLAIN ANALYZE first."),
    "vercel-deploy": ("Deploy web apps to Vercel and return the preview URL.", "Use the vercel CLI to deploy."),
    "git-commit": ("Write conventional commit messages from staged diffs.", "Read git diff --staged."),
    "react-testing": ("Write React component tests with Testing Library and Vitest.", "Render, query by role, assert."),
}


def write_skill(root: Path, name: str, desc: str, body: str = "", extra: dict[str, str] | None = None) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {desc}\n---\n\n# {name}\n\n{body}\n", encoding="utf-8"
    )
    for rel, text in (extra or {}).items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return d


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Isolated SKILL_ISSUE_HOME, CLAUDE_CONFIG_DIR and HOME with a few installed skills."""
    home = tmp_path / "home"
    claude = tmp_path / "claude"
    si = tmp_path / "si"
    for p in (home, claude, si):
        p.mkdir()
    monkeypatch.setenv("SKILL_ISSUE_HOME", str(si))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    for name, (desc, body) in SKILLS.items():
        write_skill(claude / "skills", name, desc, body)
    return {"home": home, "claude": claude, "si": si, "tmp": tmp_path}


class HashEmbedder:
    """Deterministic bag-of-words embedder for tests (no model download)."""

    name = "test-hash"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(256, dtype=np.float32)
        for tok in re.findall(r"[a-z]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % 256] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self._vec(text)


@pytest.fixture
def hash_embedder() -> HashEmbedder:
    return HashEmbedder()
