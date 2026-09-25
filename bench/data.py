"""Benchmark data access: the skill pool (with cached embeddings), core skills, prompt splits."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from bench.corpus import ROOT
from bench.select_core import load_pool, pool_embeddings
from skillissue.skill import Skill

BENCH_DIR = ROOT / "data" / "bench"
RESULTS = ROOT / "bench" / "results"


@dataclass
class Pool:
    records: list[dict[str, Any]]
    skills: list[Skill]
    vecs: np.ndarray
    index: dict[str, int]
    names: list[str]

    def near_dups(self, ids: list[str], cos: float = 0.90) -> set[int]:
        """Pool indices that duplicate any of `ids` (same normalized name, or cosine >= cos)."""
        if not ids:
            return set()
        rows = [self.index[i] for i in ids if i in self.index]
        sims = self.vecs @ self.vecs[rows].T
        out = set(np.where(sims.max(axis=1) >= cos)[0].tolist())
        want = {self.names[r] for r in rows}
        out |= {i for i, n in enumerate(self.names) if n in want}
        return out


def _norm_name(n: str) -> str:
    return re.sub(r"[^a-z0-9]", "", n.lower().split(":")[-1])


@cache
def pool() -> Pool:
    recs = load_pool()
    vecs = pool_embeddings(recs)
    skills = [
        Skill(
            id=r["id"],
            name=r["name"],
            description=r["description"],
            body=r["body"],
            source=r["repo"],
            sha256=r["sha256"],
            license=r["license"],
        )
        for r in recs
    ]
    return Pool(recs, skills, vecs, {r["id"]: i for i, r in enumerate(recs)}, [_norm_name(r["name"]) for r in recs])


@cache
def core() -> dict[str, dict[str, Any]]:
    data = json.loads((BENCH_DIR / "core_skills.json").read_text(encoding="utf-8"))
    return {s["id"]: s for s in data["skills"]}


def load_split(name: str) -> list[dict[str, Any]]:
    path = BENCH_DIR / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def required_skills(p: dict[str, Any]) -> list[str]:
    """Skills that must be in the catalog for this prompt to be a fair test: its gold labels,
    plus the anchor skill a hard negative was written against (that is what makes it hard)."""
    req = list(p["labels"])
    if p.get("anchor") and p["anchor"] not in req:
        req.append(p["anchor"])
    return req


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
