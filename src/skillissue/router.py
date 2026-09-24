"""The router: catalog -> hybrid retrieval -> gate -> threshold.

Injecting nothing is a normal outcome. The router never installs anything; in approved mode it
can return skills that are not installed, marked `installed=False`, and the caller must ask the
user before doing anything with them."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import paths
from .config import Config
from .discover import discover, skill_roots
from .gates import Gate, load_gate
from .retrieval import Candidate, EmbeddingCache, Embedder, HybridRetriever, STEmbedder
from .skill import Skill

log = logging.getLogger(__name__)


@dataclass
class RouteResult:
    prompt: str
    mode: str
    selected: list[Candidate]
    candidates: list[Candidate]
    threshold: float
    gate: str
    catalog_size: int
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self, with_candidates: bool = True) -> dict[str, Any]:
        def c2d(c: Candidate) -> dict[str, Any]:
            s = c.skill
            return {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "path": s.path,
                "source": s.source,
                "installed": s.installed,
                "tier": s.tier,
                "license": s.license,
                "sha256": s.sha256,
                "scripts": s.scripts,
                "prob": None if c.prob is None else round(float(c.prob), 4),
                "retrieval_rank": c.rank,
                "bm25": round(c.bm25, 3),
                "cosine": round(c.cosine, 4),
            }

        d: dict[str, Any] = {
            "mode": self.mode,
            "gate": self.gate,
            "threshold": self.threshold,
            "catalog_size": self.catalog_size,
            "selected": [c2d(c) for c in self.selected],
            "timings_ms": {k: round(v, 2) for k, v in self.timings.items()},
        }
        if with_candidates:
            d["candidates"] = [c2d(c) for c in self.candidates]
        return d


def _fingerprint(roots: Sequence[tuple[str, str, Path]]) -> tuple[Any, ...]:
    """Cheap change detector: every SKILL.md path with its mtime and size (depth <= 3)."""
    items: list[tuple[str, int, int]] = []
    for _, _, root in roots:
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            depth = len(Path(dirpath).relative_to(root).parts)
            if depth >= 3:
                dirnames[:] = []
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for fn in filenames:
                p = Path(dirpath) / fn
                try:
                    st = p.stat()
                except OSError:
                    continue
                items.append((str(p), int(st.st_mtime_ns), st.st_size))
    return tuple(sorted(items))


class Router:
    def __init__(
        self,
        cfg: Config | None = None,
        embedder: Embedder | None = None,
        gate: Gate | None = None,
        skills: Sequence[Skill] | None = None,
        load_models: bool = True,
    ):
        self.cfg = cfg or Config.load()
        self._lock = threading.RLock()
        self._retrievers: dict[Any, tuple[HybridRetriever, list[Skill]]] = {}
        self._fixed_skills = list(skills) if skills is not None else None
        self.embedder = embedder
        self.gate = gate
        if load_models:
            emb = str(self.cfg.get("retrieval.embed_model") or "none")
            if self.embedder is None and emb.lower() != "none":
                self.embedder = STEmbedder(emb, device=_embed_device(self.cfg))
            if self.gate is None:
                self.gate = load_gate(self.cfg)
        self._approved: list[Skill] | None = None

    # ---- catalog -------------------------------------------------------------------
    def approved_skills(self) -> list[Skill]:
        if self._approved is None:
            from .approved.registry import load_approved_index

            self._approved = load_approved_index(self.cfg)
        return self._approved

    def catalog(self, cwd: Path | None = None) -> tuple[HybridRetriever, list[Skill]]:
        with self._lock:
            if self._fixed_skills is not None:
                key: Any = ("fixed", id(self._fixed_skills))
                if key not in self._retrievers:
                    self._retrievers = {key: (self._build(self._fixed_skills), self._fixed_skills)}
                return self._retrievers[key]
            roots = skill_roots(self.cfg, cwd)
            key = (self.cfg.mode, _fingerprint(roots))
            hit = self._retrievers.get(key)
            if hit is not None:
                return hit
            installed = discover(self.cfg, cwd, roots)
            skills = list(installed)
            if self.cfg.mode == "approved":
                have = {s.name for s in installed}
                skills += [s for s in self.approved_skills() if s.name not in have]
            built = (self._build(skills), skills)
            # Keep a handful of catalogs (different project dirs) warm.
            if len(self._retrievers) > 8:
                self._retrievers.pop(next(iter(self._retrievers)))
            self._retrievers[key] = built
            return built

    def _build(self, skills: Sequence[Skill]) -> HybridRetriever:
        cache = EmbeddingCache(paths.index_dir() / "embeddings.npz") if self._fixed_skills is None else None
        return HybridRetriever(
            skills,
            self.embedder,
            body_chars=int(self.cfg.get("retrieval.body_chars", 1500)),
            rrf_k=int(self.cfg.get("retrieval.rrf_k", 60)),
            cache=cache,
        )

    # ---- routing -------------------------------------------------------------------
    def route(self, prompt: str, cwd: Path | None = None, top_k: int | None = None) -> RouteResult:
        t0 = time.perf_counter()
        retriever, skills = self.catalog(cwd)
        t1 = time.perf_counter()
        k = int(top_k or self.cfg.get("retrieval.top_k", 40))
        cands = retriever.retrieve(prompt, k)
        t2 = time.perf_counter()
        n_gate = int(self.cfg.get("gate.max_candidates", 12))
        gated = cands[:n_gate]
        gate = self.gate
        if gate is None:
            raise RuntimeError("router has no gate loaded")
        probs = gate.score(prompt, gated) if gated else []
        for c, p in zip(gated, probs):
            c.prob = float(p)
        t3 = time.perf_counter()
        thr = gate.threshold
        max_skills = int(self.cfg.get("gate.max_skills", 3))
        selected = sorted((c for c in gated if (c.prob or 0.0) >= thr), key=lambda c: -(c.prob or 0.0))[:max_skills]
        return RouteResult(
            prompt=prompt,
            mode=self.cfg.mode,
            selected=selected,
            candidates=gated,
            threshold=thr,
            gate=gate.name,
            catalog_size=len(skills),
            timings={
                "catalog": (t1 - t0) * 1000,
                "retrieve": (t2 - t1) * 1000,
                "gate": (t3 - t2) * 1000,
                "total": (t3 - t0) * 1000,
            },
        )


def _embed_device(cfg: Config) -> str | None:
    from .gates.base import resolve_device

    return resolve_device(cfg.get("gate.device", "auto"))
