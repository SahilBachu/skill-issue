"""The systems we compare. Each one = a retrieval setup + a scorer for the top-G candidates.

Scorers return raw scores; every system is then calibrated (Platt) and thresholded on the
validation split only, and frozen before it sees the test split."""

from __future__ import annotations

import gc
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from skillissue.retrieval import Candidate

RawFn = Callable[[str, Sequence[Candidate]], np.ndarray]


@dataclass
class System:
    name: str
    retrieval: str  # hybrid | bm25 | dense
    scorer: str  # laya | cross | features | skillrouter
    model: str | None = None
    n_gate: int = 12
    top_k: int = 40
    label: str = ""
    features: list[str] = field(default_factory=list)

    @property
    def cache_key(self) -> str:
        """Scores only depend on the scorer + model (not retrieval or catalog)."""
        if self.scorer == "features":
            return ""
        # v2: gate input uses a 300-char body excerpt (v1 used 600).
        return f"v2-{self.scorer}--{(self.model or '').replace('/', '--').replace(':', '_')}"


SYSTEMS: dict[str, System] = {
    s.name: s
    for s in [
        System("bm25", "bm25", "features", features=["log_bm25"], label="BM25 only"),
        System("dense", "dense", "features", features=["cosine"], label="Embeddings only (bge-small)"),
        System(
            "hybrid", "hybrid", "features", features=["cosine", "log_bm25", "rrf60"], label="Hybrid BM25 + embeddings"
        ),
        System("laya-zs", "hybrid", "laya", "convaiinnovations/laya", label="Hybrid + Laya (zero-shot)"),
        System("bge-m3-zs", "hybrid", "cross", "BAAI/bge-reranker-v2-m3", label="Hybrid + bge-reranker-v2-m3"),
        System(
            "gte-mb-zs",
            "hybrid",
            "cross",
            "Alibaba-NLP/gte-reranker-modernbert-base",
            label="Hybrid + gte-reranker-modernbert-base",
        ),
        System(
            "minilm-zs", "hybrid", "cross", "cross-encoder/ms-marco-MiniLM-L6-v2", label="Hybrid + ms-marco-MiniLM-L6"
        ),
        System("laya-ft", "hybrid", "laya", "checkpoints/laya-ft", label="Hybrid + Laya (fine-tuned)"),
        System(
            "gte-mb-ft",
            "hybrid",
            "cross",
            "checkpoints/gte-mb-ft",
            label="Hybrid + gte-reranker-modernbert (fine-tuned)",
        ),
        System("minilm-ft", "hybrid", "cross", "checkpoints/minilm-ft", label="Hybrid + MiniLM-L6 (fine-tuned)"),
    ]
}


def features(c: Candidate) -> dict[str, float]:
    return {"cosine": c.cosine, "log_bm25": math.log1p(max(c.bm25, 0.0)), "rrf60": c.rrf * 60.0}


def feature_matrix(cands: Sequence[Candidate], names: list[str]) -> np.ndarray:
    return np.array([[features(c)[n] for n in names] for c in cands], dtype=np.float64).reshape(len(cands), len(names))


class Scorer:
    """Loads the model for a system and scores candidates."""

    def __init__(self, system: System, root: Any):
        self.system = system
        self.gate: Any = None
        self.weights: np.ndarray | None = None
        if system.scorer == "features":
            return
        from skillissue import models

        ref = system.model or ""
        if ref.startswith("checkpoints/"):
            ref = str(root / ref)
        if system.scorer == "laya":
            from skillissue.gates.laya_gate import LayaGate

            self.gate = LayaGate(models.ensure_model(ref, allow_patterns=models.LAYA_FILES), device="cuda")
        elif system.scorer == "cross":
            from skillissue.gates.cross_encoder import CrossEncoderGate

            self.gate = CrossEncoderGate(ref, device="cuda")
        elif system.scorer == "skillrouter":
            from bench.skillrouter import SkillRouterReranker

            self.gate = SkillRouterReranker(device="cuda")
        else:
            raise ValueError(system.scorer)

    def raw(self, prompt: str, cands: Sequence[Candidate]) -> np.ndarray:
        if not cands:
            return np.zeros(0)
        if self.system.scorer == "features":
            x = feature_matrix(cands, self.system.features)
            if self.weights is None:
                return x[:, 0]
            return x @ self.weights[:-1] + self.weights[-1]
        return np.asarray(self.gate.raw_scores(prompt, cands), dtype=np.float64)

    def close(self) -> None:
        self.gate = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
