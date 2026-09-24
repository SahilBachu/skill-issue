"""No-model gate: a logistic score over retrieval features. The "embeddings only" baseline,
and the fallback when no gate model can load (for example CPU-only with tight latency)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ..retrieval import Candidate
from .base import Calibration, Gate

# Fitted by `bench` on the validation split; these are only starting values.
DEFAULT_WEIGHTS = {"cosine": 1.0, "log_bm25": 0.0, "rrf60": 0.0}


def features(c: Candidate) -> dict[str, float]:
    return {"cosine": c.cosine, "log_bm25": math.log1p(max(c.bm25, 0.0)), "rrf60": c.rrf * 60.0}


class RetrievalGate(Gate):
    name = "retrieval"

    def __init__(self, weights: dict[str, float] | None = None, calibration: Calibration | None = None):
        super().__init__(calibration)
        self.weights = dict(weights or DEFAULT_WEIGHTS)

    def raw_scores(self, prompt: str, candidates: Sequence[Candidate]) -> np.ndarray:
        return np.array(
            [sum(self.weights.get(k, 0.0) * v for k, v in features(c).items()) for c in candidates],
            dtype=np.float64,
        )

    def warmup(self) -> None:
        return None
