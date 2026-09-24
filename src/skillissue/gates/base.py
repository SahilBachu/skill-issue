"""Stage 2: the gate. Scores each retrieved candidate independently as a binary question
("is this skill relevant to this request?") and returns calibrated probabilities.

All gates share one interface so they can be swapped by config and compared in the bench."""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..retrieval import Candidate
from ..skill import Skill, body_excerpt

GATE_META_FILE = "skill_issue_gate.json"


@dataclass
class Calibration:
    """Platt scaling: p = sigmoid(a * raw + b). a=1, b=0 is identity on logits."""

    a: float = 1.0
    b: float = 0.0
    threshold: float = 0.5

    def apply(self, raw: np.ndarray) -> np.ndarray:
        z = np.clip(self.a * np.asarray(raw, dtype=np.float64) + self.b, -40, 40)
        return 1.0 / (1.0 + np.exp(-z))

    def to_dict(self) -> dict[str, float]:
        return {"a": self.a, "b": self.b, "threshold": self.threshold}

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> Calibration:
        d = d or {}
        return cls(float(d.get("a", 1.0)), float(d.get("b", 0.0)), float(d.get("threshold", 0.5)))


def fit_platt(raw: np.ndarray, y: np.ndarray, iters: int = 500, l2: float = 1e-4) -> tuple[float, float]:
    """Fit (a, b) by Newton's method on log loss. Small, dependency free, deterministic."""
    x = np.asarray(raw, dtype=np.float64)
    t = np.asarray(y, dtype=np.float64)
    # Platt's smoothed targets reduce overfitting on small sets.
    n_pos, n_neg = t.sum(), len(t) - t.sum()
    t = np.where(t > 0.5, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))
    a, b = 1.0, 0.0
    for _ in range(iters):
        z = np.clip(a * x + b, -40, 40)
        p = 1 / (1 + np.exp(-z))
        g_a = np.sum((p - t) * x) + l2 * a
        g_b = np.sum(p - t)
        w = p * (1 - p)
        h_aa = np.sum(w * x * x) + l2
        h_ab = np.sum(w * x)
        h_bb = np.sum(w) + 1e-12
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-18:
            break
        da = (h_bb * g_a - h_ab * g_b) / det
        db = (h_aa * g_b - h_ab * g_a) / det
        a, b = a - da, b - db
        if abs(da) < 1e-9 and abs(db) < 1e-9:
            break
    return float(a), float(b)


def gate_text(skill: Skill, body_chars: int = 300) -> str:
    """What the gate sees about a skill."""
    parts = [f"Skill: {skill.name}", f"Description: {skill.description.strip()}"]
    ex = body_excerpt(skill.body, body_chars)
    if ex:
        parts.append(f"Details: {ex}")
    return "\n".join(parts)


def clip_prompt(prompt: str, limit: int = 1200) -> str:
    prompt = prompt.strip()
    return prompt if len(prompt) <= limit else prompt[:limit] + " ..."


class Gate(ABC):
    """Scores (prompt, candidate) pairs. Subclasses implement raw_scores."""

    name: str = "gate"

    def __init__(self, calibration: Calibration | None = None, body_chars: int = 300):
        self.calibration = calibration or Calibration()
        self.body_chars = body_chars

    @abstractmethod
    def raw_scores(self, prompt: str, candidates: Sequence[Candidate]) -> np.ndarray:
        """Uncalibrated score per candidate (higher = more relevant)."""

    def score(self, prompt: str, candidates: Sequence[Candidate]) -> np.ndarray:
        if not candidates:
            return np.zeros(0)
        return self.calibration.apply(self.raw_scores(prompt, candidates))

    @property
    def threshold(self) -> float:
        return self.calibration.threshold

    def warmup(self) -> None:
        """Run one tiny forward pass so the first real request is not slow."""
        s = Skill(id="warmup", name="warmup", description="warm up the model")
        self.raw_scores("hello", [Candidate(s, 0, 0.0, 0.0, 0.0, 0, 0)])


def load_gate_meta(model_dir: Path | None) -> dict[str, Any]:
    if model_dir is None:
        return {}
    f = Path(model_dir) / GATE_META_FILE
    if f.is_file():
        try:
            data: dict[str, Any] = json.loads(f.read_text(encoding="utf-8"))
            return data
        except ValueError:
            return {}
    return {}


def resolve_device(device: str = "auto") -> str:
    if device and device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover
        pass
    return "cpu"


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, x))))
