"""Gate registry. `load_gate(cfg)` builds whatever the config names."""

from __future__ import annotations

from typing import Any

from ..config import Config
from .base import Calibration, Gate, fit_platt
from .retrieval_gate import RetrievalGate

GATE_NAMES = ("laya", "cross-encoder", "retrieval", "typesafe")

__all__ = ["GATE_NAMES", "Calibration", "Gate", "RetrievalGate", "fit_platt", "load_gate"]


def load_gate(cfg: Config, **overrides: Any) -> Gate:
    from .. import models

    name = overrides.get("name") or cfg.get("gate.name", "laya")
    ref = overrides.get("model") or cfg.get("gate.model") or models.DEFAULT_GATE_MODELS.get(name, "")
    device = overrides.get("device") or cfg.get("gate.device", "auto")
    gate: Gate
    if name == "laya":
        from .laya_gate import LayaGate

        gate = LayaGate(models.ensure_model(ref, allow_patterns=models.LAYA_FILES), device=device)
    elif name == "cross-encoder":
        from .cross_encoder import CrossEncoderGate

        gate = CrossEncoderGate(models.ensure_model(ref), device=device)
    elif name == "retrieval":
        gate = _retrieval_gate(cfg)
    elif name == "typesafe":
        from .typesafe_gate import TypeSafeGate

        gate = TypeSafeGate()
    else:
        raise ValueError(f"unknown gate {name!r}; choose one of {', '.join(GATE_NAMES)}")
    thr = cfg.get("gate.threshold")
    if thr is not None:
        gate.calibration.threshold = float(thr)
    return gate


def _retrieval_gate(cfg: Config) -> RetrievalGate:
    """Feature weights and calibration fitted by the benchmark on the validation split.
    `gate.retrieval_weights` / `gate.retrieval_calibration` (inline TOML tables) override them."""
    import json
    from importlib import resources

    data: dict[str, Any] = {}
    try:
        data = json.loads((resources.files("skillissue") / "data" / "retrieval_gate.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    weights = cfg.get("gate.retrieval_weights") or data.get("weights")
    cal = cfg.get("gate.retrieval_calibration") or data.get("calibration")
    return RetrievalGate(weights, Calibration.from_dict(cal) if cal else None)
