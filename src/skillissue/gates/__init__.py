"""Gate registry. `load_gate(cfg)` builds whatever the config names."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

from ..config import Config
from .base import Calibration, Gate, fit_platt, load_gate_meta, resolve_device
from .retrieval_gate import RetrievalGate

GATE_NAMES = ("auto", "laya", "cross-encoder", "retrieval", "typesafe")

# "auto": the accurate gate where there is an accelerator, the small fast one on CPU
# (see the latency table in the README: the 150M reranker takes ~2 s per prompt on a laptop CPU).
AUTO_GPU = ("cross-encoder", "gte")
AUTO_CPU = ("cross-encoder", "minilm")

__all__ = ["GATE_NAMES", "Calibration", "Gate", "RetrievalGate", "fit_platt", "load_gate"]


def _package_json(name: str) -> dict[str, Any]:
    try:
        data: dict[str, Any] = json.loads((resources.files("skillissue") / "data" / name).read_text(encoding="utf-8"))
        return data
    except (OSError, ValueError):
        return {}


def resolve(cfg: Config, **overrides: Any) -> tuple[str, str, str]:
    """(gate name, model ref, device) after applying "auto" and the per-gate default models."""
    from .. import models

    name = overrides.get("name") or cfg.get("gate.name", "auto")
    device = resolve_device(overrides.get("device") or cfg.get("gate.device", "auto"))
    ref = overrides.get("model") or cfg.get("gate.model") or ""
    if name == "auto":
        name, slot = AUTO_GPU if device in ("cuda", "mps") else AUTO_CPU
        ref = ref or models.DEFAULT_MODELS[slot]
    ref = ref or models.DEFAULT_GATE_MODELS.get(name, "")
    return name, ref, device


def load_gate(cfg: Config, **overrides: Any) -> Gate:
    from .. import models

    name, ref, device = resolve(cfg, **overrides)
    gate: Gate
    if name == "laya":
        from .laya_gate import LayaGate

        gate = LayaGate(models.ensure_model(ref, allow_patterns=models.LAYA_FILES), device=device)
    elif name == "cross-encoder":
        from .cross_encoder import CrossEncoderGate

        path = models.ensure_model(ref)
        gate = CrossEncoderGate(path, device=device)
        if not load_gate_meta(path).get("calibration"):
            # Base models from the Hub carry no calibration; use the values fitted on our
            # validation split (bench.run), shipped in skillissue/data/gate_calibration.json.
            cal = _package_json("gate_calibration.json").get(ref)
            if cal:
                gate.calibration = Calibration.from_dict(cal)
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
    data = _package_json("retrieval_gate.json")
    weights = cfg.get("gate.retrieval_weights") or data.get("weights")
    cal = cfg.get("gate.retrieval_calibration") or data.get("calibration")
    return RetrievalGate(weights, Calibration.from_dict(cal) if cal else None)
