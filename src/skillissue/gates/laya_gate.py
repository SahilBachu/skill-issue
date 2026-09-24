"""Laya gate (convaiinnovations/laya, ModernBERT-large decision model).

Each candidate is one row: a two-option `choice` question ("use" vs "skip") with the request
and the skill in the state. We avoid Laya's `noul` type on purpose: the model card documents a
label bias on the English checkpoint, and recommends a two-option choice instead. All rows go
through one batched forward pass. The raw score is logit(use) - logit(skip)."""

from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..retrieval import Candidate
from .base import Calibration, Gate, clip_prompt, gate_text, load_gate_meta, resolve_device

INSTRUCTIONS = "Should the assistant load this skill to handle the user request?"
OPTIONS = {
    "use": "the skill directly helps with this request",
    "skip": "the skill is unrelated or only loosely related",
}
MAX_LEN = 512
HEAD_MAX_LEN = 64


def build_state(prompt: str, skill_block: str) -> str:
    return f"User request: {clip_prompt(prompt)}\n\n{skill_block}"


def question() -> dict[str, Any]:
    """Question in laya's internal format (t / ins / crit)."""
    return {"t": "choice", "ins": INSTRUCTIONS, "crit": dict(OPTIONS)}


def encode_rows(
    tok: Any, prompt: str, skill_blocks: Sequence[str], max_len: int = MAX_LEN
) -> list[list[dict[str, Any]]]:
    from laya.common import QTYPES, build_sequence

    q = question()
    rows = []
    for block in skill_blocks:
        ids, markers = build_sequence(tok, build_state(prompt, block), q, max_len=max_len, head_max_len=HEAD_MAX_LEN)
        rows.append([{"ids": ids, "markers": markers, "qtype": QTYPES["choice"]}])
    return rows


class LayaGate(Gate):
    name = "laya"

    def __init__(
        self, model_dir: str | Path, device: str = "auto", calibration: Calibration | None = None, body_chars: int = 300
    ):
        os.environ.setdefault("USE_TF", "0")  # laya docs: avoids a hang when TensorFlow is installed
        import laya
        import torch

        meta = load_gate_meta(Path(model_dir))
        cal = calibration or Calibration.from_dict(meta.get("calibration"))
        super().__init__(cal, int(meta.get("body_chars", body_chars)))
        self.model_dir = str(model_dir)
        self.meta = meta
        dev = resolve_device(device)
        self.agent = laya.load(self.model_dir, device=dev)
        self.model = self.agent.model
        self.tok = self.agent.tok
        self.device = self.agent.device
        self.dtype = self.agent.dtype
        self.model.eval()
        self._lock = threading.Lock()
        self._torch = torch

    def raw_scores(self, prompt: str, candidates: Sequence[Candidate]) -> np.ndarray:
        if not candidates:
            return np.zeros(0)
        from laya.common import collate_items

        torch = self._torch
        rows = encode_rows(self.tok, prompt, [gate_text(c.skill, self.body_chars) for c in candidates])
        b = collate_items(rows, self.tok.pad_token_id)
        dev = self.device
        use_amp = dev.type in ("cuda",) and self.dtype != torch.float32
        with self._lock, torch.inference_mode():
            ctx = torch.autocast(device_type=dev.type, dtype=self.dtype) if use_amp else _null()
            with ctx:
                logits, _ = self.model(
                    b["input_ids"].to(dev),
                    b["attention_mask"].to(dev),
                    b["marker_pos"].to(dev),
                    b["marker_mask"].to(dev),
                    b["qtype"].to(dev),
                )
        lg = logits.float().cpu().numpy()
        diff: np.ndarray = (lg[:, 0] - lg[:, 1]).astype(np.float64)
        return diff


class _null:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *a: object) -> None:
        return None
