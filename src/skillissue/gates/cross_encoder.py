"""Cross-encoder reranker gate (sentence-transformers CrossEncoder).

Works with any HF cross-encoder: bge-reranker-v2-m3, gte-reranker-modernbert-base,
ms-marco-MiniLM, or our fine-tuned checkpoint. Raw score is the model's relevance logit."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..retrieval import Candidate
from .base import Calibration, Gate, clip_prompt, gate_text, load_gate_meta, resolve_device


class CrossEncoderGate(Gate):
    name = "cross-encoder"

    def __init__(
        self,
        model: str | Path,
        device: str = "auto",
        calibration: Calibration | None = None,
        body_chars: int = 300,
        max_length: int = 512,
    ):
        from sentence_transformers import CrossEncoder

        meta = load_gate_meta(Path(model)) if Path(str(model)).exists() else {}
        super().__init__(
            calibration or Calibration.from_dict(meta.get("calibration")), int(meta.get("body_chars", body_chars))
        )
        self.model_ref = str(model)
        # A fine-tuned checkpoint records the max length it was trained with.
        max_length = int((meta.get("args") or {}).get("max_length", max_length))
        dev = resolve_device(device)
        import torch

        kwargs = {}
        if dev == "cuda":
            kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
        self.model = CrossEncoder(self.model_ref, device=dev, max_length=max_length, trust_remote_code=False, **kwargs)
        self._lock = threading.Lock()

    def raw_scores(self, prompt: str, candidates: Sequence[Candidate]) -> np.ndarray:
        if not candidates:
            return np.zeros(0)
        import torch

        pairs = [(clip_prompt(prompt), gate_text(c.skill, self.body_chars)) for c in candidates]
        with self._lock:
            scores = self.model.predict(
                pairs,
                batch_size=64,
                show_progress_bar=False,
                activation_fn=torch.nn.Identity(),
                convert_to_numpy=True,
            )
        arr = np.asarray(scores, dtype=np.float64)
        return arr.reshape(len(candidates), -1)[:, -1] if arr.ndim > 1 else arr
