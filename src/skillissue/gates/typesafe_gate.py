"""Optional baseline: TypeSafe's hosted Jev model (POST /v1/systemone).

Only used when TYPESAFE_API_KEY is set. One request per prompt, one `noul` question per
candidate. Request/response shape follows docs.typesafe.ai/api.md."""

from __future__ import annotations

import json
import math
import os
import urllib.request
from collections.abc import Sequence

import numpy as np

from ..retrieval import Candidate
from .base import Calibration, Gate, clip_prompt, gate_text

API_URL = os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai").rstrip("/") + "/v1/systemone"


class TypeSafeGate(Gate):
    name = "typesafe"

    def __init__(self, model: str = "jev-latest", calibration: Calibration | None = None, timeout: float = 10.0):
        super().__init__(calibration)
        key = os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise RuntimeError("TYPESAFE_API_KEY is not set")
        self.key = key
        self.model = model
        self.timeout = timeout

    def raw_scores(self, prompt: str, candidates: Sequence[Candidate]) -> np.ndarray:
        questions = {
            f"s{i}": {
                "type": "noul",
                "instructions": "Would the following skill directly help the assistant handle the user's request?\n"
                + gate_text(c.skill, self.body_chars),
            }
            for i, c in enumerate(candidates)
        }
        body = json.dumps({"state": clip_prompt(prompt), "model": self.model, "questions": questions}).encode()
        req = urllib.request.Request(
            API_URL,
            data=body,
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            answers = json.loads(r.read())["answers"]
        out = []
        for i in range(len(candidates)):
            p = float(answers[f"s{i}"]["noul"])
            p = min(max(p, 1e-6), 1 - 1e-6)
            out.append(math.log(p / (1 - p)))  # back to logit so Platt scaling applies uniformly
        return np.array(out)

    def warmup(self) -> None:
        return None
