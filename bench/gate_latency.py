"""Gate latency by model, number of candidates, and input length (GPU and CPU).

    python -m bench.gate_latency --device cuda
Writes bench/results/gate_latency-<device>.json. Uses real test prompts and real skills."""

from __future__ import annotations

import argparse
import os
import random
import statistics
import time
from typing import Any

os.environ.setdefault("USE_TF", "0")

import torch

from bench.data import RESULTS, load_split, pool, save_json
from skillissue.retrieval import Candidate

MODELS = {
    "laya": ("laya", "convaiinnovations/laya"),
    "bge-m3": ("cross", "BAAI/bge-reranker-v2-m3"),
    "gte-mb": ("cross", "Alibaba-NLP/gte-reranker-modernbert-base"),
    "minilm": ("cross", "cross-encoder/ms-marco-MiniLM-L6-v2"),
}


def load(kind: str, ref: str, device: str, body_chars: int) -> Any:
    from skillissue import models

    if kind == "laya":
        from skillissue.gates.laya_gate import LayaGate

        g = LayaGate(models.ensure_model(ref, allow_patterns=models.LAYA_FILES), device=device)
    else:
        from skillissue.gates.cross_encoder import CrossEncoderGate

        g = CrossEncoderGate(ref, device=device)
    g.body_chars = body_chars
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--gs", nargs="*", type=int, default=[4, 8, 12, 20])
    ap.add_argument("--body", nargs="*", type=int, default=[0, 300, 600])
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()
    P = pool()
    prompts = [p["prompt"] for p in load_split("test")]
    rng = random.Random(0)
    out: dict[str, Any] = {
        "device": a.device,
        "gpu": torch.cuda.get_device_name(0) if a.device == "cuda" else None,
        "results": [],
    }
    for name in a.models:
        kind, ref = MODELS[name]
        g = load(kind, ref, a.device, 300)
        for body in a.body:
            g.body_chars = body
            for G in a.gs:
                times = []
                for i in range(a.reps + 3):
                    cands = [Candidate(P.skills[rng.randrange(len(P.skills))], 0, 0, 0, 0, 0, 0) for _ in range(G)]
                    prompt = prompts[rng.randrange(len(prompts))]
                    if a.device == "cuda":
                        torch.cuda.synchronize()
                    t = time.perf_counter()
                    g.raw_scores(prompt, cands)
                    if a.device == "cuda":
                        torch.cuda.synchronize()
                    if i >= 3:
                        times.append((time.perf_counter() - t) * 1000)
                times.sort()
                row = {
                    "model": name,
                    "body_chars": body,
                    "candidates": G,
                    "p50_ms": statistics.median(times),
                    "p95_ms": times[int(0.95 * (len(times) - 1))],
                }
                out["results"].append(row)
                print(row, flush=True)
        del g
        if a.device == "cuda":
            torch.cuda.empty_cache()
    save_json(RESULTS / f"gate_latency-{a.device}.json", out)


if __name__ == "__main__":
    main()
