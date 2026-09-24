"""Score (prompt, skill) requests with a gate. Runs anywhere with a GPU (or slowly on CPU).

    python -m cloud.score --model laya:/content/out/laya-ft --requests cloud_data/requests_bench.jsonl --out scores.json
    python -m cloud.score --model cross:Alibaba-NLP/gte-reranker-modernbert-base ...
    python -m cloud.score --model skillrouter ...

Output: {request key: raw score}. Requests are grouped by prompt so each gate call is one batch,
exactly like the router does at runtime."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from skillissue.retrieval import Candidate
from skillissue.skill import Skill


def load_scorer(spec: str, device: str) -> Any:
    kind, _, ref = spec.partition(":")
    if kind == "laya":
        from skillissue import models
        from skillissue.gates.laya_gate import LayaGate

        return LayaGate(models.ensure_model(ref, allow_patterns=models.LAYA_FILES), device=device)
    if kind == "cross":
        from skillissue.gates.cross_encoder import CrossEncoderGate

        return CrossEncoderGate(ref, device=device)
    if kind == "skillrouter":
        from bench.skillrouter import SkillRouterReranker

        return SkillRouterReranker(device=device)
    raise ValueError(spec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="laya:<ref> | cross:<ref> | skillrouter")
    ap.add_argument("--requests", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for line in Path(a.requests).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        groups.setdefault(r["prompt"], []).append(r)
    out_path = Path(a.out)
    scores: dict[str, float] = json.loads(out_path.read_text(encoding="utf-8")) if out_path.is_file() else {}
    scorer = load_scorer(a.model, a.device)
    t0 = time.time()
    done = 0
    for i, (prompt, reqs) in enumerate(groups.items()):
        todo = [r for r in reqs if r["key"] not in scores]
        if not todo:
            continue
        cands = [Candidate(Skill(**r["skill"]), 0, 0.0, 0.0, 0.0, 0, 0) for r in todo]
        raw = scorer.raw_scores(prompt, cands)
        for r, v in zip(todo, raw):
            scores[r["key"]] = float(v)
        done += len(todo)
        if i % 200 == 0:
            print(f"{i}/{len(groups)} prompts, {done} pairs, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
            out_path.write_text(json.dumps(scores), encoding="utf-8")  # resumable
    out_path.write_text(json.dumps(scores), encoding="utf-8")
    print(f"scored {done} pairs in {time.time() - t0:.0f}s -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
