"""Merge cloud GPU outputs back into the local benchmark caches.

    python -m cloud.ingest --from <dir with the contents of MyDrive/skill-issue-cloud/out>

Handles whatever is present:
  scores_<system>_bench.json      -> .cache/bench/scores/<cache key>.pkl  (bench.run reads these)
  scores_<system>_skillret.json   -> .cache/bench/skillret_scores/<system>.json
  sr_bench.json                   -> SkillRouter retrieval runs + SR-Rank scores for bench.run
  skillret_skillrouter.json       -> SkillRouter rows in bench/results/skillret.json
  gte-mb-ft/, minilm-ft/, laya-ft/ -> checkpoints/ (for serving and latency runs)
Then run: python -m bench.run --systems ... && python -m bench.skillret_eval && python -m bench.report"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import shutil
import sys
from pathlib import Path
from typing import Any

from bench.corpus import CACHE, ROOT
from bench.data import RESULTS, load_split, pool, save_json
from bench.run import SCORE_CACHE, retrieval_cache_path
from bench.systems import SYSTEMS
from skillissue.retrieval import Candidate


def merge_scores(path: Path, system: str) -> None:
    key = SYSTEMS[system].cache_key
    target = SCORE_CACHE / f"{key}.pkl"
    data: dict[str, float] = pickle.loads(target.read_bytes()) if target.is_file() else {}
    new = json.loads(path.read_text(encoding="utf-8"))
    data.update(new)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(pickle.dumps(data))
    print(f"{system}: +{len(new)} bench scores -> {target.name}", file=sys.stderr)


def prompts_for(split: str) -> list[dict[str, Any]]:
    if split == "handwritten":
        p = ROOT / "data" / "bench" / "handwritten.jsonl"
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    return load_split(split)


def ingest_sr_bench(path: Path) -> None:
    """SkillRouter's own retrieval results become 'srouter' retrieval caches; SR-Rank scores
    become the skillrouter system's score cache."""
    P = pool()
    data = json.loads(path.read_text(encoding="utf-8"))
    for tag, runs in data["runs"].items():
        split, size = tag.split("|")
        prompts = prompts_for(split)
        if not all(p["id"] in runs for p in prompts):
            print(f"  {tag}: incomplete ({len(runs)}/{len(prompts)}), skipped", file=sys.stderr)
            continue
        out: dict[str, dict[str, Any]] = {}
        for p in prompts:
            cands = []
            for r, (idx, cos) in enumerate(runs[p["id"]]):
                cands.append(Candidate(P.skills[idx], r, 0.0, 0.0, float(cos), len(P.skills), r))
            out[p["id"]] = {
                "cands": cands,
                "ms": float("nan"),
                "catalog": int(size) if int(size) < len(P.skills) else len(P.skills),
            }
        cp = retrieval_cache_path("srouter", split, int(size), prompts)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_bytes(pickle.dumps(out))
        print(f"  srouter retrieval {tag}: {len(out)} prompts", file=sys.stderr)
    tmp = CACHE / "cloud" / "_sr_scores.json"
    tmp.write_text(json.dumps(data["scores"]), encoding="utf-8")
    merge_scores(tmp, "skillrouter")


def ingest_skillret_sr(path: Path, seed: int = 3, n: int = 1000) -> None:
    from bench.skillret_eval import evaluate, load

    _, queries = load()
    queries = random.Random(seed).sample(queries, min(n, len(queries)))
    data = json.loads(path.read_text(encoding="utf-8"))
    out = RESULTS / "skillret.json"
    res = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {"systems": {}}
    res["systems"]["skillrouter-retrieval"] = evaluate(data["retrieval"], queries)
    res["systems"]["skillrouter"] = evaluate(data["reranked"], queries)
    save_json(out, res)
    print(f"skillret skillrouter: {res['systems']['skillrouter']}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default=str(CACHE / "cloud" / "out"))
    a = ap.parse_args()
    src = Path(a.src)
    for f in sorted(src.glob("scores_*_bench.json")):
        merge_scores(f, f.stem.removeprefix("scores_").removesuffix("_bench"))
    for f in sorted(src.glob("scores_*_skillret.json")):
        name = f.stem.removeprefix("scores_").removesuffix("_skillret")
        dst = CACHE / "bench" / "skillret_scores" / f"{name}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        print(f"{name}: skillret scores -> {dst}", file=sys.stderr)
    if (src / "sr_bench.json").is_file():
        ingest_sr_bench(src / "sr_bench.json")
    if (src / "skillret_skillrouter.json").is_file():
        ingest_skillret_sr(src / "skillret_skillrouter.json")
    for ck in ("gte-mb-ft", "minilm-ft", "laya-ft"):
        d = src / ck
        if (d / "skill_issue_gate.json").is_file() and any(d.glob("*.safetensors")):
            dst = ROOT / "checkpoints" / ck
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(d, dst, ignore=shutil.ignore_patterns("checkpoint-*"))
            print(f"checkpoint {ck} -> {dst}", file=sys.stderr)


if __name__ == "__main__":
    main()
