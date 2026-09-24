"""External benchmark: SkillRet (ThakiCloud/SKILLRET, Apache-2.0), test split.

    python -m bench.skillret_eval --systems bm25 dense hybrid gte-mb-ft skillrouter --n 1000

Their protocol: queries/test over skills/test (6,006 skills), binary qrels, document text
`name | description | skill_md`. We report nDCG@10, Recall@{1,5,10,20} and MRR@10 on a seeded
subsample of the 4,392 test queries (the full set takes hours for the larger rerankers on a
laptop GPU). SkillRet has no "no skill" queries, so this measures ranking only, not abstention.

Gate systems rerank the hybrid top-20 by gate score. SkillRet test skills are disjoint from its
train skills; our fine-tuned gates saw SkillRet *train* queries, so their numbers here are
in-distribution for query style (disclosed in the README)."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from typing import Any

import numpy as np

from bench.corpus import CACHE, ROOT
from bench.data import RESULTS, save_json
from bench.run import EMBED_MODEL, _NullEmbedder
from bench.systems import SYSTEMS, Scorer
from skillissue.retrieval import HybridRetriever, STEmbedder
from skillissue.skill import Skill, parse_skill_md

BASE = CACHE / "skillret" / "data"


def load() -> tuple[list[Skill], list[dict[str, Any]]]:
    qrels: dict[str, set[str]] = {}
    for line in (BASE / "qrels" / "test.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        qrels.setdefault(r["query_id"], set()).add(r["skill_id"])
    queries = [json.loads(x) for x in (BASE / "queries" / "test.jsonl").open(encoding="utf-8")]
    for q in queries:
        q["gold"] = qrels.get(q["id"], set(q["skill_ids"]))
    # The candidate corpus is the dataset's "skills/test" config (6,006 skills).
    skills = []
    for line in (BASE / "skills" / "test.jsonl").open(encoding="utf-8"):
        s = json.loads(line)
        _, body = parse_skill_md(s["skill_md"])
        skills.append(Skill(id=s["id"], name=s["name"], description=s["description"], body=body))
    return skills, queries


def ndcg(ranked: list[str], gold: set[str], k: int = 10) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, s in enumerate(ranked[:k]) if s in gold)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))
    return dcg / ideal if ideal else 0.0


def evaluate(rankings: dict[str, list[str]], queries: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, list[float]] = {"ndcg@10": [], "mrr@10": [], **{f"recall@{k}": [] for k in (1, 5, 10, 20)}}
    for q in queries:
        r, g = rankings[q["id"]], q["gold"]
        out["ndcg@10"].append(ndcg(r, g))
        rr = next((1 / (i + 1) for i, s in enumerate(r[:10]) if s in g), 0.0)
        out["mrr@10"].append(rr)
        for k in (1, 5, 10, 20):
            out[f"recall@{k}"].append(len(set(r[:k]) & g) / len(g))
    return {k: float(np.mean(v)) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="*", default=["bm25", "dense", "hybrid", "gte-mb-zs"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--rerank-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    skills, queries = load()
    queries = random.Random(a.seed).sample(queries, min(a.n, len(queries)))
    print(f"SkillRet: {len(skills)} skills, {len(queries)} queries", file=sys.stderr)
    emb = STEmbedder(EMBED_MODEL, device=a.device)
    vec_path = CACHE / "skillret_test_bge.npy"
    if vec_path.is_file():
        vecs = np.load(vec_path)
    else:
        vecs = emb.encode_docs([s.routing_text(1500) for s in skills])
        np.save(vec_path, vecs)
    qv = emb.model.encode(
        [emb.prefix + q["query"] for q in queries], batch_size=64, normalize_embeddings=True, show_progress_bar=False
    )
    retrievers = {
        "bm25": HybridRetriever(skills, None),
        "dense": HybridRetriever(skills, _NullEmbedder(), use_bm25=False, doc_vecs=vecs),
        "hybrid": HybridRetriever(skills, _NullEmbedder(), doc_vecs=vecs),
    }
    results: dict[str, Any] = {"n_queries": len(queries), "n_skills": len(skills), "seed": a.seed, "systems": {}}
    cands_hybrid: dict[str, list[Any]] = {}
    for mode in ("bm25", "dense", "hybrid"):
        rank: dict[str, list[str]] = {}
        for q, v in zip(queries, qv):
            cands = retrievers[mode].retrieve(q["query"], 50, query_vec=v if mode != "bm25" else None)
            rank[q["id"]] = [c.skill.id for c in cands]
            if mode == "hybrid":
                cands_hybrid[q["id"]] = cands
        if mode in a.systems:
            results["systems"][mode] = evaluate(rank, queries)
            print(mode, results["systems"][mode], file=sys.stderr)
    for name in a.systems:
        if name in ("bm25", "dense", "hybrid"):
            continue
        system = SYSTEMS[name]
        t0 = time.time()
        pre = CACHE / "bench" / "skillret_scores" / f"{name}.json"
        if pre.is_file():  # scored elsewhere (cloud/score.py), keyed "sr:<query id>|<skill id>"
            sc = json.loads(pre.read_text(encoding="utf-8"))
            rank = {}
            for q in queries:
                top = cands_hybrid[q["id"]][: a.rerank_k]
                raw = np.array([sc[f"sr:{q['id']}|{c.skill.id}"] for c in top])
                order = np.argsort(-raw, kind="stable")
                rank[q["id"]] = [top[i].skill.id for i in order] + [
                    c.skill.id for c in cands_hybrid[q["id"]][a.rerank_k :]
                ]
            results["systems"][name] = evaluate(rank, queries)
            print(name, results["systems"][name], file=sys.stderr)
            continue
        if system.model and system.model.startswith("checkpoints/") and not (ROOT / system.model).exists():
            print(f"skip {name}: not trained", file=sys.stderr)
            continue
        if system.retrieval == "srouter":
            from bench.skillrouter import SkillRouterEmbedder

            sr = SkillRouterEmbedder()
            sv_path = CACHE / "skillret_test_sr.npy"
            if sv_path.is_file():
                svecs = np.load(sv_path)
            else:
                from bench.skillrouter import doc_text

                svecs = sr.encode_docs([doc_text(s) for s in skills])
                np.save(sv_path, svecs)
            sq = sr.encode_queries([q["query"] for q in queries])
            del sr
            import torch

            torch.cuda.empty_cache()
            r = HybridRetriever(skills, _NullEmbedder(), use_bm25=False, doc_vecs=svecs)
            base = {q["id"]: r.retrieve(q["query"], 50, query_vec=v) for q, v in zip(queries, sq)}
            results["systems"]["skillrouter-retrieval"] = evaluate(
                {k: [c.skill.id for c in v] for k, v in base.items()}, queries
            )
        else:
            base = cands_hybrid
        scorer = Scorer(system, ROOT)
        rank = {}
        for q in queries:
            top = base[q["id"]][: a.rerank_k]
            raw = scorer.raw(q["query"], top)
            order = np.argsort(-raw, kind="stable")
            rank[q["id"]] = [top[i].skill.id for i in order] + [c.skill.id for c in base[q["id"]][a.rerank_k :]]
        scorer.close()
        results["systems"][name] = evaluate(rank, queries) | {"wall_s": round(time.time() - t0, 1)}
        print(name, results["systems"][name], file=sys.stderr)
    # Merge with earlier runs (same sample) so systems can be evaluated one at a time.
    out = RESULTS / "skillret.json"
    if out.is_file():
        prev = json.loads(out.read_text(encoding="utf-8"))
        if prev.get("seed") == a.seed and prev.get("n_queries") == len(queries):
            results["systems"] = {**prev.get("systems", {}), **results["systems"]}
    save_json(out, results)


if __name__ == "__main__":
    main()
