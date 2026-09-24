"""Pick the core benchmark catalog and split it into train / val / test by skill cluster.

Why by cluster: near-duplicate skills (for example two "code-review" skills from different repos)
must land in the same split, or fine-tuning on one leaks the other into the test set.

    python -m bench.select_core
Outputs data/bench/core_skills.json (ids + split + siblings) and caches pool embeddings.
"""

from __future__ import annotations

import json
import random
import sys
from typing import Any

import numpy as np

from bench.corpus import CACHE, PERMISSIVE, POOL, ROOT

OUT = ROOT / "data" / "bench" / "core_skills.json"
EMB = CACHE / "pool_emb.npy"
EMB_IDS = CACHE / "pool_emb_ids.json"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
BODY_CHARS = 1500

CORE_SIZE = 300
DUP_COS = 0.85  # above this, two skills count as duplicates for core selection
CLUSTER_COS = 0.80  # above this, two core skills must share a split (0.75 chains into one giant cluster)
SPLITS = {"train": 0.60, "val": 0.15, "test": 0.25}
SEED = 20260924
EXCLUDE_NAMES = {"template-skill", "template", "example-skill", "skill-template", "my-skill"}


def load_pool(permissive_only: bool = True) -> list[dict[str, Any]]:
    out = []
    with POOL.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if permissive_only and r.get("license") not in PERMISSIVE:
                continue
            if r["name"].lower() in EXCLUDE_NAMES or len(r["description"]) < 20 or len(r["body"]) < 200:
                continue
            out.append(r)
    return out


def pool_embeddings(pool: list[dict[str, Any]]) -> np.ndarray:
    ids = [r["id"] for r in pool]
    if EMB.is_file() and EMB_IDS.is_file() and json.loads(EMB_IDS.read_text()) == ids:
        return np.load(EMB)
    from skillissue.retrieval import STEmbedder
    from skillissue.skill import Skill

    emb = STEmbedder(EMBED_MODEL, device="cuda")
    texts = [
        Skill(id=r["id"], name=r["name"], description=r["description"], body=r["body"]).routing_text(BODY_CHARS)
        for r in pool
    ]
    vecs = emb.encode_docs(texts)
    np.save(EMB, vecs)
    EMB_IDS.write_text(json.dumps(ids))
    return vecs


def _stars(r: dict[str, Any]) -> int:
    try:
        return int(r.get("stars") or 0)
    except (TypeError, ValueError):
        return 0


def main() -> None:
    rng = random.Random(SEED)
    pool = load_pool()
    vecs = pool_embeddings(pool)
    print(f"pool (permissive, filtered): {len(pool)}", file=sys.stderr)
    idx_by_id = {r["id"]: i for i, r in enumerate(pool)}

    # Priority queues per source: tier-1 first, then round-robin across sources.
    sr_stars = {}
    for line in (CACHE / "skillret" / "data" / "skills.jsonl").open(encoding="utf-8"):
        s = json.loads(line)
        sr_stars[f"skillret:{s['id']}"] = int(s.get("stars") or 0)
    tier1 = [i for i, r in enumerate(pool) if r.get("tier") == 1]
    queues: dict[str, list[int]] = {}
    for i, r in enumerate(pool):
        if r.get("tier") == 1:
            continue
        if r["origin"] == "skillret":
            if sr_stars.get(r["id"], 0) < 200:
                continue
            key = "skillret"
        else:
            key = r["repo"]
        queues.setdefault(key, []).append(i)
    for q in queues.values():
        rng.shuffle(q)
    order = list(tier1)
    rng.shuffle(order)
    keys = sorted(queues)
    while any(queues[k] for k in keys):
        for k in keys:
            if queues[k]:
                order.append(queues[k].pop())

    chosen: list[int] = []
    names: set[str] = set()
    for i in order:
        if len(chosen) >= CORE_SIZE:
            break
        nm = pool[i]["name"].lower()
        if nm in names:
            continue
        if chosen and float(np.max(vecs[chosen] @ vecs[i])) >= DUP_COS:
            continue
        chosen.append(i)
        names.add(nm)
    print(f"core: {len(chosen)}", file=sys.stderr)

    sub = vecs[chosen]
    sim = sub @ sub.T
    np.fill_diagonal(sim, -1)
    # Union-find clusters at CLUSTER_COS.
    parent = list(range(len(chosen)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(*np.where(sim >= CLUSTER_COS)):
        if a < b:
            parent[find(a)] = find(b)
    clusters: dict[int, list[int]] = {}
    for a in range(len(chosen)):
        clusters.setdefault(find(a), []).append(a)
    comps = sorted(clusters.values(), key=len, reverse=True)
    rng.shuffle(comps)
    comps.sort(key=len, reverse=True)  # big clusters first so they don't overflow a small split
    targets = {k: v * len(chosen) for k, v in SPLITS.items()}
    fill = dict.fromkeys(SPLITS, 0)
    split_of: dict[int, str] = {}
    for comp in comps:
        # Put the cluster where it is furthest below target.
        k = max(SPLITS, key=lambda s: (targets[s] - fill[s]) / targets[s])
        for a in comp:
            split_of[a] = k
        fill[k] += len(comp)
    print(f"split sizes: {fill}; clusters: {len(comps)}, largest {len(comps[0])}", file=sys.stderr)

    skills_out = []
    for a, i in enumerate(chosen):
        r = pool[i]
        # Siblings: most similar core skills in the same split (used for hard negatives).
        same = [b for b in np.argsort(-sim[a]) if split_of[int(b)] == split_of[a] and sim[a, b] > 0.45][:3]
        skills_out.append(
            {
                "id": r["id"],
                "name": r["name"],
                "split": split_of[a],
                "repo": r["repo"],
                "license": r["license"],
                "origin": r["origin"],
                "siblings": [pool[chosen[int(b)]]["id"] for b in same],
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"seed": SEED, "embed_model": EMBED_MODEL, "skills": skills_out}, indent=1), encoding="utf-8"
    )
    print(f"wrote {OUT}", file=sys.stderr)
    _ = idx_by_id


if __name__ == "__main__":
    main()
