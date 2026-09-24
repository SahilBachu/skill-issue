"""Catalog construction for a given size N.

Prompts are packed into chunks. Each chunk gets one catalog of exactly N skills (or the whole
pool when N is larger): the chunk's required skills (gold labels, plus the anchor skill of each
hard negative), then random distractors from the pool.

Distractors exclude near-duplicates of any required skill (same normalized name, or embedding
cosine >= 0.90). Without that, a copy of the gold skill from another repo would count as a wrong
answer. This makes large catalogs slightly easier than real life, where duplicates exist; the
README says so."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from bench.data import Pool, required_skills


@dataclass
class Chunk:
    catalog: list[int]  # pool indices
    prompts: list[dict[str, Any]]


def build_chunks(prompts: list[dict[str, Any]], size: int, pool: Pool, seed: int = 0) -> list[Chunk]:
    rng = random.Random(seed * 1_000_003 + size)
    n_pool = len(pool.skills)
    size = min(size, n_pool)
    budget = max(1, size // 2)  # at most half the catalog is required skills
    order = list(prompts)
    rng.shuffle(order)
    chunks: list[tuple[list[str], list[dict[str, Any]]]] = []
    cur_req: list[str] = []
    cur: list[dict[str, Any]] = []
    for p in order:
        req = [r for r in required_skills(p) if r not in cur_req]
        if cur and len(cur_req) + len(req) > budget:
            chunks.append((cur_req, cur))
            cur_req, cur = [], []
            req = required_skills(p)
        cur_req += req
        cur.append(p)
    if cur:
        chunks.append((cur_req, cur))
    out = []
    for req_ids, ps in chunks:
        req_idx = [pool.index[i] for i in req_ids]
        blocked = pool.near_dups(req_ids) | set(req_idx)
        if size >= n_pool:
            # Whole pool: keep required skills, drop their near-duplicates.
            fill = [i for i in range(n_pool) if i not in blocked]
        else:
            need = size - len(req_idx)
            fill = []
            candidates = list(range(n_pool))
            rng.shuffle(candidates)
            for i in candidates:
                if len(fill) >= need:
                    break
                if i not in blocked:
                    fill.append(i)
        catalog = req_idx + fill
        # Shuffle: retrieval breaks score ties by position, and gold skills must not sit first.
        rng.shuffle(catalog)
        out.append(Chunk(catalog=catalog, prompts=ps))
    return out
