"""Build (request, skill, relevant?) training pairs for the gate.

    python -m training.build_pairs

Sources
  1. Our train split (data/bench/train.jsonl): 180 train-only skills, positives, hard negatives,
     "none" prompts, multi-skill prompts.
  2. SkillRet train split (Apache-2.0): a subsample of queries over SkillRet *train* skills.

Negatives are mined with the same hybrid retriever the router uses, so the gate learns to reject
exactly the near misses it will see in production.

Leakage guard: every val/test core skill and each of its near-duplicates in the pool (same
normalized name or cosine >= 0.90) is removed from every training catalog, and SkillRet queries
whose gold skill falls in that set are dropped. Nothing from the val or test prompt splits is used.
"""

from __future__ import annotations

import json
import random
import sys
from typing import Any

import numpy as np

from bench.catalogs import build_chunks
from bench.corpus import CACHE, ROOT
from bench.data import Pool, core, load_split, pool
from bench.run import EMBED_MODEL, _NullEmbedder
from skillissue.retrieval import HybridRetriever, STEmbedder
from skillissue.skill import Skill, parse_skill_md

OUT = CACHE / "train" / "pairs.jsonl"
N_CANDS = 16
CATALOG = 1000
SKILLRET_QUERIES = 3000
SEED = 11


def blocked_indices(P: Pool) -> set[int]:
    held = [sid for sid, s in core().items() if s["split"] in ("val", "test")]
    return P.near_dups(held) | {P.index[s] for s in held}


def our_pairs(P: Pool, blocked: set[int], emb: STEmbedder) -> list[dict[str, Any]]:
    prompts = load_split("train")
    # Build a pool view without held-out skills.
    keep = [i for i in range(len(P.skills)) if i not in blocked]
    sub = Pool(
        [P.records[i] for i in keep],
        [P.skills[i] for i in keep],
        P.vecs[keep],
        {P.records[i]["id"]: j for j, i in enumerate(keep)},
        [P.names[i] for i in keep],
    )
    qv = emb.model.encode(
        [emb.prefix + p["prompt"] for p in prompts], batch_size=128, normalize_embeddings=True, show_progress_bar=False
    )
    qvec = {p["id"]: v for p, v in zip(prompts, qv)}
    out = []
    for size in (100, CATALOG):
        for ch in build_chunks(prompts, size, sub, seed=size):
            r = HybridRetriever([sub.skills[i] for i in ch.catalog], _NullEmbedder(), doc_vecs=sub.vecs[ch.catalog])
            for p in ch.prompts:
                cands = r.retrieve(p["prompt"], N_CANDS, query_vec=qvec[p["id"]])
                ids = {c.skill.id for c in cands}
                for c in cands:
                    out.append(
                        {
                            "prompt": p["prompt"],
                            "skill": c.skill.id,
                            "label": int(c.skill.id in p["labels"]),
                            "src": f"ours-{size}",
                        }
                    )
                # Gold skills retrieval missed are still useful positives.
                for g in p["labels"]:
                    if g not in ids:
                        out.append({"prompt": p["prompt"], "skill": g, "label": 1, "src": f"ours-{size}-missed"})
    return out


def skillret_pairs(
    P: Pool, blocked: set[int], emb: STEmbedder
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    base = CACHE / "skillret" / "data"
    train_q = [json.loads(x) for x in (base / "queries" / "train.jsonl").open(encoding="utf-8")]
    skills_raw = {}
    for line in (base / "skills.jsonl").open(encoding="utf-8"):
        s = json.loads(line)
        skills_raw[s["id"]] = s
    train_skill_ids = {sid for q in train_q for sid in q["skill_ids"]}
    blocked_names = {P.names[i] for i in blocked}
    blocked_vecs = P.vecs[sorted(blocked)] if blocked else np.zeros((0, P.vecs.shape[1]))
    skills: list[Skill] = []
    for sid in sorted(train_skill_ids):
        s = skills_raw.get(sid)
        if not s:
            continue
        _, body = parse_skill_md(s["skill_md"])
        skills.append(Skill(id=f"skillret:{sid}", name=s["name"], description=s["description"], body=body))
    vecs = emb.encode_docs([s.routing_text(1500) for s in skills])
    # Drop SkillRet skills that duplicate a held-out skill.
    near = (vecs @ blocked_vecs.T).max(axis=1) >= 0.90 if len(blocked_vecs) else np.zeros(len(skills), bool)
    import re

    keep = [
        i for i, s in enumerate(skills) if not near[i] and re.sub(r"[^a-z0-9]", "", s.name.lower()) not in blocked_names
    ]
    dropped = {skills[i].id for i in range(len(skills)) if i not in set(keep)}
    skills = [skills[i] for i in keep]
    vecs = vecs[keep]
    rng = random.Random(SEED)
    qs = [q for q in train_q if not any(f"skillret:{x}" in dropped for x in q["skill_ids"])]
    rng.shuffle(qs)
    qs = qs[:SKILLRET_QUERIES]
    r = HybridRetriever(skills, _NullEmbedder(), doc_vecs=vecs)
    qv = emb.model.encode(
        [emb.prefix + q["query"] for q in qs], batch_size=128, normalize_embeddings=True, show_progress_bar=False
    )
    out = []
    for q, v in zip(qs, qv):
        gold = {f"skillret:{x}" for x in q["skill_ids"]}
        cands = r.retrieve(q["query"], N_CANDS, query_vec=v)
        for c in cands:
            out.append({"prompt": q["query"], "skill": c.skill.id, "label": int(c.skill.id in gold), "src": "skillret"})
    extra = {s.id: {"id": s.id, "name": s.name, "description": s.description, "body": s.body} for s in skills}
    print(f"skillret: {len(qs)} queries, dropped {len(dropped)} skills near held-out set", file=sys.stderr)
    return out, extra


def main() -> None:
    P = pool()
    blocked = blocked_indices(P)
    print(f"held-out skills + near-duplicates blocked: {len(blocked)}", file=sys.stderr)
    emb = STEmbedder(EMBED_MODEL, device="cuda")
    pairs = our_pairs(P, blocked, emb)
    sr_pairs, sr_skills = skillret_pairs(P, blocked, emb)
    pairs += sr_pairs
    # Dedupe (same prompt+skill can appear at both catalog sizes).
    seen = set()
    uniq = []
    for p in pairs:
        k = (p["prompt"], p["skill"])
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    # Skill texts needed at training time.
    skill_text: dict[str, Any] = {}
    for p in uniq:
        sid = p["skill"]
        if sid in skill_text:
            continue
        if sid in P.index:
            r = P.records[P.index[sid]]
            skill_text[sid] = {"id": sid, "name": r["name"], "description": r["description"], "body": r["body"]}
        else:
            skill_text[sid] = sr_skills[sid]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for p in uniq:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    (OUT.parent / "skills.json").write_text(json.dumps(skill_text, ensure_ascii=False), encoding="utf-8")
    pos = sum(p["label"] for p in uniq)
    by_src: dict[str, int] = {}
    for p in uniq:
        by_src[p["src"].split("-")[0]] = by_src.get(p["src"].split("-")[0], 0) + 1
    print(f"pairs: {len(uniq)} (positives {pos}, {pos / len(uniq):.1%}) by source {by_src}", file=sys.stderr)
    # Sanity: no held-out skill id anywhere.
    held = {sid for sid, s in core().items() if s["split"] in ("val", "test")}
    assert not any(p["skill"] in held for p in uniq), "held-out skill leaked into training pairs"
    _ = ROOT


if __name__ == "__main__":
    main()
