"""Package GPU work for a cloud notebook (Colab, Kaggle, any CUDA box).

    python -m cloud.prepare            # writes .cache/cloud/bundle.zip

The bundle is self-contained: this repo's code plus exactly the data each GPU job needs.
  data/pairs.jsonl, data/skills.json, data/val_pairs.jsonl   gate fine-tuning
  data/requests_bench.jsonl    every (prompt, skill) pair the benchmark scores (hybrid top-12 on
                               val@100, test at every catalog size, hand-written@100)
  data/requests_skillret.jsonl hybrid top-20 for the SkillRet sample (reranking)
  data/sr_docs.jsonl, data/sr_queries.jsonl   texts for SkillRouter's embedder
  data/skillret_docs.jsonl, data/skillret_queries.jsonl   SkillRet corpus + sample for SkillRouter
Run cloud/skill_issue_cloud.ipynb there, download results.zip, then `python -m cloud.ingest`.
Nothing here needs a local GPU."""

from __future__ import annotations

import json
import random
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from bench.corpus import CACHE, ROOT
from bench.data import load_split, pool
from bench.run import SIZES, _NullEmbedder, retrieval_runs
from skillissue.skill import Skill

OUT = CACHE / "cloud"
BODY = 2000  # gate inputs use 300 chars of body; keep a margin
SR_BODY = 6000  # SkillRouter's embedder reads up to 1,024 tokens


def skill_dict(s: Skill, body: int = BODY) -> dict[str, Any]:
    return {"id": s.id, "name": s.name, "description": s.description, "body": s.body[:body]}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def training_data(d: Path) -> None:
    from training.finetune_laya import PAIRS, SKILLS, load_pairs, val_pairs

    pairs = load_pairs(8.0, 13, PAIRS)  # the same subsample the local scripts draw
    skills = json.loads(SKILLS.read_text(encoding="utf-8"))
    used = {p["skill"] for p in pairs}
    write_jsonl(d / "pairs.jsonl", pairs)
    (d / "skills.json").write_text(
        json.dumps(
            {k: {**v, "body": v.get("body", "")[:BODY]} for k, v in skills.items() if k in used}, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    write_jsonl(
        d / "val_pairs.jsonl",
        [{"prompt": p["prompt"], "skill": skill_dict(p["skill_obj"]), "label": p["label"]} for p in val_pairs()],
    )
    print(f"training: {len(pairs)} pairs, {len(used)} skills", file=sys.stderr)


def bench_requests(d: Path, n_gate: int = 12) -> None:
    splits: list[tuple[str, int, list[dict[str, Any]]]] = [("val", 100, load_split("val"))]
    splits += [("test", size, load_split("test")) for size in SIZES]
    hand = ROOT / "data" / "bench" / "handwritten.jsonl"
    if hand.is_file():
        splits.append(
            ("handwritten", 100, [json.loads(x) for x in hand.read_text(encoding="utf-8").split("\n") if x.strip()])
        )
    seen: dict[str, dict[str, Any]] = {}
    for name, size, prompts in splits:
        runs = retrieval_runs("hybrid", name, size, prompts)
        for p in prompts:
            for c in runs[p["id"]]["cands"][:n_gate]:
                key = f"{p['id']}|{c.skill.id}"
                seen.setdefault(key, {"key": key, "prompt": p["prompt"], "skill": skill_dict(c.skill)})
    write_jsonl(d / "requests_bench.jsonl", list(seen.values()))
    print(f"bench requests: {len(seen)}", file=sys.stderr)


def skillret(d: Path, n: int = 1000, seed: int = 3, k: int = 20) -> None:
    from bench.skillret_eval import load
    from skillissue.retrieval import HybridRetriever, STEmbedder

    skills, queries = load()
    queries = random.Random(seed).sample(queries, min(n, len(queries)))
    vec_path = CACHE / "skillret_test_bge.npy"
    emb = STEmbedder("BAAI/bge-small-en-v1.5", device="cpu")
    if vec_path.is_file():
        vecs = np.load(vec_path)
    else:
        vecs = emb.encode_docs([s.routing_text(1500) for s in skills])
        np.save(vec_path, vecs)
    qv = emb.model.encode(
        [emb.prefix + q["query"] for q in queries], batch_size=64, normalize_embeddings=True, show_progress_bar=False
    )
    r = HybridRetriever(skills, _NullEmbedder(), doc_vecs=vecs)
    reqs = []
    for q, v in zip(queries, qv):
        for c in r.retrieve(q["query"], k, query_vec=v):
            reqs.append({"key": f"sr:{q['id']}|{c.skill.id}", "prompt": q["query"], "skill": skill_dict(c.skill)})
    write_jsonl(d / "requests_skillret.jsonl", reqs)
    write_jsonl(d / "skillret_docs.jsonl", [skill_dict(s, SR_BODY) for s in skills])
    write_jsonl(d / "skillret_queries.jsonl", [{"id": q["id"], "text": q["query"]} for q in queries])
    print(f"skillret: {len(reqs)} rerank requests, {len(skills)} docs, {len(queries)} queries", file=sys.stderr)


def skillrouter_inputs(d: Path) -> None:
    from bench.catalogs import build_chunks

    P = pool()
    # The exact catalogs every other system is evaluated on (same seeds as bench.run).
    entries = []
    splits = [("val", 100, load_split("val"))] + [("test", s, load_split("test")) for s in SIZES]
    hand = ROOT / "data" / "bench" / "handwritten.jsonl"
    if hand.is_file():
        splits.append(
            ("handwritten", 100, [json.loads(x) for x in hand.read_text(encoding="utf-8").split("\n") if x.strip()])
        )
    for name, size, prompts in splits:
        chunks = build_chunks(prompts, size, P)
        entries.append(
            {
                "split": name,
                "size": size,
                "chunks": [{"catalog": ch.catalog, "prompts": [p["id"] for p in ch.prompts]} for ch in chunks],
            }
        )
    (d / "chunks.json").write_text(json.dumps(entries), encoding="utf-8")
    write_jsonl(d / "sr_docs.jsonl", [skill_dict(s, SR_BODY) for s in P.skills])
    qs = load_split("val") + load_split("test")
    hand = ROOT / "data" / "bench" / "handwritten.jsonl"
    if hand.is_file():
        qs += [json.loads(x) for x in hand.read_text(encoding="utf-8").split("\n") if x.strip()]
    write_jsonl(d / "sr_queries.jsonl", [{"id": p["id"], "text": p["prompt"]} for p in qs])
    print(f"skillrouter: {len(P.skills)} docs, {len(qs)} queries", file=sys.stderr)


def main() -> None:
    stage = OUT / "bundle"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "data").mkdir(parents=True)
    training_data(stage / "data")
    bench_requests(stage / "data")
    skillret(stage / "data")
    skillrouter_inputs(stage / "data")
    zpath = OUT / "bundle.zip"
    import time as _time

    version = _time.strftime("%Y%m%d-%H%M%S")
    (OUT / "bundle_version.txt").write_text(version, encoding="utf-8")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("skill-issue/BUNDLE_VERSION", version)
        for sub in ("src", "bench", "training", "cloud"):
            for p in (ROOT / sub).rglob("*"):
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc" and "results" not in p.parts:
                    z.write(p, f"skill-issue/{p.relative_to(ROOT).as_posix()}")
        for name in ("pyproject.toml", "README.md", "LICENSE", "NOTICE"):
            z.write(ROOT / name, f"skill-issue/{name}")
        for p in (stage / "data").iterdir():
            z.write(p, f"skill-issue/cloud_data/{p.name}")
    print(f"wrote {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
