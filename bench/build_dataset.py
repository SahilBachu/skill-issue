"""Validate generated prompts and write the benchmark splits.

    python -m bench.build_dataset

Leakage precautions (see docs/benchmark.md):
  * skills are split by similarity cluster (bench.select_core), prompts inherit their skill's split
  * generators for one split never saw another split's skills
  * exact duplicates are removed everywhere; near-duplicate prompts (cosine >= 0.92) that cross
    splits are dropped from the training side
  * "none" prompts are shuffled with a fixed seed and split 60/15/25
Outputs data/bench/{train,val,test}.jsonl and data/bench/stats.json.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from bench.corpus import CACHE, ROOT

GEN_OUT = CACHE / "gen" / "out"
GEN_IN = CACHE / "gen" / "in"
CORE = ROOT / "data" / "bench" / "core_skills.json"
OUT_DIR = ROOT / "data" / "bench"
SEED = 7
NEAR_DUP = 0.92
KINDS = {"positive", "hard_negative_none", "hard_negative_sibling", "multi", "none"}


def _norm(p: str) -> str:
    return re.sub(r"\s+", " ", p.strip().lower())


def load_batch(path: Path, allowed: set[str], split: str, errors: list[str]) -> list[dict[str, Any]]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            errors.append(f"{path.name}:{n} bad json")
            continue
        prompt = str(r.get("prompt") or "").strip()
        labels = r.get("labels") or []
        kind = r.get("kind")
        if not prompt or kind not in KINDS or not isinstance(labels, list):
            errors.append(f"{path.name}:{n} bad row")
            continue
        bad = [x for x in labels if x not in allowed]
        if bad:
            errors.append(f"{path.name}:{n} labels outside split: {bad}")
            continue
        if kind in ("none", "hard_negative_none") and labels:
            errors.append(f"{path.name}:{n} none-kind with labels")
            continue
        if kind in ("positive", "hard_negative_sibling", "multi") and not labels:
            errors.append(f"{path.name}:{n} {kind} without labels")
            continue
        rows.append(
            {
                "prompt": prompt,
                "labels": sorted(set(labels)),
                "kind": kind,
                "style": r.get("style"),
                "anchor": r.get("anchor"),
                "split": split,
                "source": "synthetic",
                "batch": path.stem,
            }
        )
    return rows


def main() -> None:
    core = json.loads(CORE.read_text(encoding="utf-8"))["skills"]
    split_of = {s["id"]: s["split"] for s in core}
    names = {s["id"]: s["name"] for s in core}
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    for path in sorted(GEN_OUT.glob("*.jsonl")):
        stem = path.stem
        if stem.startswith("none-"):
            rows += load_batch(path, set(), "none", errors)
        else:
            split = stem.split("-", 1)[0]
            allowed = {i for i, s in split_of.items() if s == split}
            rows += load_batch(path, allowed, split, errors)
    # Exact duplicates.
    seen: set[str] = set()
    uniq = []
    for r in rows:
        k = _norm(r["prompt"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    dup_exact = len(rows) - len(uniq)
    rows = uniq
    # Assign "none" prompts to splits.
    rng = random.Random(SEED)
    nones = [r for r in rows if r["split"] == "none"]
    rng.shuffle(nones)
    n = len(nones)
    for i, r in enumerate(nones):
        r["split"] = "train" if i < 0.6 * n else ("val" if i < 0.75 * n else "test")
    # Near-duplicates across splits: drop the train/val copy, keep the test copy.
    from skillissue.retrieval import STEmbedder

    emb = STEmbedder("BAAI/bge-small-en-v1.5", device="cuda")
    vecs = emb.encode_docs([r["prompt"] for r in rows])
    order = {"test": 0, "val": 1, "train": 2}
    drop: set[int] = set()
    sim = vecs @ vecs.T
    idx_a, idx_b = np.where(np.triu(sim, 1) >= NEAR_DUP)
    for a, b in zip(idx_a.tolist(), idx_b.tolist()):
        if rows[a]["split"] == rows[b]["split"]:
            continue
        loser = a if order[rows[a]["split"]] > order[rows[b]["split"]] else b
        drop.add(loser)
    rows = [r for i, r in enumerate(rows) if i not in drop]
    # Stable ids.
    out: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for r in rows:
        out[r["split"]].append(r)
    stats: dict[str, Any] = {
        "errors": len(errors),
        "exact_duplicates_removed": dup_exact,
        "cross_split_near_dups_removed": len(drop),
    }
    for split, rs in out.items():
        for i, r in enumerate(rs):
            r["id"] = f"{split}-{i:05d}"
        path = OUT_DIR / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rs:
                f.write(
                    json.dumps(
                        {k: r[k] for k in ("id", "prompt", "labels", "kind", "style", "anchor", "split", "source")},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        pos = [r for r in rs if r["kind"] == "positive"]
        named = sum(
            1 for r in pos if names[r["labels"][0]].lower().replace("-", " ") in r["prompt"].lower().replace("-", " ")
        )
        stats[split] = {
            "total": len(rs),
            "kinds": dict(Counter(r["kind"] for r in rs)),
            "skills": len({lab for r in rs for lab in r["labels"]}),
            "positives_naming_skill": f"{named}/{len(pos)}",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    for e in errors[:20]:
        print("  ", e, file=sys.stderr)
    print(json.dumps(stats, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
