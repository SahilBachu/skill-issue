"""Run the benchmark.

    python -m bench.run                           # default systems, all sizes
    python -m bench.run --systems hybrid laya-zs  # a subset
    python -m bench.run --sizes 10 100 --quick    # fast smoke run

For every system: fit calibration + threshold on the validation split at catalog size 100, freeze
them, then evaluate the test split at every catalog size. Results go to bench/results/*.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from typing import Any

import numpy as np

from bench import metrics
from bench.catalogs import build_chunks
from bench.corpus import CACHE, ROOT
from bench.data import RESULTS, load_split, pool, save_json
from bench.systems import SYSTEMS, Scorer, System, feature_matrix
from skillissue.gates.base import Calibration, fit_platt
from skillissue.retrieval import Candidate, HybridRetriever, STEmbedder

SIZES = [10, 100, 1000, 10000, 100000]  # 100000 = whole pool
CAL_SIZE = 100
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
SCORE_CACHE = CACHE / "bench" / "scores"
RETR_CACHE = CACHE / "bench" / "retrieval"

os.environ.setdefault("USE_TF", "0")


class _NullEmbedder:
    """Marks dense retrieval as on while query vectors come precomputed."""

    name = EMBED_MODEL

    def encode_docs(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        raise RuntimeError("doc vectors are precomputed")

    def encode_query(self, text: str) -> np.ndarray:  # pragma: no cover
        raise RuntimeError("query vectors are precomputed")


_QVECS: dict[str, np.ndarray] = {}


def query_vecs(prompts: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    missing = [p for p in prompts if p["id"] not in _QVECS]
    if missing:
        emb = STEmbedder(EMBED_MODEL, device="cuda")
        vecs = emb.model.encode(
            [emb.prefix + p["prompt"] for p in missing],
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        for p, v in zip(missing, vecs):
            _QVECS[p["id"]] = np.asarray(v, dtype=np.float32)
    return _QVECS


def retrieval_runs(
    mode: str, split: str, size: int, prompts: list[dict[str, Any]], top_k: int = 40
) -> dict[str, dict[str, Any]]:
    """{prompt id: {"cands": [...], "ms": retrieval ms}} for one retrieval mode and catalog size."""
    key = hashlib.sha1(json.dumps([mode, split, size, top_k, [p["id"] for p in prompts]]).encode()).hexdigest()[:16]
    path = RETR_CACHE / f"{mode}-{split}-{size}-{key}.pkl"
    if path.is_file():
        with path.open("rb") as f:
            cached: dict[str, dict[str, Any]] = pickle.load(f)
            return cached
    P = pool()
    qv = query_vecs(prompts)
    out: dict[str, dict[str, Any]] = {}
    chunks = build_chunks(prompts, size, P)
    for ch in chunks:
        skills = [P.skills[i] for i in ch.catalog]
        use_dense = mode in ("hybrid", "dense")
        r = HybridRetriever(
            skills,
            _NullEmbedder() if use_dense else None,
            use_bm25=mode in ("hybrid", "bm25"),
            doc_vecs=P.vecs[ch.catalog] if use_dense else None,
        )
        for p in ch.prompts:
            t = time.perf_counter()
            cands = r.retrieve(p["prompt"], top_k, query_vec=qv[p["id"]] if use_dense else None)
            ms = (time.perf_counter() - t) * 1000
            out[p["id"]] = {"cands": cands, "ms": ms, "catalog": len(skills)}
    RETR_CACHE.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(out, f)
    return out


class ScoreCache:
    def __init__(self, key: str):
        self.path = SCORE_CACHE / f"{key}.pkl" if key else None
        self.data: dict[str, float] = {}
        self.dirty = False
        if self.path and self.path.is_file():
            with self.path.open("rb") as f:
                self.data = pickle.load(f)

    def save(self) -> None:
        if self.path and self.dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("wb") as f:
                pickle.dump(self.data, f)
            self.dirty = False


def score_prompts(
    system: System,
    scorer: Scorer | None,
    cache: ScoreCache,
    prompts: list[dict[str, Any]],
    runs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Raw scores for each prompt's top-G candidates. Uses the cache; loads the model lazily."""
    out = {}
    for p in prompts:
        cands: list[Candidate] = runs[p["id"]]["cands"][: system.n_gate]
        keys = [f"{p['id']}|{c.skill.id}" for c in cands]
        gate_ms = 0.0
        if system.scorer == "features":
            assert scorer is not None
            raw = scorer.raw(p["prompt"], cands)
        elif all(k in cache.data for k in keys):
            raw = np.array([cache.data[k] for k in keys])
            gate_ms = float("nan")
        else:
            assert scorer is not None
            t = time.perf_counter()
            raw = scorer.raw(p["prompt"], cands)
            gate_ms = (time.perf_counter() - t) * 1000
            for k, v in zip(keys, raw):
                cache.data[k] = float(v)
            cache.dirty = True
        out[p["id"]] = {"cands": cands, "raw": raw, "gate_ms": gate_ms, "retr_ms": runs[p["id"]]["ms"]}
    return out


def to_records(
    prompts: list[dict[str, Any]], scored: dict[str, dict[str, Any]], runs: dict[str, dict[str, Any]], cal: Calibration
) -> list[dict[str, Any]]:
    recs = []
    for p in prompts:
        s = scored[p["id"]]
        probs = cal.apply(s["raw"]) if len(s["raw"]) else np.zeros(0)
        gated = [(c.skill.id, float(pr)) for c, pr in zip(s["cands"], probs)]
        rec = {
            "id": p["id"],
            "kind": p["kind"],
            "labels": p["labels"],
            "retrieved": [c.skill.id for c in runs[p["id"]]["cands"]],
            "gated": gated,
            "selected": metrics.select(gated, cal.threshold),
        }
        if s["gate_ms"] == s["gate_ms"]:  # not NaN: measured this run
            rec["ms"] = s["retr_ms"] + s["gate_ms"]
        recs.append(rec)
    return recs


def calibrate(
    system: System,
    scorer: Scorer | None,
    scored: dict[str, dict[str, Any]],
    prompts: list[dict[str, Any]],
    runs: dict[str, dict[str, Any]],
) -> Calibration:
    labels = {p["id"]: set(p["labels"]) for p in prompts}
    if system.scorer == "features" and len(system.features) > 1:
        from sklearn.linear_model import LogisticRegression

        X, y = [], []
        for pid, s in scored.items():
            X.append(feature_matrix(s["cands"], system.features))
            y += [c.skill.id in labels[pid] for c in s["cands"]]
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(np.vstack(X), np.array(y))
        assert scorer is not None
        scorer.weights = np.concatenate([lr.coef_[0], lr.intercept_])
        for s in scored.values():  # rescore with the fitted weights
            s["raw"] = scorer.raw("", s["cands"])
    xs, ys = [], []
    for pid, s in scored.items():
        xs += list(s["raw"])
        ys += [c.skill.id in labels[pid] for c in s["cands"]]
    a, b = fit_platt(np.array(xs), np.array(ys, dtype=float))
    cal = Calibration(a, b, 0.5)
    best = (-1.0, 0.5)
    for thr in np.arange(0.05, 0.96, 0.01):
        cal.threshold = float(thr)
        em = metrics.compute(to_records(prompts, scored, runs, cal))["exact_match"]
        if em > best[0] + 1e-9:
            best = (em, float(thr))
    cal.threshold = round(best[1], 2)
    return cal


def run_system(system: System, sizes: list[int], quick: bool = False) -> dict[str, Any]:
    val = load_split("val")
    test = load_split("test")
    hand = ROOT / "data" / "bench" / "handwritten.jsonl"
    extra = (
        {"handwritten": [json.loads(x) for x in hand.read_text(encoding="utf-8").splitlines() if x.strip()]}
        if hand.is_file()
        else {}
    )
    if quick:
        import random

        val, test = random.Random(1).sample(val, 120), random.Random(2).sample(test, 150)
    cache = ScoreCache(system.cache_key)
    scorer: Scorer | None = None

    def need_model(prompts: list[dict[str, Any]], runs: dict[str, dict[str, Any]]) -> None:
        nonlocal scorer
        if scorer is not None:
            return
        if system.scorer == "features":
            scorer = Scorer(system, ROOT)
            return
        for p in prompts:
            for c in runs[p["id"]]["cands"][: system.n_gate]:
                if f"{p['id']}|{c.skill.id}" not in cache.data:
                    scorer = Scorer(system, ROOT)
                    return

    t0 = time.time()
    vruns = retrieval_runs(system.retrieval, "val", CAL_SIZE, val, system.top_k)
    need_model(val, vruns)
    vscored = score_prompts(system, scorer, cache, val, vruns)
    cal = calibrate(system, scorer, vscored, val, vruns)
    val_metrics = metrics.compute(to_records(val, vscored, vruns, cal))
    result: dict[str, Any] = {
        "system": system.name,
        "label": system.label,
        "retrieval": system.retrieval,
        "scorer": system.scorer,
        "model": system.model,
        "n_gate": system.n_gate,
        "calibration": cal.to_dict(),
        "val": val_metrics,
        "test": {},
    }
    if scorer is not None and scorer.weights is not None:
        result["feature_weights"] = dict(zip([*system.features, "bias"], scorer.weights.tolist()))
    all_records: dict[str, Any] = {}
    for size in sizes:
        runs = retrieval_runs(system.retrieval, "test", size, test, system.top_k)
        need_model(test, runs)
        scored = score_prompts(system, scorer, cache, test, runs)
        cache.save()
        recs = to_records(test, scored, runs, cal)
        m = metrics.compute(recs)
        m["catalog_size"] = int(np.median([runs[p["id"]]["catalog"] for p in test]))
        result["test"][str(size)] = m
        all_records[str(size)] = recs
        print(
            f"  {system.name:12s} N={m['catalog_size']:>6} exact={m['exact_match']:.3f} top1={m['top1']:.3f} "
            f"r@10={m['recall@10']:.3f} none={m['none_accuracy']:.3f} fir={m['false_injection_rate']:.3f} ece={m['ece']:.3f}",
            file=sys.stderr,
        )
    for name, prompts in extra.items():
        runs = retrieval_runs(system.retrieval, name, CAL_SIZE, prompts, system.top_k)
        need_model(prompts, runs)
        scored = score_prompts(system, scorer, cache, prompts, runs)
        cache.save()
        recs = to_records(prompts, scored, runs, cal)
        result[name] = metrics.compute(recs)
        all_records[name] = recs
    cache.save()
    if scorer is not None and scorer.gate is not None:
        try:
            import torch

            result["gpu_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
        except Exception:
            pass
        scorer.close()
    result["wall_s"] = round(time.time() - t0, 1)
    save_json(RESULTS / f"{system.name}.json", result)
    save_json(CACHE / "bench" / "records" / f"{system.name}.json", all_records)
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m bench.run")
    ap.add_argument(
        "--systems", nargs="*", default=["bm25", "dense", "hybrid", "laya-zs", "bge-m3-zs", "gte-mb-zs", "minilm-zs"]
    )
    ap.add_argument("--sizes", nargs="*", type=int, default=SIZES)
    ap.add_argument("--n-gate", type=int, help="override how many candidates the gate scores")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    for name in a.systems:
        system = SYSTEMS[name]
        if a.n_gate:
            system = System(**{**system.__dict__, "n_gate": a.n_gate, "name": f"{system.name}-g{a.n_gate}"})
        if system.model and system.model.startswith("checkpoints/") and not (ROOT / system.model).exists():
            print(f"skip {name}: {system.model} not found (train it first)", file=sys.stderr)
            continue
        print(f"== {system.name}", file=sys.stderr)
        run_system(system, a.sizes, a.quick)


if __name__ == "__main__":
    main()
