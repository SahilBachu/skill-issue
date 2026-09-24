"""Metrics over per-prompt records.

A record: {"id", "kind", "labels": [...], "retrieved": [skill ids in retrieval order],
           "gated": [[skill id, prob], ...] in retrieval order, "selected": [skill ids]}
"""

from __future__ import annotations

from typing import Any

import numpy as np


def select(gated: list[tuple[str, float]], threshold: float, max_skills: int = 3) -> list[str]:
    keep = sorted((g for g in gated if g[1] >= threshold), key=lambda g: -g[1])[:max_skills]
    return [g[0] for g in keep]


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> tuple[float, list[dict[str, float]]]:
    """Expected calibration error with equal-width bins, plus the reliability diagram data."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if len(probs) == 0:
        return float("nan"), []
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    diagram = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs >= lo) & (probs < hi) if hi < 1 else (probs >= lo) & (probs <= hi)
        if not m.any():
            continue
        conf, acc, n = float(probs[m].mean()), float(labels[m].mean()), int(m.sum())
        total += abs(conf - acc) * n
        diagram.append({"lo": float(lo), "hi": float(hi), "confidence": conf, "accuracy": acc, "n": n})
    return total / len(probs), diagram


def compute(records: list[dict[str, Any]], ks: tuple[int, ...] = (1, 3, 5, 10, 20, 40)) -> dict[str, Any]:
    labeled = [r for r in records if r["labels"]]
    unlabeled = [r for r in records if not r["labels"]]
    out: dict[str, Any] = {"n": len(records), "n_labeled": len(labeled), "n_none": len(unlabeled)}

    # Retrieval recall@k: fraction of gold skills found in the top k (averaged per prompt).
    for k in ks:
        vals = [len(set(r["labels"]) & set(r["retrieved"][:k])) / len(r["labels"]) for r in labeled]
        out[f"recall@{k}"] = float(np.mean(vals)) if vals else float("nan")

    # Top-1 after the gate: is the highest-probability candidate a gold skill?
    def top1(r: dict[str, Any]) -> bool:
        if not r["gated"]:
            return False
        best = max(r["gated"], key=lambda g: g[1])[0]
        return best in r["labels"]

    out["top1"] = float(np.mean([top1(r) for r in labeled])) if labeled else float("nan")

    # Decisions (what actually gets injected).
    tp = fp = fn = 0
    exact = 0
    false_inject = 0
    for r in records:
        gold, sel = set(r["labels"]), set(r["selected"])
        tp += len(gold & sel)
        fp += len(sel - gold)
        fn += len(gold - sel)
        exact += gold == sel
        false_inject += bool(sel - gold)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    out["precision"] = prec
    out["recall"] = rec
    out["f1"] = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec else float("nan")
    out["exact_match"] = exact / len(records) if records else float("nan")
    out["false_injection_rate"] = false_inject / len(records) if records else float("nan")
    out["none_accuracy"] = float(np.mean([not r["selected"] for r in unlabeled])) if unlabeled else float("nan")
    out["hit_rate"] = float(np.mean([bool(set(r["labels"]) & set(r["selected"])) for r in labeled])) if labeled else float("nan")

    # Calibration over every gated (prompt, candidate) pair.
    probs, ys = [], []
    for r in records:
        for sid, p in r["gated"]:
            probs.append(p)
            ys.append(sid in r["labels"])
    e, diagram = ece(np.array(probs), np.array(ys))
    out["ece"] = e
    out["reliability"] = diagram
    out["brier"] = float(np.mean((np.array(probs) - np.array(ys, dtype=float)) ** 2)) if probs else float("nan")

    # Per-kind exact match, to see where a system wins or loses.
    kinds: dict[str, list[bool]] = {}
    for r in records:
        kinds.setdefault(r["kind"], []).append(set(r["labels"]) == set(r["selected"]))
    out["exact_by_kind"] = {k: float(np.mean(v)) for k, v in sorted(kinds.items())}

    lat = [r["ms"] for r in records if "ms" in r]
    if lat:
        out["latency_ms_p50"] = float(np.percentile(lat, 50))
        out["latency_ms_p95"] = float(np.percentile(lat, 95))
    return out
