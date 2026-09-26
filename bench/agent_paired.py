"""Paired comparison of the agent arms: both run the same sampled prompts, so compare them prompt by
prompt (exact McNemar test on the discordant pairs).

    python -m bench.agent_paired

Writes bench/results/agent/paired.json."""

from __future__ import annotations

import json
from math import comb
from typing import Any

from bench.corpus import CACHE
from bench.data import RESULTS, save_json


def exact(r: dict[str, Any]) -> bool:
    return set(r["selected"]) == set(r["labels"])


def mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: binomial test of b successes in b + c trials at 0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def main() -> None:
    out: dict[str, Any] = {}
    for size in (100, 1000):
        runs = {}
        for arm in ("vanilla", "router"):
            p = CACHE / "bench" / "agent" / f"{arm}-N{size}-n100-records.json"
            if p.is_file():
                d = json.loads(p.read_text(encoding="utf-8"))
                rows = d if isinstance(d, list) else d.get("raw", d)
                runs[arm] = {r["id"]: exact(r) for r in rows}
        if len(runs) < 2:
            continue
        ids = sorted(set(runs["vanilla"]) & set(runs["router"]))
        only_router = sum(1 for i in ids if runs["router"][i] and not runs["vanilla"][i])
        only_vanilla = sum(1 for i in ids if runs["vanilla"][i] and not runs["router"][i])
        out[str(size)] = {
            "n_paired": len(ids),
            "vanilla_exact": sum(runs["vanilla"][i] for i in ids) / len(ids),
            "router_exact": sum(runs["router"][i] for i in ids) / len(ids),
            "router_only_correct": only_router,
            "vanilla_only_correct": only_vanilla,
            "mcnemar_exact_p": float(f"{mcnemar_p(only_router, only_vanilla):.2g}"),
        }
        print(size, out[str(size)])
    save_json(RESULTS / "agent" / "paired.json", out)


if __name__ == "__main__":
    main()
