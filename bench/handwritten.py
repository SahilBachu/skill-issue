"""Convert data/bench/handwritten_draft.txt (human-readable) into handwritten.jsonl.

Labels are exact test-split skill names. The hand-written
set targets test-split skills only, so it is held out exactly like the synthetic test split."""

from __future__ import annotations

import json
import sys

from bench.corpus import ROOT
from bench.data import core, pool

SRC = ROOT / "data" / "bench" / "handwritten_draft.txt"
OUT = ROOT / "data" / "bench" / "handwritten.jsonl"


def main() -> None:
    P = pool()
    test = [sid for sid, s in core().items() if s["split"] == "test"]
    by_name: dict[str, list[str]] = {}
    for sid in test:
        by_name.setdefault(P.records[P.index[sid]]["name"].lower(), []).append(sid)
    rows, bad = [], []
    for n, line in enumerate(SRC.read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip() or line.startswith("#"):
            continue
        labels_s, kind, prompt = line.split("\t", 2)
        labels = []
        for lab in [] if labels_s == "-" else labels_s.split(","):
            cands = by_name.get(lab.strip().lower(), [])
            if len(cands) != 1:
                bad.append(f"line {n}: {lab!r} -> {cands}")
                continue
            labels.append(cands[0])
        rows.append(
            {
                "id": f"hand-{len(rows):03d}",
                "prompt": prompt,
                "labels": labels,
                "kind": kind,
                "style": None,
                "anchor": None,
                "split": "handwritten",
                "source": "handwritten-draft",
            }
        )
    if bad:
        print("\n".join(bad), file=sys.stderr)
        print("known names:", sorted(by_name), file=sys.stderr)
        sys.exit(1)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
