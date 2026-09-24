"""SkillRouter's full pipeline on the SkillRet sample: SR-Emb retrieval top-50, SR-Rank rerank top-20.

    python -m cloud.skillret_sr --vecs skillret_sr_vecs.npz --docs cloud_data/skillret_docs.jsonl \
        --queries cloud_data/skillret_queries.jsonl --out skillret_skillrouter.json

Writes {"retrieval": {qid: [ids]}, "reranked": {qid: [ids]}} for bench.skillret_eval to score."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from bench.skillrouter import SkillRouterReranker
from skillissue.retrieval import Candidate
from skillissue.skill import Skill


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vecs", required=True)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=20)
    a = ap.parse_args()
    z = np.load(a.vecs)
    docs = {d["id"]: d for d in (json.loads(x) for x in Path(a.docs).read_text(encoding="utf-8").splitlines())}
    queries = [json.loads(x) for x in Path(a.queries).read_text(encoding="utf-8").splitlines()]
    doc_ids = [str(x) for x in z["doc_ids"]]
    dv = z["doc_vecs"].astype(np.float32)
    qvec = {str(i): v.astype(np.float32) for i, v in zip(z["query_ids"], z["query_vecs"])}
    rr = SkillRouterReranker()
    out: dict[str, dict[str, list[str]]] = {"retrieval": {}, "reranked": {}}
    t0 = time.time()
    for n, q in enumerate(queries):
        order = np.argsort(-(dv @ qvec[q["id"]]), kind="stable")[:50]
        top = [doc_ids[i] for i in order]
        out["retrieval"][q["id"]] = top
        cands = [Candidate(Skill(**docs[i]), 0, 0.0, 0.0, 0.0, 0, 0) for i in top[: a.k]]
        raw = rr.raw_scores(q["text"], cands)
        out["reranked"][q["id"]] = [top[i] for i in np.argsort(-raw, kind="stable")] + top[a.k :]
        if n % 100 == 0:
            print(f"{n}/{len(queries)} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    Path(a.out).write_text(json.dumps(out), encoding="utf-8")


if __name__ == "__main__":
    main()
