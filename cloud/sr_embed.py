"""Embed documents and queries with SkillRouter's SR-Emb-0.6B (for the SkillRouter baseline).

python -m cloud.sr_embed --docs cloud_data/sr_docs.jsonl --queries cloud_data/sr_queries.jsonl --out sr_vecs.npz"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from bench.skillrouter import SkillRouterEmbedder, doc_text
from skillissue.skill import Skill


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    docs = [json.loads(x) for x in Path(a.docs).read_text(encoding="utf-8").splitlines()]
    queries = [json.loads(x) for x in Path(a.queries).read_text(encoding="utf-8").splitlines()]
    emb = SkillRouterEmbedder()
    t0 = time.time()
    dv = emb.encode_docs([doc_text(Skill(**d)) for d in docs])
    qv = emb.encode_queries([q["text"] for q in queries])
    np.savez(
        a.out,
        doc_ids=np.array([d["id"] for d in docs]),
        doc_vecs=dv.astype(np.float16),
        query_ids=np.array([q["id"] for q in queries]),
        query_vecs=qv.astype(np.float16),
    )
    print(f"embedded {len(docs)} docs and {len(queries)} queries in {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
