"""SkillRouter on the skill-issue benchmark: SR-Emb retrieval over the same catalogs the other
systems see (exported by cloud.prepare), then SR-Rank over the top 20.

    python -m cloud.sr_bench --vecs sr_vecs.npz --chunks cloud_data/chunks.json \
        --docs cloud_data/sr_docs.jsonl --queries cloud_data/sr_queries.jsonl --out sr_bench.json

Output: {"runs": {"<split>|<size>": {prompt id: [[pool index, cosine], ...top 40]}},
         "scores": {"<prompt id>|<skill id>": raw SR-Rank score}}"""

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
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--rerank", type=int, default=20)
    a = ap.parse_args()
    z = np.load(a.vecs)
    dv = z["doc_vecs"].astype(np.float32)
    qvec = {str(i): v.astype(np.float32) for i, v in zip(z["query_ids"], z["query_vecs"])}
    docs = [json.loads(x) for x in Path(a.docs).read_text(encoding="utf-8").split("\n") if x.strip()]
    texts = {
        q["id"]: q["text"]
        for q in (json.loads(x) for x in Path(a.queries).read_text(encoding="utf-8").split("\n") if x.strip())
    }
    chunks = json.loads(Path(a.chunks).read_text(encoding="utf-8"))
    out_path = Path(a.out)
    out = json.loads(out_path.read_text(encoding="utf-8")) if out_path.is_file() else {"runs": {}, "scores": {}}
    rr = SkillRouterReranker()
    t0 = time.time()
    for entry in chunks:
        tag = f"{entry['split']}|{entry['size']}"
        runs = out["runs"].setdefault(tag, {})
        for ch in entry["chunks"]:
            cat = np.array(ch["catalog"])
            sub = dv[cat]
            for pid in ch["prompts"]:
                if pid in runs:
                    continue
                cos = sub @ qvec[pid]
                order = np.lexsort((np.arange(len(cos)), -cos))[: a.top_k]
                runs[pid] = [[int(cat[i]), float(cos[i])] for i in order]
                top = [docs[int(cat[i])] for i in order[: a.rerank]]
                todo = [d for d in top if f"{pid}|{d['id']}" not in out["scores"]]
                if todo:
                    raw = rr.raw_scores(texts[pid], [Candidate(Skill(**d), 0, 0.0, 0.0, 0.0, 0, 0) for d in todo])
                    for d, v in zip(todo, raw):
                        out["scores"][f"{pid}|{d['id']}"] = float(v)
        print(f"{tag}: done ({len(out['scores'])} scores, {time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
        out_path.write_text(json.dumps(out), encoding="utf-8")


if __name__ == "__main__":
    main()
