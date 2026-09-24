"""Prepare input files for prompt generation (done by LLM subagents; see docs/benchmark.md).

Each batch file holds the skills a generator may see. Generators for one split never see skills
from another split. Outputs go to .cache/gen/out/<batch>.jsonl and are validated by
bench.build_dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bench.corpus import CACHE, POOL, ROOT
from skillissue.skill import body_excerpt

GEN = CACHE / "gen"
CORE = ROOT / "data" / "bench" / "core_skills.json"

PLAN = {  # split: (batch size, positives per skill)
    "train": (30, 6),
    "val": (23, 3),
    "test": (19, 3),
}


def _pool_index() -> dict[str, dict[str, Any]]:
    out = {}
    with POOL.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r
    return out


def brief(r: dict[str, Any], body_chars: int = 1200) -> dict[str, Any]:
    return {"id": r["id"], "name": r["name"], "description": r["description"][:1200], "body_excerpt": body_excerpt(r["body"], body_chars)}


def main() -> None:
    core = json.loads(CORE.read_text(encoding="utf-8"))["skills"]
    pool = _pool_index()
    (GEN / "in").mkdir(parents=True, exist_ok=True)
    (GEN / "out").mkdir(parents=True, exist_ok=True)
    batches = []
    for split, (size, n_pos) in PLAN.items():
        skills = [s for s in core if s["split"] == split]
        for b in range(0, len(skills), size):
            chunk = skills[b : b + size]
            name = f"{split}-skills-{b // size:02d}"
            data = {
                "batch": name,
                "split": split,
                "positives_per_skill": n_pos,
                "hard_negatives_per_skill": 2,
                "skills": [
                    {**brief(pool[s["id"]]), "siblings": [brief(pool[x], 300) for x in s["siblings"]]}
                    for s in chunk
                ],
            }
            (GEN / "in" / f"{name}.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
            batches.append(name)
        # Multi-skill batch: the whole split's skill list, short form.
        multi = {
            "batch": f"{split}-multi",
            "split": split,
            "count": {"train": 80, "val": 20, "test": 40}[split],
            "skills": [brief(pool[s["id"]], 200) for s in skills],
        }
        (GEN / "in" / f"{split}-multi.json").write_text(json.dumps(multi, indent=1, ensure_ascii=False), encoding="utf-8")
        batches.append(f"{split}-multi")
    # "None" batches see every core skill (name + short description) so they can avoid all of them.
    allskills = [{"id": s["id"], "name": pool[s["id"]]["name"], "description": pool[s["id"]]["description"][:240]} for s in core]
    focus = {
        "none-a": "everyday code edits and debugging in the user's own code: refactors, renames, small bug fixes on pasted snippets, adding a parameter, fixing an off-by-one, reading a stack trace from their own app",
        "none-b": "questions and conversation: explain a concept, compare two approaches, what does this regex/line do, chit-chat, thanks, meta questions about the session, planning talk that needs no special workflow",
        "none-c": "keyword traps and small ops chores: requests that mention words like PDF, deploy, test, Docker, API, Slack, database, security, notebook, git, release, but ask for something trivial or general that no specialized skill is needed for",
    }
    for name, f in focus.items():
        data = {"batch": name, "count": 150, "focus": f, "skills_to_avoid": allskills}
        (GEN / "in" / f"{name}.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        batches.append(name)
    print("\n".join(batches))


if __name__ == "__main__":
    main()
