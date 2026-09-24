"""Agent-level baseline: does Claude Code load the right skill on its own, and with skill-issue?

    python -m bench.agent_baseline --arm vanilla --size 100 --n 20      # pilot
    python -m bench.agent_baseline --arm router  --size 100 --n 20

For each catalog chunk, the skills are written as *project* skills into a fresh temp directory,
and `claude -p` runs there with:
  --setting-sources project   no user settings, user plugins, or user hooks
  --strict-mcp-config         no MCP servers
  --no-session-persistence    nothing written to the session history
so the only skills Claude sees are the catalog plus Claude Code's own built-in skills (the same
in both arms). Auth is the existing Claude Code login. The "router" arm adds this repo as a
plugin (--plugin-dir) whose hook talks to a daemon that can only see the temp project's skills.

We record every Skill tool call in the stream-json output. Runs are bounded by --max-turns (the
skill decision happens in the first turn) and always finish so the reported cost is captured.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from bench import metrics
from bench.catalogs import build_chunks
from bench.corpus import CACHE, ROOT
from bench.data import RESULTS, load_split, pool, save_json

OUT = RESULTS / "agent"
CLAUDE = shutil.which("claude") or "claude"


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9-]+", "-", name.lower().split(":")[-1]).strip("-")
    return s[:60] or "skill"


def write_catalog(root: Path, skill_idx: list[int]) -> dict[str, str]:
    """Write SKILL.md files; return {skill dir name: pool skill id}."""
    P = pool()
    skills_dir = root / ".claude" / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    names: dict[str, str] = {}
    for i in skill_idx:
        s = P.skills[i]
        base = slug(s.name)
        name, k = base, 2
        while name in names:
            name, k = f"{base}-{k}", k + 1
        names[name] = s.id
        d = skills_dir / name
        d.mkdir()
        desc = " ".join(s.description.split())[:1500]
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {json.dumps(desc)}\n---\n\n{s.body}\n", encoding="utf-8"
        )
    (root / ".git").mkdir(exist_ok=True)  # mark it a project root
    return names


def run_claude(
    prompt: str, cwd: Path, env: dict[str, str], model: str, plugin_dir: Path | None, max_turns: int, timeout: int = 240
) -> dict[str, Any]:
    cmd = [
        CLAUDE, "-p", prompt,
        "--setting-sources", "project",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--output-format", "stream-json", "--verbose",
        "--model", model,
        "--max-turns", str(max_turns),
        "--permission-mode", "dontAsk",
        "--disallowedTools", "Bash", "PowerShell", "Edit", "Write", "NotebookEdit", "WebFetch", "WebSearch", "Task",
    ]  # fmt: skip
    if plugin_dir is not None:
        cmd += ["--plugin-dir", str(plugin_dir), "--include-hook-events"]
    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    skills: list[str] = []
    tools: list[str] = []
    result: dict[str, Any] = {}
    injected = None
    init_skills: list[str] = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("type") == "system" and ev.get("subtype") == "init":
                init_skills = list(ev.get("skills") or [])
            if ev.get("type") == "system" and ev.get("subtype") == "hook_response":
                injected = ev.get("output") or ev.get("stdout")
            if ev.get("type") == "assistant":
                for c in ev["message"].get("content", []):
                    if c.get("type") == "tool_use":
                        tools.append(c["name"])
                        if c["name"] == "Skill":
                            skills.append(
                                str((c.get("input") or {}).get("skill") or (c.get("input") or {}).get("command") or "")
                            )
            if ev.get("type") == "result":
                result = {k: ev.get(k) for k in ("total_cost_usd", "num_turns", "duration_ms", "is_error", "subtype")}
                result["usage"] = ev.get("usage")
                result["text"] = str(ev.get("result") or "")[:300]
            if time.time() - t0 > timeout:
                break
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    return {
        "skills": skills,
        "tools": tools,
        "result": result,
        "seconds": round(time.time() - t0, 1),
        "injected": injected,
        "n_init_skills": len(init_skills),
    }


def api_failed(r: dict[str, Any]) -> bool:
    """The request never reached the model (usage limit, overload, network): zero tokens used."""
    res = r["result"]
    if not res:
        return not r["tools"]
    usage = res.get("usage") or {}
    return bool(res.get("is_error")) and not usage.get("input_tokens") and not usage.get("cache_read_input_tokens")


def run_with_retry(*args: Any, retries: int = 4, **kwargs: Any) -> dict[str, Any]:
    """Retry API-level failures with backoff; the result says how many attempts it took."""
    r: dict[str, Any] = {}
    for attempt in range(retries + 1):
        r = run_claude(*args, **kwargs)
        r["attempts"] = attempt + 1
        if not api_failed(r):
            return r
        time.sleep(min(600, 30 * 2**attempt))
    r["api_error"] = True
    return r


def start_daemon(home: Path, empty_claude: Path) -> subprocess.Popen[bytes]:
    cfg = (ROOT / "bench" / "agent_daemon.toml").read_text(encoding="utf-8")
    (home / "config.toml").write_text(cfg, encoding="utf-8")
    env = {**os.environ, "SKILL_ISSUE_HOME": str(home), "CLAUDE_CONFIG_DIR": str(empty_claude)}
    p = subprocess.Popen(
        [sys.executable, "-m", "skillissue", "daemon", "run"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 300
    while time.time() < deadline:
        r = subprocess.run(
            [sys.executable, "-m", "skillissue", "daemon", "status"], env=env, capture_output=True, text=True
        )
        if '"ready"' in r.stdout:
            return p
        time.sleep(2)
    p.kill()
    raise RuntimeError("daemon did not become ready")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["vanilla", "router"], required=True)
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--n", type=int, default=20, help="number of test prompts (stratified sample)")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--max-turns", type=int, default=2)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()

    test = load_split("test")
    rng = random.Random(a.seed)
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for p in test:
        by_kind.setdefault(p["kind"], []).append(p)
    sample: list[dict[str, Any]] = []
    for kind in sorted(by_kind):
        k = max(1, round(a.n * len(by_kind[kind]) / len(test)))
        sample += rng.sample(by_kind[kind], min(k, len(by_kind[kind])))
    sample = sample[: a.n]
    P = pool()
    chunks = build_chunks(sample, a.size, P, seed=a.seed)
    work = Path(tempfile.mkdtemp(prefix="skill-issue-agent-"))
    env = {k: v for k, v in os.environ.items() if not k.startswith("SKILL_ISSUE")}
    daemon = None
    records: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    try:
        if a.arm == "router":
            home, empty = work / "si-home", work / "empty-claude"
            home.mkdir()
            empty.mkdir()
            daemon = start_daemon(home, empty)
            env["SKILL_ISSUE_HOME"] = str(home)
        for ci, ch in enumerate(chunks):
            proj = work / f"chunk{ci}"
            name_to_id = write_catalog(proj, ch.catalog)

            def job(p: dict[str, Any], proj: Path = proj, name_to_id: dict[str, str] = name_to_id) -> dict[str, Any]:
                r = run_with_retry(p["prompt"], proj, env, a.model, ROOT if a.arm == "router" else None, a.max_turns)
                chosen = [name_to_id[s.split(":")[-1]] for s in r["skills"] if s.split(":")[-1] in name_to_id]
                other = [s for s in r["skills"] if s.split(":")[-1] not in name_to_id]
                return {"prompt": p, "run": r, "selected": sorted(set(chosen)), "other_skills": other}

            with ThreadPoolExecutor(a.workers) as ex:
                for res in ex.map(job, ch.prompts):
                    p = res["prompt"]
                    if res["run"].get("api_error"):
                        errors.append({"id": p["id"], "result": res["run"]["result"]})
                        print(
                            f"  {p['id']} API error, excluded: {res['run']['result'].get('text', '')[:120]}",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
                    records.append(
                        {
                            "id": p["id"],
                            "kind": p["kind"],
                            "labels": p["labels"],
                            "retrieved": [],
                            "gated": [],
                            "selected": res["selected"],
                            "ms": res["run"]["seconds"] * 1000,
                        }
                    )
                    raw.append(
                        {
                            "id": p["id"],
                            "kind": p["kind"],
                            "labels": p["labels"],
                            "selected": res["selected"],
                            "other_skills": res["other_skills"],
                            "tools": res["run"]["tools"],
                            "result": res["run"]["result"],
                            "seconds": res["run"]["seconds"],
                            "n_init_skills": res["run"]["n_init_skills"],
                            "attempts": res["run"]["attempts"],
                            "injected": res["run"]["injected"],
                        }
                    )
                    print(
                        f"  {p['id']} {p['kind']:22s} gold={len(p['labels'])} picked={res['selected']} other={res['other_skills']} {res['run']['seconds']}s",
                        file=sys.stderr,
                        flush=True,
                    )
    finally:
        if daemon is not None:
            daemon.terminate()
        shutil.rmtree(work, ignore_errors=True)
    m = metrics.compute(records)
    costs = [r["result"].get("total_cost_usd") or 0 for r in raw]
    m.pop("reliability", None)
    summary = {
        "arm": a.arm,
        "size": a.size,
        "n": len(records),
        "n_requested": len(sample),
        "n_api_errors_excluded": len(errors),
        "model": a.model,
        "metrics": m,
        "reported_cost_usd_total": round(sum(costs), 4),
        "reported_cost_usd_mean": round(sum(costs) / max(1, len(costs)), 4),
        "note": "Cost is Claude Code's own reported total_cost_usd; the runs used a subscription login, not an API key.",
    }
    tag = f"{a.arm}-N{a.size}-n{len(sample)}"
    save_json(OUT / f"{tag}.json", summary)
    save_json(CACHE / "bench" / "agent" / f"{tag}-records.json", raw)
    print(
        json.dumps(
            {k: summary[k] for k in ("arm", "size", "n", "reported_cost_usd_total")}
            | {
                "exact": m["exact_match"],
                "hit": m["hit_rate"],
                "none_acc": m["none_accuracy"],
                "fir": m["false_injection_rate"],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
