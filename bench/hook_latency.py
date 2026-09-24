"""End-to-end hook latency: the time Claude Code waits on our UserPromptSubmit hook.

    python -m bench.hook_latency --gate cross-encoder --model checkpoints/gte-mb-ft --catalog 100

Starts a real daemon (real models) in an isolated SKILL_ISSUE_HOME, writes a catalog of N skills
as project skills, then runs `sh hooks/run-hook.sh prompt` with Claude Code's hook JSON for real
test prompts. Wall time per call includes shell + Python start, HTTP, retrieval, gate, and output.
Also records daemon RSS and GPU memory."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from bench.agent_baseline import write_catalog
from bench.corpus import ROOT
from bench.data import RESULTS, load_split, pool, save_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default="cross-encoder")
    ap.add_argument("--model", default="")
    ap.add_argument("--max-candidates", type=int, default=12)
    ap.add_argument("--catalog", type=int, default=100)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    import psutil

    work = Path(tempfile.mkdtemp(prefix="skill-issue-lat-"))
    home, empty, proj = work / "si", work / "claude", work / "proj"
    for p in (home, empty):
        p.mkdir()
    rng = random.Random(0)
    P = pool()
    write_catalog(proj, rng.sample(range(len(P.skills)), a.catalog))
    model = a.model
    if model.startswith("checkpoints/"):
        model = str(ROOT / model)
    (home / "config.toml").write_text(
        "[sources]\nclaude_user = false\nclaude_plugins = false\nagents = false\n\n"
        f'[gate]\nname = "{a.gate}"\nmodel = "{model.replace(chr(92), "/")}"\nmax_candidates = {a.max_candidates}\ndevice = "{a.device}"\n',
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "SKILL_ISSUE_HOME": str(home),
        "CLAUDE_CONFIG_DIR": str(empty),
        "SKILL_ISSUE_HOOK_PYTHON": sys.executable,
    }
    env["CLAUDE_PLUGIN_ROOT"] = str(ROOT)
    daemon = subprocess.Popen(
        [sys.executable, "-m", "skillissue", "daemon", "run"],
        env=env,
        cwd=proj,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    t_start = time.time()
    try:
        while True:
            r = subprocess.run(
                [sys.executable, "-m", "skillissue", "daemon", "status"], env=env, capture_output=True, text=True
            )
            try:
                status = json.loads(r.stdout).get("status")
            except ValueError:
                status = None
            if status == "ready":
                break
            if status == "error" or time.time() - t_start > 600:
                raise RuntimeError(r.stdout)
            time.sleep(1)
        cold_start_s = time.time() - t_start
        prompts = [p["prompt"] for p in load_split("test")]
        rng.shuffle(prompts)
        sh = shutil.which("sh") or "sh"
        times, injected = [], 0
        for i, prompt in enumerate(prompts[: a.n + 5]):
            payload = json.dumps(
                {"session_id": "lat", "hook_event_name": "UserPromptSubmit", "prompt": prompt, "cwd": str(proj)}
            )
            t = time.perf_counter()
            out = subprocess.run(
                [sh, str(ROOT / "hooks" / "run-hook.sh"), "prompt"],
                input=payload,
                env=env,
                capture_output=True,
                text=True,
            ).stdout
            ms = (time.perf_counter() - t) * 1000
            if i >= 5:  # warm-up
                times.append(ms)
                injected += bool(out)
        # Floor: the same launcher with the daemon answering nothing (SKILL_ISSUE_DISABLE) = pure process overhead.
        floor = []
        for _ in range(30):
            t = time.perf_counter()
            subprocess.run(
                [sh, str(ROOT / "hooks" / "run-hook.sh"), "prompt"],
                input="{}",
                env={**env, "SKILL_ISSUE_DISABLE": "1"},
                capture_output=True,
                text=True,
            )
            floor.append((time.perf_counter() - t) * 1000)
        # On Windows the venv's python.exe is a launcher; the daemon is its child.
        procs = [psutil.Process(daemon.pid), *psutil.Process(daemon.pid).children(recursive=True)]
        rss_mb = max(p.memory_info().rss for p in procs) / 1e6
        pids = {p.pid for p in procs}
        gpu_mb = None
        try:
            q = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
            ).stdout
            for line in q.splitlines():
                pid, mem = (x.strip() for x in line.split(","))
                if int(pid) in pids:
                    gpu_mb = float(mem)
        except (OSError, ValueError):
            pass
        times.sort()
        res = {
            "gate": a.gate,
            "model": a.model,
            "max_candidates": a.max_candidates,
            "catalog": a.catalog,
            "n": len(times),
            "device": a.device,
            "p50_ms": statistics.median(times),
            "p95_ms": times[int(0.95 * (len(times) - 1))],
            "mean_ms": statistics.mean(times),
            "process_floor_p50_ms": statistics.median(floor),
            "injected_fraction": injected / len(times),
            "daemon_rss_mb": round(rss_mb, 1),
            "daemon_gpu_mb": gpu_mb,
            "cold_start_s": round(cold_start_s, 1),
            "platform": sys.platform,
        }
        print(json.dumps(res, indent=1))
        tag = a.tag or f"{a.gate}-{Path(a.model).name or 'default'}-{a.device}-g{a.max_candidates}-N{a.catalog}"
        save_json(RESULTS / "hook_latency" / f"{tag}.json", res)
    finally:
        daemon.terminate()
        try:
            daemon.wait(timeout=20)
        except subprocess.TimeoutExpired:
            daemon.kill()
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
