"""End to end: real daemon process + the plugin's real hook launcher (hooks/run-hook.sh), fed the
same JSON Claude Code sends on UserPromptSubmit. Model-free (BM25 + feature gate) so it runs in CI."""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ..conftest import SKILLS, write_skill

pytestmark = pytest.mark.e2e
ROOT = Path(__file__).resolve().parents[2]
SH = shutil.which("sh")

CONFIG = """
[retrieval]
embed_model = "none"

[gate]
name = "retrieval"
max_candidates = 5
retrieval_weights = { log_bm25 = 1.0 }
retrieval_calibration = { a = 4.0, b = -3.0, threshold = 0.5 }
"""


@pytest.fixture
def stack(tmp_path: Path):
    si, claude, home = tmp_path / "si", tmp_path / "claude", tmp_path / "home"
    for p in (si, claude, home):
        p.mkdir()
    for name, (desc, body) in SKILLS.items():
        write_skill(claude / "skills", name, desc, body)
    (si / "config.toml").write_text(CONFIG, encoding="utf-8")
    env = {
        **os.environ,
        "SKILL_ISSUE_HOME": str(si),
        "CLAUDE_CONFIG_DIR": str(claude),
        "HOME": str(home),
        "USERPROFILE": str(home),
    }
    env.pop("CLAUDE_PLUGIN_DATA", None)
    env["CLAUDE_PLUGIN_ROOT"] = str(ROOT)
    env["SKILL_ISSUE_HOOK_PYTHON"] = sys.executable
    proc = subprocess.Popen(
        [sys.executable, "-m", "skillissue", "daemon", "run"],
        env=env,
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    rt = si / "run" / "daemon.json"
    deadline = time.time() + 60
    ready = False
    while time.time() < deadline and not ready:
        time.sleep(0.2)
        if rt.is_file():
            r = subprocess.run(
                [sys.executable, "-m", "skillissue", "daemon", "status"], env=env, capture_output=True, text=True
            )
            ready = '"ready"' in r.stdout
    assert ready, "daemon did not become ready"
    yield {"env": env, "proc": proc, "tmp": tmp_path}
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def run_hook(env: dict, prompt: str, cwd: Path) -> tuple[str, int, float]:
    payload = json.dumps(
        {"session_id": "e2e", "hook_event_name": "UserPromptSubmit", "prompt": prompt, "cwd": str(cwd)}
    )
    t = time.perf_counter()
    r = subprocess.run(
        [SH, str(ROOT / "hooks" / "run-hook.sh"), "prompt"],
        input=payload,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return r.stdout, r.returncode, (time.perf_counter() - t) * 1000


@pytest.mark.skipif(SH is None, reason="needs a POSIX sh (Claude Code runs hooks with sh/Git Bash)")
def test_hook_injects_matching_skill(stack):
    out, code, _ = run_hook(stack["env"], "merge these two pdf files and extract the tables", stack["tmp"])
    assert code == 0
    ctx = json.loads(out)["hookSpecificOutput"]["additionalContext"]
    assert "pdf-tools" in ctx
    assert "sql-tuning" not in ctx


@pytest.mark.skipif(SH is None, reason="needs a POSIX sh")
def test_hook_injects_nothing_when_nothing_fits(stack):
    out, code, _ = run_hook(stack["env"], "thanks, that's all for today", stack["tmp"])
    assert code == 0 and out == ""


@pytest.mark.skipif(SH is None, reason="needs a POSIX sh")
def test_hook_latency_is_reasonable(stack):
    times = [run_hook(stack["env"], "why is my sql query slow, need an index", stack["tmp"])[2] for _ in range(7)]
    # Loose bound for CI runners; the real latency benchmark lives in bench/hook_latency.py.
    assert statistics.median(times) < 1500


@pytest.mark.skipif(SH is None, reason="needs a POSIX sh")
def test_hook_fails_open_when_daemon_dies(stack):
    stack["proc"].kill()
    stack["proc"].wait(timeout=10)
    env = {**stack["env"], "SKILL_ISSUE_DISABLE": ""}
    # Keep the hook from spawning a replacement daemon during the test.
    (Path(env["SKILL_ISSUE_HOME"]) / "run").mkdir(exist_ok=True)
    (Path(env["SKILL_ISSUE_HOME"]) / "run" / "spawn.lock").write_text("test", encoding="utf-8")
    out, code, ms = run_hook(env, "merge these two pdf files", stack["tmp"])
    assert code == 0
    assert out == ""
    assert ms < 5000
