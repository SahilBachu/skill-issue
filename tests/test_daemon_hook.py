"""Daemon + hook integration, in-process with a fake router (no models)."""

import json
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from skillissue import daemon, hook
from skillissue.config import Config
from skillissue.gates import Calibration, RetrievalGate
from skillissue.router import Router


@pytest.fixture
def running(env, hash_embedder, monkeypatch):
    """A daemon server on a free port with a model-free router, plus its runtime file."""
    cfg = Config()
    state = daemon.State(cfg)
    gate = RetrievalGate({"cosine": 10.0, "log_bm25": 1.0}, Calibration(1.0, -4.0, 0.5))
    state.router = Router(cfg, embedder=hash_embedder, gate=gate, load_models=False)
    state.status = "ready"
    ref: dict = {}
    server = ThreadingHTTPServer(("127.0.0.1", 0), daemon.make_handler(state, "tok", ref))
    ref["server"] = server
    daemon._write_runtime(server.server_address[1], "tok")
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    spawned: list = []
    monkeypatch.setattr(hook, "start_daemon", lambda: spawned.append(1))
    yield {"state": state, "server": server, "spawned": spawned}
    server.shutdown()
    server.server_close()


def event(prompt: str, cwd: Path) -> str:
    return json.dumps({"hook_event_name": "UserPromptSubmit", "prompt": prompt, "cwd": str(cwd), "session_id": "s"})


def test_health_and_auth(running):
    h = daemon.health()
    assert h is not None and h["status"] == "ready"
    rt = daemon.read_runtime()
    rt["token"] = "wrong"
    daemon.paths.runtime_file().write_text(json.dumps(rt), encoding="utf-8")
    assert daemon.request("GET", "/health")["http_status"] == 403


def test_hook_injects_context(running, env):
    out = hook.cmd_prompt(event("extract the tables from this pdf file", env["tmp"]))
    data = json.loads(out)
    ctx = data["hookSpecificOutput"]["additionalContext"]
    assert data["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "pdf-tools" in ctx


def test_hook_accepts_user_input_field(running, env):
    raw = json.dumps({"user_input": "extract the tables from this pdf file", "cwd": str(env["tmp"])})
    assert "pdf-tools" in hook.cmd_prompt(raw)


def test_hook_injects_nothing_for_unrelated_prompt(running, env):
    assert hook.cmd_prompt(event("thanks, looks good", env["tmp"])) == ""


def test_hook_skips_slash_commands_and_bad_input(running, env):
    assert hook.cmd_prompt(event("/compact", env["tmp"])) == ""
    assert hook.cmd_prompt("not json") == ""
    assert hook.cmd_prompt("") == ""


def test_hook_disabled_by_env(running, env, monkeypatch):
    monkeypatch.setenv("SKILL_ISSUE_DISABLE", "1")
    assert hook.cmd_prompt(event("extract the tables from this pdf file", env["tmp"])) == ""


def test_hook_fails_open_while_loading(running, env):
    running["state"].status = "loading"
    assert hook.cmd_prompt(event("extract the tables from this pdf file", env["tmp"])) == ""
    assert running["spawned"] == []  # daemon is up, just not ready: don't spawn another


def test_hook_fails_open_and_restarts_when_daemon_is_gone(running, env):
    running["server"].shutdown()
    running["server"].server_close()
    t = time.perf_counter()
    assert hook.cmd_prompt(event("extract the tables from this pdf file", env["tmp"])) == ""
    assert time.perf_counter() - t < 2.0
    assert running["spawned"] == [1]
    assert not daemon.paths.runtime_file().exists()  # stale runtime file removed


def test_hook_fails_open_on_timeout(running, env, monkeypatch):
    def slow_route(*a, **k):
        time.sleep(2)
        raise RuntimeError("too slow")

    monkeypatch.setattr(running["state"].router, "route", slow_route)
    monkeypatch.setenv("SKILL_ISSUE_HOOK_TIMEOUT_MS", "200")
    t = time.perf_counter()
    assert hook.cmd_prompt(event("extract the tables from this pdf file", env["tmp"])) == ""
    assert time.perf_counter() - t < 1.5


def test_hook_without_runtime_starts_daemon(env, monkeypatch):
    spawned: list = []
    monkeypatch.setattr(hook, "start_daemon", lambda: spawned.append(1))
    assert hook.cmd_prompt(event("anything", env["tmp"])) == ""
    assert spawned == [1]


def test_hook_main_never_raises(env, monkeypatch, capsys):
    monkeypatch.setattr(hook, "cmd_prompt", lambda s: 1 / 0)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("{}"))
    assert hook.main(["prompt"]) == 0
    assert capsys.readouterr().out == ""
