"""The warm daemon: keeps the embedder and gate loaded and answers route requests over
localhost HTTP. Requests need the per-start token from ~/.skill-issue/run/daemon.json, so other
local users and web pages cannot query it.

It starts listening immediately and loads models in the background; until then /route returns
503 and the hook injects nothing (fail open)."""

from __future__ import annotations

import json
import logging
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import __version__, paths
from .config import Config

log = logging.getLogger("skillissue.daemon")


class State:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.router: Any = None
        self.status = "loading"
        self.error: str | None = None
        self.started = time.time()
        self.last_request = time.time()
        self.requests = 0
        self.lock = threading.Lock()

    def load(self) -> None:
        try:
            from .router import Router

            t = time.perf_counter()
            router = Router(self.cfg)
            assert router.gate is not None
            router.gate.warmup()
            router.catalog(Path.cwd())
            self.router = router
            self.status = "ready"
            log.info("models ready in %.1fs (gate=%s)", time.perf_counter() - t, router.gate.name)
        except Exception as e:  # keep serving /health so doctor can show the error
            self.status = "error"
            self.error = f"{type(e).__name__}: {e}"
            log.exception("model load failed")


def make_handler(state: State, token: str, server_ref: dict[str, Any]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            log.debug(fmt, *args)

        def _send(self, code: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authed(self) -> bool:
            return secrets.compare_digest(self.headers.get("X-Skill-Issue-Token", ""), token)

        def do_GET(self) -> None:
            if not self._authed():
                return self._send(403, {"error": "bad token"})
            if self.path == "/health":
                r = state.router
                return self._send(
                    200,
                    {
                        "status": state.status,
                        "error": state.error,
                        "version": __version__,
                        "pid": os.getpid(),
                        "uptime_s": round(time.time() - state.started, 1),
                        "requests": state.requests,
                        "mode": state.cfg.mode,
                        "gate": r.gate.name if r is not None and r.gate is not None else None,
                    },
                )
            self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if not self._authed():
                return self._send(403, {"error": "bad token"})
            n = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._send(400, {"error": "bad json"})
            state.last_request = time.time()
            if self.path == "/route":
                if state.status != "ready":
                    return self._send(503, {"status": state.status, "error": state.error})
                return self._route(payload)
            if self.path == "/reload":
                state.cfg = Config.load()
                state.status = "loading"
                threading.Thread(target=state.load, daemon=True).start()
                return self._send(200, {"status": "reloading"})
            if self.path == "/shutdown":
                self._send(200, {"status": "bye"})
                threading.Thread(target=server_ref["server"].shutdown, daemon=True).start()
                return None
            self._send(404, {"error": "not found"})

        def _route(self, payload: dict[str, Any]) -> None:
            from .inject import render_context

            prompt = str(payload.get("prompt") or "")
            cwd = payload.get("cwd")
            try:
                with state.lock:
                    state.requests += 1
                    res = state.router.route(prompt, Path(cwd) if cwd else None)
            except Exception as e:
                log.exception("route failed")
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})
            out = res.to_dict(with_candidates=bool(payload.get("candidates")))
            out["context"] = render_context(res)
            self._send(200, out)

    return Handler


def _write_runtime(port: int, token: str) -> None:
    rt = paths.runtime_file()
    paths.ensure(rt.parent)
    tmp = rt.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {"port": port, "pid": os.getpid(), "token": token, "version": __version__, "python": sys.executable}
        ),
        encoding="utf-8",
    )
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    tmp.replace(rt)
    # Lets the stdlib hook find a Python with skillissue installed when there is no plugin venv.
    (paths.home() / "python.txt").write_text(sys.executable, encoding="utf-8")
    try:
        (paths.runtime_dir() / "spawn.lock").unlink()  # the hook's "starting" marker
    except OSError:
        pass


def run(cfg: Config | None = None, port: int | None = None) -> None:
    """Run in the foreground until shutdown or idle timeout."""
    cfg = cfg or Config.load()
    paths.ensure(paths.runtime_dir())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(paths.daemon_log(), encoding="utf-8"), logging.StreamHandler(sys.stderr)],
    )
    state = State(cfg)
    token = secrets.token_urlsafe(24)
    server_ref: dict[str, Any] = {}
    host = str(cfg.get("daemon.host", "127.0.0.1"))
    server = ThreadingHTTPServer(
        (host, int(port if port is not None else cfg.get("daemon.port", 0))), make_handler(state, token, server_ref)
    )
    server.daemon_threads = True
    server_ref["server"] = server
    _write_runtime(server.server_address[1], token)
    log.info("listening on %s:%s (pid %s)", host, server.server_address[1], os.getpid())
    threading.Thread(target=state.load, daemon=True).start()

    idle = float(cfg.get("daemon.idle_timeout_min", 240)) * 60

    def reaper() -> None:
        while True:
            time.sleep(30)
            if idle > 0 and time.time() - state.last_request > idle:
                log.info("idle for %.0f min, exiting", idle / 60)
                server.shutdown()
                return

    threading.Thread(target=reaper, daemon=True).start()
    try:
        signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
    except (ValueError, AttributeError):  # pragma: no cover - not main thread / platform
        pass
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        try:
            data = json.loads(paths.runtime_file().read_text(encoding="utf-8"))
            if data.get("pid") == os.getpid():
                paths.runtime_file().unlink()
        except (OSError, ValueError):
            pass


# ---- client side ----------------------------------------------------------------------


def read_runtime() -> dict[str, Any] | None:
    try:
        data: dict[str, Any] = json.loads(paths.runtime_file().read_text(encoding="utf-8"))
        return data
    except (OSError, ValueError):
        return None


def request(method: str, route: str, payload: dict[str, Any] | None = None, timeout: float = 5.0) -> dict[str, Any]:
    rt = read_runtime()
    if not rt:
        raise ConnectionError("daemon not running")
    url = f"http://127.0.0.1:{rt['port']}{route}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method, headers={"X-Skill-Issue-Token": rt["token"], "Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body: dict[str, Any] = json.loads(r.read())
            return body
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read())
        except ValueError:
            body = {"error": str(e)}
        body.setdefault("http_status", e.code)
        return body


def health(timeout: float = 2.0) -> dict[str, Any] | None:
    try:
        return request("GET", "/health", timeout=timeout)
    except (OSError, ConnectionError, ValueError):
        return None


def spawn(python: str | None = None) -> int:
    """Start the daemon detached from this process. Returns the child pid."""
    py = python or sys.executable
    paths.ensure(paths.runtime_dir())
    log_f = paths.daemon_log().open("ab")
    kwargs: dict[str, Any] = {"stdin": subprocess.DEVNULL, "stdout": log_f, "stderr": log_f, "close_fds": True}
    if os.name == "nt":
        kwargs["creationflags"] = 0x00000008 | 0x00000200 | 0x08000000  # DETACHED | NEW_GROUP | NO_WINDOW
    else:
        kwargs["start_new_session"] = True
    p = subprocess.Popen([py, "-m", "skillissue", "daemon", "run"], **kwargs)
    return p.pid


def ensure_running(wait_ready: float = 0.0) -> dict[str, Any] | None:
    """Start the daemon if it is not up. Optionally wait until models are loaded."""
    h = health()
    if h is None:
        spawn()
        deadline = time.time() + 20
        while time.time() < deadline and h is None:
            time.sleep(0.3)
            h = health()
    if wait_ready > 0:
        deadline = time.time() + wait_ready
        while h is not None and h.get("status") == "loading" and time.time() < deadline:
            time.sleep(0.5)
            h = health()
    return h


def stop(timeout: float = 10.0) -> bool:
    if health() is None:
        return False
    try:
        request("POST", "/shutdown", {}, timeout=3)
    except (OSError, ConnectionError):
        pass
    deadline = time.time() + timeout
    while time.time() < deadline:
        if health(timeout=0.5) is None:
            return True
        time.sleep(0.2)
    return False
