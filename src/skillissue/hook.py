#!/usr/bin/env python3
"""Claude Code hook entry point. Standard library only, and no package-relative imports, so the
plugin can run this file directly (`python -I -S hook.py prompt`) before any venv exists.

Rules:
- Never block the user. Any error, timeout, or missing daemon means: print nothing, exit 0.
- Start the daemon lazily (and bootstrap its venv on first use) in the background.

Subcommands:
    prompt         UserPromptSubmit: read hook JSON on stdin, print additionalContext JSON.
    session-start  SessionStart: make sure the venv and daemon are coming up. Prints nothing
                   except a one-time notice while the first install runs.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()
# In the plugin layout this file is <root>/src/skillissue/hook.py.
PLUGIN_ROOT = Path(os.environ.get("CLAUDE_PLUGIN_ROOT") or HERE.parents[2])


def home() -> Path:
    env = os.environ.get("SKILL_ISSUE_HOME")
    return Path(env).expanduser() if env else Path.home() / ".skill-issue"


def runtime_file() -> Path:
    return home() / "run" / "daemon.json"


def venv_dir() -> Path:
    data = os.environ.get("CLAUDE_PLUGIN_DATA")
    return Path(data) / "venv" if data else home() / "venv"


def venv_python() -> Path | None:
    v = venv_dir()
    for cand in (v / "bin" / "python", v / "Scripts" / "python.exe"):
        if cand.is_file():
            return cand
    return None


def _timeout_s() -> float:
    try:
        return max(0.1, float(os.environ.get("SKILL_ISSUE_HOOK_TIMEOUT_MS", "1500")) / 1000)
    except ValueError:
        return 1.5


def _detached_popen(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("ab")
    kwargs: dict[str, Any] = {"stdin": subprocess.DEVNULL, "stdout": log_f, "stderr": log_f, "close_fds": True}
    if os.name == "nt":
        kwargs["creationflags"] = 0x00000008 | 0x00000200 | 0x08000000
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(cmd, **kwargs)


def _lock(name: str, stale_s: float) -> bool:
    """Best-effort cross-process lock so concurrent prompts don't spawn twice."""
    p = home() / "run" / f"{name}.lock"
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        if p.exists() and time.time() - p.stat().st_mtime > stale_s:
            p.unlink()
        fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except OSError:
        return False


def daemon_python() -> str | None:
    """A Python that has skillissue installed: env override, plugin venv, or the last daemon's."""
    env = os.environ.get("SKILL_ISSUE_PYTHON")
    if env:
        return env
    vp = venv_python()
    if vp is not None and (venv_dir() / ".skill-issue-installed").is_file():
        return str(vp)
    marker = home() / "python.txt"  # written by `skill-issue daemon run` / `skill-issue doctor`
    try:
        p = marker.read_text(encoding="utf-8").strip()
        if p and Path(p).is_file():
            return p
    except OSError:
        pass
    return None


def install_in_background() -> bool:
    """First-run bootstrap: create the venv and install this plugin's package into it."""
    if not _lock("install", stale_s=1800):
        return False
    v = venv_dir()
    log = home() / "run" / "install.log"
    uv = shutil.which("uv")
    root = str(PLUGIN_ROOT)
    stamp = str(v / ".skill-issue-installed")
    if uv:
        script = (
            f'"{uv}" venv --allow-existing "{v}" --python 3.12 && '
            f'"{uv}" pip install --python "{v}" --torch-backend auto "{root}" && '
            f'echo ok > "{stamp}"'
        )
    else:
        py = sys.executable
        vpy = v / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        script = f'"{py}" -m venv "{v}" && "{vpy}" -m pip install -q "{root}" && echo ok > "{stamp}"'
    lockfile = home() / "run" / "install.lock"
    script += f'; rm -f "{lockfile}"'
    shell = (
        ["cmd", "/c", script.replace("rm -f", "del /f")]
        if os.name == "nt" and not shutil.which("sh")
        else ["sh", "-c", script]
    )
    _detached_popen(shell, log)
    return True


def start_daemon() -> None:
    py = daemon_python()
    if py is None:
        install_in_background()
        return
    if not _lock("spawn", stale_s=60):
        return
    _detached_popen([py, "-m", "skillissue", "daemon", "run"], home() / "run" / "daemon.log")


def post(route: str, payload: dict[str, Any], timeout: float) -> dict[str, Any] | None:
    try:
        rt = json.loads(runtime_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        start_daemon()
        return None
    req = urllib.request.Request(
        f"http://127.0.0.1:{rt['port']}{route}",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"X-Skill-Issue-Token": rt.get("token", ""), "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body: dict[str, Any] = json.loads(r.read())
            return body
    except urllib.error.HTTPError:
        return None  # 503 while loading, 403, 500: fail open
    except (urllib.error.URLError, OSError) as e:
        if _daemon_is_gone(e, rt.get("pid")):
            # Stale runtime file (daemon died): remove it and start a new one.
            try:
                runtime_file().unlink()
            except OSError:
                pass
            start_daemon()
        # Anything else, above all a timeout on a slow request, just fails open. The daemon is
        # alive and busy; restarting it would make things worse.
        return None
    except ValueError:
        return None


def _daemon_is_gone(e: BaseException, pid: Any) -> bool:
    """True when the daemon is really dead, not just slow.

    A refused connection means nothing is listening. A timeout is ambiguous: on Windows even a
    dead localhost port times out instead of refusing, so check the recorded pid."""
    reason = getattr(e, "reason", e)
    if isinstance(reason, (ConnectionRefusedError, ConnectionResetError)):
        return True
    return isinstance(pid, int) and not _pid_alive(pid)


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not handle:
            return False
        try:
            return bool(kernel32.WaitForSingleObject(handle, 0) == 0x102)  # WAIT_TIMEOUT: still running
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)  # POSIX: signal 0 only checks existence
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def cmd_prompt(stdin_text: str) -> str:
    """Returns what to print on stdout ('' for nothing)."""
    if os.environ.get("SKILL_ISSUE_DISABLE") == "1":
        return ""
    try:
        event = json.loads(stdin_text or "{}")
    except ValueError:
        return ""
    prompt = str(event.get("prompt") or event.get("user_input") or "").strip()
    if not prompt or prompt.startswith("/"):
        return ""  # slash commands already say what they want
    res = post("/route", {"prompt": prompt, "cwd": event.get("cwd")}, _timeout_s())
    ctx = (res or {}).get("context") or ""
    if not ctx:
        return ""
    return json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": ctx}})


def cmd_session_start() -> str:
    if os.environ.get("SKILL_ISSUE_DISABLE") == "1":
        return ""
    if daemon_python() is None:
        if install_in_background():
            return json.dumps(
                {
                    "systemMessage": "skill-issue: first run, installing the router in the background (a few minutes). Prompts run normally meanwhile."
                }
            )
        return ""
    try:
        json.loads(runtime_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        start_daemon()
    return ""


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else "prompt"
    try:
        if cmd == "prompt":
            out = cmd_prompt(sys.stdin.read())
        elif cmd == "session-start":
            out = cmd_session_start()
        else:
            return 0
        if out:
            sys.stdout.write(out)
    except Exception:
        pass  # fail open, always
    return 0


if __name__ == "__main__":
    sys.exit(main())
