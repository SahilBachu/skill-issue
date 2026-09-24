"""Filesystem locations. Everything lives under one home dir so the stdlib hook can find it."""

from __future__ import annotations

import os
from pathlib import Path


def home() -> Path:
    """$SKILL_ISSUE_HOME, or ~/.skill-issue."""
    env = os.environ.get("SKILL_ISSUE_HOME")
    return Path(env).expanduser() if env else Path.home() / ".skill-issue"


def config_file() -> Path:
    return home() / "config.toml"


def models_dir() -> Path:
    return home() / "models"


def index_dir() -> Path:
    return home() / "index"


def runtime_dir() -> Path:
    return home() / "run"


def runtime_file() -> Path:
    """Written by the daemon: {"port", "pid", "token", "version"}."""
    return runtime_dir() / "daemon.json"


def daemon_log() -> Path:
    return runtime_dir() / "daemon.log"


def approved_dir() -> Path:
    return home() / "approved"


def claude_home() -> Path:
    env = os.environ.get("CLAUDE_CONFIG_DIR")
    return Path(env).expanduser() if env else Path.home() / ".claude"


def ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
