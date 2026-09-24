"""Config file (TOML) with defaults. Unknown keys are kept so newer configs survive older code."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

from . import paths

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

MODES = ("installed", "approved")

DEFAULTS: dict[str, Any] = {
    "mode": "installed",
    "retrieval": {
        # How many candidates retrieval hands to the gate stage.
        "top_k": 40,
        # Any sentence-transformers model, or "none" for BM25 only (no model download).
        "embed_model": "BAAI/bge-small-en-v1.5",
        # Characters of SKILL.md body used for routing (the body carries real signal).
        "body_chars": 1500,
        "rrf_k": 60,
    },
    "gate": {
        # laya | cross-encoder | retrieval | typesafe
        "name": "laya",
        # HF repo id or local path. Empty means the gate's built-in default.
        "model": "",
        # Only the top-N retrieval candidates are scored by the gate (latency bound).
        "max_candidates": 12,
        # Inject a skill when calibrated P(relevant) >= threshold. Empty = model's tuned value.
        "threshold": None,
        "max_skills": 3,
        "device": "auto",
    },
    "daemon": {
        "host": "127.0.0.1",
        "port": 0,
        "idle_timeout_min": 240,
    },
    "hook": {
        # The hook gives up and injects nothing after this long.
        "timeout_ms": 1500,
    },
    "sources": {
        "claude_user": True,
        "claude_project": True,
        "claude_plugins": True,
        "agents": True,
        "extra_dirs": [],
    },
    "approved": {
        # Where the vetted index comes from. A path or URL to approved-index.json.
        "index": "",
    },
}


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    def __init__(self, data: dict[str, Any] | None = None, path: Path | None = None):
        self.data = _merge(DEFAULTS, data or {})
        self.path = path

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        path = path or paths.config_file()
        data: dict[str, Any] = {}
        if path.is_file():
            with path.open("rb") as f:
                data = tomllib.load(f)
        return cls(data, path)

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        cur = self.data
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value

    @property
    def mode(self) -> str:
        m = str(self.data.get("mode", "installed"))
        return m if m in MODES else "installed"

    def save(self, path: Path | None = None) -> Path:
        path = path or self.path or paths.config_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps_toml(self.data), encoding="utf-8")
        self.path = path
        return path


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"cannot write {type(v).__name__} to TOML")


def dumps_toml(data: dict[str, Any]) -> str:
    """Minimal TOML writer for our config shape (scalars, lists, one level of tables)."""
    lines: list[str] = []
    tables: list[tuple[str, dict[str, Any]]] = []
    for k, v in data.items():
        if isinstance(v, dict):
            tables.append((k, v))
        elif v is not None:
            lines.append(f"{k} = {_toml_value(v)}")
    for name, table in tables:
        lines.append("")
        lines.append(f"[{name}]")
        for k, v in table.items():
            if v is None or isinstance(v, dict):
                continue
            lines.append(f"{k} = {_toml_value(v)}")
    return "\n".join(lines).lstrip("\n") + "\n"
