"""Model weights: resolve an HF repo id or local path to a local directory.

Downloads go to ~/.skill-issue/models/<org>--<name> with local_dir (no symlinks, which keeps
Windows without developer mode working). Progress bars come from huggingface_hub."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from . import paths

log = logging.getLogger(__name__)

# Default weights per gate. The fine-tuned checkpoint is used when present locally or once
# published; until then the base model is the fallback (see `resolve_gate_model`).
DEFAULT_GATE_MODELS = {
    "laya": "convaiinnovations/laya",
    "cross-encoder": "Alibaba-NLP/gte-reranker-modernbert-base",
}
LAYA_FILES = ["model.safetensors", "encoder/*", "tokenizer/*", "rl_agent_config.json", "skill_issue_gate.json"]


def local_name(repo_id: str) -> str:
    return repo_id.replace("/", "--")


def is_local(ref: str) -> bool:
    return Path(ref).expanduser().exists()


def ensure_model(ref: str, allow_patterns: list[str] | None = None, quiet: bool = False) -> Path:
    """Return a local dir for `ref`, downloading from Hugging Face on first use."""
    p = Path(ref).expanduser()
    if p.exists():
        return p
    target = paths.models_dir() / local_name(ref)
    marker = target / ".complete"
    if marker.is_file():
        return target
    from huggingface_hub import snapshot_download

    if quiet:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    log.info("downloading %s to %s", ref, target)
    snapshot_download(ref, local_dir=str(target), allow_patterns=allow_patterns)
    marker.write_text("ok", encoding="utf-8")
    return target


def is_downloaded(ref: str) -> bool:
    if is_local(ref):
        return True
    return (paths.models_dir() / local_name(ref) / ".complete").is_file()
