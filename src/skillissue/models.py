"""Model weights: resolve an HF repo id, a release URL or a local path to a local directory.

Downloads go to ~/.skill-issue/models/<name> as plain files (no symlinks, which keeps Windows
without developer mode working). Hugging Face repos come through huggingface_hub; the fine-tuned
gates are zip files attached to a GitHub release, pinned by SHA-256."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from . import paths

log = logging.getLogger(__name__)

RELEASE_URL = "https://github.com/SahilBachu/skill-issue/releases/download/gates-v1"
GTE_FT = f"{RELEASE_URL}/gte-skill-gate-v1.zip"
MINILM_FT = f"{RELEASE_URL}/minilm-skill-gate-v1.zip"

# Fine-tuned gates (docs/model-card.md). Each download must match its pinned hash. If it cannot be
# fetched (offline, or no access to the repository) the router uses the public base model instead,
# with the calibration fitted for it on our validation split (skillissue/data/gate_calibration.json).
RELEASE_MODELS: dict[str, dict[str, str]] = {
    GTE_FT: {
        "sha256": "83437e8deff57700e0fe380be8176468e01240fac43e6c03bd8c5f87c0d08c2f",
        "fallback": "Alibaba-NLP/gte-reranker-modernbert-base",
    },
    MINILM_FT: {
        "sha256": "46b7a62c0dbcb9597b295c2a14620fe66280924462c4a131e3e8a46446047200",
        "fallback": "cross-encoder/ms-marco-MiniLM-L6-v2",
    },
}

# Model behind each "auto" slot.
DEFAULT_MODELS = {"gte": GTE_FT, "minilm": MINILM_FT}
# Default weights when a gate is named explicitly.
DEFAULT_GATE_MODELS = {
    "laya": "convaiinnovations/laya",
    "cross-encoder": GTE_FT,
}
LAYA_FILES = ["model.safetensors", "encoder/*", "tokenizer/*", "rl_agent_config.json", "skill_issue_gate.json"]


def is_url(ref: str) -> bool:
    return ref.startswith(("https://", "http://"))


def local_name(ref: str) -> str:
    if is_url(ref):
        return "release--" + ref.rstrip("/").rsplit("/", 1)[-1].removesuffix(".zip")
    return ref.replace("/", "--")


def is_local(ref: str) -> bool:
    return not is_url(ref) and Path(ref).expanduser().exists()


def fallback_for(ref: str) -> str | None:
    return RELEASE_MODELS.get(ref, {}).get("fallback")


def ensure_model(ref: str, allow_patterns: list[str] | None = None, quiet: bool = False) -> Path:
    """Return a local dir for `ref`, downloading it on first use."""
    if is_local(ref):
        return Path(ref).expanduser()
    target = paths.models_dir() / local_name(ref)
    marker = target / ".complete"
    if marker.is_file():
        return target
    if is_url(ref):
        _download_zip(ref, target, RELEASE_MODELS.get(ref, {}).get("sha256"))
    else:
        from huggingface_hub import snapshot_download

        if quiet:
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        log.info("downloading %s to %s", ref, target)
        snapshot_download(ref, local_dir=str(target), allow_patterns=allow_patterns)
    marker.write_text("ok", encoding="utf-8")
    return target


def ensure_gate_model(ref: str) -> tuple[Path, str]:
    """Like ensure_model, but a release model that cannot be fetched falls back to its base model.
    Returns (local dir, the ref actually used)."""
    try:
        return ensure_model(ref), ref
    except Exception as e:
        fb = fallback_for(ref)
        if not fb:
            raise
        log.warning("could not fetch %s (%s); using the base model %s", ref, e, fb)
        return ensure_model(fb), fb


def _download_zip(url: str, target: Path, sha256: str | None) -> None:
    """Fetch a zip, check its hash, and unpack it into target (atomically, no path traversal)."""
    if sha256 is None:
        log.warning("no pinned hash for %s; the download is not verified", url)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".dl-", dir=target.parent))
    try:
        zpath = tmp / "model.zip"
        h = hashlib.sha256()
        log.info("downloading %s", url)
        req = urllib.request.Request(url, headers={"User-Agent": "skill-issue"})
        with urllib.request.urlopen(req, timeout=60) as r, zpath.open("wb") as f:
            while chunk := r.read(1 << 20):
                h.update(chunk)
                f.write(chunk)
        if sha256 and h.hexdigest() != sha256:
            raise ValueError(f"hash mismatch for {url}: got {h.hexdigest()}, expected {sha256}")
        out = tmp / "out"
        with zipfile.ZipFile(zpath) as z:
            for info in z.infolist():
                dest = (out / info.filename).resolve()
                if not dest.is_relative_to(out.resolve()):
                    raise ValueError(f"unsafe path in {url}: {info.filename}")
            z.extractall(out)
        # Accept both a flat zip and one with a single top-level folder.
        entries = list(out.iterdir())
        src = entries[0] if len(entries) == 1 and entries[0].is_dir() else out
        if target.exists():
            shutil.rmtree(target)
        src.rename(target)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def is_downloaded(ref: str) -> bool:
    if is_local(ref):
        return True
    return (paths.models_dir() / local_name(ref) / ".complete").is_file()
