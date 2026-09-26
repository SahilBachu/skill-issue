"""Package a fine-tuned cross-encoder gate for a GitHub release.

    python -m training.export_gate checkpoints/gte-mb-ft dist/gte-skill-gate-v1.zip --card docs/model-card.md

Weights are stored in fp16 (half the download; the router upcasts on CPU and runs fp16 on GPU).
The zip keeps skill_issue_gate.json, which carries the calibration bench.run fitted on the
validation split. Prints the SHA-256 to pin in skillissue/models.py."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

KEEP = {
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "skill_issue_gate.json",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--card", help="markdown model card, stored as README.md")
    a = ap.parse_args()
    import torch
    from safetensors.torch import load_file, save_file

    src, out = Path(a.checkpoint), Path(a.out)
    meta = json.loads((src / "skill_issue_gate.json").read_text(encoding="utf-8"))
    if not meta.get("calibration"):
        raise SystemExit(f"{src} has no calibration; run bench.run for it first")
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t) / out.stem
        tmp.mkdir()
        for f in src.iterdir():
            if f.name in KEEP:
                shutil.copy2(f, tmp / f.name)
        state = load_file(str(src / "model.safetensors"))
        state = {k: v.half() if v.is_floating_point() else v for k, v in state.items()}
        save_file(state, str(tmp / "model.safetensors"), metadata={"format": "pt"})
        if a.card:
            shutil.copy2(a.card, tmp / "README.md")
        out.parent.mkdir(parents=True, exist_ok=True)
        # Fixed timestamps so the same inputs give the same zip (and hash).
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(tmp.iterdir()):
                info = zipfile.ZipInfo(f"{out.stem}/{f.name}", date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(info, f.read_bytes())
    del torch
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"{out} {out.stat().st_size / 1e6:.1f} MB sha256 {digest}")


if __name__ == "__main__":
    main()
