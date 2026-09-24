"""Generate cloud/skill_issue_cloud.ipynb (run once after editing the cells below)."""

from __future__ import annotations

import json
from pathlib import Path

D = "cloud_data"
PAIRS = f"--pairs {D}/pairs.jsonl --skills {D}/skills.json --val {D}/val_pairs.jsonl"
CELLS: list[tuple[str, str]] = [
    (
        "md",
        """# skill-issue: cloud GPU jobs

Runs the GPU-heavy parts of the skill-issue benchmark on a cloud GPU (a free Colab T4 works).

1. Locally: `python -m cloud.prepare` writes `.cache/cloud/bundle.zip`.
2. Here: *Runtime > Change runtime type > GPU*, then run the cells in order and upload the bundle when asked.
3. Results land in Google Drive under `skill-issue-cloud/out` (and in `results_part1.zip` / `results_part2.zip`). Put them in `.cache/cloud/` and run `python -m cloud.ingest` locally.

Every step writes to `/content/out` and skips work that is already done, so you can rerun a cell after a disconnect.""",
    ),
    ("code", "!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    (
        "code",
        """# Everything is written to Google Drive, so a disconnect or a sleeping laptop loses nothing.
import os, shutil
from google.colab import drive, files
drive.mount('/content/drive')
W = '/content/drive/MyDrive/skill-issue-cloud'
os.makedirs(f'{W}/out', exist_ok=True)
os.chdir('/content')
if not os.path.exists('bundle.zip'):
    if os.path.exists(f'{W}/bundle.zip'):
        shutil.copy(f'{W}/bundle.zip', 'bundle.zip')
    else:
        files.upload()  # choose bundle.zip
        shutil.copy('bundle.zip', f'{W}/bundle.zip')
if not os.path.islink('/content/out'):
    shutil.rmtree('/content/out', ignore_errors=True)
    os.symlink(f'{W}/out', '/content/out')
print(os.listdir('/content/out'))""",
    ),
    (
        "code",
        """!unzip -oq bundle.zip
%cd /content/skill-issue
!pip install -q -e . bitsandbytes datasets 2>&1 | tail -2
os.environ['USE_TF'] = '0'""",
    ),
    ("md", "## Part 1: gte and MiniLM fine-tunes, SkillRouter baseline, SkillRet reranking"),
    (
        "code",
        f"""!test -f /content/out/gte-mb-ft/skill_issue_gate.json || python -m training.finetune_cross --base Alibaba-NLP/gte-reranker-modernbert-base --out /content/out/gte-mb-ft --runs-dir /content/runs/gte --micro 16 {PAIRS} 2>&1 | grep -vi warn | tail -4
!python -m cloud.score --model cross:/content/out/gte-mb-ft --requests {D}/requests_bench.jsonl --out /content/out/scores_gte-mb-ft_bench.json 2>&1 | tail -1
!python -m cloud.score --model cross:/content/out/gte-mb-ft --requests {D}/requests_skillret.jsonl --out /content/out/scores_gte-mb-ft_skillret.json 2>&1 | tail -1""",
    ),
    (
        "code",
        f"""!test -f /content/out/minilm-ft/skill_issue_gate.json || python -m training.finetune_cross --base cross-encoder/ms-marco-MiniLM-L6-v2 --out /content/out/minilm-ft --runs-dir /content/runs/minilm {PAIRS} 2>&1 | grep -vi warn | tail -4
!python -m cloud.score --model cross:/content/out/minilm-ft --requests {D}/requests_bench.jsonl --out /content/out/scores_minilm-ft_bench.json 2>&1 | tail -1
!python -m cloud.score --model cross:/content/out/minilm-ft --requests {D}/requests_skillret.jsonl --out /content/out/scores_minilm-ft_skillret.json 2>&1 | tail -1
!python -m cloud.score --model cross:Alibaba-NLP/gte-reranker-modernbert-base --requests {D}/requests_skillret.jsonl --out /content/out/scores_gte-mb-zs_skillret.json 2>&1 | tail -1""",
    ),
    (
        "code",
        f"""!test -f /content/out/sr_vecs.npz || python -m cloud.sr_embed --docs {D}/sr_docs.jsonl --queries {D}/sr_queries.jsonl --out /content/out/sr_vecs.npz 2>&1 | tail -1
!python -m cloud.sr_bench --vecs /content/out/sr_vecs.npz --chunks {D}/chunks.json --docs {D}/sr_docs.jsonl --queries {D}/sr_queries.jsonl --out /content/out/sr_bench.json 2>&1 | tail -3
!test -f /content/out/skillret_sr_vecs.npz || python -m cloud.sr_embed --docs {D}/skillret_docs.jsonl --queries {D}/skillret_queries.jsonl --out /content/out/skillret_sr_vecs.npz 2>&1 | tail -1
!test -f /content/out/skillret_skillrouter.json || python -m cloud.skillret_sr --vecs /content/out/skillret_sr_vecs.npz --docs {D}/skillret_docs.jsonl --queries {D}/skillret_queries.jsonl --out /content/out/skillret_skillrouter.json 2>&1 | tail -1""",
    ),
    (
        "code",
        """!cd /content/out && rm -f results_part1.zip && zip -qr results_part1.zip scores_*.json sr_*.json sr_vecs.npz skillret_*.json skillret_sr_vecs.npz minilm-ft gte-mb-ft -x '*/checkpoint-*' && ls -la results_part1.zip
files.download('/content/out/results_part1.zip')""",
    ),
    ("md", "## Part 2: Laya fine-tune (the long one)"),
    (
        "code",
        f"""!test -f /content/out/laya-ft/skill_issue_gate.json || python -m training.finetune_laya --out /content/out/laya-ft {PAIRS} 2>&1 | grep -E 'step|val|saved|done|precision|trainable|Error' | tail -40
!python -m cloud.score --model laya:/content/out/laya-ft --requests {D}/requests_bench.jsonl --out /content/out/scores_laya-ft_bench.json 2>&1 | tail -1
!python -m cloud.score --model laya:/content/out/laya-ft --requests {D}/requests_skillret.jsonl --out /content/out/scores_laya-ft_skillret.json 2>&1 | tail -1""",
    ),
    (
        "code",
        """!cd /content/out && rm -f results_part2.zip && zip -qr results_part2.zip scores_laya-ft_*.json laya-ft-trainlog.json laya-ft/skill_issue_gate.json && ls -la results_part2.zip
files.download('/content/out/results_part2.zip')""",
    ),
    (
        "md",
        "Optional: the Laya checkpoint itself (about 850 MB), only needed to serve the fine-tuned Laya gate locally.",
    ),
    (
        "code",
        """!cd /content/out && rm -f laya-ft.zip && zip -qr laya-ft.zip laya-ft && ls -la laya-ft.zip
files.download('/content/out/laya-ft.zip')""",
    ),
]


def main() -> None:
    cells = []
    for kind, src in CELLS:
        lines = src.split("\n")
        source = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
        if kind == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": source})
        else:
            cells.append(
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": source}
            )
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }
    out = Path(__file__).with_name("skill_issue_cloud.ipynb")
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
