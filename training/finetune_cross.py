"""Fine-tune a cross-encoder reranker as a skill gate on the same pairs as Laya.

    python -m training.finetune_cross --base Alibaba-NLP/gte-reranker-modernbert-base --out checkpoints/gte-mb-ft
    python -m training.finetune_cross --base cross-encoder/ms-marco-MiniLM-L6-v2 --out checkpoints/minilm-ft

Same inputs as inference: (clip_prompt(request), gate_text(skill)). BCE with a positive weight,
bf16, best checkpoint by validation loss on our val split."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import torch  # noqa: E402

from bench.corpus import ROOT  # noqa: E402
from skillissue.gates.base import clip_prompt, gate_text  # noqa: E402
from skillissue.skill import Skill  # noqa: E402
from training.finetune_laya import SKILLS, load_pairs, val_pairs  # noqa: E402


def to_columns(pairs: list[dict], skills: dict, body_chars: int) -> dict[str, list]:
    q, d, y = [], [], []
    for p in pairs:
        if "skill_obj" in p:
            s = p["skill_obj"]
        else:
            r = skills[p["skill"]]
            s = Skill(id=r["id"], name=r["name"], description=r["description"], body=r.get("body", ""))
        q.append(clip_prompt(p["prompt"]))
        d.append(gate_text(s, body_chars))
        y.append(float(p["label"]))
    return {"query": q, "passage": d, "label": y}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--neg-ratio", type=float, default=8.0)
    ap.add_argument("--pos-weight", type=float, default=2.0)
    ap.add_argument("--body-chars", type=int, default=300)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args(argv)

    from datasets import Dataset
    from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    skills = json.loads(SKILLS.read_text(encoding="utf-8"))
    train = Dataset.from_dict(to_columns(load_pairs(a.neg_ratio, a.seed), skills, a.body_chars))
    val = Dataset.from_dict(to_columns(val_pairs(), skills, a.body_chars))
    print(f"train {len(train)} val {len(val)}", file=sys.stderr)

    model = CrossEncoder(a.base, num_labels=1, max_length=a.max_length)
    loss = BinaryCrossEntropyLoss(model, pos_weight=torch.tensor(a.pos_weight))
    out = ROOT / a.out
    args = CrossEncoderTrainingArguments(
        output_dir=str(out.parent / f"{out.name}-runs"),
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.bs,
        per_device_eval_batch_size=64,
        learning_rate=a.lr,
        warmup_ratio=0.05,
        bf16=True,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=100,
        seed=a.seed,
        report_to="none",
        dataloader_num_workers=0,
    )
    trainer = CrossEncoderTrainer(model=model, args=args, train_dataset=train, eval_dataset=val, loss=loss)
    trainer.train()
    model.save_pretrained(str(out))
    meta = {"base": a.base, "template_version": 1, "body_chars": a.body_chars, "task": "skill-relevance gate", "args": vars(a),
            "log": [h for h in trainer.state.log_history if "eval_loss" in h]}
    (Path(out) / "skill_issue_gate.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
