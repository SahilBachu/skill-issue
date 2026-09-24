"""Fine-tune the Laya decision model as a skill-relevance gate on a 6 GB laptop GPU.

    python -m training.finetune_laya --out checkpoints/laya-ft

Memory plan (RTX 4050 Laptop, 6 GB): fp32 master weights, bf16 autocast, gradient
checkpointing in the encoder, 8-bit AdamW (bitsandbytes), and the token embeddings plus the
lowest encoder layers frozen. The task framing is exactly what inference uses
(skillissue.gates.laya_gate.encode_rows): one two-option choice question per (request, skill).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("USE_TF", "0")

from bench.corpus import CACHE, ROOT
from skillissue import models
from skillissue.gates.base import gate_text
from skillissue.gates.laya_gate import encode_rows
from skillissue.skill import Skill

PAIRS = CACHE / "train" / "pairs.jsonl"
SKILLS = CACHE / "train" / "skills.json"


def load_pairs(max_neg_ratio: float, seed: int) -> list[dict[str, Any]]:
    rows = [json.loads(x) for x in PAIRS.open(encoding="utf-8")]
    pos = [r for r in rows if r["label"]]
    neg = [r for r in rows if not r["label"]]
    rng = random.Random(seed)
    rng.shuffle(neg)
    neg = neg[: int(len(pos) * max_neg_ratio)]
    out = pos + neg
    rng.shuffle(out)
    return out


def val_pairs() -> list[dict[str, Any]]:
    """Validation pairs: our val split, hybrid retrieval at catalog size 100, top 12."""
    from bench.data import load_split
    from bench.run import retrieval_runs

    val = load_split("val")
    runs = retrieval_runs("hybrid", "val", 100, val)
    out = []
    for p in val:
        for c in runs[p["id"]]["cands"][:12]:
            s = c.skill
            out.append({"prompt": p["prompt"], "skill_obj": s, "label": int(s.id in p["labels"])})
    return out


def tokenize(tok: Any, pairs: list[dict[str, Any]], skills: dict[str, Any], body_chars: int) -> list[dict[str, Any]]:
    items = []
    for p in pairs:
        if "skill_obj" in p:
            s = p["skill_obj"]
        else:
            d = skills[p["skill"]]
            s = Skill(id=d["id"], name=d["name"], description=d["description"], body=d.get("body", ""))
        row = encode_rows(tok, p["prompt"], [gate_text(s, body_chars)])[0][0]
        row["label"] = 0 if p["label"] else 1  # option 0 = "use", option 1 = "skip"
        items.append(row)
    return items


def batches(
    items: list[dict[str, Any]], bs: int, rng: random.Random, shuffle: bool = True
) -> list[list[dict[str, Any]]]:
    """Length-bucketed batches: sort within shuffled windows so padding stays small."""
    idx = list(range(len(items)))
    if shuffle:
        rng.shuffle(idx)
    window = bs * 50
    out = []
    for w in range(0, len(idx), window):
        chunk = sorted(idx[w : w + window], key=lambda i: len(items[i]["ids"]))
        out += [[items[i] for i in chunk[j : j + bs]] for j in range(0, len(chunk), bs)]
    if shuffle:
        rng.shuffle(out)
    return out


def collate(batch: list[dict[str, Any]], pad_id: int, dev: torch.device) -> dict[str, torch.Tensor]:
    from laya.common import collate_items

    b = collate_items([[it] for it in batch], pad_id)
    return {k: v.to(dev) for k, v in b.items() if isinstance(v, torch.Tensor)}


@torch.no_grad()
def evaluate(model: Any, items: list[dict[str, Any]], pad_id: int, dev: torch.device, bs: int = 32) -> dict[str, float]:
    model.eval()
    raws, ys = [], []
    for b in batches(items, bs, random.Random(0), shuffle=False):
        t = collate(b, pad_id, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(t["input_ids"], t["attention_mask"], t["marker_pos"], t["marker_mask"], t["qtype"])
        lg = logits.float()
        raws += (lg[:, 0] - lg[:, 1]).cpu().tolist()
        ys += [1 - it["label"] for it in b]
    model.train()
    raw, y = np.array(raws), np.array(ys)
    from sklearn.metrics import average_precision_score, roc_auc_score

    p = 1 / (1 + np.exp(-np.clip(raw, -30, 30)))
    ll = float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))
    return {"auc": float(roc_auc_score(y, raw)), "ap": float(average_precision_score(y, raw)), "logloss": ll}


def save(model: Any, base_dir: Path, out: Path, meta: dict[str, Any]) -> None:
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    for sub in ("encoder", "tokenizer"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
        shutil.copytree(base_dir / sub, out / sub)
    shutil.copy2(base_dir / "rl_agent_config.json", out / "rl_agent_config.json")
    sd = {
        k: (v.detach().to(torch.float16) if v.is_floating_point() and k != "temperature" else v.detach())
        .contiguous()
        .cpu()
        for k, v in model.state_dict().items()
    }
    save_file(sd, str(out / "model.safetensors"))
    (out / "skill_issue_gate.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="checkpoints/laya-ft")
    ap.add_argument("--base", default="convaiinnovations/laya")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--neg-ratio", type=float, default=8.0)
    ap.add_argument("--pos-weight", type=float, default=2.0)
    ap.add_argument("--freeze-layers", type=int, default=8)
    ap.add_argument("--body-chars", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=400)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max-steps", type=int, default=0)
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    import bitsandbytes as bnb
    import laya

    base_dir = models.ensure_model(a.base, allow_patterns=models.LAYA_FILES)
    agent = laya.load(str(base_dir), device="cuda")
    model, tok = agent.model.float(), agent.tok
    dev = torch.device("cuda")
    enc = model.encoder
    enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    for p in enc.embeddings.parameters():
        p.requires_grad = False
    for layer in enc.layers[: a.freeze_layers]:
        for p in layer.parameters():
            p.requires_grad = False
    enc_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    n_train = sum(p.numel() for p in enc_params + head_params)
    print(
        f"trainable params: {n_train / 1e6:.1f}M of {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M",
        file=sys.stderr,
    )
    opt = bnb.optim.AdamW8bit(
        [
            {"params": enc_params, "lr": a.lr, "weight_decay": 0.01},
            {"params": head_params, "lr": a.head_lr, "weight_decay": 0.01},
        ]
    )

    skills = json.loads(SKILLS.read_text(encoding="utf-8"))
    t0 = time.time()
    train_items = tokenize(tok, load_pairs(a.neg_ratio, a.seed), skills, a.body_chars)
    val_items = tokenize(tok, val_pairs(), skills, a.body_chars)
    print(
        f"train {len(train_items)} rows, val {len(val_items)} rows, tokenized in {time.time() - t0:.0f}s; "
        f"mean len {np.mean([len(i['ids']) for i in train_items]):.0f}",
        file=sys.stderr,
    )

    accum = a.bs // a.micro
    steps_per_epoch = math.ceil(len(train_items) / a.bs)
    total = int(steps_per_epoch * a.epochs) if not a.max_steps else a.max_steps
    warm = max(1, int(total * 0.05))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm))
    )
    cw = torch.tensor([a.pos_weight, 1.0], device=dev)
    base_eval = evaluate(model, val_items, tok.pad_token_id, dev)
    print(f"step 0 val {base_eval}", file=sys.stderr)
    best = base_eval["ap"]
    log = [{"step": 0, **base_eval}]
    out = ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
    meta = {
        "base": a.base,
        "template_version": 1,
        "body_chars": a.body_chars,
        "task": "skill-relevance gate",
        "train_rows": len(train_items),
        "args": vars(a),
    }
    step = 0
    mb_i = 0
    model.train()
    t0 = time.time()
    done = False
    while not done:
        for mb in batches(train_items, a.micro, rng):
            t = collate(mb, tok.pad_token_id, dev)
            labels = torch.tensor([it["label"] for it in mb], device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(t["input_ids"], t["attention_mask"], t["marker_pos"], t["marker_mask"], t["qtype"])
            loss = F.cross_entropy(logits.float()[:, :2], labels, weight=cw) / accum
            loss.backward()
            mb_i += 1
            if mb_i % accum == 0:
                torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0:
                    el = time.time() - t0
                    print(
                        f"step {step}/{total} loss {loss.item() * accum:.4f} {el / step:.2f}s/step "
                        f"mem {torch.cuda.max_memory_allocated() / 1e9:.2f}GB",
                        file=sys.stderr,
                        flush=True,
                    )
                if step % a.eval_every == 0 or step == total:
                    ev = evaluate(model, val_items, tok.pad_token_id, dev)
                    log.append({"step": step, **ev})
                    print(f"step {step} val {ev}", file=sys.stderr, flush=True)
                    if ev["ap"] > best:
                        best = ev["ap"]
                        meta.update({"best_step": step, "val": ev})
                        save(model, base_dir, out, meta)
                        print(f"  saved (best AP {best:.4f})", file=sys.stderr, flush=True)
                if step >= total:
                    done = True
                    break
    (out.parent / f"{out.name}-trainlog.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"done in {(time.time() - t0) / 60:.1f} min, best val AP {best:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
