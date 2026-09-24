"""Fine-tune the Laya decision model as a skill-relevance gate on a 6 GB laptop GPU.

    python -m training.finetune_laya --out checkpoints/laya-ft

Memory plan (RTX 4050 Laptop, 6 GB): fp32 master weights, bf16 autocast (fp16 + loss scaling
on GPUs without bf16, such as a Colab T4), gradient
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
# bf16 where the GPU supports it (Ampere+), else fp16 with loss scaling (for example a Colab T4).
AMP = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
SKILLS = CACHE / "train" / "skills.json"


def load_pairs(max_neg_ratio: float, seed: int, path: Path = PAIRS) -> list[dict[str, Any]]:
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    pos = [r for r in rows if r["label"]]
    neg = [r for r in rows if not r["label"]]
    rng = random.Random(seed)
    rng.shuffle(neg)
    neg = neg[: int(len(pos) * max_neg_ratio)]
    out = pos + neg
    rng.shuffle(out)
    return out


def load_val(path: Path | None) -> list[dict[str, Any]]:
    """Validation pairs from a file (as written by cloud/prepare.py) or computed locally."""
    if path is None:
        return val_pairs()
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out.append({"prompt": r["prompt"], "skill_obj": Skill(**r["skill"]), "label": r["label"]})
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
        with torch.autocast("cuda", dtype=AMP):
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
    ap.add_argument("--pairs", default=str(PAIRS), help="training pairs jsonl")
    ap.add_argument("--skills", default=str(SKILLS), help="skill texts json")
    ap.add_argument("--val", default=None, help="validation pairs jsonl (default: computed from the val split)")
    ap.add_argument("--state", default=None, help="resume file; written at every evaluation, read on start")
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

    skills = json.loads(Path(a.skills).read_text(encoding="utf-8"))
    t0 = time.time()
    train_items = tokenize(tok, load_pairs(a.neg_ratio, a.seed, Path(a.pairs)), skills, a.body_chars)
    val_items = tokenize(tok, load_val(Path(a.val) if a.val else None), skills, a.body_chars)
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
    scaler = torch.amp.GradScaler("cuda", enabled=AMP is torch.float16)
    print(f"mixed precision: {AMP}", file=sys.stderr)
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    state_path = Path(a.state) if a.state else None
    step = 0
    mb_i = 0
    if state_path and state_path.is_file():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        with torch.no_grad():
            for n, v in st["trainable"].items():
                trainable[n].copy_(v.to(dev))
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        step, mb_i, best, log = st["step"], st["mb_i"], st["best"], st["log"]
        print(f"resumed from {state_path} at step {step}", file=sys.stderr, flush=True)
    else:
        base_eval = evaluate(model, val_items, tok.pad_token_id, dev)
        print(f"step 0 val {base_eval}", file=sys.stderr)
        best = base_eval["ap"]
        log = [{"step": 0, **base_eval}]

    def save_state() -> None:
        if state_path is None:
            return
        tmp = state_path.with_suffix(".tmp")
        torch.save(
            {
                "trainable": {n: p.detach().cpu() for n, p in trainable.items()},
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "mb_i": mb_i,
                "best": best,
                "log": log,
            },
            tmp,
        )
        tmp.replace(state_path)

    out = ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
    meta = {
        "base": a.base,
        "template_version": 1,
        "body_chars": a.body_chars,
        "task": "skill-relevance gate",
        "train_rows": len(train_items),
        "args": vars(a),
    }
    model.train()
    t0 = time.time()
    t_step0 = step
    done = step >= total
    seen = 0  # micro-batches replayed; the batch order is deterministic, so resume skips mb_i of them
    while not done:
        for mb in batches(train_items, a.micro, rng):
            seen += 1
            if seen <= mb_i:
                continue
            t = collate(mb, tok.pad_token_id, dev)
            labels = torch.tensor([it["label"] for it in mb], device=dev)
            with torch.autocast("cuda", dtype=AMP):
                logits, _ = model(t["input_ids"], t["attention_mask"], t["marker_pos"], t["marker_mask"], t["qtype"])
            loss = F.cross_entropy(logits.float()[:, :2], labels, weight=cw) / accum
            scaler.scale(loss).backward()
            mb_i += 1
            if mb_i % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0:
                    el = time.time() - t0
                    print(
                        f"step {step}/{total} loss {loss.item() * accum:.4f} {el / max(1, step - t_step0):.2f}s/step "
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
                    save_state()
                if step >= total:
                    done = True
                    break
    if not (out / "model.safetensors").is_file():
        # Fine-tuning never beat the base model on validation: ship the final weights, and say so.
        meta.update({"best_step": step, "note": "no evaluation beat the base model; final weights saved"})
        save(model, base_dir, out, meta)
    (out.parent / f"{out.name}-trainlog.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"done in {(time.time() - t0) / 60:.1f} min, best val AP {best:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
