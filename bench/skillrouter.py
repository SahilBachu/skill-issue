"""SkillRouter baseline (zhengyanzhao1997/SkillRouter, models pipizhao/SkillRouter-*-0.6B, Apache-2.0).

Implemented from the model cards: SR-Emb-0.6B retrieves (last-token pooling, instruction-prefixed
query, documents formatted `name | description | body`), SR-Rank-0.6B reranks with the Qwen
reranker yes/no template, score = logit(yes) - logit(no).

One deviation, for a 6 GB GPU: documents are truncated to 1,024 tokens for the embedder
(the card allows up to 4,096). The reranker uses the card's own limits (description 500 chars,
body 2,000 chars). Their pipeline has no abstention step; we give it the same calibration and
threshold treatment as every other system (fit on the validation split)."""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence

import numpy as np
import torch

from bench.corpus import CACHE
from skillissue.retrieval import Candidate
from skillissue.skill import Skill

EMB_ID = "pipizhao/SkillRouter-Embedding-0.6B"
RANK_ID = "pipizhao/SkillRouter-Reranker-0.6B"
QUERY_INSTRUCTION = "Instruct: Given a task description, retrieve the most relevant skill document that would help an agent complete the task\nQuery:"
DOC_MAX_TOKENS = 1024
EMB_CACHE = CACHE / "pool_sr_emb.npy"
EMB_IDS = CACHE / "pool_sr_emb_ids.json"


def _last_token_pool(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if bool((mask[:, -1].sum() == mask.shape[0]).item()):
        return h[:, -1]
    lens = mask.sum(dim=1) - 1
    return h[torch.arange(h.shape[0], device=h.device), lens]


class SkillRouterEmbedder:
    name = EMB_ID

    def __init__(self, device: str = "cuda"):
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(EMB_ID, padding_side="left")
        self.model = AutoModel.from_pretrained(EMB_ID, dtype=torch.bfloat16).eval().to(device)
        self.device = device

    @torch.no_grad()
    def _encode(self, texts: list[str], max_length: int, bs: int) -> np.ndarray:
        out = []
        order = np.argsort([-len(t) for t in texts])  # long first: stable memory peak
        for i in range(0, len(texts), bs):
            batch = [texts[j] for j in order[i : i + bs]]
            enc = self.tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(
                self.device
            )
            h = self.model(**enc).last_hidden_state
            e = torch.nn.functional.normalize(_last_token_pool(h, enc["attention_mask"]).float(), p=2, dim=1)
            out.append(e.cpu().numpy())
        vecs = np.concatenate(out)
        res = np.empty_like(vecs)
        res[order] = vecs
        return res

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, DOC_MAX_TOKENS, 8)

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode([QUERY_INSTRUCTION + text], 512, 1)[0]

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([QUERY_INSTRUCTION + t for t in texts], 512, 32)


def doc_text(s: Skill) -> str:
    return f"{s.name} | {s.description} | {s.body}"


def pool_vectors(skills: list[Skill], ids: list[str]) -> np.ndarray:
    if EMB_CACHE.is_file() and EMB_IDS.is_file() and json.loads(EMB_IDS.read_text()) == ids:
        return np.load(EMB_CACHE)
    emb = SkillRouterEmbedder()
    vecs = emb.encode_docs([doc_text(s) for s in skills])
    np.save(EMB_CACHE, vecs)
    EMB_IDS.write_text(json.dumps(ids))
    del emb
    torch.cuda.empty_cache()
    return vecs


class SkillRouterReranker:
    """Gate-compatible scorer: raw_scores(prompt, candidates) -> logit(yes) - logit(no)."""

    name = "skillrouter"
    INSTRUCTION = (
        "Given a task description, judge whether the skill document is relevant and useful for completing the task"
    )
    PREFIX = (
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
        'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, device: str = "cuda", max_length: int = 2048):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(RANK_ID, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(RANK_ID, dtype=torch.bfloat16).eval().to(device)
        self.device = device
        self.max_length = max_length
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")
        self.pre = self.tok.encode(self.PREFIX, add_special_tokens=False)
        self.suf = self.tok.encode(self.SUFFIX, add_special_tokens=False)
        self._lock = threading.Lock()

    def _ids(self, query: str, s: Skill) -> list[int]:
        prompt = f"<Instruct>: {self.INSTRUCTION}\n\n<Query>: {query}\n\n<Document>: {s.name} | {s.description[:500]} | {s.body[:2000]}"
        body = self.tok(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - len(self.pre) - len(self.suf),
        )["input_ids"]
        return self.pre + body + self.suf

    @torch.no_grad()
    def raw_scores(self, prompt: str, candidates: Sequence[Candidate]) -> np.ndarray:
        seqs = [self._ids(prompt, c.skill) for c in candidates]
        out: list[float] = []
        with self._lock:
            for i in range(0, len(seqs), 4):
                chunk = seqs[i : i + 4]
                L = max(len(s) for s in chunk)
                pad = self.tok.pad_token_id
                ids = torch.tensor([[pad] * (L - len(s)) + s for s in chunk], device=self.device)
                att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in chunk], device=self.device)
                # Only the last position is needed; full logits (seq x 152k vocab) do not fit in 6 GB.
                logits = self.model(input_ids=ids, attention_mask=att, logits_to_keep=1).logits[:, -1, :].float()
                out += (logits[:, self.yes] - logits[:, self.no]).cpu().tolist()
        return np.array(out)
