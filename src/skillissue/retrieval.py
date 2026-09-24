"""Stage 1: hybrid retrieval. BM25 over name/description/body plus a small embedding model,
fused with reciprocal rank fusion (RRF). Returns the top-k candidates with their features."""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .skill import Skill

log = logging.getLogger(__name__)

_SPLIT = re.compile(r"[-_/.:]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

try:
    import Stemmer

    _STEMMER: Any = Stemmer.Stemmer("english")
except Exception:  # pragma: no cover - optional
    _STEMMER = None


def normalize_text(text: str) -> str:
    """Split identifiers (pdf-processing, camelCase) so BM25 can match their parts."""
    text = _CAMEL.sub(" ", text)
    return _SPLIT.sub(" ", text)


def bm25_doc(skill: Skill, body_chars: int) -> str:
    """Field weighting by repetition: name x3, description x2, body excerpt x1."""
    name = normalize_text(skill.name)
    desc = normalize_text(skill.description)
    return f"{name} {name} {name} {desc} {desc} {normalize_text(skill.routing_text(body_chars))}"


def _tokenize(texts: list[str]) -> list[list[str]]:
    import bm25s

    out = bm25s.tokenize(texts, stopwords="en", stemmer=_STEMMER, return_ids=False, show_progress=False)
    return out  # type: ignore[return-value]


class BM25Index:
    def __init__(self, skills: Sequence[Skill], body_chars: int = 1500):
        import bm25s

        self.n = len(skills)
        self.retriever = bm25s.BM25()
        docs = [bm25_doc(s, body_chars) for s in skills]
        tokens = _tokenize(docs) if docs else []
        if self.n:
            self.retriever.index(tokens, show_progress=False)

    def scores(self, query: str) -> np.ndarray:
        if not self.n:
            return np.zeros(0, dtype=np.float32)
        q = _tokenize([normalize_text(query)])[0]
        if not q:
            return np.zeros(self.n, dtype=np.float32)
        return np.asarray(self.retriever.get_scores(q), dtype=np.float32)


class Embedder(Protocol):
    name: str

    def encode_docs(self, texts: list[str]) -> np.ndarray: ...
    def encode_query(self, text: str) -> np.ndarray: ...


# Query prefixes recommended by each model card.
_QUERY_PREFIX = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "Snowflake/snowflake-arctic-embed-s": "Represent this sentence for searching relevant passages: ",
    "Snowflake/snowflake-arctic-embed-xs": "Represent this sentence for searching relevant passages: ",
}


class STEmbedder:
    """sentence-transformers embedder with normalized outputs."""

    def __init__(self, model: str, device: str | None = None, local_path: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.name = model
        self.prefix = _QUERY_PREFIX.get(model, "")
        self.model = SentenceTransformer(local_path or model, device=device)
        self._lock = threading.Lock()

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            v = self.model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        with self._lock:
            v = self.model.encode([self.prefix + text], normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(v[0], dtype=np.float32)


@dataclass
class Candidate:
    skill: Skill
    rank: int
    rrf: float
    bm25: float
    cosine: float
    bm25_rank: int
    dense_rank: int
    prob: float | None = None
    gate_logit: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def embedding_cache_key(skill: Skill, model: str, body_chars: int) -> str:
    raw = f"{model}|{body_chars}|{skill.sha256 or ''}|{skill.name}|{skill.description}|{len(skill.body)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Doc vectors on disk, keyed by content, so re-indexing only embeds changed skills."""

    def __init__(self, path: Path | None):
        self.path = path
        self.vecs: dict[str, np.ndarray] = {}
        if path and path.is_file():
            try:
                data = np.load(path, allow_pickle=False)
                keys = data["keys"]
                mat = data["vecs"]
                self.vecs = {str(k): mat[i] for i, k in enumerate(keys)}
            except Exception as e:  # corrupt cache is not fatal
                log.warning("ignoring embedding cache %s: %s", path, e)

    def save(self, keep: set[str]) -> None:
        if not self.path:
            return
        keys = [k for k in self.vecs if k in keep]
        if not keys:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez(tmp, keys=np.array(keys), vecs=np.stack([self.vecs[k] for k in keys]))
        tmp.replace(self.path)


class HybridRetriever:
    """BM25 + dense retrieval with RRF. Either side can be disabled for baselines."""

    def __init__(
        self,
        skills: Sequence[Skill],
        embedder: Embedder | None,
        body_chars: int = 1500,
        rrf_k: int = 60,
        use_bm25: bool = True,
        cache: EmbeddingCache | None = None,
        doc_vecs: np.ndarray | None = None,
    ):
        self.skills = list(skills)
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.body_chars = body_chars
        self.bm25 = BM25Index(self.skills, body_chars) if use_bm25 else None
        self.doc_vecs: np.ndarray | None = doc_vecs
        if embedder is not None and self.skills and doc_vecs is None:
            self.doc_vecs = self._embed_docs(embedder, cache)

    def _embed_docs(self, embedder: Embedder, cache: EmbeddingCache | None) -> np.ndarray:
        cache = cache if cache is not None else EmbeddingCache(None)
        keys = [embedding_cache_key(s, embedder.name, self.body_chars) for s in self.skills]
        missing = [i for i, k in enumerate(keys) if k not in cache.vecs]
        if missing:
            vecs = embedder.encode_docs([self.skills[i].routing_text(self.body_chars) for i in missing])
            for i, v in zip(missing, vecs):
                cache.vecs[keys[i]] = v
            cache.save(set(keys))
        return np.stack([cache.vecs[k] for k in keys])

    def retrieve(self, query: str, top_k: int = 40, query_vec: np.ndarray | None = None) -> list[Candidate]:
        n = len(self.skills)
        if n == 0:
            return []
        bm = self.bm25.scores(query) if self.bm25 is not None else np.zeros(n, dtype=np.float32)
        if self.embedder is not None and self.doc_vecs is not None:
            qv = query_vec if query_vec is not None else self.embedder.encode_query(query)
            cos = self.doc_vecs @ qv
        else:
            cos = np.zeros(n, dtype=np.float32)
        # Ranks (0 = best). Ties broken by index for determinism.
        bm_rank = _ranks(bm) if self.bm25 is not None else np.full(n, n)
        de_rank = _ranks(cos) if self.embedder is not None else np.full(n, n)
        rrf = np.zeros(n, dtype=np.float64)
        if self.bm25 is not None:
            rrf += 1.0 / (self.rrf_k + 1 + bm_rank)
        if self.embedder is not None:
            rrf += 1.0 / (self.rrf_k + 1 + de_rank)
        order = np.lexsort((np.arange(n), -rrf))[:top_k]
        return [
            Candidate(
                skill=self.skills[i],
                rank=r,
                rrf=float(rrf[i]),
                bm25=float(bm[i]),
                cosine=float(cos[i]),
                bm25_rank=int(bm_rank[i]),
                dense_rank=int(de_rank[i]),
            )
            for r, i in enumerate(order)
        ]


def _ranks(scores: np.ndarray) -> np.ndarray:
    order = np.lexsort((np.arange(len(scores)), -scores))
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(len(scores))
    return ranks
