"""Neural retrieval baselines: a sentence-embedding bi-encoder and a cross-encoder reranker.

These are evaluation components. They are compared against BM25 on an identical corpus,
candidate scope and question set; the default application path stays lexical unless the
measured results justify a change. Requires ``sentence-transformers`` (optional extra).
"""
from __future__ import annotations
import hashlib
from pathlib import Path

import numpy as np

from .corpus import ROOT, Chunk

CACHE = ROOT / 'runtime/embeddings'


class DenseIndex:
    """Cosine-similarity search over normalised sentence embeddings (exact, no ANN)."""

    def __init__(self, chunks: list[Chunk], model_name: str, revision: str | None = None,
                 query_prefix: str = '', batch_size: int = 64, cache: bool = True):
        from sentence_transformers import SentenceTransformer
        self.chunks, self.model_name, self.query_prefix = chunks, model_name, query_prefix
        self.model = SentenceTransformer(model_name, revision=revision, device='cpu')
        self.doc_indices: dict[str, list[int]] = {}
        for i, c in enumerate(chunks):
            self.doc_indices.setdefault(c.document, []).append(i)
        key = hashlib.sha256(('\n'.join(c.id + '\t' + c.text for c in chunks) + model_name + str(revision)).encode()).hexdigest()[:24]
        path = CACHE / f'{key}.npy'
        if cache and path.exists():
            self.vectors = np.load(path)
        else:
            self.vectors = self.model.encode([c.text for c in chunks], batch_size=batch_size,
                                             normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
            if cache:
                CACHE.mkdir(parents=True, exist_ok=True)
                np.save(path, self.vectors)

    def scores(self, query: str) -> np.ndarray:
        q = self.model.encode([self.query_prefix + query], normalize_embeddings=True, show_progress_bar=False)[0]
        return self.vectors @ q.astype(np.float32)

    def rank(self, query: str, k: int, document: str | None = None) -> list[int]:
        scores = self.scores(query)
        eligible = np.array(self.doc_indices.get(document, []) if document else range(len(self.chunks)), dtype=int)
        if not len(eligible):
            return []
        return [int(i) for i in eligible[np.argsort(-scores[eligible], kind='stable')][:k]]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int, constant: int = 60) -> list[int]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1 / (constant + rank + 1)
    return sorted(fused, key=lambda i: (-fused[i], i))[:k]


class CrossEncoderReranker:
    """Scores (question, passage) pairs jointly; reranks a candidate list from another retriever."""

    def __init__(self, model_name: str, revision: str | None = None):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, revision=revision, device='cpu', max_length=256)

    def rerank(self, query: str, candidates: list[int], chunks: list[Chunk], k: int) -> list[int]:
        if not candidates:
            return []
        scores = self.model.predict([(query, chunks[i].text) for i in candidates], batch_size=64,
                                    show_progress_bar=False)
        order = sorted(range(len(candidates)), key=lambda j: (-float(scores[j]), candidates[j]))
        return [candidates[j] for j in order[:k]]
