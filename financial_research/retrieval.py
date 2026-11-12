from __future__ import annotations
import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from .corpus import Chunk, tokens

METHODS = ('bm25', 'vector', 'hybrid', 'reranked')
FEATURE_NAMES = ['bm25_saturation', 'lsa_cosine', 'query_coverage', 'exact_number_coverage',
                 'table_indicator', 'query_bigram_coverage', 'query_density']


def content_tokens(text):
    return [t for t in tokens(text) if t not in ENGLISH_STOP_WORDS]


class Engine:
    def __init__(self, chunks: list[Chunk], model: dict | None = None):
        if not chunks:
            raise ValueError('Cannot index an empty corpus')
        self.chunks = chunks
        self.model = model
        self.by_id = {c.id: i for i, c in enumerate(chunks)}
        self.doc_indices = {}
        for i, c in enumerate(chunks):
            self.doc_indices.setdefault(c.document, []).append(i)
        self.docs = [tokens(c.text) for c in chunks]
        self.counts = [Counter(d) for d in self.docs]
        self.lengths = np.array([len(d) for d in self.docs], dtype=float)
        self.avglen = max(1, float(self.lengths.mean()))
        df = Counter(t for d in self.counts for t in d)
        n = len(chunks)
        self.idf = {t: np.log(1 + (n - v + .5) / (v + .5)) for t, v in df.items()}
        self.postings = {}
        for i, counts in enumerate(self.counts):
            for term, freq in counts.items():
                self.postings.setdefault(term, []).append((i, freq))
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=24000,
                                    sublinear_tf=True, stop_words='english')
        matrix = self.tfidf.fit_transform(c.text for c in chunks)
        dims = min(64, matrix.shape[0] - 1, matrix.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=max(1, dims), random_state=42)
        self.vectors = normalize(self.svd.fit_transform(matrix)).astype(np.float32)
        self.fingerprint = hashlib.sha256(json.dumps([asdict(c) for c in chunks], sort_keys=True).encode()).hexdigest()

    def scores(self, query: str):
        bm25 = np.zeros(len(self.chunks))
        for term in set(content_tokens(query)):
            pairs = self.postings.get(term, [])
            if not pairs:
                continue
            idx, freq = np.array(pairs).T
            idx = idx.astype(int)
            den = freq + 1.5 * (1 - .75 + .75 * self.lengths[idx] / self.avglen)
            bm25[idx] += self.idf[term] * freq * 2.5 / den
        q = normalize(self.svd.transform(self.tfidf.transform([query])))[0]
        vector = self.vectors @ q
        return bm25, vector

    def features(self, query: str, idx: int, bm25: float, vector: float):
        q = set(content_tokens(query))
        d = set(self.docs[idx])
        nums = {t for t in q if re.fullmatch(r'\d+(?:\.\d+)?', t)}
        qt = content_tokens(query)
        dt = content_tokens(self.chunks[idx].text)
        bigrams = set(zip(qt, qt[1:]))
        db = set(zip(dt, dt[1:]))
        return [bm25 / (bm25 + 10), float(vector), len(q & d) / max(1, len(q)),
                len(nums & d) / max(1, len(nums)), float(self.chunks[idx].kind == 'table'),
                len(bigrams & db) / max(1, len(bigrams)), len(q & d) / max(1, len(d))]

    def search(self, query: str, method='reranked', k=5, document: str | None = None):
        if method not in METHODS:
            raise ValueError('Unknown retrieval method')
        bm, vec = self.scores(query)
        eligible = np.array(self.doc_indices.get(document, []) if document else range(len(self.chunks)), dtype=int)
        if not len(eligible):
            return []
        # Stable deterministic tie-breaking through corpus order.
        rb = eligible[np.argsort(-bm[eligible], kind='stable')]
        rv = eligible[np.argsort(-vec[eligible], kind='stable')]
        if method == 'bm25':
            ranked = [(int(i), float(bm[i])) for i in rb[:k] if bm[i] > 0]
        elif method == 'vector':
            ranked = [(int(i), float(vec[i])) for i in rv[:k] if vec[i] > 0]
        else:
            rrf = {}
            for ranking in (rb[:40], rv[:40]):
                for rank, idx in enumerate(ranking):
                    if bm[idx] <= 0 and vec[idx] <= 0:
                        continue
                    rrf[int(idx)] = rrf.get(int(idx), 0) + 1 / (60 + rank + 1)
            candidates = sorted(rrf, key=lambda i: (-rrf[i], i))[:40]
            if method == 'reranked' and self.model:
                coef = np.array(self.model['coef'])
                ranked = [(i, float(expit(np.dot(self.features(query, i, bm[i], vec[i]), coef) + self.model['intercept']))) for i in candidates]
                ranked.sort(key=lambda item: (-item[1], item[0]))
            else:
                ranked = [(i, rrf[i]) for i in candidates]
            ranked = ranked[:k]
        return [{**asdict(self.chunks[i]), 'score': round(score, 6),
                 'bm25': round(float(bm[i]), 6), 'vector': round(float(vec[i]), 6),
                 'coverage': self.features(query, i, bm[i], vec[i])[2]} for i, score in ranked]


def train_reranker(engine: Engine, queries: list[dict]) -> dict:
    xs, ys = [], []
    ids = []
    for q in queries:
        if q['split'] != 'train':
            continue
        ids.append(q['id'])
        bm, vec = engine.scores(q['question'])
        candidates = engine.search(q['question'], 'hybrid', 20, q['document'])
        candidate_ids = {c['id'] for c in candidates} | set(q['gold'])
        for ident in sorted(candidate_ids):
            idx = engine.by_id[ident]
            xs.append(engine.features(q['question'], idx, bm[idx], vec[idx]))
            ys.append(int(ident in q['gold']))
    classifier = LogisticRegression(C=1, class_weight='balanced', max_iter=1000, random_state=42)
    classifier.fit(xs, ys)
    return {'type': 'logistic_pointwise_relevance_classifier', 'feature_names': FEATURE_NAMES,
            'coef': classifier.coef_[0].tolist(), 'intercept': float(classifier.intercept_[0]),
            'training_examples': len(ys), 'positive_examples': sum(ys), 'training_queries': len(ids),
            'training_id_sha256': hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest(),
            'corpus_fingerprint': engine.fingerprint,
            'note': 'A supervised feature-based reranker, not a fine-tuned LLM or neural cross-encoder. Fit on train labels only.'}
