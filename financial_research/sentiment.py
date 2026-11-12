"""Financial sentiment model: pinned DistilBERT + locally trained LoRA adapter."""
from __future__ import annotations
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
import numpy as np
from scipy.special import softmax
from .corpus import ROOT, DATA

os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')

LABELS = ['negative', 'neutral', 'positive']
BASE = ROOT / 'runtime/base-model'
ADAPTER = DATA / 'sentiment-adapter'
load_lock = threading.RLock()
_model = None
_tokenizer = None


def load_phrasebank():
    meta = json.loads((DATA / 'phrasebank_upstream.json').read_text())
    raw = (DATA / 'phrasebank-allagree.txt').read_bytes()
    if hashlib.sha256(raw).hexdigest() != meta['text_sha256']:
        raise ValueError('Financial PhraseBank snapshot hash mismatch')
    records = []
    seen = {}
    for line in raw.decode('latin-1').splitlines():
        text, label = line.rsplit('@', 1)
        canonical = ' '.join(re.findall(r'[a-z0-9]+', text.lower()))
        if canonical in seen:
            if seen[canonical] != label:
                raise ValueError('Conflicting duplicate labels in dataset')
            continue
        seen[canonical] = label
        records.append({'id': hashlib.sha256(canonical.encode()).hexdigest()[:20], 'text': text.strip(), 'label': LABELS.index(label.strip())})
    return records


def make_splits(records):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import StratifiedGroupKFold
    # Form connected components of normalized character-ngram similarity >= .85.
    # Replace numbers so trivial numeric changes do not straddle partitions.
    normalized = [re.sub(r'\d+(?:[.,]\d+)*', '#', r['text'].lower()) for r in records]
    vectors = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=1).fit_transform(normalized)
    parents = list(range(len(records)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    similarities = (vectors @ vectors.T).tocoo()
    for i, j, score in zip(similarities.row, similarities.col, similarities.data):
        if i < j and score >= .85:
            parents[find(int(j))] = find(int(i))
    groups = [find(i) for i in range(len(records))]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = np.zeros(len(records), dtype=int)
    y = [r['label'] for r in records]
    for fold, (_, test) in enumerate(splitter.split(np.zeros(len(records)), y, groups)):
        folds[test] = fold
    result = []
    for i, r in enumerate(records):
        result.append({**r, 'group': groups[i], 'split': 'test' if folds[i] == 0 else 'validation' if folds[i] == 1 else 'train'})
    return result


def ensure_base():
    import httpx
    meta = json.loads((DATA / 'base_model_upstream.json').read_text())
    BASE.mkdir(parents=True, exist_ok=True)
    names = ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt', 'LICENSE', 'README.md']
    for name in names:
        target = BASE / name
        if target.exists() and target.stat().st_size > 0:
            continue
        url = f"https://huggingface.co/{meta['repo']}/resolve/{meta['revision']}/{name}"
        partial = target.with_suffix(target.suffix + '.part')
        # Pinned revision, no custom remote code or pickle weights.
        with httpx.stream('GET', url, timeout=180, follow_redirects=True) as response:
            response.raise_for_status()
            with partial.open('wb') as out:
                for block in response.iter_bytes(1024 * 1024):
                    out.write(block)
        partial.replace(target)
    return BASE


def load_model():
    global _model, _tokenizer
    with load_lock:
        if _model is None:
            if not (ADAPTER / 'adapter_config.json').exists():
                raise RuntimeError('Trained sentiment adapter missing. Run scripts/train_sentiment.py.')
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            from peft import PeftModel
            torch.set_num_threads(min(4, os.cpu_count() or 1))
            base = ensure_base()
            _tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
            transformer = AutoModelForSequenceClassification.from_pretrained(base, num_labels=3,
                            id2label=dict(enumerate(LABELS)), label2id={s: i for i, s in enumerate(LABELS)}, local_files_only=True)
            _model = PeftModel.from_pretrained(transformer, ADAPTER, local_files_only=True).eval()
        return _model, _tokenizer


def predict_transformer(texts):
    import torch
    model, tokenizer = load_model()
    outputs = []
    with load_lock, torch.inference_mode():
        for start in range(0, len(texts), 16):
            batch = tokenizer(texts[start:start + 16], padding=True, truncation=True, max_length=96, return_tensors='pt')
            outputs.extend(model(**batch).logits.float().numpy().tolist())
    temperature_path = ADAPTER / 'calibration.json'
    temperature = json.loads(temperature_path.read_text())['temperature'] if temperature_path.exists() else 1.
    return softmax(np.array(outputs) / temperature, axis=1)


def predict_baseline(texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    # Fit only on the frozen training partition; no unsafe pickled model artifacts.
    records = json.loads((DATA / 'sentiment-splits.json').read_text())
    train = [r for r in records if r['split'] == 'train']
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    x = vectorizer.fit_transform(r['text'] for r in train)
    model = LogisticRegression(C=1, max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(x, [r['label'] for r in train])
    return model.predict_proba(vectorizer.transform(texts))


def predict(texts, model='transformer'):
    start = time.perf_counter()
    if model not in ('transformer', 'tfidf'):
        raise ValueError('Unknown sentiment model')
    probs = predict_transformer(texts) if model == 'transformer' else predict_baseline(texts)
    return {'model': 'DistilBERT + trained LoRA adapter' if model == 'transformer' else 'TF-IDF + logistic regression',
            'results': [{'text': t, 'label': LABELS[int(p.argmax())], 'probabilities': {label: round(float(p[i]), 5) for i, label in enumerate(LABELS)}} for t, p in zip(texts, probs)],
            'elapsed_ms': round((time.perf_counter() - start) * 1000, 2),
            'notice': 'Financial sentence sentiment, not an investment recommendation or return forecast. The transformer reads at most 96 tokens; longer text is truncated. Trained on news sentences; filing passages are a different domain.'}
