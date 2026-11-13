"""Train the learned numerical-QA proposer on DEVELOPMENT data only.

Fit pool: FinQA train.json minus every report in the frozen blind set and minus the
reports of this repository's dev validation/test splits, plus the dev train split.
Model selection and the abstention threshold use the dev validation split only.
The blind set is never opened here beyond its file names (for the exclusion).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from financial_research.blind import TRAIN_URL, corpus_from_rows, development_pool, numeric_correct, report_of
from financial_research.corpus import load_finqa
from financial_research.numeric_qa import (MAX_CANDIDATES, MODEL_PATH, PASSAGES, NumericQA, canonical_template,
                                           clean_number, extract_candidates, pair_features, slots_in,
                                           unary_features)
from financial_research.retrieval import Engine

MIN_TEMPLATE_COUNT = 12
SEED = 20261113


def fetch_train(path: Path) -> Path:
    if not path.exists():
        import httpx
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(httpx.get(TRAIN_URL, timeout=300, follow_redirects=True).raise_for_status().content)
    return path


def gold_matches(candidates, slot, gold_chunks):
    value = clean_number(slot)[1]
    hits = [i for i, c in enumerate(candidates) if c['value'] == value]
    preferred = [i for i in hits if candidates[i]['chunk'] in gold_chunks or candidates[i]['chunk'] == 'question']
    return preferred or hits


def prepare(engine, questions):
    rows = []
    for q in questions:
        hits = engine.search(q['question'], 'bm25', PASSAGES, q['document'])
        cands = extract_candidates(q['question'], hits)
        parsed = canonical_template(q['program'])
        rows.append({'q': q, 'cands': cands, 'template': parsed[0] if parsed else None,
                     'slots': parsed[1] if parsed else [],
                     'matches': [gold_matches(cands, s, set(q['gold'])) for s in parsed[1]] if parsed else []})
    return rows


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    import joblib

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=str(ROOT / 'runtime/finqa-train.json'))
    args = parser.parse_args()
    started = time.time()
    fit_rows, validation_rows, accounting = development_pool(fetch_train(Path(args.train)))
    chunks, questions = corpus_from_rows(fit_rows)
    print(f'fit pool: {len(questions)} questions, {len(chunks)} chunks', flush=True)
    rows = prepare(Engine(chunks), questions)

    counts = Counter(r['template'] for r in rows if r['template'] and len(r['slots']) <= 2)
    templates = sorted(t for t, n in counts.items() if n >= MIN_TEMPLATE_COUNT and 1 <= slots_in(t) <= 2)
    label = lambda r: r['template'] if r['template'] in templates else 'OTHER'
    print(f'{len(templates)} supported templates cover {sum(label(r) != "OTHER" for r in rows) / len(rows):.3f}', flush=True)

    template_model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, min_df=2),
        LogisticRegression(C=8, max_iter=3000))
    template_model.fit([r['q']['question'] for r in rows], [label(r) for r in rows])

    unit_model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
                               LogisticRegression(C=4, max_iter=2000))
    unit_model.fit([f"{r['q']['question']} || {label(r)}" for r in rows], [int('%' in r['q']['answer']) for r in rows])

    # --- operand scorer (cross-fitted by report so the pair model sees honest scores)
    ux, uy, ugroup, urow = [], [], [], []
    for n, r in enumerate(rows):
        positive = {i for m in r['matches'] for i in m}
        if not r['template'] or not positive:
            continue
        for i, c in enumerate(r['cands']):
            ux.append(unary_features(r['q']['question'], c)); uy.append(int(i in positive))
            ugroup.append(report_of(r['q']['document'])); urow.append((n, i))
    ux, uy = np.array(ux), np.array(uy)
    make_unary = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_leaf_nodes=31,
                                                        l2_regularization=1.0, random_state=SEED)
    crossfit = np.zeros(len(uy))
    for train_idx, test_idx in GroupKFold(5).split(ux, uy, ugroup):
        crossfit[test_idx] = make_unary().fit(ux[train_idx], uy[train_idx]).predict_proba(ux[test_idx])[:, 1]
    unary_model = make_unary().fit(ux, uy)
    score_of = {key: s for key, s in zip(urow, crossfit)}
    print(f'unary examples {len(uy)} positives {int(uy.sum())}', flush=True)

    # --- ordered-pair scorer
    px, py = [], []
    rng = np.random.default_rng(SEED)
    for n, r in enumerate(rows):
        if r['template'] not in templates or slots_in(r['template']) != 2 or not all(r['matches']):
            continue
        scored = sorted(((score_of.get((n, i), 0.0), i) for i in range(len(r['cands']))), reverse=True)[:MAX_CANDIDATES]
        pool = [(r['cands'][i], s, i) for s, i in scored]
        first, second = set(r['matches'][0]), set(r['matches'][1])
        tid = templates.index(r['template'])
        for a, sa, ia in pool:
            for b, sb, ib in pool:
                if ia == ib:
                    continue
                positive = ia in first and ib in second
                if positive or rng.random() < .35:
                    px.append(pair_features(r['q']['question'], a, b, sa, sb, tid)); py.append(int(positive))
    pair_model = HistGradientBoostingClassifier(max_iter=400, learning_rate=.06, max_leaf_nodes=31,
                                                l2_regularization=1.0, random_state=SEED,
                                                categorical_features=[4]).fit(np.array(px), np.array(py))
    print(f'pair examples {len(py)} positives {int(sum(py))}', flush=True)

    bundle = {'templates': templates, 'template_model': template_model, 'unit_model': unit_model,
              'unary_model': unary_model, 'pair_model': pair_model, 'threshold': 0.0}
    qa = NumericQA(bundle)

    # --- validation: dev validation split, real retrieval, both scopes
    dev_chunks, _ = load_finqa()
    dev_engine = Engine(dev_chunks)
    _, val_questions = corpus_from_rows(validation_rows)
    outcomes = []
    for scope in ('page', 'collection'):
        for q in val_questions:
            hits = dev_engine.search(q['question'], 'bm25', PASSAGES, q['document'] if scope == 'page' else None)
            out = qa.answer(q['question'], hits, threshold=0.0)
            answered = out['status'] == 'answered'
            outcomes.append({'scope': scope, 'answered': answered, 'confidence': out.get('confidence', 0.0),
                             'correct': answered and numeric_correct(out['value'], q['exe_ans'])})
    answered = [o for o in outcomes if o['answered']]
    best, threshold = -10**9, 1.0
    for t in sorted({round(o['confidence'], 3) for o in answered}):
        gain = sum((1 if o['correct'] else -1) for o in answered if o['confidence'] >= t)
        if gain > best:
            best, threshold = gain, t
    bundle['threshold'] = float(threshold)
    summary = {}
    for scope in ('page', 'collection'):
        sub = [o for o in outcomes if o['scope'] == scope]
        kept = [o for o in sub if o['answered'] and o['confidence'] >= threshold]
        summary[scope] = {'questions': len(sub),
                          'accuracy_without_abstention': round(sum(o['correct'] for o in sub) / len(sub), 4),
                          'answered': len(kept), 'correct': sum(o['correct'] for o in kept),
                          'answered_precision': round(sum(o['correct'] for o in kept) / max(1, len(kept)), 4)}
    card = {'trained_at_unix': int(started), 'seed': SEED, 'accounting': accounting, 'templates': templates,
            'template_coverage_of_fit_pool': round(sum(label(r) != 'OTHER' for r in rows) / len(rows), 4),
            'unary_examples': int(len(uy)), 'pair_examples': int(len(py)),
            'threshold': bundle['threshold'], 'threshold_rule': 'maximise (correct - wrong) on dev validation, both scopes pooled',
            'validation': summary, 'passages': PASSAGES, 'max_candidates': MAX_CANDIDATES,
            'note': 'Validation numbers are development figures used for selection; they are not the blind result.'}
    bundle['card'] = card
    joblib.dump(bundle, MODEL_PATH, compress=3)
    card['model_sha256'] = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    (ROOT / 'reports/numeric-qa-training.json').write_text(json.dumps(card, indent=2) + '\n')
    print(json.dumps({k: card[k] for k in ('accounting', 'template_coverage_of_fit_pool', 'threshold', 'validation')}, indent=2))
    print(f'{time.time() - started:.0f}s')


if __name__ == '__main__':
    main()
