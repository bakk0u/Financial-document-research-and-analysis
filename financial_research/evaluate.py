"""Reproducible retrieval evaluation. Gold answers are never generator inputs."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
from .corpus import ROOT, DATA, load_finqa
from .retrieval import Engine, METHODS, train_reranker


def metrics(predicted, gold, k=5):
    selected = predicted[:k]
    relevance = [int(x in gold) for x in selected]
    hit = sum(relevance)
    dcg = sum(v / np.log2(i + 2) for i, v in enumerate(relevance))
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(gold), k)))
    return {'recall_at_5': hit / len(gold), 'precision_at_5': hit / k,
            'mrr_at_5': next((1 / (i + 1) for i, v in enumerate(relevance) if v), 0),
            'ndcg_at_5': dcg / ideal if ideal else 0,
            'all_evidence_at_5': float(set(gold).issubset(selected))}


def evaluate(engine, queries, split, scope):
    records = []
    for method in METHODS:
        rows, latencies = [], []
        for q in queries:
            if q['split'] != split:
                continue
            start = time.perf_counter()
            found = engine.search(q['question'], method, 5, q['document'] if scope == 'page' else None)
            elapsed = (time.perf_counter() - start) * 1000
            m = metrics([x['id'] for x in found], q['gold'])
            rows.append(m)
            latencies.append(elapsed)
            records.append({'query_id': q['id'], 'method': method, 'scope': scope,
                            'question': q['question'], 'gold': q['gold'],
                            'predicted': [c['id'] for c in found], **m})
        aggregate = {key: round(float(np.mean([row[key] for row in rows])), 5) for key in rows[0]}
        aggregate.update({'method': method, 'scope': scope, 'queries': len(rows),
                          'latency_p50_ms': round(float(np.median(latencies)), 2),
                          'latency_p95_ms': round(float(np.percentile(latencies, 95)), 2)})
        yield aggregate, [r for r in records if r['method'] == method]


def main():
    chunks, queries = load_finqa()
    print(f'Indexing {len(chunks)} chunks from {len(set(c.document for c in chunks))} pages', flush=True)
    engine = Engine(chunks)
    engine.model = train_reranker(engine, queries)
    (DATA / 'reranker.json').write_text(json.dumps(engine.model, indent=2))
    (DATA / 'queries.json').write_text(json.dumps(queries, indent=2))
    summary, details = [], []
    for split in ('validation', 'test'):
        for scope in ('page', 'collection'):
            for row, records in evaluate(engine, queries, split, scope):
                row['split'] = split
                summary.append(row)
                details.extend({**r, 'split': split} for r in records)
                print(split, scope, row['method'], row['recall_at_5'], flush=True)
    result = {'corpus_fingerprint': engine.fingerprint, 'chunks': len(chunks),
              'pages': len(engine.doc_indices), 'reports': len(set('/'.join(c.document.split('/')[:2]) for c in chunks)),
              'queries': len(queries), 'split_counts': {s: sum(q['split'] == s for q in queries) for s in ('train', 'validation', 'test')},
              'model': engine.model, 'results': summary,
              'limitations': ['Custom repartition of the public FinQA dev set, not an official FinQA score.',
                             'Report-disjoint query splits; companies may appear in different years across splits.',
                             'Unsupervised LSA vocabulary and vectors are fitted on the searchable collection, including held-out documents.',
                             'Page scope assumes the relevant source page is supplied; collection scope does not.',
                             'Historical report excerpts, not full filings or current market information.',
                             'Retrieval metrics do not establish numerical answer accuracy or narrative faithfulness.',
                             'Supervised feature reranker; no LLM or neural embedding weights are fine-tuned.',
                             'No hyperparameter search performed; test results observed once for this frozen configuration.']}
    out = ROOT / 'reports'
    out.mkdir(exist_ok=True)
    (out / 'evaluation.json').write_text(json.dumps(result, indent=2))
    (out / 'predictions.json').write_text(json.dumps(details, indent=2))
    lines = ['# Financial Document Research & Analysis — retrieval evaluation', '', 'All numbers below are measured, not illustrative.', '',
             '| Split | Scope | Method | Recall@5 | MRR@5 | All evidence@5 | Median ms |',
             '|---|---|---|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['split']} | {r['scope']} | {r['method']} | {r['recall_at_5']:.3f} | {r['mrr_at_5']:.3f} | {r['all_evidence_at_5']:.3f} | {r['latency_p50_ms']:.1f} |")
    lines.extend(['', '## Interpretation', '', 'Compare reranked against hybrid in BOTH scopes. A loss is a finding, not a reason to hide a baseline.', '', '## Limitations', ''])
    lines.extend('- ' + x for x in result['limitations'])
    (out / 'EVALUATION.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
