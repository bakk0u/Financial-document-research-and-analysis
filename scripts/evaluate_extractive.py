"""Measure actual retrieval -> extractive-answer behavior on a frozen FinQA split.

This deliberately NEVER uses FinQA answers or gold arithmetic programs as system inputs.
Gold passage IDs are consulted only after prediction, for the evaluation labels.
Metric is evidence selection, NOT numerical answer accuracy or semantic faithfulness.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from financial_research.answer import synthesize
from financial_research.corpus import DATA, load_finqa
from financial_research.retrieval import Engine


def check_one(query: dict, hits: list[dict], answer: dict) -> dict:
    """Measure passage selection without using labels to choose an answer."""
    gold = set(query['gold'])
    available = {h['id']: h['text'] for h in hits}
    claims = answer.get('claims', [])
    selected = [c['citation'] for c in claims]
    exact = all(
        isinstance(c.get('quote'), str)
        and c.get('statement') == c['quote']
        and c['citation'] in available
        and ' '.join(c['quote'].split()) in ' '.join(available[c['citation']].split())
        for c in claims
    )
    top = {h['id'] for h in hits}
    return {
        'query_id': query['id'], 'source_page': query['document'],
        'status': answer['status'], 'retrieved_ids': [h['id'] for h in hits],
        'selected_ids': selected, 'selected_claim_count': len(claims),
        'exact_quote_check': exact, 'any_gold_in_top5': bool(gold & top),
        'all_gold_in_top5': gold <= top,
        'any_gold_selected': bool(gold & set(selected)),
        'all_gold_selected': gold <= set(selected),
        'failure_reason': (
            'abstained_or_blocked' if answer['status'] not in ('evidence', 'answered')
            else 'no_gold_in_retrieval' if not gold & top
            else 'gold_retrieved_not_selected' if not gold & set(selected)
            else 'gold_selected'
        ),
    }


def cluster_ci(rows: list[dict], field: str, seed: int = 4217, draws: int = 10000):
    """Page-cluster percentile interval for question-weighted retrieval/selection rates."""
    by_page: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_page[row['source_page']].append(row)
    pages = sorted(by_page)
    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(draws):
        sampled = rng.integers(0, len(pages), len(pages))
        total = sum(len(by_page[pages[i]]) for i in sampled)
        good = sum(sum(int(r[field]) for r in by_page[pages[i]]) for i in sampled)
        rates.append(good / total)
    return [round(float(x), 6) for x in np.percentile(rates, [2.5, 97.5])]


def summarize(rows: list[dict], *, draws=10000):
    n = len(rows)
    assert n > 0
    out = {
        'test_queries': n, 'source_pages': len({r['source_page'] for r in rows}),
        'output_type': 'retrieved exact excerpts, not generated numerical answers',
        'any_gold_in_top5': sum(r['any_gold_in_top5'] for r in rows),
        'all_gold_in_top5': sum(r['all_gold_in_top5'] for r in rows),
        'any_gold_selected': sum(r['any_gold_selected'] for r in rows),
        'all_gold_selected': sum(r['all_gold_selected'] for r in rows),
        'claims_exact_quote_valid': sum(r['exact_quote_check'] for r in rows),
        'emitted_excerpts': sum(r['selected_claim_count'] for r in rows),
        'abstained_or_blocked': sum(r['status'] in ('abstained', 'blocked') for r in rows),
        'failure_taxonomy': {
            label: sum(r['failure_reason'] == label for r in rows)
            for label in ('gold_selected', 'gold_retrieved_not_selected', 'no_gold_in_retrieval', 'abstained_or_blocked')
        },
        'any_gold_selected_page_cluster_ci_95': cluster_ci(rows, 'any_gold_selected', draws=draws),
        'any_gold_in_top5_page_cluster_ci_95': cluster_ci(rows, 'any_gold_in_top5', draws=draws),
        'limit': 'FinQA gold passage IDs are evaluation labels only; selected gold passage is NOT an accurate answer, a faithful numerical computation, or a model-generated question-answer score.',
        'uncertainty_limit': 'Page-cluster bootstrap describes fixed observations; it does not measure variation across retraining seeds or new banks.',
    }
    assert sum(out['failure_taxonomy'].values()) == n
    return out


def evaluate():
    chunks, queries = load_finqa()
    engine = Engine(chunks, json.loads((DATA / 'reranker.json').read_text()))
    rows = []
    for query in queries:
        if query['split'] != 'test':
            continue
        hits = engine.search(query['question'], method='bm25', k=5)
        response = synthesize(query['question'], hits, use_llm=False)
        rows.append(check_one(query, hits, response))
    rows.sort(key=lambda r: r['query_id'])
    summary = summarize(rows)
    payload = {'method': 'bm25', 'scope': 'entire_collection', 'mode': 'extractive_no_llm',
               'summary': summary, 'examples': rows}
    reports = ROOT / 'reports'
    (reports / 'extractive-evaluation.json').write_text(json.dumps(payload, indent=2) + '\n')
    lines = [
        '# End-to-end extractive evidence selection (no LLM)', '',
        'The actual retrieval engine takes each **question only**, searches the entire corpus with BM25, then runs the same local extractive answer selection as the API. This evaluates **retrieval-to-quotation coverage**, not numerical question answering. FinQA gold passage IDs are read **after** selection and are evaluation labels only.', '',
        f"Benchmark: {summary['test_queries']} held-out questions on {summary['source_pages']} pages (the previously inspected custom FinQA split).", '',
        '| Check | Count | Denominator |', '|---|---:|---:|',
    ]
    for name in ('any_gold_in_top5', 'all_gold_in_top5', 'any_gold_selected', 'all_gold_selected', 'claims_exact_quote_valid', 'abstained_or_blocked'):
        lines.append(f"| {name.replace('_', ' ')} | {summary[name]} | {summary['test_queries']} |")
    lines.extend(['',
        f"- Any gold passage selected: {summary['any_gold_selected'] / summary['test_queries']:.1%}; 95% page-cluster bootstrap CI {summary['any_gold_selected_page_cluster_ci_95']}.",
        f"- Any gold passage retrieved top-five: {summary['any_gold_in_top5'] / summary['test_queries']:.1%}; 95% page-cluster bootstrap CI {summary['any_gold_in_top5_page_cluster_ci_95']}.",
        f"- Total excerpts emitted: {summary['emitted_excerpts']}. The exact quote check is purely textual; it cannot establish factual completeness or relevance.", '',
        '## Error taxonomy', '',
    ])
    for key, count in summary['failure_taxonomy'].items():
        lines.append(f'- {key.replace("_", " ")}: {count}')
    lines.extend(['', '## Limitations', '',
        '- There is **no** independently predicted arithmetic program and **no** evaluated numerical final answer. Do not represent these coverage metrics as FinQA answer accuracy.',
        '- A correct source passage is only a prerequisite for a correct answer; selecting one does not entail its evidence is enough for a multi-hop question.',
        '- Excerpts are verbatim substrings by construction. That check does not establish semantic faithfulness, correct financial units or absence of cherry-picking.',
        '- The held-out split has already been inspected in earlier project revisions. Treat this as a transparent audit of a frozen system, not an untouched model-selection benchmark.',
        '- All metrics include abstentions, count each question once and use page-cluster bootstrap intervals. No new supervised training occurs here.',
        '- Real model-generated answer evaluation still requires a separate, untouched set of report questions and verified free-text/numerical reference answers.',
        '', 'Reproduce: `python scripts/evaluate_extractive.py`. Example-level outcomes are in `reports/extractive-evaluation.json`.'])
    (reports / 'EXTRACTIVE_EVALUATION.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    evaluate()
