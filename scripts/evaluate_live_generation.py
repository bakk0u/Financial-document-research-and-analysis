"""Live evaluation of the optional cited-answer generation mode (exact-quotation claims).

The application's "Generate a cited answer" path asks a hosted model to select verbatim
excerpts that answer the question; the server rejects anything that is not an exact quote
from a retrieved passage. Until now that path was covered only by mocked contract tests.

This script runs it for real on a fixed subset of the frozen blind set (the first 150
questions by id, collection-wide BM25 top five, exactly as the UI does) and measures:
  * outcome: answered, abstained (model returned no claim), error (output failed validation);
  * supported claims: every accepted claim is re-checked to be a verbatim substring of its source;
  * evidence quality: share of answers citing at least one gold passage, versus retrieval's ceiling;
  * latency and token usage.
It does not score numerical correctness: this mode never computes. See evaluate_blind.py for that.
"""
from __future__ import annotations
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from financial_research.answer import synthesize
from financial_research.blind import corpus_from_rows, load_blind, load_dev_rows
from financial_research.retrieval import Engine

SUBSET = 150


def main():
    if not (os.environ.get('OPENAI_API_KEY') and os.environ.get('FINANCIAL_RESEARCH_MODEL')):
        sys.exit('Set OPENAI_API_KEY and FINANCIAL_RESEARCH_MODEL.')
    out_path = ROOT / 'reports/live-generation.json'
    if out_path.exists():
        sys.exit('Live generation already evaluated on this fixed subset; refusing to overwrite.')
    dev_chunks, _ = corpus_from_rows(load_dev_rows())
    blind_chunks, questions = corpus_from_rows(load_blind())
    engine = Engine(sorted(dev_chunks + blind_chunks, key=lambda c: c.id))
    questions = sorted(questions, key=lambda q: q['id'])[:SUBSET]

    def run(q):
        hits = engine.search(q['question'], 'bm25', 5)
        started = time.perf_counter()
        result = synthesize(q['question'], hits, use_llm=True)
        elapsed = (time.perf_counter() - started) * 1000
        by_id = {h['id']: h['text'] for h in hits}
        squeeze = lambda s: ' '.join(s.split())
        claims = result.get('claims', [])
        return {'query_id': q['id'], 'status': result['status'], 'claims': len(claims),
                'all_claims_verbatim': all(squeeze(c['quote']) in squeeze(by_id.get(c['citation'], '')) and c['statement'] == c['quote'] for c in claims),
                'cites_gold': bool(set(q['gold']) & {c['citation'] for c in claims}),
                'gold_in_top5': bool(set(q['gold']) & set(by_id)), 'latency_ms': round(elapsed, 1),
                'usage': result.get('usage', {})}

    with ThreadPoolExecutor(6) as pool:
        rows = list(pool.map(run, questions))
    answered = [r for r in rows if r['status'] == 'answered']
    tokens = lambda key: sum(r['usage'].get(key, 0) for r in rows)
    summary = {'model': os.environ['FINANCIAL_RESEARCH_MODEL'], 'questions': len(rows), 'scope': 'collection, BM25 top 5',
               'status_counts': {s: sum(r['status'] == s for r in rows) for s in sorted({r['status'] for r in rows})},
               'claims_emitted': sum(r['claims'] for r in rows),
               'answers_with_all_claims_verbatim': sum(r['all_claims_verbatim'] for r in answered),
               'answers_citing_gold': sum(r['cites_gold'] for r in answered),
               'questions_with_gold_in_top5': sum(r['gold_in_top5'] for r in rows),
               'answered_when_gold_absent_from_top5': sum(not r['gold_in_top5'] for r in answered),
               'latency_p50_ms': round(float(np.median([r['latency_ms'] for r in rows])), 1),
               'latency_p95_ms': round(float(np.percentile([r['latency_ms'] for r in rows], 95)), 1),
               'prompt_tokens': tokens('prompt_tokens'), 'completion_tokens': tokens('completion_tokens')}
    out_path.write_text(json.dumps({'summary': summary, 'rows': rows}, indent=1) + '\n')
    s, a = summary, max(1, len(answered))
    lines = ['# Live evaluation: cited-answer generation (exact quotations)', '',
             f"Model `{s['model']}`; first {s['questions']} questions of the frozen blind set; whole-collection BM25 top five passed to the model, as in the application. Run once.", '',
             '| Measure | Result |', '|---|---:|',
             *[f"| Outcome: {k} | {v} / {s['questions']} |" for k, v in s['status_counts'].items()],
             f"| Claims emitted | {s['claims_emitted']} |",
             f"| Answers whose every claim is a verbatim source quotation | {s['answers_with_all_claims_verbatim']} / {len(answered)} |",
             f"| Answers citing at least one gold passage | {s['answers_citing_gold']} / {len(answered)} ({100 * s['answers_citing_gold'] / a:.1f}%) |",
             f"| Questions where retrieval put a gold passage in the top five | {s['questions_with_gold_in_top5']} / {s['questions']} |",
             f"| Answers given although no gold passage was retrieved | {s['answered_when_gold_absent_from_top5']} |",
             f"| Latency, median / 95th percentile | {s['latency_p50_ms']} / {s['latency_p95_ms']} ms |",
             f"| Tokens, prompt / completion | {s['prompt_tokens']:,} / {s['completion_tokens']:,} |",
             '', '## Reading the result', '',
             '- "error" means the model output failed server-side validation (not an exact quote, or an unknown source) and nothing was shown. That is the guardrail working, and its rate is the cost of the guardrail.',
             '- Verbatim quotation guarantees the text exists in the source. It does not guarantee the excerpt answers the question; the gold-citation rate is the measured proxy for that.',
             '- An answer given when no gold passage was retrieved is a selection of related but insufficient evidence: the missing-evidence failure this mode can still make.',
             '- Cost is reported as tokens; multiply by the current price of the model used.',
             '', 'Per-question rows: `reports/live-generation.json`.']
    (ROOT / 'reports/LIVE_GENERATION.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
