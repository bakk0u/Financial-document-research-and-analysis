"""Run the pre-registered blind benchmark (docs/BLIND_PROTOCOL.md).

    python scripts/evaluate_blind.py --freeze            # record system hashes, before any blind run
    python scripts/evaluate_blind.py --run retrieval     # BM25 vs neural retrieval
    python scripts/evaluate_blind.py --run learned       # local numerical QA
    python scripts/evaluate_blind.py --run llm           # hosted-model proposer (needs credentials)
    python scripts/evaluate_blind.py --report            # write reports/blind/BLIND_RESULTS.md
    python scripts/evaluate_blind.py --verify-learned    # CI: recompute the local run, compare with the record

A run refuses to start unless the frozen hashes match the working tree, and refuses to
overwrite an existing result: each frozen system is evaluated once.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from financial_research.arithmetic import grounded_literal, literal_atoms
from financial_research.blind import corpus_from_rows, load_blind, load_dev_rows, numeric_correct
from financial_research.evaluate import metrics
from financial_research.numeric_qa import PASSAGES, NumericQA, canonical_template, clean_number
from financial_research.retrieval import Engine
from financial_research.uncertainty import paired_cluster_bootstrap

OUT = ROOT / 'reports/blind'
CONFIG = ROOT / 'data/blind/system_config.json'
FROZEN_FILES = ['financial_research/numeric_qa.py', 'financial_research/llm_program.py',
                'financial_research/arithmetic.py', 'financial_research/retrieval.py',
                'financial_research/neural.py', 'financial_research/blind.py', 'financial_research/corpus.py',
                'scripts/evaluate_blind.py', 'data/numeric-qa-model.joblib', 'data/blind/system_config.json',
                'data/blind/finqa-blind-test.json']
SCOPES = ('collection', 'page')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_hashes() -> dict:
    return {name: sha(ROOT / name) for name in FROZEN_FILES}


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'SYSTEM_FREEZE.json'
    if any(OUT.glob('*-predictions.json')):
        sys.exit('Blind results already exist; the system cannot be re-frozen.')
    target.write_text(json.dumps({'frozen_at_unix': int(time.time()), 'files': current_hashes(),
                                  'config': json.loads(CONFIG.read_text())}, indent=2) + '\n')
    print('frozen', target)


def require_freeze():
    target = OUT / 'SYSTEM_FREEZE.json'
    if not target.exists():
        sys.exit('Freeze the system first: python scripts/evaluate_blind.py --freeze')
    frozen = json.loads(target.read_text())['files']
    changed = [n for n, h in current_hashes().items() if frozen.get(n) != h]
    if changed:
        sys.exit(f'Working tree differs from the frozen system: {changed}')


DRY_RUN = False


def build():
    if DRY_RUN:
        # Development rehearsal of the identical code path on the dev validation split.
        from financial_research.corpus import split_for
        dev = load_dev_rows()
        chunks, _ = corpus_from_rows(dev)
        _, questions = corpus_from_rows([r for r in dev if split_for(r['filename']) == 'validation'])
        return Engine(chunks), questions
    rows = load_blind()
    dev_chunks, _ = corpus_from_rows(load_dev_rows())
    blind_chunks, questions = corpus_from_rows(rows)
    assert not {c.id for c in dev_chunks} & {c.id for c in blind_chunks}
    chunks = sorted(dev_chunks + blind_chunks, key=lambda c: c.id)
    return Engine(chunks), questions


def cluster_ci(rows: list[dict], field: str, seed: int = 20261113, draws: int = 10000) -> list[float]:
    by_page = defaultdict(list)
    for row in rows:
        by_page[row['document']].append(float(row[field]))
    sums = np.array([sum(v) for v in by_page.values()])
    counts = np.array([len(v) for v in by_page.values()])
    pick = np.random.default_rng(seed).integers(0, len(sums), size=(draws, len(sums)))
    rates = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return [round(float(x), 4) for x in np.quantile(rates, [.025, .975])]


# ------------------------------------------------------------------ retrieval
def run_retrieval():
    from financial_research.neural import CrossEncoderReranker, DenseIndex, reciprocal_rank_fusion
    config = json.loads(CONFIG.read_text())['retrieval']
    engine, questions = build()
    chunks = engine.chunks
    dense = DenseIndex(chunks, config['bi_encoder'], config.get('bi_encoder_revision'),
                       config.get('query_prefix', ''))
    cross = CrossEncoderReranker(config['cross_encoder'], config.get('cross_encoder_revision'))
    bm25 = lambda q, k, doc: [engine.by_id[h['id']] for h in engine.search(q, 'bm25', k, doc)]
    methods = {
        'bm25': lambda q, doc: bm25(q, 5, doc),
        'dense': lambda q, doc: dense.rank(q, 5, doc),
        'bm25+dense_rrf': lambda q, doc: reciprocal_rank_fusion([bm25(q, 40, doc), dense.rank(q, 40, doc)], 5),
        'bm25->cross_encoder': lambda q, doc: cross.rerank(q, bm25(q, config['rerank_depth'], doc), chunks, 5),
    }
    records, latency = [], defaultdict(list)
    for scope in SCOPES:
        for name, fn in methods.items():
            for q in questions:
                started = time.perf_counter()
                found = [chunks[i].id for i in fn(q['question'], q['document'] if scope == 'page' else None)]
                latency[(scope, name)].append((time.perf_counter() - started) * 1000)
                records.append({'query_id': q['id'], 'document': q['document'], 'scope': scope, 'method': name,
                                'predicted': found, 'gold': q['gold'], **metrics(found, q['gold'])})
            print(scope, name, flush=True)
    summary, contrasts = [], []
    for scope in SCOPES:
        per_query = defaultdict(dict)
        for r in records:
            if r['scope'] == scope:
                per_query[r['query_id']][r['method']] = r
        rows = [{'page': v['bm25']['document'], **v} for _, v in sorted(per_query.items())]
        for name in methods:
            sub = [r for r in records if r['scope'] == scope and r['method'] == name]
            summary.append({'scope': scope, 'method': name, 'queries': len(sub),
                            **{m: round(float(np.mean([r[m] for r in sub])), 4)
                               for m in ('recall_at_5', 'mrr_at_5', 'all_evidence_at_5')},
                            'latency_p50_ms': round(float(np.median(latency[(scope, name)])), 1)})
            if name != 'bm25':
                for metric in ('recall_at_5', 'mrr_at_5', 'all_evidence_at_5'):
                    contrasts.append({'scope': scope, 'contrast': f'{name} minus bm25', 'metric': metric,
                                      **paired_cluster_bootstrap(rows, 'page', name, 'bm25', metric)})
    save('retrieval', {'config': config, 'corpus_chunks': len(chunks), 'questions': len(questions),
                       'summary': summary, 'paired_page_cluster_bootstrap': contrasts}, records)


# ------------------------------------------------------------------ numerical QA
def score(q: dict, hits: list[dict], out: dict, scope: str) -> dict:
    gold = set(q['gold'])
    retrieved = {h['id'] for h in hits}
    answered = out['status'] == 'answered'
    row = {'query_id': q['id'], 'document': q['document'], 'scope': scope, 'status': out['status'],
           'answered': answered, 'gold_retrieved_any': bool(gold & retrieved), 'gold_retrieved_all': gold <= retrieved,
           'exe_ans': q['exe_ans'], 'reference_answer': q['answer'], 'gold_program': q['program'],
           'reason': out.get('reason'), 'strict': False, 'lenient': False}
    candidate = out if answered else out.get('unverified_candidate')
    if candidate:
        row.update(program=candidate.get('program'), value=candidate.get('value'), confidence=candidate.get('confidence'),
                   candidate_strict=numeric_correct(candidate.get('value'), q['exe_ans']))
    for key in ('usage', 'latency_ms', 'model'):
        if key in out:
            row[key] = out[key]
    if not answered:
        row['failure'] = 'abstained' if out['status'] == 'abstained' else out['status']
        return row
    cited = {c['id'] for c in out['citations']}
    texts = [c['quote'] for c in out['citations']] + [q['question']]
    parsed = canonical_template(q['program'])
    gold_values = sorted(clean_number(s)[1] for s in parsed[1]) if parsed else None
    used_values = sorted(clean_number(a)[1] for a in dict.fromkeys(literal_atoms(out['program'])))
    row.update(strict=numeric_correct(out['value'], q['exe_ans']),
               lenient=numeric_correct(out['value'], q['exe_ans'], lenient=True),
               unit=out.get('unit'), display=out.get('display'),
               unit_agrees=(out.get('unit') == 'percent') == ('%' in q['answer']),
               cited=sorted(cited), cites_any_gold=bool(gold & cited), cites_all_gold=gold <= cited,
               operands_grounded=all(grounded_literal(a, texts) for a in literal_atoms(out['program'])))
    row['failure'] = ('correct' if row['strict'] else 'scale_only' if row['lenient']
                      else 'gold_evidence_not_retrieved' if not row['gold_retrieved_any']
                      else 'right_operands_wrong_operation_or_order' if gold_values == used_values
                      else 'wrong_operands')
    return row


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    answered = [r for r in rows if r['answered']]
    a = max(1, len(answered))
    out = {'questions': n, 'answered': len(answered), 'abstention_rate': round(1 - len(answered) / n, 4),
           'strict_accuracy': round(sum(r['strict'] for r in rows) / n, 4),
           'strict_accuracy_page_cluster_ci_95': cluster_ci(rows, 'strict'),
           'lenient_accuracy': round(sum(r['lenient'] for r in rows) / n, 4),
           'answered_precision': round(sum(r['strict'] for r in answered) / a, 4),
           'wrong_answer_rate': round(sum(not r['strict'] for r in answered) / n, 4),
           'unit_agreement_on_answered': round(sum(r['unit_agrees'] for r in answered) / a, 4),
           'cites_any_gold_on_answered': round(sum(r['cites_any_gold'] for r in answered) / a, 4),
           'cites_all_gold_on_answered': round(sum(r['cites_all_gold'] for r in answered) / a, 4),
           'operands_grounded_on_answered': round(sum(r['operands_grounded'] for r in answered) / a, 4),
           'gold_evidence_retrieved_any': round(sum(r['gold_retrieved_any'] for r in rows) / n, 4),
           'failure_taxonomy': dict(Counter(r['failure'] for r in rows).most_common())}
    if any('candidate_strict' in r for r in rows):
        out['strict_accuracy_if_never_abstaining'] = round(sum(r.get('candidate_strict', False) for r in rows) / n, 4)
    return out


def run_learned(verify: bool = False):
    engine, questions = build()
    qa = NumericQA.load()
    rows = []
    for scope in SCOPES:
        for q in questions:
            hits = engine.search(q['question'], 'bm25', PASSAGES, q['document'] if scope == 'page' else None)
            rows.append(score(q, hits, qa.answer(q['question'], hits), scope))
        print('learned', scope, flush=True)
    summary = {scope: summarise([r for r in rows if r['scope'] == scope]) for scope in SCOPES}
    if verify:
        recorded = json.loads((OUT / 'numeric-learned.json').read_text())['summary']
        for scope in SCOPES:
            drift = abs(recorded[scope]['strict_accuracy'] - summary[scope]['strict_accuracy'])
            print(scope, 'recorded', recorded[scope]['strict_accuracy'], 'recomputed', summary[scope]['strict_accuracy'])
            if drift > 0.01:
                sys.exit('Recomputed blind accuracy drifts from the committed record by more than one point.')
        return
    save('numeric-learned', {'proposer': 'learned', 'threshold': qa.threshold, 'passages': PASSAGES,
                             'summary': summary}, rows)


def run_llm():
    from financial_research.llm_program import answer_with_llm
    config = json.loads(CONFIG.read_text())['llm']
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        sys.exit('OPENAI_API_KEY is required for the hosted-model run.')
    engine, questions = build()
    jobs = []
    for scope in SCOPES:
        for q in questions:
            hits = engine.search(q['question'], 'bm25', PASSAGES, q['document'] if scope == 'page' else None)
            jobs.append((scope, q, hits))
    # Raw provider outputs are checkpointed so an interrupted run resumes instead of re-asking.
    checkpoint = ROOT / ('runtime/dryrun-llm-checkpoint.jsonl' if DRY_RUN else 'runtime/blind-llm-checkpoint.jsonl')
    checkpoint.parent.mkdir(exist_ok=True)
    done = {}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            item = json.loads(line)
            done[(item['scope'], item['id'])] = item['out']

    def work(job):
        scope, q, hits = job
        out = done.get((scope, q['id'])) or answer_with_llm(q['question'], hits, model=config['model'], key=key)
        return scope, q, hits, out

    rows = []
    with ThreadPoolExecutor(config.get('concurrency', 6)) as pool, checkpoint.open('a') as handle:
        for n, (scope, q, hits, out) in enumerate(pool.map(work, jobs), 1):
            if (scope, q['id']) not in done and out['status'] != 'error':
                handle.write(json.dumps({'scope': scope, 'id': q['id'], 'out': out}) + '\n')
                handle.flush()
            rows.append(score(q, hits, out, scope))
            if n % 100 == 0:
                print('llm', n, len(jobs), flush=True)
    summary = {scope: summarise([r for r in rows if r['scope'] == scope]) for scope in SCOPES}
    for scope in SCOPES:
        sub = [r for r in rows if r['scope'] == scope]
        usage = [r['usage'] for r in sub if r.get('usage')]
        summary[scope].update(
            provider_errors=sum(r['status'] == 'error' for r in sub),
            latency_p50_ms=round(float(np.median([r['latency_ms'] for r in sub if r.get('latency_ms')])), 1),
            latency_p95_ms=round(float(np.percentile([r['latency_ms'] for r in sub if r.get('latency_ms')], 95)), 1),
            prompt_tokens=sum(u['prompt_tokens'] for u in usage),
            completion_tokens=sum(u['completion_tokens'] for u in usage),
            mean_total_tokens_per_question=round(sum(u['total_tokens'] for u in usage) / max(1, len(usage)), 1))
    save('numeric-llm', {'proposer': 'llm', 'model': config['model'], 'passages': PASSAGES, 'summary': summary}, rows)


def save(name: str, summary: dict, rows: list[dict]):
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / f'{name}.json').exists():
        sys.exit(f'{name} already evaluated on the blind set; refusing to overwrite.')
    summary['blind_sha256'] = sha(ROOT / 'data/blind/finqa-blind-test.json')
    summary['evaluated_at_unix'] = int(time.time())
    (OUT / f'{name}.json').write_text(json.dumps(summary, indent=2) + '\n')
    (OUT / f'{name}-predictions.json').write_text(json.dumps(rows, indent=1) + '\n')
    print(json.dumps(summary.get('summary', summary), indent=2)[:6000])


# ------------------------------------------------------------------ report
def report():
    pct = lambda x: f'{100 * x:.1f}%'
    lines = ['# Blind benchmark results', '',
             'Protocol: [docs/BLIND_PROTOCOL.md](../../docs/BLIND_PROTOCOL.md). The 710 questions were frozen before any of '
             'the evaluated code existed, the system was hash-frozen before the run (`SYSTEM_FREEZE.json`), and each system '
             'was run once. Systems receive the question and the searchable corpus only; gold programs, gold evidence and '
             'answers are read by the scorer afterwards.', '']
    for name, title in (('numeric-learned', 'Local learned proposer (no LLM, no network)'),
                        ('numeric-llm', 'Hosted-model proposer behind the same verifier')):
        path = OUT / f'{name}.json'
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        lines += [f'## Numerical answers: {title}', '']
        if 'model' in data:
            lines += [f"Model: `{data['model']}`. The model proposes a program only; grounding and arithmetic are local.", '']
        lines += ['| Metric | Collection scope | Page scope |', '|---|---:|---:|']
        s = data['summary']
        both = lambda fn: ' | '.join(fn(s[scope]) for scope in SCOPES)
        lines += [
            f"| **Strict numeric accuracy** (all 710) | {both(lambda x: '**' + pct(x['strict_accuracy']) + '**')} |",
            f"| 95% page-cluster interval | {both(lambda x: pct(x['strict_accuracy_page_cluster_ci_95'][0]) + ' to ' + pct(x['strict_accuracy_page_cluster_ci_95'][1]))} |",
            f"| Scale-lenient accuracy | {both(lambda x: pct(x['lenient_accuracy']))} |",
            f"| Abstention rate | {both(lambda x: pct(x['abstention_rate']))} |",
            f"| Answered precision | {both(lambda x: pct(x['answered_precision']))} |",
            f"| Wrong-answer rate (of all) | {both(lambda x: pct(x['wrong_answer_rate']))} |",
            f"| Unit agreement (answered) | {both(lambda x: pct(x['unit_agreement_on_answered']))} |",
            f"| Cites at least one gold passage (answered) | {both(lambda x: pct(x['cites_any_gold_on_answered']))} |",
            f"| Cites every gold passage (answered) | {both(lambda x: pct(x['cites_all_gold_on_answered']))} |",
            f"| Every operand found in a cited passage | {both(lambda x: pct(x['operands_grounded_on_answered']))} |",
            f"| Gold evidence among retrieved passages | {both(lambda x: pct(x['gold_evidence_retrieved_any']))} |"]
        if 'strict_accuracy_if_never_abstaining' in s['collection']:
            lines.append(f"| Diagnostic: accuracy if it never abstained | {both(lambda x: pct(x['strict_accuracy_if_never_abstaining']))} |")
        if 'latency_p50_ms' in s['collection']:
            lines += [f"| Median / p95 model latency | {both(lambda x: str(x['latency_p50_ms']) + ' / ' + str(x['latency_p95_ms']) + ' ms')} |",
                      f"| Mean tokens per question | {both(lambda x: str(x['mean_total_tokens_per_question']))} |",
                      f"| Provider errors | {both(lambda x: str(x['provider_errors']))} |"]
        lines += ['', 'Outcome of every question:', '', '| Outcome | Collection | Page |', '|---|---:|---:|']
        keys = list(dict.fromkeys(list(s['collection']['failure_taxonomy']) + list(s['page']['failure_taxonomy'])))
        lines += [f"| {k.replace('_', ' ')} | {s['collection']['failure_taxonomy'].get(k, 0)} | {s['page']['failure_taxonomy'].get(k, 0)} |" for k in keys]
        lines.append('')
    path = OUT / 'retrieval.json'
    if path.exists():
        data = json.loads(path.read_text())
        lines += ['## Retrieval: BM25 against neural methods', '',
                  f"Identical corpus ({data['corpus_chunks']:,} passages), questions and candidate scope. Bi-encoder "
                  f"`{data['config']['bi_encoder']}`; cross-encoder `{data['config']['cross_encoder']}` reranking the BM25 "
                  f"top {data['config']['rerank_depth']}.", '',
                  '| Scope | Method | Recall@5 | MRR@5 | All evidence@5 | Median ms |', '|---|---|---:|---:|---:|---:|']
        lines += [f"| {r['scope']} | {r['method']} | {pct(r['recall_at_5'])} | {pct(r['mrr_at_5'])} | {pct(r['all_evidence_at_5'])} | {r['latency_p50_ms']} |"
                  for r in data['summary']]
        lines += ['', 'Paired difference in Recall@5 versus BM25 (page-cluster bootstrap, 10,000 draws):', '',
                  '| Scope | Contrast | Delta | 95% interval | Excludes zero |', '|---|---|---:|---:|:---:|']
        for c in data['paired_page_cluster_bootstrap']:
            if c['metric'] == 'recall_at_5':
                lines.append(f"| {c['scope']} | {c['contrast']} | {100 * c['delta']:+.1f} pp | [{100 * c['lower_95']:+.1f}, {100 * c['upper_95']:+.1f}] pp | "
                             f"{'yes' if c['lower_95'] > 0 or c['upper_95'] < 0 else 'no'} |")
        lines.append('')
    lines += ['## Limits', '',
              '- Report-separated, not company-separated: a company in the blind set can appear in development data for a different year.',
              '- FinQA `exe_ans` is the reference. Its annotations contain some errors; none were corrected.',
              '- Unit agreement only distinguishes percent from non-percent, the one unit signal FinQA answers carry.',
              '- Citing a gold passage shows the right evidence was used, not that the cited cell is the semantically right one.',
              '- One benchmark of historical US filings. It says nothing about other document types, live market data or advice.',
              '', 'Per-question outputs: `*-predictions.json` in this directory.']
    (OUT / 'BLIND_RESULTS.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--run', choices=['retrieval', 'learned', 'llm'])
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--verify-learned', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='rehearse on the dev validation split; writes to runtime/')
    args = parser.parse_args()
    if args.dry_run:
        DRY_RUN = True
        OUT = ROOT / 'runtime/blind-dryrun'
    if args.freeze:
        freeze()
    if args.run:
        if not DRY_RUN:
            require_freeze()
        {'retrieval': run_retrieval, 'learned': run_learned, 'llm': run_llm}[args.run]()
    if args.verify_learned:
        run_learned(verify=True)
    if args.report:
        report()
