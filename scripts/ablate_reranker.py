"""Exploratory validation-only reranker feature ablation.

No test-set measurement or production model selection is performed here.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import hashlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from financial_research.corpus import ROOT, DATA, load_finqa
from financial_research.retrieval import Engine,FEATURE_NAMES
from financial_research.evaluate import metrics

ABLATIONS={
    'all_seven_features': [],
    'without_lsa_cosine': ['lsa_cosine'],
    'without_exact_number_coverage': ['exact_number_coverage'],
    'without_table_indicator': ['table_indicator'],
    'without_bigram_coverage': ['query_bigram_coverage'],
    'without_lexical_query_coverage': ['query_coverage','query_density','query_bigram_coverage'],
}

def main():
    chunks, queries=load_finqa(); engine=Engine(chunks)
    x=[]; y=[]; ids=[]
    for q in queries:
        if q['split']!='train':continue
        ids.append(q['id'])
        bm,vec=engine.scores(q['question'])
        cand=engine.search(q['question'],'hybrid',20,q['document'])
        cids={c['id'] for c in cand}|set(q['gold'])
        for ident in sorted(cids):
            idx=engine.by_id[ident]
            x.append(engine.features(q['question'],idx,bm[idx],vec[idx]));y.append(int(ident in q['gold']))
    x=np.asarray(x);y=np.asarray(y)
    results=[]
    for name,excluded in ABLATIONS.items():
        include=[i for i,n in enumerate(FEATURE_NAMES) if n not in excluded]
        m=LogisticRegression(C=1,class_weight='balanced',max_iter=1000,random_state=42).fit(x[:,include],y)
        coef=np.zeros(len(FEATURE_NAMES));coef[include]=m.coef_[0]
        engine.model={'coef':coef.tolist(),'intercept':float(m.intercept_[0])}
        for scope in ('collection','page'):
            scores=[]
            for q in queries:
                if q['split']!='validation':continue
                hit=engine.search(q['question'],'reranked',5,q['document'] if scope=='page' else None)
                scores.append(metrics([h['id'] for h in hit],q['gold']))
            results.append({'variant':name,'excluded_features':excluded,'scope':scope,
                'split':'validation','queries':len(scores),'train_pairs':len(y),
                'recall_at_5':round(float(np.mean([v['recall_at_5'] for v in scores])),5),
                'mrr_at_5':round(float(np.mean([v['mrr_at_5'] for v in scores])),5),
                'all_evidence_at_5':round(float(np.mean([v['all_evidence_at_5'] for v in scores])),5)})
        print(name,[(r['scope'],r['recall_at_5']) for r in results[-2:]],flush=True)
    output={'design':'Post-hoc exploratory feature ablation on validation only; no test-set results used to choose variants.',
            'trained_query_count':len(ids),'train_pairs':len(y),'validation_questions':sum(q['split']=='validation' for q in queries),
            'results':results,'caution':'The test set was inspected before this audit. Do not select a new final model from these results and reuse the old test set as fresh confirmation.'}
    path=ROOT/'reports/ablation.json';path.write_text(json.dumps(output,indent=2))
    lines=['# Seven-feature logistic reranker: exploratory validation ablations','',
        '**Validation only**; no new frozen-test model comparison. The baseline is a supervised seven-feature classifier, not a cross-encoder or fine-tuned neural retriever.','','| Feature set | Page Recall@5 | Collection Recall@5 |', '|---|---:|---:|']
    for name in ABLATIONS:
        p=next(x for x in results if x['variant']==name and x['scope']=='page')
        c=next(x for x in results if x['variant']==name and x['scope']=='collection')
        lines.append(f"| {name.replace('_',' ')} | {p['recall_at_5']*100:.1f}% | {c['recall_at_5']*100:.1f}% |")
    lines+=['','## Design','',
        '- Refit the exact same training query–passage pairs with a fixed seed, class weights, C=1, and identical retrieval candidate sets. Excluded features have zero coefficient; all other features retain their original positions.','- Evaluate on the previously designated report-separated validation queries; do not use the inspected test set to select a new variant.','- Correlated features make leave-one-out ablations an imperfect causal explanation. Different negative sampling or retrieval candidate pools can change the conclusions.','- The number of training examples and all measured values are in `ablation.json`; keep weak/negative ablations visible.']
    (ROOT/'reports/ABLATION.md').write_text('\n'.join(lines)+'\n')
    return output

if __name__=='__main__':main()
