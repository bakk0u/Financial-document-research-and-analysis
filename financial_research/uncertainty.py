"""Pre-specified, deterministic paired uncertainty analysis of frozen predictions.

This reads only existing test predictions; it never trains or selects a model.
Cluster resampling makes within-document dependence explicit.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import f1_score


def paired_cluster_bootstrap(rows, cluster_by, key_a, key_b, metric='recall_at_5', repetitions=10000, seed=20261112):
    """Return percentile CI for mean A minus B; resample independent clusters.

    rows: one query per row, each with a value for both methods.
    The bootstrap CI is *descriptive*, not proof of a superpopulation sampling model.
    """
    if not rows or repetitions < 100:
        raise ValueError('Need rows and >=100 bootstrap draws')
    clusters = defaultdict(list)
    for row in rows:
        clusters[row[cluster_by]].append(row)
    if len(clusters) < 2:
        raise ValueError('At least two independent clusters required')
    names = sorted(clusters)
    nums = np.array([sum(x[key_a][metric] - x[key_b][metric] for x in clusters[n]) for n in names])
    denoms = np.array([len(clusters[n]) for n in names])
    generator = np.random.default_rng(seed)
    sampled = generator.integers(0, len(names), size=(repetitions, len(names)))
    samples = nums[sampled].sum(axis=1) / denoms[sampled].sum(axis=1)
    diff = float(nums.sum() / denoms.sum())
    return {'delta': round(diff, 5), 'lower_95': round(float(np.quantile(samples, .025)), 5),
            'upper_95': round(float(np.quantile(samples, .975)), 5),
            'resampling_unit': cluster_by, 'clusters': len(names), 'queries': len(rows),
            'repetitions': repetitions, 'seed': seed}


def retrieval_uncertainty(records):
    result=[]
    for scope in ('collection', 'page'):
        records_scope = [x for x in records if x['split']=='test' and x['scope']==scope]
        by_query=defaultdict(dict)
        for item in records_scope:
            by_query[item['query_id']][item['method']]=item
        rows=[]
        for qid, v in sorted(by_query.items()):
            if not {'bm25','reranked','hybrid','vector'}.issubset(v):
                raise ValueError(f'Incomplete method predictions: {qid}')
            doc=v['bm25']['gold'][0].split('::')[0]
            rows.append({'page':doc,'report':'/'.join(doc.split('/')[:2]),**v})
        for competitor in ('reranked','hybrid','vector'):
            for metric in ('recall_at_5','all_evidence_at_5','mrr_at_5'):
                for unit in ('page','report'):
                    ci=paired_cluster_bootstrap(rows,unit,competitor,'bm25',metric)
                    result.append({'scope':scope,'contrast':f'{competitor} minus bm25','metric':metric,**ci})
    return result


def sentiment_uncertainty(predictions, splits, repeats=5000, seed=20261112):
    """Grouped bootstrap of fixed test predictions, NOT multi-seed training variance."""
    group={x['id']:x['group'] for x in splits if x['split']=='test'}
    by_model=defaultdict(dict)
    for row in predictions:
        if row['split']=='test' and row['model'] in ('distilbert_raw','tfidf'):
            by_model[row['model']][row['id']]=row
    ids=sorted(set(by_model['distilbert_raw']) & set(by_model['tfidf']))
    if len(ids) != len(group) or len(ids)!=452:
        raise ValueError('Test prediction/split integrity mismatch')
    groups=defaultdict(list)
    for i, ident in enumerate(ids):groups[group[ident]].append(i)
    group_ids=sorted(groups)
    truth=np.array([by_model['tfidf'][i]['truth'] for i in ids])
    pred_a=np.array([by_model['distilbert_raw'][i]['prediction'] for i in ids])
    pred_b=np.array([by_model['tfidf'][i]['prediction'] for i in ids])
    if any(truth[i]!=by_model['distilbert_raw'][ident]['truth'] for i,ident in enumerate(ids)):
        raise ValueError('Ground-truth mismatch between models')
    r=np.random.default_rng(seed)
    deltas=[]; scores=[]
    indices=[np.asarray(groups[g],dtype=int) for g in group_ids]
    for _ in range(repeats):
        chosen=r.integers(len(indices),size=len(indices))
        sample=np.concatenate([indices[i] for i in chosen])
        a=f1_score(truth[sample],pred_a[sample],average='macro',labels=['negative','neutral','positive'],zero_division=0)
        b=f1_score(truth[sample],pred_b[sample],average='macro',labels=['negative','neutral','positive'],zero_division=0)
        deltas.append(a-b);scores.append(a)
    f1a=f1_score(truth,pred_a,average='macro',labels=['negative','neutral','positive'])
    f1b=f1_score(truth,pred_b,average='macro',labels=['negative','neutral','positive'])
    return {'frozen_single_seed_only':True,'test_sentences':len(ids),'template_groups':len(groups),
        'negative_test_sentences':int(np.sum(truth=='negative')), 'repetitions':repeats,'seed':seed,
        'distilbert_macro_f1':round(float(f1a),5),'tfidf_macro_f1':round(float(f1b),5),
        'macro_f1_difference':round(float(f1a-f1b),5),
        'difference_group_bootstrap_95':list(np.round(np.quantile(deltas,[.025,.975]),5)),
        'distilbert_group_bootstrap_95':list(np.round(np.quantile(scores,[.025,.975]),5)),
        'limits':'Grouped resampling estimates test-sample variation at fixed trained checkpoints. It does not account for retraining seeds, distribution shift, repeated test inspection or annotator disagreement.'}


def generate(root: Path, repetitions=5000):
    retrieval=json.loads((root/'reports/predictions.json').read_text())
    pred=json.loads((root/'reports/sentiment-predictions.json').read_text())
    splits=json.loads((root/'data/sentiment-splits.json').read_text())
    summary={'retrieval':retrieval_uncertainty(retrieval),
             'sentiment':sentiment_uncertainty(pred,splits,repeats=repetitions),
             'disclosure':'Post-hoc uncertainty audit of published test results: confidence intervals are descriptive; no new model choice is justified by this analysis.'}
    (root/'reports/uncertainty.json').write_text(json.dumps(summary,indent=2,default=lambda x:float(x)))
    lines=['# Frozen-test uncertainty audit','','All values computed from committed per-question/per-sentence predictions; no models were retrained.','',
        f"**Sentiment:** {summary['sentiment']['test_sentences']} test sentences in {summary['sentiment']['template_groups']} template groups, including {summary['sentiment']['negative_test_sentences']} negative examples. Single training seed.",
        f"Macro-F1: LoRA {summary['sentiment']['distilbert_macro_f1']:.3f}; TF-IDF {summary['sentiment']['tfidf_macro_f1']:.3f}. Paired difference {summary['sentiment']['macro_f1_difference']:+.3f}, group-bootstrap 95% CI [{summary['sentiment']['difference_group_bootstrap_95'][0]:+.3f}, {summary['sentiment']['difference_group_bootstrap_95'][1]:+.3f}].",'',
        '**Retrieval:** delta is *competitor minus BM25* (negative means BM25 ahead). Cluster bootstrap resamples the 50 test pages or report-years, not 162 independent questions.','','| Scope | Competitor | Metric | Unit | Delta | Cluster-bootstrap 95% CI | Clusters |',
        '|---|---|---|---|---:|---:|---:|']
    for x in summary['retrieval']:
        if x['metric']=='recall_at_5':
            lines.append(f"| {x['scope']} | {x['contrast'].split(' minus ')[0]} | Recall@5 | {x['resampling_unit']} | {100*x['delta']:+.1f} pp | [{100*x['lower_95']:+.1f}, {100*x['upper_95']:+.1f}] pp | {x['clusters']} |")
    lines+=['','## Interpretation','',
         'Differences with intervals crossing zero do not support a confident winner. Resampling at report level is more conservative about dependence across pages of the same report.',
         'Percentile bootstrap intervals are descriptive, not a guarantee of 95% coverage for nonrandom report selection.',
         'The original test set has been viewed repeatedly. Further model tuning would require a genuinely new untouched held-out benchmark.',
         'Sentiment estimates exclude multi-seed training variability; especially avoid overinterpreting per-class effects with only 61 negative test sentences.',
         'No claims about live LLM answer correctness, banking-grade security, or investment performance follow from these intervals.']
    (root/'reports/UNCERTAINTY.md').write_text('\n'.join(lines)+'\n')
    return summary
