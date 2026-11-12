import json
import numpy as np
import pytest
from financial_research.corpus import DATA
from financial_research.sentiment import LABELS, load_phrasebank, make_splits, predict_baseline


def test_exact_duplicates_removed():
    rows=load_phrasebank()
    assert len(rows)==2259
    assert len({r['id'] for r in rows})==len(rows)
    assert {r['label'] for r in rows}=={0,1,2}


def test_near_duplicate_groups_do_not_cross_splits():
    rows=json.loads((DATA/'sentiment-splits.json').read_text())
    groups={}
    for r in rows:groups.setdefault(r['group'],set()).add(r['split'])
    assert all(len(s)==1 for s in groups.values())
    assert all(any(r['label']==label and r['split']==split for r in rows) for label in range(3) for split in ['train','validation','test'])


def test_group_assignment_is_reproducible():
    rows=load_phrasebank()
    assert make_splits(rows)==json.loads((DATA/'sentiment-splits.json').read_text())


def test_tfidf_probabilities_are_valid():
    texts=['Operating profit increased as revenue grew.', 'The company announced an ordinary shareholder meeting.']
    p=predict_baseline(texts)
    assert p.shape==(2,3)
    assert np.all(np.isfinite(p)) and np.all(p>=0)
    assert np.allclose(p.sum(1),1)


def test_model_report_and_adapter_exist():
    report=json.loads((DATA.parent/'reports/sentiment-evaluation.json').read_text())
    assert report['trainable_parameters']>0
    assert report['base_model']['repo']=='distilbert/distilbert-base-uncased'
    assert report['selected_epoch'] in [1,2,3]
    assert (DATA/'sentiment-adapter/adapter_model.safetensors').stat().st_size>100000
    assert report['split_counts']['test']==452


def test_actual_adapter_inference():
    # Real saved checkpoint, not a mocked prediction or upstream sentiment model.
    from financial_research.sentiment import predict_transformer
    p=predict_transformer(['Revenue and operating profit increased substantially.'])
    assert p.shape==(1,3)
    assert np.allclose(p.sum(1),1)
    assert p[0,LABELS.index('positive')]>p[0,LABELS.index('negative')]
