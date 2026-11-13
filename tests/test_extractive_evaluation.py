"""Unit tests for the real retrieval-to-excerpt evaluation procedure."""
from scripts.evaluate_extractive import check_one, cluster_ci, summarize


def example(qid, page='p1', gold=('a',)):
    return {'id': qid, 'document': page, 'gold': list(gold)}


def hit(ident):
    return {'id': ident, 'text': 'Revenue in 2017 was 120 million.', 'coverage': 0.8, 'bm25': 5}


def test_no_gold_leaks_into_selection_and_quotes_are_exact():
    q = example('q1', gold=('b',))
    a = {'status': 'evidence', 'claims': [{'citation': 'a', 'statement': 'Revenue in 2017 was 120 million.',
                                          'quote': 'Revenue in 2017 was 120 million.'}]}
    result = check_one(q, [hit('a'), hit('b')], a)
    assert result['any_gold_in_top5']
    assert not result['any_gold_selected']
    assert result['failure_reason'] == 'gold_retrieved_not_selected'
    assert result['exact_quote_check']
    altered = {'status': 'evidence', 'claims': [{**a['claims'][0], 'statement': 'Revenue doubled.'}]}
    assert not check_one(q, [hit('a')], altered)['exact_quote_check']


def test_abstentions_denominator_and_page_clusters():
    a = check_one(example('q1','p1',('a',)),[hit('a')],{'status':'evidence','claims':[
        {'citation':'a','quote':'Revenue in 2017 was 120 million.','statement':'Revenue in 2017 was 120 million.'}]})
    b = check_one(example('q2','p1',('z',)),[hit('a')],{'status':'evidence','claims':[]})
    c = check_one(example('q3','p2',('z',)),[],{'status':'abstained','claims':[]})
    summ=summarize([a,b,c],draws=500)
    assert summ['test_queries']==3
    assert summ['source_pages']==2
    assert summ['any_gold_selected']==1
    assert summ['abstained_or_blocked']==1
    assert sum(summ['failure_taxonomy'].values())==3
    assert cluster_ci([a,b,c], 'any_gold_selected',draws=500)==cluster_ci([a,b,c], 'any_gold_selected',draws=500)
