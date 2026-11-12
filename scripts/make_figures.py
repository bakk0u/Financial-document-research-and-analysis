"""Make GitHub README figures from frozen evaluation artifacts.

Optional: requires matplotlib (not needed by the runtime or tests).
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
a=json.loads((ROOT/'reports/evaluation.json').read_text())
u=json.loads((ROOT/'reports/uncertainty.json').read_text())
ar=json.loads((ROOT/'reports/arithmetic-evaluation.json').read_text())['summary']
by={r['method']:r for r in a['results'] if r['split']=='test' and r['scope']=='collection'}
methods=['bm25','reranked','hybrid','vector']
labels=['BM25','Trained reranker','Hybrid RRF','LSA (64-D)']
fig,ax=plt.subplots(figsize=(8.6,4.4))
bar=ax.barh(labels[::-1],[100*by[x]['recall_at_5'] for x in methods[::-1]])
for b in bar:ax.text(b.get_width()+.9,b.get_y()+b.get_height()/2,f'{b.get_width():.1f}%',va='center',fontsize=10)
ax.set_xlim(0,85);ax.set_xlabel('Recall@5 (%)');ax.set_title('Frozen FinQA retrieval test — collection-wide (162 questions)')
ax.spines[['top','right']].set_visible(False);fig.tight_layout()
fig.savefig(ROOT/'assets/retrieval-recall.png',dpi=180);plt.close(fig)
ci=next(x for x in u['retrieval'] if x['scope']=='collection' and x['contrast']=='reranked minus bm25' and x['metric']=='recall_at_5' and x['resampling_unit']=='page')
fig,ax=plt.subplots(figsize=(8.6,2.7))
mean=100*ci['delta'];lo=100*ci['lower_95'];hi=100*ci['upper_95']
ax.errorbar([mean],[0],xerr=[[mean-lo],[hi-mean]],fmt='o',capsize=6,markersize=9)
ax.axvline(0,linestyle='--',alpha=.55)
ax.set_xlim(-12,8);ax.set_ylim(-.6,.6);ax.set_yticks([])
ax.set_xlabel('Difference in Recall@5 (percentage points)')
ax.set_title('Reranker minus BM25: page-cluster bootstrap 95% CI')
ax.annotate(f'{mean:+.1f} pp  [{lo:+.1f}, {hi:+.1f}] pp',(mean,.17),ha='center')
fig.tight_layout();fig.savefig(ROOT/'assets/paired-uncertainty.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(8.6,3.0))
steps=[('All test questions',ar['test_questions']),('Supported gold programs',ar['supported_programs']),('Operands in gold evidence',ar['oracle_gold_evidence_operand_coverage']),('Operands in BM25 top 5',ar['collection_bm25_operand_coverage'])]
bar=ax.barh([x[0] for x in steps][::-1],[x[1] for x in steps][::-1]);ax.set_xlim(0,195)
for b in bar:ax.text(b.get_width()+2,b.get_y()+b.get_height()/2,str(int(b.get_width())),va='center')
ax.set_title('FinQA arithmetic *oracle-input diagnostic* — NOT generated-answer accuracy')
ax.set_xlabel('Questions (n=162)');ax.spines[['top','right']].set_visible(False);fig.tight_layout()
fig.savefig(ROOT/'assets/arithmetic-bottleneck.png',dpi=180);plt.close(fig)
print('Wrote 3 data-driven README figures.')
