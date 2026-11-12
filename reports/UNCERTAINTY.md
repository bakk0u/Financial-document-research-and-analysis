# Frozen-test uncertainty audit

All values computed from committed per-question/per-sentence predictions; no models were retrained.

**Sentiment:** 452 test sentences in 424 template groups, including 61 negative examples. Single training seed.
Macro-F1: LoRA 0.927; TF-IDF 0.838. Paired difference +0.089, group-bootstrap 95% CI [+0.047, +0.135].

**Retrieval:** delta is *competitor minus BM25* (negative means BM25 ahead). Cluster bootstrap resamples the 50 test pages or report-years, not 162 independent questions.

| Scope | Competitor | Metric | Unit | Delta | Cluster-bootstrap 95% CI | Clusters |
|---|---|---|---|---:|---:|---:|
| collection | reranked | Recall@5 | page | -2.4 pp | [-8.8, +4.2] pp | 50 |
| collection | reranked | Recall@5 | report | -2.4 pp | [-8.4, +4.4] pp | 40 |
| collection | hybrid | Recall@5 | page | -25.7 pp | [-34.8, -16.9] pp | 50 |
| collection | hybrid | Recall@5 | report | -25.7 pp | [-34.9, -16.6] pp | 40 |
| collection | vector | Recall@5 | page | -43.2 pp | [-52.2, -34.8] pp | 50 |
| collection | vector | Recall@5 | report | -43.2 pp | [-51.6, -35.1] pp | 40 |
| page | reranked | Recall@5 | page | -3.2 pp | [-8.4, +1.5] pp | 50 |
| page | reranked | Recall@5 | report | -3.2 pp | [-8.1, +1.9] pp | 40 |
| page | hybrid | Recall@5 | page | -5.9 pp | [-11.2, -0.8] pp | 50 |
| page | hybrid | Recall@5 | report | -5.9 pp | [-10.7, -1.2] pp | 40 |
| page | vector | Recall@5 | page | -17.4 pp | [-25.9, -9.3] pp | 50 |
| page | vector | Recall@5 | report | -17.4 pp | [-24.8, -10.4] pp | 40 |

## Interpretation

Differences with intervals crossing zero do not support a confident winner. Resampling at report level is more conservative about dependence across pages of the same report.
Percentile bootstrap intervals are descriptive, not a guarantee of 95% coverage for nonrandom report selection.
The original test set has been viewed repeatedly. Further model tuning would require a genuinely new untouched held-out benchmark.
Sentiment estimates exclude multi-seed training variability; especially avoid overinterpreting per-class effects with only 61 negative test sentences.
No claims about live LLM answer correctness, banking-grade security, or investment performance follow from these intervals.
