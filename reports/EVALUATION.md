# Financial Document Research & Analysis — retrieval evaluation

All numbers below are measured, not illustrative.

| Split | Scope | Method | Recall@5 | MRR@5 | All evidence@5 | Median ms |
|---|---|---|---:|---:|---:|---:|
| validation | page | bm25 | 0.812 | 0.726 | 0.687 | 3.7 |
| validation | page | vector | 0.679 | 0.550 | 0.524 | 3.4 |
| validation | page | hybrid | 0.767 | 0.675 | 0.610 | 3.4 |
| validation | page | reranked | 0.857 | 0.842 | 0.744 | 4.4 |
| validation | collection | bm25 | 0.518 | 0.470 | 0.423 | 4.6 |
| validation | collection | vector | 0.204 | 0.167 | 0.159 | 4.5 |
| validation | collection | hybrid | 0.337 | 0.291 | 0.244 | 5.0 |
| validation | collection | reranked | 0.538 | 0.544 | 0.423 | 5.8 |
| test | page | bm25 | 0.858 | 0.798 | 0.765 | 3.0 |
| test | page | vector | 0.683 | 0.528 | 0.586 | 3.0 |
| test | page | hybrid | 0.799 | 0.688 | 0.704 | 3.1 |
| test | page | reranked | 0.825 | 0.817 | 0.710 | 3.8 |
| test | collection | bm25 | 0.650 | 0.621 | 0.543 | 4.5 |
| test | collection | vector | 0.218 | 0.233 | 0.160 | 4.4 |
| test | collection | hybrid | 0.392 | 0.340 | 0.315 | 4.9 |
| test | collection | reranked | 0.626 | 0.585 | 0.519 | 5.8 |

## Interpretation

Compare reranked against hybrid in BOTH scopes. A loss is a finding, not a reason to hide a baseline.

## Limitations

- Custom repartition of the public FinQA dev set, not an official FinQA score.
- Report-disjoint query splits; companies may appear in different years across splits.
- Unsupervised LSA vocabulary and vectors are fitted on the searchable collection, including held-out documents.
- Page scope assumes the relevant source page is supplied; collection scope does not.
- Historical report excerpts, not full filings or current market information.
- Retrieval metrics do not establish numerical answer accuracy or narrative faithfulness.
- Supervised feature reranker; no LLM or neural embedding weights are fine-tuned.
- No hyperparameter search performed; test results observed once for this frozen configuration.
