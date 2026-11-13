# End-to-end extractive evidence selection (no LLM)

The actual retrieval engine takes each **question only**, searches the entire corpus with BM25, then runs the same local extractive answer selection as the API. This evaluates **retrieval-to-quotation coverage**, not numerical question answering. FinQA gold passage IDs are read **after** selection and are evaluation labels only.

Benchmark: 162 held-out questions on 50 pages (the previously inspected custom FinQA split).

| Check | Count | Denominator |
|---|---:|---:|
| any gold in top5 | 126 | 162 |
| all gold in top5 | 88 | 162 |
| any gold selected | 119 | 162 |
| all gold selected | 75 | 162 |
| claims exact quote valid | 162 | 162 |
| abstained or blocked | 0 | 162 |

- Any gold passage selected: 73.5%; 95% page-cluster bootstrap CI [0.649718, 0.813953].
- Any gold passage retrieved top-five: 77.8%; 95% page-cluster bootstrap CI [0.697531, 0.850932].
- Total excerpts emitted: 486. The exact quote check is purely textual; it cannot establish factual completeness or relevance.

## Error taxonomy

- gold selected: 119
- gold retrieved not selected: 7
- no gold in retrieval: 36
- abstained or blocked: 0

## Limitations

- There is **no** independently predicted arithmetic program and **no** evaluated numerical final answer. Do not represent these coverage metrics as FinQA answer accuracy.
- A correct source passage is only a prerequisite for a correct answer; selecting one does not entail its evidence is enough for a multi-hop question.
- Excerpts are verbatim substrings by construction. That check does not establish semantic faithfulness, correct financial units or absence of cherry-picking.
- The held-out split has already been inspected in earlier project revisions. Treat this as a transparent audit of a frozen system, not an untouched model-selection benchmark.
- All metrics include abstentions, count each question once and use page-cluster bootstrap intervals. No new supervised training occurs here.
- Real model-generated answer evaluation still requires a separate, untouched set of report questions and verified free-text/numerical reference answers.

Reproduce: `python scripts/evaluate_extractive.py`. Example-level outcomes are in `reports/extractive-evaluation.json`.
