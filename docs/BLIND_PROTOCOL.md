# Blind benchmark protocol (pre-registered)

This protocol was committed together with the frozen blind set and **before** any
end-to-end numerical question answering code, neural retrieval experiment or tuning
existed in this repository. The git history is the evidence for that ordering.

## Why a new set

The original 162-question test repartition of `finqa-dev.json` has been inspected
repeatedly across earlier revisions. It can no longer serve as an untouched test.

## The frozen set

- Source: the official FinQA `test.json` (upstream blob pinned in `data/blind/FREEZE.json`).
- Membership rule: every question whose company/report-year is absent from the bundled
  `finqa-dev.json`. The rule reads file names only; no question, answer or evidence was
  inspected to decide membership.
- Size: 710 questions on 238 pages from 185 report-years.
- Rebuild check: `python scripts/freeze_blind.py` must reproduce the committed SHA-256.

## Development data

- FinQA `train.json`, **excluding every row that shares a company/report-year with the
  blind set**, plus the existing dev snapshot. Gold programs may be used as training
  labels there. Companies can recur across years, so this is report separation, not
  company separation; that limit is reported with the results.
- Gold programs, gold evidence indices and answers of the blind set are read only by the
  scorer, after the system has produced its output.

## What the evaluated system receives

Only the question text and the searchable corpus (dev pages plus blind pages, raw report
text and table rows). It receives no program, no gold evidence and no answer. Two scopes
are reported and never mixed: **collection** (search everything) and **page** (the source
page is supplied, as a user who has opened a filing would do).

## Metrics fixed in advance

Numerical answering, each over all 710 questions (abstentions count as not correct):

1. **Strict numeric accuracy** (primary): the returned value matches FinQA `exe_ans`
   within a relative tolerance of 1e-3 or an absolute tolerance of 5e-4 after rounding
   both to five decimals; yes/no answers must match exactly.
2. **Scale-lenient accuracy**: as above, also accepting a factor of 100 (percent written
   as a fraction or the reverse). Reported separately, never as the headline.
3. **Abstention rate**, **answered precision** (correct / answered) and **wrong-answer
   rate** (answered and incorrect / all).
4. **Unit agreement** on answered questions: whether the returned unit class
   (percent versus not percent) matches the form of the written reference answer.
5. **Evidence correctness** on answered questions: share whose citations include at
   least one gold evidence item, and share whose citations include all of them. Every
   operand must occur literally in a cited passage (checked, not assumed).
6. **Failure taxonomy**: abstained, gold evidence not retrieved, gold retrieved but
   wrong operands, right operands but wrong operation or order, scale only.

Retrieval comparison, on the identical corpus, candidate scope and questions:
Recall@5, MRR@5 and all-evidence@5 for BM25, a neural sentence-embedding retriever, a
BM25+neural fusion, and a neural cross-encoder reranking BM25 candidates. Differences
versus BM25 use the existing paired page-cluster bootstrap (10,000 draws, fixed seed).

Uncertainty for accuracy figures: page-cluster percentile bootstrap, 95%.

## Rules

- All model, feature, threshold and template choices are made on development data.
- Before the blind run the exact system (code, configuration, trained weights) is
  committed; `reports/blind/SYSTEM_FREEZE.json` records the hashes. The blind run happens
  once per frozen system. Nothing is tuned afterwards.
- Every result is published, including a loss to BM25 or a low accuracy.
- If a defect makes a run invalid (a crash or a scoring bug, not a disappointing
  number), the defect, the fix and both results are disclosed.
