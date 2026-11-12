# Reproducibility and verification ledger

## Changes in revision 1.1 — 12 November 2026

The archived version 1.0 evidence is retained, but does not constitute a fresh CI run for this revision. This revision adds:

1. Deterministic paired bootstrap intervals for the frozen 162-query retrieval test set; clusters are 50 distinct source pages or 40 report-years. The reranker–BM25 collection-scope Recall@5 difference is -2.4 percentage points with page-cluster percentile 95% CI [-8.8, +4.2] points (10,000 draws, fixed seed).
2. A grouped bootstrap on 452 frozen sentiment predictions, 424 template groups and 61 negative cases (single training seed). An interval on fixed predictions is **not** a multiple-seed training replication.
3. A FinQA oracle-program audit: 155/162 arithmetic programs executed by the safe DSL; all literal operands appeared in collection-wide BM25 top-5 evidence on 89/162. This is not independent question-to-program prediction, end-to-end arithmetic QA or faithfulness evaluation.
4. A cited arithmetic API and GUI; the server resolves references from its own index and validates exact source quotation, operand occurrence, finite arithmetic, operation trace and audit record.
5. Generation restricted to exact supporting quotations. No free-form paraphrase can pass provider output validation. Screening handles the three earlier regression probes but is not robust to arbitrary injection.

6. Six exploratory post-hoc reranker ablations on the pre-designated validation partition, with training pairs and parameters held fixed. The original production model and held-out test results are unchanged.

## How to verify

```sh
python -m pip install -r requirements.lock.txt
python -m pytest -q
python scripts/redteam.py
python scripts/evaluate_uncertainty.py
python scripts/evaluate_arithmetic.py
python scripts/ablate_reranker.py
python scripts/verify_manifest.py
python run.py
```

The original archived repository reported 49 passing tests including actual saved LoRA-adapter inference on an environment with downloaded pinned base-model weights. This revised project must be tested separately; do not combine these counts or claim remote workflow status until GitHub Actions actually runs.

## Revision 1.1 verification actually executed locally

On 12 November 2026, this editing workspace ran **Python 3.13** with NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, PyTorch 2.10.0+cpu, FastAPI 0.128.2, and pytest 9.0.2. This differs from the archived Python 3.12 training environment and pinned GitHub CI matrix.

- **73 tests passed, 1 deselected, 1 scikit-learn numerical warning**. The deselected test is `tests/test_sentiment.py::test_actual_adapter_inference`, because `transformers` and `peft` are unavailable in this isolated runtime and downloads cannot complete. See `reports/revision-tests.log`. Earlier build logs record the separate environment's passing transformer inference but are not a substitute for this revision's full run.
- `node --check static/app.js` passed.
- `scripts/redteam.py` passed 13/13 known prompt-screening checks and rejected 2/2 citation forgeries. New unseen injections remain untested.
- `scripts/evaluate_uncertainty.py`, `scripts/evaluate_arithmetic.py`, and six validation-only ablations executed. Their JSON and Markdown outputs are committed as reproducible artifacts.
- A running local server returned **HTTP 200** for `/api/status` with 8,778 chunks; `/api/chunk` retrieved a real PM 2017 table; `/api/calculate` returned **688** for `subtract(11503,10815)` with an exact cited quote; both added evaluation endpoints returned HTTP 200.
- Browser automation of the local server was attempted but blocked by the execution environment's network administrator (`ERR_BLOCKED_BY_ADMINISTRATOR`). **The revised frontend therefore has not been browser-verified**. A static JavaScript syntax check is not a UX or browser test.

## Known gaps (not hidden)

- Live OpenAI generation is **not verified**. There is no model-generated FinQA answer set, independently annotated answer correctness/faithfulness check or numerical QA accuracy claim. Mocked response contracts only verify the I/O boundary.
- Model training has a **single seed** and the retained published test set has been viewed. Valid new model selection would require a genuinely unused evaluation set. Training a new seed later cannot erase the fact this test was inspected.
- The vector method is unsupervised LSA, not a neural embedding or cross-encoder, and the learned reranker uses seven logistic features.
- Page-scope retrieval assumes that the correct page is already known. Report templates, companies and time periods can be dependent; clustering doesn't guarantee random sampling.
- PDF parsing is not sandboxed. No authentication, multi-tenant controls, PII guarantees, tool execution or investment advice is provided.
- Docker, a live deployed backend, remote GitHub CI, Safari/Firefox and independent operating-system execution are not verified unless later testing documents specific artifacts or logs.

## Legacy 1.0 observations

The original archive contained `reports/tests.log`, `reports/clean-tests.log`, and model-training logs. These were generated before revision 1.1 and must not be represented as validation of newly modified code.
