# Independent technical reviewer quickstart

## Five-minute audit

```bash
python -m pip install -r requirements.lock.txt
python -m pytest -q
python scripts/redteam.py
python scripts/evaluate_uncertainty.py
python scripts/evaluate_arithmetic.py
```

Compare the outputs against the committed JSON and Markdown files in `reports/`; inspect `MANIFEST.json` for content hashes. Optional `python scripts/ablate_reranker.py` retrains six variations of the seven-feature classifier on the frozen original training pairs and reports **validation only**.

## Reproduce an observable calculation

Run `python run.py`, visit `http://127.0.0.1:8765`, choose the PM 2017 page 38 source (or locate its ID `PM/2017/page_38.pdf::table_3` through the source ledger), and verify `subtract(11503,10815)` using that citation. The server must compute **688**, return an explicit trace and an audit event; substituting an unsupported operand must fail. The result uses a provided formula, not a generated FinQA solution.

## Questions to ask

- Show why the 162 retrieval questions correspond to 50 pages and 40 report-years, and why query-IID bootstrap overstates effective sample size.
- Why might a logistic reranker help validation but lose to BM25 on the frozen test set? Inspect the training-negative candidate pool, corpus scope and feature ablations.
- Does the Financial PhraseBank all-agreement partition support a 92.7% real-world bank accuracy claim? **No**: 61 negative examples, one training seed, same-domain grouped news sentences, no temporal split.
- Demonstrate the oracle leakage distinction in `scripts/evaluate_arithmetic.py`: the script receives gold programs; the deployed API does **not**.
- What does exact quote checking exclude? Model-generated unsupported paraphrases; not ambiguous, cherry-picked, out-of-context or outdated cited facts.
- How would you benchmark independent financial numerical question answering? Freeze a completely new report-separated set, annotate gold arithmetic and citation faithfulness, run the *actual* configured generator, score exact numeric/units and evidence entailment, and document abstentions/costs.
- Where is the remote CI URL? **Not available until publication and a passing workflow run**.

## Methodology boundaries

This is an AI-assisted engineering/research prototype. The documented experiments and original logs provide reproducibility evidence, not proof of the candidate's personal code authorship, long-term operations, or production security. Public review should focus on demonstrated understanding, error discovery, independent reproduction, honest baseline comparisons, and concrete fixes.
