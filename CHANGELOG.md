## Unreleased verification improvements (13 November 2026)

- Force LF checkouts on Windows for reproducible SHA-256 integrity checks (`.gitattributes`).
- Benchmark the real BM25-to-extractive-quotation pipeline on the frozen 162-question FinQA split; report gold-source selection rates, page-cluster intervals, and failure cases without claiming numerical answer accuracy.
- Run the extractive audit in Linux CI with output-drift checks, and run the Docker smoke job independently of the cross-platform Python matrix.
- Surface extractive evaluation results in the FastAPI API and Evaluation tab.
- Document actual previous GitHub Actions failure and the exact limitations of extractive text-selection scores.

# Changelog

## 1.1.0 — 2026-11-12

- Added reproducible page/report-clustered, paired bootstrap estimates for retrieval and grouped, frozen-prediction uncertainty for sentiment.
- Implemented a finite, whitelisted arithmetic language with source-literal validation and traceable operation steps; added a cited-calculation API and UI workbench.
- Audited FinQA gold arithmetic programs separately from natural-language question answering, with explicit oracle-input and annotation mismatch disclaimers.
- Rejected optional model claims with paraphrased statements even when the quote exists, lowering unsupported-generation risk.
- Regressed the three original prompt-screening bypasses; retained the limitations of rule-based screening.
- Surfaced statistical and numerical diagnostics in the Evaluation view, extended tests and interview documentation.
- Deployment and live-provider checks remain pending external credentials/infrastructure; no unverified check is described as passed.

# Changelog

## 1.0.0 — 2026-11-11

- Financial evidence research UI, source inspector, evaluation, source library, methodology and audit views.
- Four measured retrieval methods and a supervised pointwise relevance reranker.
- Reproducible FinQA snapshot with gold-label exclusion and report-separated query partitions.
- PDF ingestion, persisted uploads, bounded query cache and local audit verification.
- Optional strict-schema generated answers with citation and exact-quote validation.
- Automated tests, bounded adversarial evaluation and a clean-environment installation check.
- Complete methodological limitations, model card, CV wording and interview study guide.

At original release, live provider generation and cross-platform runtime were unverified; see docs/VERIFICATION.md.

## Completed financial NLP integration — 2026-11-11

- Replaced the product branding with the descriptive title Financial Document Research & Analysis.
- Added trained DistilBERT LoRA adapters and a three-class financial sentiment head.
- Added a deduplicated, near-duplicate-grouped Financial PhraseBank evaluation, lexical and majority baselines, validation checkpoint selection and validation-only temperature scaling.
- Added local sentiment inference, integration from retrieved evidence, audit events and measured comparison in the interface.
- Included trained adapters, dataset attribution, model setup, training code, evaluation artifacts and interview documentation.
