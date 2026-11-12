# Research portfolio: evidence-backed CV wording

**Financial Document Research & Analysis**  
Python · PyTorch/LoRA · scikit-learn · FastAPI · retrieval evaluation · source auditing

- Fine-tuned a DistilBERT LoRA financial-sentiment classifier (92.7% macro-F1 vs 83.8% TF-IDF; 452 held-out near-duplicate-grouped sentences); published single-seed uncertainty and error limitations.
- Benchmarked BM25, LSA, hybrid search and a trained seven-feature reranker against FinQA retrieval labels (162 report-separated test questions); BM25 reached 65.0% Recall@5, and the reranker difference was statistically unclear under page-cluster bootstrap.
- Built a local evidence-linked financial-report research tool with PDF ingestion, immutable-source citation checks, a bounded financial arithmetic interpreter, audit logs, and adversarial regression tests.

**Important:** the safe numeric interpreter verifies *provided* arithmetic programs and cited literals; its gold-program FinQA audit is not independent numerical QA. Optional generated answers have only mocked-provider coverage and exact-excerpt validation. Do not claim banking deployment, verified end-to-end FinQA answers, live-model evaluation, or multi-seed training.

Reproduce the frozen tests and study the source code and failures before using this project in an interview. The implementation included AI coding assistance; describe the work accurately if asked.
