# Model card — retrieval relevance reranker

**Purpose:** reorder retrieved report evidence for financial questions. It is not an investment model or a numerical reasoning system.

**Model:** scikit-learn logistic regression, C=1, balanced class weights, max_iter=1000, seed=42. Seven engineered features: saturated BM25, LSA cosine, query token coverage, numeric overlap, table indicator, query bigram coverage and query density. Probabilities are uncalibrated relevance scores, not confidence in answer truth.

**Training:** 475 queries, 8,888 candidate pairs, 821 relevant pairs. Training candidates come from page-scoped hybrid search (top 20), supplemented with any missing annotated positives. Supervision comes only from the training partition's FinQA gold IDs. At inference, the classifier rescores up to 40 RRF candidates. Negative sampling is therefore easier than collection-wide inference and may cause distribution shift.

**Partition:** SHA-256 of company/report-year modulo 10: buckets 0–5 train, 6–7 validation, 8–9 test. This keeps pages within a report together. It does not keep companies across years together. Source collection: 299 unique pages, 224 report-years. Original upstream split: FinQA dev. Local test: 162 questions.

**Unsupervised representation:** TF-IDF unigrams/bigrams, English stopwords, sublinear TF, maximum 24,000 vocabulary entries. Truncated SVD, 64 components, seed=42; normalized vectors. Vocabulary and LSA are fitted on the full searchable collection. No held-out query labels enter this fit.

**Observed behavior:** collection test Recall@5 is 62.55%, versus hybrid 39.25% and BM25 64.97%. Page test Recall@5 is 82.51%, versus hybrid 79.89% and BM25 85.75%. The model does not establish superiority over BM25. There are no confidence intervals or significance claims.

**Failure modes:** entity ambiguity, stale historic information, rare numeric phrases, financial tables with lost context, false positives on broad questions, weak low-rank semantic baseline, and incomplete candidate recall. A reranker cannot recover evidence missing from its candidate set. Uploading new sources changes IDF and LSA geometry; the original weights can be used experimentally but benchmark numbers no longer describe the changed index.

**Deployment:** serialized as transparent JSON weights in `data/reranker.json`; no untrusted pickle loading. Training corpus fingerprint and hashed training IDs recorded. No online learning or automatic retraining.

**No claims:** LLM fine-tuning, neural cross-encoder training, guaranteed financial correctness, production banking suitability, verified absence of PII leakage, or current market coverage.


The separate DistilBERT sentiment model is documented in `SENTIMENT_MODEL_CARD.md`. This card describes retrieval only.
