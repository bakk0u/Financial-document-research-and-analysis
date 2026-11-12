# Interview guide

## Your 60-second explanation

“I built Financial Document Research & Analysis to study how retrieval choices affect financial document research. It indexes report passages and table rows, compares BM25, LSA vectors and hybrid retrieval, then trains a small supervised reranker on human-annotated evidence. I separated reports across training and evaluation, exposed every citation in a research interface and kept failure cases visible. The reranker improved hybrid recall, but BM25 remained strongest on held-out recall. That result shaped my understanding of when extra model complexity actually helps.”

Explain AI assistance honestly if asked. Do not claim to have manually implemented or independently mastered parts you have not reviewed.

## Learn in this order

1. Follow one page through `corpus.py`: source fields, table serialization, ID assignment and label exclusion.
2. Follow one query through `Engine.scores`: BM25 postings, TF-IDF query projection, cosine similarity.
3. Work through reciprocal rank fusion on two lists by hand: score = sum(1 / (60 + rank)).
4. Inspect the seven features and JSON weights; explain the linear log-odds and sigmoid.
5. Read the partition function and metric definitions. Calculate Recall@5 and MRR@5 on a small example.
6. Run a query in the UI, inspect evidence, switch retrieval methods and explain a changed ordering.
7. Read the provider validator and adversarial report. Explain exactly what is and is not verified.
8. Run tests and reproduce the benchmark before claiming ownership of the results.

## Questions you should be able to answer

**Why BM25?** Financial queries often reuse exact terms and numbers. BM25 weights rare matching terms and normalizes document length. It is a serious baseline, not a token comparison to make ML look good.

**How is LSA different from transformer embeddings?** LSA is a linear low-rank approximation to the corpus term matrix. It captures co-occurrence patterns, but has no language-model pretraining. This project does not call it a neural embedding model.

**Why can hybrid perform worse?** Equal RRF weights allow a weaker vector ranking to displace good lexical results. Combining methods does not guarantee improvement.

**What exactly was trained?** A seven-feature logistic relevance classifier. Separately, DistilBERT attention adapters and a sentiment head are fine-tuned on financial sentences; the retrieval embeddings and answer-generating LLM are not fine-tuned. Training data consists of labeled query–passage pairs, not raw filings alone.

**Why not split rows randomly?** Nearby passages and multiple questions from the same source could leak across splits. Report-year partitioning reduces this overlap. The same company can still occur in a different year.

**Is indexing held-out documents leakage?** This is transductive retrieval: the source collection is available for searching, but held-out question labels are not available for training. That setting must be disclosed. It is not a test of generalizing an unchanged representation to unseen documents.

**Why train on page-level candidates?** It gives clear local hard negatives and a manageable fit. It also creates candidate distribution shift for collection-wide search, a real limitation documented in the model card.

**Does a valid citation prove correctness?** No. It proves the ID exists and the quote occurs in the retrieved source. The statement could still misinterpret the quote, omit context or perform invalid arithmetic.

**What does the audit chain protect?** Each event hashes the previous digest plus canonical JSON. Editing a retained event breaks verification. An owner can still rewrite the whole database or delete its tail; there is no external anchor.

**Why did your ML model lose to BM25?** Inspect candidate recall, query scope, representation quality, negative sampling and metric choice. Do not invent an explanation without an experiment. The observed fact is sufficient: more complexity did not consistently improve recall.

**How would you improve it?** Establish a new evaluation set before tuning. Consider neural embeddings, a neural cross-encoder, collection-wide hard negatives, table-aware parsing and calibrated answerability. These are proposed experiments, not implemented capabilities.

**What would public deployment need?** Authentication, tenant-separated indexes and documents, request quotas, background ingestion, hardened PDF parsing, sensitive-data handling, monitoring and independent security evaluation.

## Exercises before an interview

- Calculate BM25 for two tiny passages and explain IDF and length normalization.
- Change only the requested search method and inspect one result difference.
- Identify a test failure where the reranker misses evidence BM25 finds.
- Show that an invented citation is rejected, then explain why an unrelated valid quote is still a problem.
- Trace a query from frontend to audit and identify every boundary.

Do not memorize the pitch without understanding these steps.

## Financial NLP component

Explain the difference between the two trained models: the relevance reranker predicts whether a passage supports a query; the DistilBERT sentiment classifier predicts positive/neutral/negative tone. They solve different tasks and have separate datasets and evaluation.

**What is LoRA?** A pretrained linear map W is frozen and a low-rank update BA is learned. Here rank 8 adapters are added to attention query and value projections. The classification head is trained too. This saves optimizer memory and reduces the distributed checkpoint size; it does not mean all base-model parameters were updated.

**Why not use an existing FinBERT sentiment checkpoint?** Such a model may already have been trained on Financial PhraseBank. Starting with a general English checkpoint makes the downstream training and evaluation easier to interpret. It still does not prove the base pretraining corpus contains no overlapping text.

**Why macro-F1?** It gives each sentiment class equal weight, exposing failures on the minority negative class. Accuracy alone can reward predicting neutral too often.

**Why group near duplicates?** Repeated sentence templates with changed numbers can make a random split too easy. Character-ngram similarity after numeric normalization creates groups that are kept together. It is a heuristic and cannot substitute for source-event or company metadata.

**What does temperature scaling change?** It rescales logits before softmax. The positive scalar changes probabilities and calibration, but not the argmax class or F1. Its temperature is selected on validation NLL, not test scores.

**Can sentiment predict returns?** This experiment does not test that. It classifies sentence tone. A returns study would need timestamps, entity resolution, chronological validation, execution assumptions and transaction costs.


## New technical-defense questions (revision 1.1)

1. Why use a *paired* cluster bootstrap instead of resampling 162 queries as IID? Reproduce the 50-page and 40-report intervals from `reports/predictions.json` and explain why each resampling unit changes uncertainty. The collection-scope reranker–BM25 interval includes zero; never call its ranking statistically decisive.
2. Why is LoRA’s 92.7% single-seed macro-F1 not a generalization guarantee? Explain the 61 negative test samples, grouped sentences, domain shift, the 424 test template groups and why fixed-prediction bootstrapping does **not** capture training seed variation.
3. Why do the seven-feature validation ablations produce different answers for page and collection retrieval? Use the actual source code in `scripts/ablate_reranker.py` and the six rows in `reports/ABLATION.md`. Name the training-pair labeling strategy and why appending gold positives to candidates is supervised.
4. Walk through `financial_research/arithmetic.py`. Explain why programs are **oracle inputs** in the FinQA diagnostic, what the 89/162 collection-grounding count establishes, what it does not establish, and why source presence does not validate sign, column or scale.
5. Demonstrate a real calculation in the dashboard and then deliberately substitute an invented value. Show the 422 rejection, immutable server-side source lookup and audit trace.
6. Why can a valid source quotation still fail answer quality? What types of unsupported claim do `validate_claims(..., extractive_only=True)` reject, and what semantic errors can still pass?
7. Which checks were actually completed on this revision, on which runtime, and which are only configured in CI? Never say GitHub CI, Docker or live-model generation passed without corresponding job logs / results.

**Ownership exercise:** before listing the project in a competitive interview, rerun the ablation, explain each table row without reading notes, choose one real retrieval failure and analyze why the matched financial value was absent; then implement and test one independent improvement without tuning on the already-inspected held-out test set.
