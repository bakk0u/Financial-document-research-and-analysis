# Seven-feature logistic reranker: exploratory validation ablations

**Validation only**; no new frozen-test model comparison. The baseline is a supervised seven-feature classifier, not a cross-encoder or fine-tuned neural retriever.

| Feature set | Page Recall@5 | Collection Recall@5 |
|---|---:|---:|
| all seven features | 85.7% | 53.8% |
| without lsa cosine | 86.2% | 52.4% |
| without exact number coverage | 86.0% | 52.5% |
| without table indicator | 82.5% | 48.0% |
| without bigram coverage | 87.2% | 47.9% |
| without lexical query coverage | 85.5% | 43.2% |

## Design

- Refit the exact same training query–passage pairs with a fixed seed, class weights, C=1, and identical retrieval candidate sets. Excluded features have zero coefficient; all other features retain their original positions.
- Evaluate on the previously designated report-separated validation queries; do not use the inspected test set to select a new variant.
- Correlated features make leave-one-out ablations an imperfect causal explanation. Different negative sampling or retrieval candidate pools can change the conclusions.
- The number of training examples and all measured values are in `ablation.json`; keep weak/negative ablations visible.
