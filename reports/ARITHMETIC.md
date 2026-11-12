# Numerical reasoning: oracle-program audit

This is a **gold-program diagnostic**, *not* end-to-end QA. The evaluator receives the answer program given by FinQA; the live application does not read these programs.

| Check | Count | Denominator |
|---|---:|---:|
| supported programs | 155 | 162 |
| program agrees with exe ans | 155 | 162 |
| program agrees with reported answer | 96 | 162 |
| oracle gold evidence operand coverage | 154 | 162 |
| collection bm25 operand coverage | 89 | 162 |
| collection reranked operand coverage | 85 | 162 |
| page bm25 operand coverage | 124 | 162 |

## Interpretation

- Arithmetic is executed by a bounded, pure whitelist DSL; no Python `eval`, no arbitrary code, no network.
- Operand coverage checks whether each literal number occurs in at least one cited passage; it does **not** confirm cell/column attribution, sign, scale or unit consistency.
- FinQA has internally inconsistent example annotations; report disagreements and retain examples. Do not silently overwrite the benchmark labels.
- A production-worthy numerical QA system must select operands and operations from the *question* without oracle programs; this audit cannot establish that capability.
- Real answer-quality and faithfulness evaluation for model-generated answers remains unverified without real generated model outputs.
