# Model card — financial sentence sentiment

## Intended use

Noncommercial research and portfolio demonstration of supervised financial NLP. Input: one English financial sentence. Output: negative, neutral or positive sentiment and probabilities. This is not a buy/sell recommender, asset valuation model, client-flow predictor or return forecast.

## Model and training

- Base: `distilbert/distilbert-base-uncased`, pinned upstream revision recorded in `data/base_model_upstream.json`. General-English masked-language pretraining; not an upstream financial-sentiment classifier.
- Adaptation: PEFT LoRA, rank 8, alpha 16, dropout 0.1, attention `q_lin` and `v_lin`; the `pre_classifier` and `classifier` heads are trained as well.
- Trainable parameters: 740,355. Original DistilBERT weights remain frozen. This is parameter-efficient transformer fine-tuning, not full-weight fine-tuning.
- Optimizer: AdamW, learning rate 0.0002, weight decay 0.01, class-balanced cross entropy, gradient clipping 1.0, batch 16, maximum 96 tokens, three epochs, random seed 42.
- Checkpoint selection: best validation macro-F1. Temperature scaling: fixed 60-point geometric grid from 0.5 to 4.0, minimum validation negative log likelihood. Positive temperature scaling cannot change predicted classes.
- Artifacts: trained adapters, trained classification head and tokenizer files in `data/sentiment-adapter/`; base downloaded separately from a pinned public revision. No custom remote code or pickle weights are loaded.

## Dataset and split

Financial PhraseBank, highest annotator agreement subset: 2,264 original sentences; 2,259 after exact normalized deduplication. Sentences are annotated from a financial perspective by humans. Negative, neutral and positive labels are original annotations, not model-generated supervision.

Near-duplicate grouping uses character-within-word 3–5-gram TF-IDF cosine at least 0.85 after numeric normalization. Connected components are kept together with seeded stratified group partitioning. The exact split file is `data/sentiment-splits.json`: 1,355 train / 452 validation / 452 test. This is a custom split, not an official benchmark score. Grouping is unsupervised; all sentence text is used to establish groups before splitting, but held-out labels are not used to fit model weights.

Stratification uses labels to create balanced partitions; it is separate from model training. This does not prove company or news-event separation. No publication timestamps or source-event IDs are available for a chronological study. Base pretraining overlap with publicly available sentences cannot independently be excluded.

## Evaluation

The test split is evaluated only after validation checkpoint selection and calibration. Compared methods: majority predictor, TF-IDF/logistic regression, raw DistilBERT adaptation and calibrated DistilBERT adaptation. Outputs include accuracy, macro-F1, per-class precision/recall/F1, confusion matrices, NLL, 10-bin expected calibration error and multiclass Brier score. See `reports/SENTIMENT_EVALUATION.md` and machine-readable predictions.

A single seed and single three-epoch run do not establish statistical superiority. Scores are classification metrics, not forecasting performance or RAG answer quality. The numerical evaluation is kept separate from the retrieval benchmark.

## Limitations and operation

News-sentence training does not establish accuracy on complete report paragraphs, chats or contemporary market language. At most 96 tokenizer tokens are read; longer inputs are truncated. Calibration is evaluated on the held-out set and is not guaranteed on arbitrary input. Probabilities are not confidence that the source statement is factually true.

All sentiment inference is local. First use may download the pinned base weights (about 268 MB); input sentences are not sent to Hugging Face for inference. Query text hashes, model identity, predicted labels and latency are logged locally; raw sentences are not audit-log fields.

## Licenses and attribution

Base model: Apache-2.0. Financial PhraseBank: CC BY-NC-SA 3.0, attribution and original license retained under `data/PHRASEBANK_*`. Dataset-derived artifacts are distributed for noncommercial research; source-code MIT licensing does not override dataset terms. Refer to the upstream terms before other use.

## Observed qualitative failure

The UI example about increasing operating profit/revenue and declining costs was predicted negative with 83.0% probability. It is retained in `reports/QUALITATIVE_CHECKS.md`. No weights or checkpoint selection changed after this observation. A strong aggregate score does not eliminate simple failures.
