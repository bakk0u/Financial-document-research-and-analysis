# Financial sentence sentiment — trained DistilBERT adapters

Locally trained LoRA adapters and classification-head weights for `distilbert/distilbert-base-uncased`. The original base model is downloaded separately from the pinned revision in `data/base_model_upstream.json`.

Training: Financial PhraseBank all-agreement subset, deduplicated and group-partitioned. Selected epoch: 2 of 3, chosen on validation macro-F1. Test: 452 sentences, macro-F1 0.92736, accuracy 0.94469. TF-IDF baseline test macro-F1: 0.83803.

These are classification scores on a custom split, not trading performance. Full training, calibration, limitations and data terms are in `docs/SENTIMENT_MODEL_CARD.md` and `reports/SENTIMENT_EVALUATION.md`.

Financial PhraseBank is CC BY-NC-SA 3.0. Dataset-derived artifacts are distributed for noncommercial research; source-code MIT terms do not override upstream terms. The base DistilBERT model is Apache-2.0.
