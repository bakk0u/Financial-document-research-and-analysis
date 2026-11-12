# Financial sentiment: measured evaluation

| Split | Model | Accuracy | Macro-F1 | ECE | NLL |
|---|---|---:|---:|---:|---:|
| validation | majority | 0.6128 | 0.2533 | 0.0012 | 0.9178 |
| validation | tfidf | 0.8695 | 0.8184 | 0.1888 | 0.4860 |
| validation | distilbert_raw | 0.9580 | 0.9449 | 0.0204 | 0.1402 |
| validation | distilbert_calibrated | 0.9580 | 0.9449 | 0.0105 | 0.1381 |
| test | majority | 0.6128 | 0.2533 | 0.0012 | 0.9178 |
| test | tfidf | 0.8761 | 0.8380 | 0.2050 | 0.5039 |
| test | distilbert_raw | 0.9447 | 0.9274 | 0.0289 | 0.1242 |
| test | distilbert_calibrated | 0.9447 | 0.9274 | 0.0210 | 0.1205 |

## Training
Trainable parameters: 740,355. Selected epoch: 2 of 3. Temperature: 0.879.

## Limitations
- Financial PhraseBank all-agreement subset; custom group split, not an official leaderboard.
- Near-duplicate grouping uses character n-gram cosine >=0.85 with numbers normalized; it does not guarantee event/company separation.
- No publication timestamps or source-event IDs: this is not a temporal market-prediction evaluation.
- News-sentence sentiment may not transfer to report paragraphs. Sentiment is not future return or investment advice.
- Only one seed and one three-epoch run; no statistical superiority claim.
- Base model pretraining overlap with public financial sentences cannot be independently ruled out.
- Calibration selected on validation; held-out calibration is measured, not guaranteed.
