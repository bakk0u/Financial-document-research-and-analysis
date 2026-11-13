"""Multi-seed sentiment training and evaluation on harder, shifted data.

Same data split, architecture and hyperparameters as ``train_sentiment.py``; only the
random seed changes. The shipped adapter is not modified. Nothing is tuned here: the
two harder sets (``scripts/prepare_sentiment_shift.py``) are evaluation-only.

Reports mean, standard deviation and range across seeds for macro-F1 and per-class F1 on
(a) the original held-out test split, (b) contested PhraseBank sentences and (c) financial
tweets from a later period. The deterministic TF-IDF baseline is scored on the same sets.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from financial_research.corpus import DATA, ROOT
from financial_research.sentiment import LABELS, ensure_base, predict_baseline

SEEDS = [42, 7, 123, 2026, 31337]
SETS = {'in_distribution_test': 'PhraseBank all-agreement held-out split (template-grouped)',
        'phrasebank_contested': 'PhraseBank sentences without full annotator agreement',
        'twitter_financial_news': 'Financial tweets, later period, different companies and medium'}


def load_sets():
    splits = json.loads((DATA / 'sentiment-splits.json').read_text())
    parts = {s: [r for r in splits if r['split'] == s] for s in ('train', 'validation', 'test')}
    shifted = {name: [{'text': r['text'], 'label': LABELS.index(r['label'])}
                      for r in json.loads((DATA / f'sentiment-shift/{file}.json').read_text())]
               for name, file in (('phrasebank_contested', 'phrasebank-contested'),
                                  ('twitter_financial_news', 'twitter-financial-news'))}
    return parts, {'in_distribution_test': parts['test'], **shifted}


def scores(truth, pred):
    per_class = f1_score(truth, pred, average=None, labels=[0, 1, 2], zero_division=0)
    return {'macro_f1': float(f1_score(truth, pred, average='macro', labels=[0, 1, 2], zero_division=0)),
            'accuracy': float(accuracy_score(truth, pred)),
            **{f'f1_{label}': float(per_class[i]) for i, label in enumerate(LABELS)}}


def train_once(seed, parts, eval_sets, base, tokenizer):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=3, id2label=dict(enumerate(LABELS)), label2id={s: i for i, s in enumerate(LABELS)}, local_files_only=True)
    model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=.1,
                                             target_modules=['q_lin', 'v_lin'], modules_to_save=['pre_classifier', 'classifier']))
    encode = lambda rows: tokenizer([r['text'] for r in rows], padding='max_length', truncation=True, max_length=96, return_tensors='pt')
    train, y_train = encode(parts['train']), torch.tensor([r['label'] for r in parts['train']])
    weights = torch.tensor(len(y_train) / (3 * np.bincount(y_train.numpy(), minlength=3)), dtype=torch.float32)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=2e-4, weight_decay=.01)
    loader = DataLoader(list(range(len(y_train))), batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(seed))

    def predict(rows):
        model.eval(); out = []
        encoded = encode(rows)
        with torch.inference_mode():
            for start in range(0, len(rows), 64):
                out.append(model(**{k: v[start:start + 64] for k, v in encoded.items()}).logits.argmax(1).numpy())
        return np.concatenate(out)

    best, best_epoch, best_predictions = -1.0, 0, None
    val_truth = [r['label'] for r in parts['validation']]
    for epoch in range(1, 4):
        model.train()
        for indices in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(**{k: v[indices] for k, v in train.items()}).logits, y_train[indices])
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1); optimizer.step()
        score = f1_score(val_truth, predict(parts['validation']), average='macro')
        if score > best:        # checkpoint selection on validation only, as in the shipped run
            best, best_epoch = float(score), epoch
            best_predictions = {name: predict(rows) for name, rows in eval_sets.items()}
        print(f'seed={seed} epoch={epoch} validation_macro_f1={score:.4f}', flush=True)
    return {'seed': seed, 'selected_epoch': best_epoch, 'validation_macro_f1': best,
            'sets': {name: scores([r['label'] for r in eval_sets[name]], pred) for name, pred in best_predictions.items()}}


def main():
    started = time.perf_counter()
    torch.set_num_threads(4)
    parts, eval_sets = load_sets()
    base = ensure_base()
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    runs = [train_once(seed, parts, eval_sets, base, tokenizer) for seed in SEEDS]
    metrics = ['macro_f1', 'accuracy'] + [f'f1_{l}' for l in LABELS]
    summary = {}
    for name, rows in eval_sets.items():
        truth = [r['label'] for r in rows]
        values = {m: np.array([run['sets'][name][m] for run in runs]) for m in metrics}
        summary[name] = {
            'description': SETS[name], 'examples': len(rows),
            'class_counts': {l: int(sum(t == i for t in truth)) for i, l in enumerate(LABELS)},
            'distilbert_lora': {m: {'mean': round(float(v.mean()), 4), 'sd': round(float(v.std(ddof=1)), 4),
                                    'min': round(float(v.min()), 4), 'max': round(float(v.max()), 4)} for m, v in values.items()},
            'tfidf_baseline': {m: round(v, 4) for m, v in scores(truth, predict_baseline([r['text'] for r in rows]).argmax(1)).items()},
            'majority_macro_f1': round(float(f1_score(truth, [1] * len(truth), average='macro', labels=[0, 1, 2], zero_division=0)), 4)}
    report = {'seeds': SEEDS, 'runs': runs, 'summary': summary, 'elapsed_seconds': round(time.perf_counter() - started, 1),
              'protocol': 'Identical split and hyperparameters to the shipped run; only the seed varies. Shifted sets are evaluation-only; no tuning followed these results.',
              'limits': ['Five seeds describe optimisation variance, not variance across datasets or annotators.',
                         'Contested sentences have majority-vote labels that annotators themselves disputed; an accuracy ceiling below 100% is expected.',
                         'Tweet labels (bearish/bullish/neutral) are mapped to negative/positive/neutral; the label concepts are close but not identical.',
                         'PhraseBank has no company or date fields, so strict company- or time-separated splits of it are impossible; the tweet set supplies the time and company shift instead.']}
    (ROOT / 'reports/sentiment-generalization.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Sentiment generalization: five seeds, three test sets', '',
             'Same split and hyperparameters as the shipped model; only the random seed changes. The two harder sets were never used for training, selection or calibration. Values are mean ± sample standard deviation across seeds (min to max).', '']
    for name, s in summary.items():
        d = s['distilbert_lora']
        lines += [f"## {s['description']}", '',
                  f"{s['examples']:,} sentences: {s['class_counts']['negative']} negative, {s['class_counts']['neutral']} neutral, {s['class_counts']['positive']} positive.", '',
                  '| Metric | DistilBERT + LoRA (5 seeds) | TF-IDF baseline |', '|---|---:|---:|']
        for m, label in (('macro_f1', 'Macro-F1'), ('accuracy', 'Accuracy'), ('f1_negative', 'F1 negative'), ('f1_neutral', 'F1 neutral'), ('f1_positive', 'F1 positive')):
            lines.append(f"| {label} | {100 * d[m]['mean']:.1f} ± {100 * d[m]['sd']:.1f} ({100 * d[m]['min']:.1f} to {100 * d[m]['max']:.1f}) | {100 * s['tfidf_baseline'][m]:.1f} |")
        lines += ['', f"Majority-class macro-F1: {100 * s['majority_macro_f1']:.1f}.", '']
    lines += ['## Limits', ''] + ['- ' + x for x in report['limits']]
    (ROOT / 'reports/SENTIMENT_GENERALIZATION.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
