"""Build the two harder sentiment evaluation sets from pinned public sources.

1. ``phrasebank-contested``: Financial PhraseBank sentences on which annotators did NOT
   all agree (the 50%-agreement file minus the all-agreement file used for training),
   labelled by majority vote, with near-duplicates of any training/validation sentence
   removed. Same source and era as training, but ambiguous by construction.
2. ``twitter-financial-news``: the validation split of zeroshot/twitter-financial-news-sentiment
   (MIT). Different medium, different companies and a later period than PhraseBank.
   Bearish/Bullish/Neutral are mapped to negative/positive/neutral.

Neither set is used for training, model selection or calibration.
"""
from __future__ import annotations
import csv
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'data/sentiment-shift'
TWITTER = {'repo': 'zeroshot/twitter-financial-news-sentiment', 'revision': 'ccbe24de388e287beb92dd393a335c376b350ac3',
           'file': 'sent_valid.csv', 'sha256': '31ea8274c98b52212daa68f2fb962811bf1913c1efe51b99765c8e2311806de6',
           'license': 'MIT', 'label_map': {'0': 'negative', '1': 'positive', '2': 'neutral'}}


def fetch(url: str) -> bytes:
    import httpx
    return httpx.get(url, timeout=180, follow_redirects=True).raise_for_status().content


def canonical(text: str) -> str:
    return ' '.join(re.findall(r'[a-z0-9]+', text.lower()))


def main(zip_path: str | None = None, twitter_path: str | None = None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    meta = json.loads((ROOT / 'data/phrasebank_upstream.json').read_text())
    raw_zip = Path(zip_path).read_bytes() if zip_path else fetch(meta['source_url'])
    assert hashlib.sha256(raw_zip).hexdigest() == meta['zip_sha256'], 'PhraseBank archive hash mismatch'
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as z:
        fifty = z.read('FinancialPhraseBank-v1.0/Sentences_50Agree.txt').decode('latin-1').splitlines()
    splits = json.loads((ROOT / 'data/sentiment-splits.json').read_text())
    known = {canonical(r['text']) for r in splits}
    rows, seen = [], set()
    for line in fifty:
        text, label = line.rsplit('@', 1)
        key = canonical(text)
        if key in known or key in seen:
            continue
        seen.add(key)
        rows.append({'text': text.strip(), 'label': label.strip()})
    # Remove near-duplicates of sentences the model was fitted or selected on (same rule as the split).
    norm = lambda t: re.sub(r'\d+(?:[.,]\d+)*', '#', t.lower())
    seen_texts = [norm(r['text']) for r in splits if r['split'] != 'test']
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=1).fit(seen_texts + [norm(r['text']) for r in rows])
    similarity = (vectorizer.transform([norm(r['text']) for r in rows]) @ vectorizer.transform(seen_texts).T).max(axis=1).toarray().ravel()
    kept = [r for r, s in zip(rows, similarity) if s < .85]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'phrasebank-contested.json').write_text(json.dumps(kept, indent=1, ensure_ascii=False) + '\n')

    raw_tw = Path(twitter_path).read_bytes() if twitter_path else fetch(
        f"https://huggingface.co/datasets/{TWITTER['repo']}/resolve/{TWITTER['revision']}/{TWITTER['file']}")
    assert hashlib.sha256(raw_tw).hexdigest() == TWITTER['sha256'], 'Twitter snapshot hash mismatch'
    tweets = [{'text': r['text'], 'label': TWITTER['label_map'][r['label']]}
              for r in csv.DictReader(io.StringIO(raw_tw.decode('utf-8')))]
    (OUT / 'twitter-financial-news.json').write_text(json.dumps(tweets, indent=1, ensure_ascii=False) + '\n')
    count = lambda items: {l: sum(r['label'] == l for r in items) for l in ('negative', 'neutral', 'positive')}
    provenance = {'phrasebank_contested': {'source': meta['source_url'], 'zip_sha256': meta['zip_sha256'],
                                           'rule': '50%-agreement sentences absent from the all-agreement file; near-duplicates (char n-gram cosine >= 0.85) of training/validation sentences removed',
                                           'candidates': len(rows), 'kept': len(kept), 'class_counts': count(kept), 'license': 'CC BY-NC-SA 3.0'},
                  'twitter_financial_news': {**TWITTER, 'rows': len(tweets), 'class_counts': count(tweets)}}
    (OUT / 'PROVENANCE.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main(*sys.argv[1:3])
