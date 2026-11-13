"""Freeze the blind FinQA benchmark BEFORE any end-to-end system development.

Selection rule (structural only; no question, answer or evidence content is read
to decide membership): every question of the official FinQA ``test.json`` whose
company/report-year does not occur in the bundled ``finqa-dev.json`` snapshot.
Development data for the numerical pipeline must exclude every report that
occurs in this blind set (see ``financial_research/blind.py``).

Upstream model outputs (``model_input``, ``table_retrieved`` ...) are dropped.
The script is deterministic: rerunning it against the pinned upstream blob must
reproduce the committed SHA-256 exactly, otherwise it refuses to write.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json'
QA_FIELDS = ('question', 'answer', 'exe_ans', 'program', 'gold_inds')


def report(filename: str) -> str:
    return '/'.join(filename.split('/')[:2])


def build(test_rows: list[dict], dev_rows: list[dict]) -> list[dict]:
    seen = {report(r['filename']) for r in dev_rows}
    rows = []
    for r in test_rows:
        if report(r['filename']) in seen:
            continue
        rows.append({'id': r['id'], 'filename': r['filename'], 'pre_text': r['pre_text'],
                     'post_text': r['post_text'], 'table': r['table'],
                     'qa': {k: r['qa'][k] for k in QA_FIELDS}})
    return sorted(rows, key=lambda r: r['id'])


def main(path: str | None = None) -> None:
    if path:
        raw = Path(path).read_bytes()
    else:
        import httpx
        raw = httpx.get(SOURCE, timeout=120, follow_redirects=True).raise_for_status().content
    blob = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
    dev = json.loads((ROOT / 'data/finqa-dev.json').read_text())
    rows = build(json.loads(raw), dev)
    canonical = json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    manifest_path = ROOT / 'data/blind/FREEZE.json'
    if manifest_path.exists():
        frozen = json.loads(manifest_path.read_text())
        if (frozen['blind_sha256'], frozen['upstream_blob_sha']) != (digest, blob):
            sys.exit('Refusing to overwrite: rebuilt blind set differs from the frozen manifest.')
        (ROOT / 'data/blind/finqa-blind-test.json').write_bytes(canonical)
        print('Blind set reproduced bit-identically:', digest)
        return
    (ROOT / 'data/blind/finqa-blind-test.json').write_bytes(canonical)
    manifest = {
        'name': 'FinQA report-separated blind test', 'source_url': SOURCE,
        'upstream_blob_sha': blob, 'upstream_sha256': hashlib.sha256(raw).hexdigest(),
        'blind_sha256': digest, 'questions': len(rows),
        'pages': len({r['filename'] for r in rows}),
        'reports': len({report(r['filename']) for r in rows}),
        'selection_rule': 'official FinQA test.json rows whose company/report-year is absent from data/finqa-dev.json',
        'development_exclusion': 'any FinQA train.json row sharing a company/report-year with this set is excluded from development',
        'frozen_before': 'any end-to-end numerical QA code, neural-retrieval experiment or tuning existed in this repository',
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
