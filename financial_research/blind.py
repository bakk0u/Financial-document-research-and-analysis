"""Data access for development and for the frozen blind benchmark.

Development rows and blind rows share a format but never a code path that could
hand labels to the system: ``corpus_from_rows`` indexes raw report text only, and
``labels`` are returned separately for training (development) or scoring (blind).
"""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path

from .corpus import DATA, Chunk, split_for, tokens

BLIND = DATA / 'blind/finqa-blind-test.json'
FREEZE = DATA / 'blind/FREEZE.json'
TRAIN_URL = 'https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json'
TRAIN_SHA256 = '49f237eb9779b569473b26b08048867d04635a7cc39ad6a7a5664c55bb428db6'


def report_of(filename: str) -> str:
    return '/'.join(filename.split('/')[:2])


def corpus_from_rows(rows: list[dict]) -> tuple[list[Chunk], list[dict]]:
    """Same serialisation as ``corpus.load_finqa``; returns (chunks, labelled questions)."""
    chunks: dict[str, Chunk] = {}
    questions = []
    for row in rows:
        doc = row['filename']
        company, year = doc.split('/')[0], doc.split('/')[1]
        found = re.search(r'page_(\d+)', doc)
        page = found.group(1) if found else '?'
        items = [(f'text_{i}', text, 'text') for i, text in enumerate(row['pre_text'] + row['post_text'])
                 if len(tokens(text)) >= 1]
        table = row['table']
        if table:
            header = table[0]
            for i, values in enumerate(table):
                text = ' ; '.join(f'{header[j] if j < len(header) else j}: {v}' for j, v in enumerate(values))
                items.append((f'table_{i}', text, 'table'))
        for locator, text, kind in items:
            ident = f'{doc}::{locator}'
            chunks[ident] = Chunk(ident, doc, company, year, page, locator, text, kind, 'FinQA')
        qa = row['qa']
        questions.append({'id': row['id'], 'question': qa['question'], 'document': doc,
                          'gold': [f'{doc}::{k}' for k in qa['gold_inds'] if f'{doc}::{k}' in chunks],
                          'program': qa['program'], 'exe_ans': qa['exe_ans'], 'answer': str(qa['answer'])})
    return sorted(chunks.values(), key=lambda c: c.id), questions


def load_blind() -> list[dict]:
    raw = BLIND.read_bytes()
    if hashlib.sha256(raw).hexdigest() != json.loads(FREEZE.read_text())['blind_sha256']:
        raise ValueError('Blind benchmark does not match its frozen hash.')
    return json.loads(raw)


def load_dev_rows() -> list[dict]:
    return json.loads((DATA / 'finqa-dev.json').read_text())


def development_pool(train_path: Path) -> tuple[list[dict], list[dict], dict]:
    """(fit rows, validation rows, accounting). Blind reports are excluded from both;
    validation is the repository's dev ``validation`` split and its reports are excluded from fit."""
    raw = train_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != TRAIN_SHA256:
        raise ValueError('FinQA train.json differs from the pinned upstream snapshot.')
    blind_reports = {report_of(r['filename']) for r in load_blind()}
    dev = load_dev_rows()
    held_reports = {report_of(r['filename']) for r in dev if split_for(r['filename']) != 'train'}
    train = json.loads(raw)
    fit = [r for r in train if report_of(r['filename']) not in blind_reports | held_reports]
    fit += [r for r in dev if split_for(r['filename']) == 'train']
    validation = [r for r in dev if split_for(r['filename']) == 'validation']
    assert not {report_of(r['filename']) for r in fit} & (blind_reports | held_reports)
    accounting = {'upstream_train_rows': len(train), 'fit_rows': len(fit), 'validation_rows': len(validation),
                  'dropped_for_blind_report_overlap': sum(report_of(r['filename']) in blind_reports for r in train),
                  'dropped_for_dev_holdout_overlap': sum(report_of(r['filename']) in held_reports - blind_reports for r in train),
                  'fit_reports': len({report_of(r['filename']) for r in fit})}
    return fit, validation, accounting


def numeric_correct(value, exe_ans, *, lenient: bool = False) -> bool:
    """Protocol metric: strict by default; ``lenient`` also accepts a factor of 100."""
    if isinstance(value, str) or isinstance(exe_ans, str):
        return str(value).strip().lower() == str(exe_ans).strip().lower()
    try:
        gold = round(float(exe_ans), 5)
        pred = round(float(value), 5)
    except (TypeError, ValueError):
        return False
    scales = (1, 100, .01) if lenient else (1,)
    return any(abs(pred * s - gold) <= max(5e-4, 1e-3 * abs(gold)) for s in scales)
