from __future__ import annotations
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?", text.lower())


@dataclass(frozen=True)
class Chunk:
    id: str
    document: str
    company: str
    year: str
    page: str
    locator: str
    text: str
    kind: str
    source: str


def split_for(document: str) -> str:
    # Split at company/report level, not page level: nearby pages cannot cross splits.
    report = '/'.join(document.split('/')[:2])
    bucket = int(hashlib.sha256(report.encode()).hexdigest()[:8], 16) % 10
    return 'train' if bucket < 6 else 'validation' if bucket < 8 else 'test'


def load_finqa(path: Path = DATA / 'finqa-dev.json') -> tuple[list[Chunk], list[dict]]:
    raw = path.read_bytes()
    provenance = DATA / 'provenance.json'
    if path == DATA / 'finqa-dev.json' and provenance.exists():
        expected = json.loads(provenance.read_text())['snapshot_sha256']
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Bundled data snapshot hash mismatch. Restore it with scripts/download_data.py.')
    rows = json.loads(raw)
    chunks: dict[str, Chunk] = {}
    queries = []
    for row in rows:
        doc = row['filename']
        bits = doc.split('/')
        company, year = bits[0], bits[1]
        page = re.search(r'page_(\d+)', doc)
        page = page.group(1) if page else '?'
        source = 'https://github.com/czyssrs/FinQA/blob/main/dataset/dev.json'
        # Only raw report fields are indexed. qa, model_input, retrieved scores, and
        # gold evidence are not read until after corpus construction for this page.
        evidence = []
        for i, text in enumerate(row['pre_text'] + row['post_text']):
            if len(tokens(text)) >= 1:
                evidence.append((f'text_{i}', text, 'text'))
        table = row['table']
        if table:
            header = table[0]
            for i, values in enumerate(table):
                # Uniform serialization of ALL rows, independent of relevance labels.
                text = ' ; '.join(f'{header[j] if j < len(header) else j}: {v}' for j, v in enumerate(values))
                evidence.append((f'table_{i}', text, 'table'))
        for locator, text, kind in evidence:
            ident = f'{doc}::{locator}'
            chunk = Chunk(ident, doc, company, year, page, locator, text, kind, source)
            if ident in chunks and chunks[ident] != chunk:
                raise ValueError(f'Conflicting source content: {ident}')
            chunks[ident] = chunk
        qa = row['qa']
        queries.append({'id': row['id'], 'question': qa['question'], 'document': doc,
                        'split': split_for(doc),
                        'gold': [f'{doc}::{key}' for key in qa['gold_inds']],
                        'reference_answer': str(qa['answer'])})
    missing = {g for q in queries for g in q['gold'] if g not in chunks}
    if missing:
        raise ValueError(f'{len(missing)} missing gold evidence chunks; fix parser before evaluation')
    return sorted(chunks.values(), key=lambda c: c.id), queries


def chunk_pdf(blob: bytes, filename: str) -> list[Chunk]:
    from io import BytesIO
    from pypdf import PdfReader
    reader = PdfReader(BytesIO(blob))
    if reader.is_encrypted:
        raise ValueError('Encrypted PDFs are not supported. Upload an unlocked text PDF.')
    if len(reader.pages) > 400:
        raise ValueError('Maximum PDF length is 400 pages.')
    digest = hashlib.sha256(blob).hexdigest()[:16]
    doc = f'upload/{digest}/{Path(filename).name}'
    chunks = []
    for page_no, page in enumerate(reader.pages, 1):
        text = page.extract_text(extraction_mode='layout') or ''
        words = text.split()
        # Chunks never cross physical PDF page boundaries; stable content IDs.
        for start in range(0, len(words), 180):
            segment = ' '.join(words[start:start + 220])
            if len(segment) < 30:
                continue
            loc = f'p{page_no}-w{start}'
            chunks.append(Chunk(f'{doc}::{loc}', doc, 'Uploaded', '', str(page_no), loc,
                                segment, 'pdf', f'/api/pdf/{digest}#page={page_no}'))
    if not chunks:
        raise ValueError('No selectable text found. Scanned PDFs require OCR before upload.')
    return chunks


def dump_chunks(chunks: list[Chunk], path: Path) -> None:
    path.write_text(json.dumps([asdict(c) for c in chunks]))
