from __future__ import annotations
import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from .answer import synthesize
from .audit import Audit
from .corpus import ROOT, DATA, Chunk, chunk_pdf, load_finqa
from .retrieval import Engine, METHODS

lock = threading.RLock()
cache = OrderedDict()
state = {}
RUNTIME = Path(os.environ.get('FINANCIAL_RESEARCH_RUNTIME', str(ROOT / 'runtime')))


def initialize():
    chunks, queries = load_finqa()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    model_path = DATA / 'reranker.json'
    model = json.loads(model_path.read_text()) if model_path.exists() else None
    uploads = RUNTIME / 'uploads.json'
    if uploads.exists():
        chunks.extend(Chunk(**c) for c in json.loads(uploads.read_text()))
    engine = Engine(chunks, model)
    state.update(engine=engine, queries=queries, audit=Audit(RUNTIME / 'audit.sqlite3'))


@asynccontextmanager
async def lifespan(app):
    initialize()
    yield


app = FastAPI(title='Financial Document Research & Analysis', version='1.1.0', lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=['localhost', '127.0.0.1', '[::1]', 'testserver'])


@app.middleware('http')
async def local_boundary(request: Request, call_next):
    # Loopback application, not an authenticated multi-tenant service.
    origin = request.headers.get('origin')
    host = request.headers.get('host', '')
    if origin and origin not in (f'http://{host}', f'https://{host}'):
        raise_response = __import__('fastapi.responses', fromlist=['JSONResponse']).JSONResponse
        return raise_response({'detail': 'Cross-origin requests are not allowed'}, status_code=403)
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'; base-uri 'self'"
    return response


class Query(BaseModel):
    question: str = Field(min_length=3, max_length=1200)
    method: str = 'reranked'
    document: str | None = None
    k: int = Field(default=5, ge=1, le=10)
    generate: bool = False


@app.get('/api/status')
def status():
    engine = state['engine']
    docs = []
    for doc, indices in sorted(engine.doc_indices.items()):
        c = engine.chunks[indices[0]]
        docs.append({'id': doc, 'company': c.company, 'year': c.year, 'page': c.page, 'chunks': len(indices)})
    return {'name': 'Financial Document Research & Analysis', 'chunks': len(engine.chunks), 'documents': docs,
            'fingerprint': engine.fingerprint, 'methods': METHODS,
            'model_trained': bool(engine.model), 'model': engine.model,
            'generation_configured': bool(os.environ.get('OPENAI_API_KEY') and os.environ.get('FINANCIAL_RESEARCH_MODEL')),
            'sentiment_ready': (DATA / 'sentiment-adapter/adapter_config.json').exists(),
            'sentiment_base_cached': (ROOT / 'runtime/base-model/model.safetensors').exists(),
            'vector_backend': 'TF-IDF + 64-dimensional latent semantic analysis (LSA); not neural embeddings',
            'examples': [{'question': q['question'], 'document': q['document']} for q in state['queries'] if q['split'] == 'test'][:8]}


@app.post('/api/ask')
def ask(query: Query):
    if query.method not in METHODS:
        raise HTTPException(422, 'Unknown method')
    if not query.question.strip():
        raise HTTPException(422, 'Question must contain text')
    with lock:
        engine = state['engine']
        if query.document and query.document not in engine.doc_indices:
            raise HTTPException(404, 'Unknown source document')
        start = time.perf_counter()
        key = (engine.fingerprint, query.question, query.method, query.k, query.document)
        cached = key in cache
        if cached:
            hits = cache[key]
            cache.move_to_end(key)
        else:
            hits = engine.search(query.question, query.method, query.k, query.document)
            cache[key] = hits
            if len(cache) > 128:
                cache.popitem(last=False)
        retrieval_ms = (time.perf_counter() - start) * 1000
    answer = synthesize(query.question, hits, query.generate)
    elapsed = (time.perf_counter() - start) * 1000
    event_id = state['audit'].append({'action': 'query', 'question_sha256': hashlib.sha256(query.question.encode()).hexdigest(),
         'method': query.method, 'document': query.document, 'retrieved_ids': [h['id'] for h in hits],
         'answer_status': answer['status'], 'mode': answer['mode'], 'index_fingerprint': engine.fingerprint,
         'retrieval_ms': round(retrieval_ms, 2), 'total_ms': round(elapsed, 2), 'cached': cached})
    return {**answer, 'evidence': hits, 'retrieval_ms': round(retrieval_ms, 2), 'total_ms': round(elapsed, 2),
            'audit_id': event_id, 'cached': cached, 'question': query.question, 'method': query.method}


class EvidenceQuote(BaseModel):
    id: str = Field(min_length=1, max_length=300)
    quote: str = Field(min_length=3, max_length=2000)


class ArithmeticRequest(BaseModel):
    program: str = Field(min_length=5, max_length=700)
    evidence: list[EvidenceQuote] = Field(min_length=1, max_length=8)


@app.post('/api/calculate')
def calculate(request: ArithmeticRequest):
    """User-proposed arithmetic is never treated as an inferred answer.

    Citation IDs resolve against the server-side index, not client-supplied content.
    """
    from .arithmetic import InvalidCalculation, verify_manual_calculation
    with lock:
        engine = state['engine']
        quotes=[]
        for supplied in request.evidence:
            if supplied.id not in engine.by_id:
                raise HTTPException(422,'Citation not present in indexed sources')
            c=engine.chunks[engine.by_id[supplied.id]]
            quotes.append({'id':supplied.id,'quote':supplied.quote,'text':c.text})
        try:
            verified=verify_manual_calculation(request.program,quotes)
        except InvalidCalculation as exc:
            raise HTTPException(422,str(exc)) from exc
        verified['audit_id']=state['audit'].append({'action':'calculation',
            'program_sha256':hashlib.sha256(request.program.encode()).hexdigest(),
            'cited_ids':[q['id'] for q in quotes], 'result_type':type(verified['value']).__name__})
        return verified


@app.get('/api/uncertainty')
def uncertainty():
    path=ROOT / 'reports/uncertainty.json'
    if not path.exists():raise HTTPException(503,'Uncertainty analysis not yet run')
    return json.loads(path.read_text())


@app.get('/api/arithmetic-evaluation')
def arithmetic_evaluation():
    path=ROOT / 'reports/arithmetic-evaluation.json'
    if not path.exists():raise HTTPException(503,'Arithmetic evaluation not yet run')
    return json.loads(path.read_text())['summary']


@app.get('/api/chunk')
def get_chunk(id: str):
    engine = state['engine']
    if id not in engine.by_id:
        raise HTTPException(404, 'Unknown evidence identifier')
    from dataclasses import asdict
    return asdict(engine.chunks[engine.by_id[id]])


@app.get('/api/evaluation')
def evaluation():
    path = ROOT / 'reports/evaluation.json'
    if not path.exists():
        raise HTTPException(503, 'Run python -m financial_research.evaluate to produce measured results')
    return json.loads(path.read_text())


@app.get('/api/audit')
def audit():
    return state['audit'].read()


@app.post('/api/upload')
async def upload(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(422, 'Upload a PDF file')
    blob = await file.read(12 * 1024 * 1024 + 1)
    if len(blob) > 12 * 1024 * 1024:
        raise HTTPException(413, 'PDF exceeds the 12 MiB limit')
    if not blob.startswith(b'%PDF-'):
        raise HTTPException(422, 'File is not a PDF')
    # PDF parsing and index rebuilding are CPU-bound: run outside the event loop.
    import asyncio
    return await asyncio.to_thread(ingest_upload, blob, file.filename)


def ingest_upload(blob, filename):
    try:
        chunks = chunk_pdf(blob, filename)
    except Exception:
        raise HTTPException(422, 'Cannot parse this PDF. Use an unlocked PDF with selectable text, under 400 pages.')
    digest = hashlib.sha256(blob).hexdigest()[:16]
    with lock:
        engine = state['engine']
        if chunks[0].id in engine.by_id:
            return {'status': 'already_indexed', 'chunks': len(chunks), 'document': chunks[0].document}
        uploaded = [c for c in engine.chunks if c.kind == 'pdf']
        if len(uploaded) + len(chunks) > 6000:
            raise HTTPException(413, 'Local upload index limit reached (6000 chunks).')
        new_engine = Engine(engine.chunks + chunks, engine.model)
        from .corpus import dump_chunks
        (RUNTIME / f'{digest}.pdf').write_bytes(blob)
        tmp = RUNTIME / 'uploads.tmp'
        dump_chunks(uploaded + chunks, tmp)
        tmp.replace(RUNTIME / 'uploads.json')
        state['engine'] = new_engine
        cache.clear()
        state['audit'].append({'action': 'upload', 'content_sha256': hashlib.sha256(blob).hexdigest(), 'chunks': len(chunks)})
    return {'status': 'indexed', 'chunks': len(chunks), 'document': chunks[0].document}


@app.get('/api/pdf/{digest}')
def pdf(digest: str):
    import re
    if not re.fullmatch('[a-f0-9]{16}', digest) or not (RUNTIME / f'{digest}.pdf').exists():
        raise HTTPException(404, 'PDF not found')
    return FileResponse(RUNTIME / f'{digest}.pdf', media_type='application/pdf')


class SentimentRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=8)
    model: str = 'transformer'


@app.post('/api/sentiment')
def sentiment(request: SentimentRequest):
    if request.model not in ('transformer', 'tfidf') or any(not 5 <= len(t.strip()) <= 2000 for t in request.texts):
        raise HTTPException(422, 'Provide 1–8 sentences of 5–2000 characters and a supported model.')
    from .sentiment import predict
    try:
        result = predict(request.texts, request.model)
    except Exception:
        raise HTTPException(503, 'Sentiment model could not load. Check the model files and first-use network access on the server.')
    result['audit_id'] = state['audit'].append({'action': 'sentiment', 'model': result['model'],
         'input_sha256': [hashlib.sha256(t.encode()).hexdigest() for t in request.texts],
         'predictions': [r['label'] for r in result['results']], 'elapsed_ms': result['elapsed_ms']})
    return result


@app.get('/api/sentiment/evaluation')
def sentiment_evaluation():
    path = ROOT / 'reports/sentiment-evaluation.json'
    if not path.exists():
        raise HTTPException(503, 'Sentiment training has not completed.')
    return json.loads(path.read_text())


app.mount('/', StaticFiles(directory=ROOT / 'static', html=True), name='frontend')
