"""End-to-end numerical question answering without gold programs.

    question -> retrieve passages -> extract number candidates -> infer the
    operation template -> select and order operands -> execute in the bounded
    arithmetic DSL -> verify grounding -> cited answer, or abstain.

Nothing here reads FinQA programs, gold evidence or answers at inference time.
Those are used only as training labels on development data
(``scripts/train_numeric_qa.py``) and by the scorer (``scripts/evaluate_blind.py``).

Two proposers share one verifier:
* ``learned``: three small scikit-learn models (operation template classifier,
  operand scorer, ordered-pair scorer). Local, deterministic, no network.
* ``llm`` (optional, see ``llm_program.py``): a hosted model proposes a program;
  it is accepted only if every operand is grounded in the retrieved evidence.
"""
from __future__ import annotations
import math
import re
from pathlib import Path

import numpy as np

from .arithmetic import InvalidCalculation, execute, grounded_literal, literal_atoms
from .corpus import DATA
from .retrieval import content_tokens

MODEL_PATH = DATA / 'numeric-qa-model.joblib'
NUMBER = re.compile(r'(?<![\w.,])-?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:\s?%)?(?![\w])')
YEAR = re.compile(r'\b(?:19|20)\d{2}\b')
STEP = re.compile(r'([a-z_]+)\(([^()]*)\)')
MAX_CANDIDATES = 12          # operands are chosen among the top unary-scored numbers
PASSAGES = 12                # retrieved passages searched for operands
CUES = ('percent', 'percentage', 'change', 'increase', 'decrease', 'growth', 'ratio', 'average',
        'total', 'sum', 'difference', 'portion', 'much', 'many', 'rate', 'net', 'from', 'to',
        'between', 'compared', 'exceed', 'greater', 'more', 'less', 'higher', 'lower', 'of',
        'per', 'margin', 'return', 'decline', 'grow', 'did', 'was', 'what', 'in')


def stem(token: str) -> str:
    return token[:-1] if len(token) > 3 and token.endswith('s') else token


def words(text: str) -> set[str]:
    return {stem(t) for t in content_tokens(text) if not t[0].isdigit()}


def clean_number(raw: str) -> tuple[str, float, bool] | None:
    """Surface number -> (DSL atom without percent sign, value, is_percent)."""
    percent = raw.rstrip().endswith('%')
    body = raw.replace('%', '').replace(',', '').strip()
    if body.startswith('.'):
        body = '0' + body
    elif body.startswith('-.'):
        body = '-0' + body[1:]
    try:
        value = float(body)
    except ValueError:
        return None
    if not math.isfinite(value) or len(body) > 24:
        return None
    return body, value, percent


def parse_table_row(text: str) -> list[tuple[str, str]]:
    """Invert the corpus serialisation ``header: value ; header: value``."""
    cells = []
    for piece in text.split(' ; '):
        header, sep, value = piece.partition(': ')
        cells.append((header.strip(), value.strip()) if sep else ('', piece.strip()))
    return cells


def extract_candidates(question: str, hits: list[dict]) -> list[dict]:
    """Every distinct number in the retrieved passages, with its local context."""
    out, seen = [], set()

    def push(raw, chunk, kind, row, col, col_idx, context, rank, bm25, n_cells=0, text_pos=0.0):
        parsed = clean_number(raw)
        if not parsed:
            return
        atom, value, percent = parsed
        key = (chunk, atom, row, col)
        if key in seen:
            return
        seen.add(key)
        out.append({'atom': atom, 'value': value, 'percent': percent, 'chunk': chunk, 'kind': kind,
                    'row': row, 'col': col, 'col_idx': col_idx, 'context': context, 'rank': rank,
                    'bm25': bm25, 'n_cells': n_cells, 'text_pos': text_pos, 'surface': raw.strip(),
                    'dollar': False})

    for rank, hit in enumerate(hits):
        text = hit['text']
        if hit['kind'] == 'table':
            cells = parse_table_row(text)
            row = cells[0][1] if cells else ''
            for j, (header, value) in enumerate(cells[1:], 1):
                found = NUMBER.search(value)       # first number only: "-12 ( 12 )" repeats itself
                if found:
                    push(found[0], hit['id'], 'table', row, header, j, f'{row} {header}', rank,
                         hit.get('bm25', 0.0), len(cells))
                    out[-1]['dollar'] = '$' in value
        else:
            toks = text.split()
            joined = ' '.join(toks)
            for found in NUMBER.finditer(joined):
                left = joined[max(0, found.start() - 90):found.start()]
                right = joined[found.end():found.end() + 70]
                if re.search(r'\(\s*$', left) and re.match(r'\s*%?\s*\)', right):
                    continue                       # "12% ( 12 % )" duplicate rendering
                push(found[0], hit['id'], 'text', '', '', 0, f'{left} {right}', rank,
                     hit.get('bm25', 0.0), 0, found.start() / max(1, len(joined)))
                out[-1]['dollar'] = left.rstrip().endswith('$')
    for found in NUMBER.finditer(question):
        if YEAR.fullmatch(found[0].strip()):
            continue
        push(found[0], 'question', 'question', '', '', 0, question, 0, 0.0)
    return out


def question_features(question: str) -> list[float]:
    q = set(content_tokens(question)) | set(re.findall(r'[a-z]+', question.lower()))
    return [float(c in q or stem(c) in q) for c in CUES]


def unary_features(question: str, cand: dict) -> list[float]:
    q = words(question)
    years = set(YEAR.findall(question))
    row, col, ctx = words(cand['row']), words(cand['col']), words(cand['context'])
    ctx_years = set(YEAR.findall(cand['context']))
    col_years = set(YEAR.findall(cand['col']))
    value = cand['value']
    integer = float(value).is_integer()
    return [
        len(q & row) / max(1, len(q)), len(q & row) / max(1, len(row)),
        len(q & col) / max(1, len(q)), len(q & col) / max(1, len(col)),
        len(q & ctx) / max(1, len(q)), len(q & ctx) / max(1, len(ctx)),
        float(bool(years & col_years)), float(bool(years & ctx_years)),
        float(bool(col_years)), float(len(years)),
        float(cand['kind'] == 'table'), float(cand['kind'] == 'text'), float(cand['kind'] == 'question'),
        float(cand['percent']), float(cand['dollar']), float(value < 0), float(integer),
        float(integer and 1900 <= value <= 2100 and not cand['dollar'] and not cand['percent']),
        math.log10(abs(value) + 1e-6) if value else -6.0,
        float(cand['rank']), cand['bm25'] / (cand['bm25'] + 10), float(cand['col_idx']),
        float(cand['n_cells']), cand['text_pos'],
        float(cand['atom'] in question.replace(',', '')),
    ] + question_features(question)


def pair_features(question: str, a: dict, b: dict, sa: float, sb: float, template_id: int) -> list[float]:
    ya, yb = YEAR.findall(a['context']), YEAR.findall(b['context'])
    year_a = max(map(int, ya)) if ya else 0
    year_b = max(map(int, yb)) if yb else 0
    qyears = [int(y) for y in YEAR.findall(question)]
    first_q = qyears[0] if qyears else 0
    last_q = qyears[-1] if qyears else 0
    same_row = float(a['kind'] == 'table' and a['chunk'] == b['chunk'])
    same_col = float(a['kind'] == b['kind'] == 'table' and a['col'] == b['col'] and a['chunk'] != b['chunk'])
    va, vb = a['value'], b['value']
    ratio = math.log10((abs(va) + 1e-6) / (abs(vb) + 1e-6))
    return [
        sa, sb, sa * sb, min(sa, sb), float(template_id),
        same_row, same_col, float(a['chunk'] == b['chunk']),
        float(a['col_idx'] - b['col_idx']), float(a['rank'] - b['rank']),
        float(year_a > year_b), float(year_a < year_b), float(year_a == year_b and year_a > 0),
        float(year_a == first_q and first_q > 0), float(year_b == first_q and first_q > 0),
        float(year_a == last_q and last_q > 0), float(year_b == last_q and last_q > 0),
        float(va > vb), ratio, float(a['percent']), float(b['percent']),
        float(a['dollar']), float(b['dollar']), float(a['kind'] == 'question'), float(b['kind'] == 'question'),
        float(a['kind'] == 'table'), float(b['kind'] == 'table'),
        float(a['text_pos'] < b['text_pos']),
        len(words(a['row']) & words(b['row'])) / max(1, len(words(a['row']) | words(b['row']))),
        float(va == vb),
    ] + question_features(question)


def canonical_template(program: str) -> tuple[str, list[str]] | None:
    """Abstract literal operands to N1, N2 ... (identity aware); keep constants and step refs."""
    steps = STEP.findall(program.replace(' ', ''))
    if not steps or ', '.join(f'{o}({a})' for o, a in steps).replace(' ', '') != program.replace(' ', ''):
        return None
    slots: list[str] = []
    rendered = []
    for op, args in steps:
        parts = []
        for atom in args.split(','):
            if atom.startswith(('#', 'const_')):
                parts.append(atom)
                continue
            parsed = clean_number(atom)
            if not parsed:
                return None                        # table_* row-name programs are unsupported
            key = parsed[0] + ('%' if parsed[2] else '')
            if key not in slots:
                slots.append(key)
            parts.append(f'N{slots.index(key) + 1}')
        rendered.append(f'{op}({",".join(parts)})')
    return ', '.join(rendered), slots


def instantiate(template: str, operands: list[dict]) -> str:
    program = template
    for i, cand in reversed(list(enumerate(operands, 1))):
        program = program.replace(f'N{i}', cand['atom'])
    return program


def slots_in(template: str) -> int:
    return len(set(re.findall(r'N\d', template)))


def percent_scaled(template: str, operands: list[dict]) -> str:
    """Operands written as percentages in the source are used as fractions (12% -> 0.12),
    matching the DSL percent atom, only when the operation mixes them with non-percent values."""
    if all(o['percent'] for o in operands) or not any(o['percent'] for o in operands):
        return instantiate(template, operands)
    program = template
    for i, cand in reversed(list(enumerate(operands, 1))):
        program = program.replace(f'N{i}', cand['atom'] + ('%' if cand['percent'] else ''))
    return program


class NumericQA:
    """Learned proposer plus the shared verifier. Load with ``NumericQA.load()``."""

    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.templates: list[str] = bundle['templates']
        self.threshold: float = bundle['threshold']

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> 'NumericQA':
        import joblib
        return cls(joblib.load(path))

    # ---- proposal -------------------------------------------------------
    def template_probabilities(self, question: str) -> list[tuple[str, float]]:
        proba = self.bundle['template_model'].predict_proba([question])[0]
        classes = self.bundle['template_model'].classes_
        return sorted(zip(classes, proba), key=lambda item: -item[1])

    def score_candidates(self, question: str, candidates: list[dict]) -> np.ndarray:
        if not candidates:
            return np.zeros(0)
        x = np.array([unary_features(question, c) for c in candidates])
        return self.bundle['unary_model'].predict_proba(x)[:, 1]

    def propose(self, question: str, hits: list[dict], top_templates: int = 3) -> dict | None:
        candidates = extract_candidates(question, hits)
        scores = self.score_candidates(question, candidates)
        order = np.argsort(-scores, kind='stable')[:MAX_CANDIDATES]
        pool = [(candidates[i], float(scores[i])) for i in order]
        best = None
        for template, p_template in self.template_probabilities(question)[:top_templates]:
            if template == 'OTHER':
                continue
            tid = self.templates.index(template)
            need = slots_in(template)
            if need == 1 and pool:
                operands, p_operands = [pool[0][0]], pool[0][1]
            elif need == 2 and len(pool) >= 2:
                pairs = [(i, j) for i in range(len(pool)) for j in range(len(pool)) if i != j]
                x = np.array([pair_features(question, pool[i][0], pool[j][0], pool[i][1], pool[j][1], tid)
                              for i, j in pairs])
                p = self.bundle['pair_model'].predict_proba(x)[:, 1]
                k = int(np.argmax(p))
                operands, p_operands = [pool[pairs[k][0]][0], pool[pairs[k][1]][0]], float(p[k])
            else:
                continue
            confidence = float(p_template) * p_operands
            if best is None or confidence > best['confidence']:
                best = {'template': template, 'operands': operands, 'confidence': confidence,
                        'template_probability': float(p_template), 'operand_probability': p_operands,
                        'program': percent_scaled(template, operands)}
        return best

    def percent_answer(self, question: str, template: str) -> bool:
        return bool(self.bundle['unit_model'].predict([f'{question} || {template}'])[0])

    # ---- full pipeline --------------------------------------------------
    def answer(self, question: str, hits: list[dict], threshold: float | None = None) -> dict:
        threshold = self.threshold if threshold is None else threshold
        proposal = self.propose(question, hits)
        if proposal is None:
            return abstain('No supported calculation could be formed from the retrieved evidence.')
        result = verify_and_execute(proposal['program'], proposal['operands'], hits, question)
        if result['status'] != 'answered':
            return result
        result.update(confidence=round(proposal['confidence'], 4), proposer='learned',
                      template=proposal['template'])
        if proposal['confidence'] < threshold:
            return abstain(f"Confidence {proposal['confidence']:.2f} is below the fixed threshold "
                           f"{threshold:.2f}; the candidate calculation is shown but not asserted.",
                           candidate=result)
        result['unit'] = unit_label(question, result, self.percent_answer(question, proposal['template']))
        result['display'] = display(result)
        return result


def abstain(reason: str, candidate: dict | None = None) -> dict:
    out = {'status': 'abstained', 'reason': reason}
    if candidate:
        out['unverified_candidate'] = {k: candidate[k] for k in ('program', 'value', 'citations', 'confidence')
                                       if k in candidate}
    return out


def verify_and_execute(program: str, operands: list[dict], hits: list[dict], question: str) -> dict:
    """Shared trust boundary: every literal must occur in a cited retrieved passage or the question."""
    by_id = {h['id']: h for h in hits}
    try:
        literals = literal_atoms(program)
    except InvalidCalculation as exc:
        return abstain(f'Proposed calculation rejected: {exc}')
    citations, used = [], set()
    for cand in operands:
        if cand['chunk'] == 'question':
            if not grounded_literal(cand['atom'], [question]):
                return abstain('An operand attributed to the question does not occur in it.')
            continue
        hit = by_id.get(cand['chunk'])
        if hit is None or not grounded_literal(cand['atom'], [hit['text']]):
            return abstain('An operand is not present in its cited passage.')
        if hit['id'] not in used:
            used.add(hit['id'])
            citations.append({'id': hit['id'], 'quote': hit['text'], 'page': hit.get('page'),
                              'document': hit.get('document')})
    grounding_texts = [c['quote'] for c in citations] + [question]
    if not all(grounded_literal(atom, grounding_texts) for atom in literals):
        return abstain('A number in the calculation is not grounded in the cited evidence.')
    if not citations:
        return abstain('No operand comes from the retrieved documents.')
    try:
        value, trace = execute(program)
    except InvalidCalculation as exc:
        return abstain(f'Calculation could not be executed safely: {exc}')
    return {'status': 'answered', 'value': value, 'program': program, 'trace': trace, 'citations': citations,
            'operands': [{'atom': c['atom'], 'surface': c['surface'], 'source': c['chunk'],
                          'row': c['row'], 'column': c['col']} for c in operands]}


def unit_label(question: str, result: dict, percent: bool) -> str:
    if result['value'] in ('yes', 'no'):
        return 'boolean'
    if percent:
        return 'percent'
    scale = re.search(r'\bin (thousands|millions|billions)\b', question.lower())
    if any('$' in c['quote'] for c in result['citations']) and 'divide' not in result['program']:
        return f"USD {scale[1]}" if scale else 'USD (scale as reported in the source)'
    return f'{scale[1]} (as reported)' if scale else 'number (unit as reported in the source)'


def display(result: dict) -> str:
    value = result['value']
    if isinstance(value, str):
        return value
    if result.get('unit') == 'percent':
        # A ratio of two reported amounts is a fraction; values already expressed in percent
        # (percent operands, or an explicit multiplication by 100) are shown unchanged.
        already = (all(str(o['surface']).rstrip().endswith('%') for o in result['operands'])
                   or re.search(r'multiply\([^()]*const_100\)', result['program']))
        return f'{value if already else value * 100:,.2f}%'
    return f'{value:,.4f}'.rstrip('0').rstrip('.')
