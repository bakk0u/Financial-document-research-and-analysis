"""Minimal financial arithmetic DSL: finite, traceable, no eval/code execution.

A program is a list of arithmetic functions with literal operands or backward #step
references. This module is *not* a natural-language semantic parser; a program must
be supplied by a trusted operator or by an explicitly labelled oracle evaluator.
"""
from __future__ import annotations
import math
import re
from dataclasses import dataclass

ATOM = re.compile(r'(?:(?:-?\d+(?:\.\d+)?%?)|const_-?\d+(?:\.\d+)?|#\d+)\Z')
STEP = re.compile(r'([a-z_]+)\((.*)\)\Z')
OPS = {'add', 'subtract', 'multiply', 'divide', 'greater', 'exp'}


class InvalidCalculation(ValueError):
    pass


def parse_program(program: str) -> list[tuple[str, str, str]]:
    if not isinstance(program,str) or len(program)>700:
        raise InvalidCalculation('Invalid program length')
    parts=[]; depth=0; last=0
    for i,char in enumerate(program):
        if char=='(':depth+=1
        elif char==')':
            depth-=1
            if depth<0:raise InvalidCalculation('Unbalanced parentheses')
        elif char==',' and depth==0:
            parts.append(program[last:i].strip());last=i+1
    if depth:raise InvalidCalculation('Unbalanced parentheses')
    parts.append(program[last:].strip())
    if not 1<=len(parts)<=12:raise InvalidCalculation('Invalid step count')
    result=[]
    for statement in parts:
        match=STEP.fullmatch(statement)
        if not match or match[1] not in OPS:
            raise InvalidCalculation('Unsupported arithmetic operation')
        args=[x.strip() for x in match[2].split(',')]
        if len(args)!=2 or not all(ATOM.fullmatch(x) for x in args):
            raise InvalidCalculation('Invalid arithmetic operands')
        result.append((match[1],args[0],args[1]))
    return result


def operand(atom, results):
    if atom.startswith('#'):
        pos=int(atom[1:])
        if pos>=len(results):raise InvalidCalculation('Forward/cyclic reference forbidden')
        return results[pos]
    if atom.startswith('const_'):
        return float(atom[6:])
    if atom.endswith('%'):
        return float(atom[:-1])/100
    return float(atom)


def execute(program: str):
    """Return finite result and explicit deterministic operation trace."""
    steps=parse_program(program); values=[]; trace=[]
    for i,(op,a,b) in enumerate(steps):
        x=operand(a,values); y=operand(b,values)
        if not all(math.isfinite(z) for z in (x,y)):
            raise InvalidCalculation('Nonfinite operands')
        try:
            if op=='add': result=x+y
            elif op=='subtract':result=x-y
            elif op=='multiply':result=x*y
            elif op=='divide':
                if y==0:raise InvalidCalculation('Division by zero')
                result=x/y
            elif op=='exp':
                if abs(y)>20:raise InvalidCalculation('Exponent exceeds bounded range')
                result=x**y
            elif op=='greater':result='yes' if x>y else 'no'
        except (OverflowError, ZeroDivisionError, ValueError) as exc:
            raise InvalidCalculation('Unsafe arithmetic') from exc
        if isinstance(result,float) and not math.isfinite(result):
            raise InvalidCalculation('Nonfinite arithmetic result')
        if op=='greater' and i!=len(steps)-1:
            raise InvalidCalculation('Boolean steps cannot feed arithmetic')
        values.append(result)
        trace.append({'step':i,'operation':op,'input_atoms':[a,b], 'resolved_operands':[x,y], 'result':result})
    return values[-1], trace


def literal_atoms(program):
    """Only values requiring evidence grounding, not constants or earlier steps."""
    return [atom for _,a,b in parse_program(program) for atom in (a,b)
            if not atom.startswith(('const_','#'))]


def grounded_literal(atom: str, texts: list[str]) -> bool:
    """Conservative numerical-token grounding (not semantic cell attribution)."""
    value=float(atom.rstrip('%'))
    for text in texts:
        # accounting negatives in parentheses may be ambiguous, intentionally not inferred
        for found in re.finditer(r'(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?\s*%?(?![\w])', text):
            token=found[0].replace(',','').strip().rstrip('%').strip()
            try:
                if math.isclose(float(token),value,rel_tol=1e-9,abs_tol=1e-9):
                    return True
            except ValueError:
                continue
    return False


def all_literals_grounded(program: str, texts: list[str]) -> bool:
    try:return all(grounded_literal(a,texts) for a in literal_atoms(program))
    except (InvalidCalculation,ValueError):return False


def reference_agrees(value, reference) -> bool | None:
    """Diagnostics only: FinQA's answer-format and exe_ans annotations can disagree."""
    if value in ('yes','no'):
        return str(reference).lower()==value
    if isinstance(reference,str):
        x=reference.strip().replace(',','').replace('$','')
        x=x.rstrip('%').strip()
        try: ref=float(x)
        except ValueError:return None
    else:
        try:ref=float(reference)
        except (TypeError,ValueError):return None
    # The corpus has inconsistent percentage conventions: compare both shapes.
    return any(abs(v-ref)<=max(.015,.001*abs(ref)) for v in (value, value*100, value/100))


def verify_manual_calculation(program: str, source_quotes: list[dict]):
    """Trust boundary for manually specified arithmetic: citations and literals are checked.

    source_quotes: each has id, text, quote. The caller must fetch text from the
    server-side corpus, NOT accept client-supplied source text as authoritative.
    """
    if not source_quotes or len(source_quotes)>8:
        raise InvalidCalculation('Provide one to eight cited evidence snippets')
    for item in source_quotes:
        if not item.get('id') or not isinstance(item.get('quote'),str) or len(item['quote'])<3:
            raise InvalidCalculation('Missing evidence citation/quote')
        if ' '.join(item['quote'].split()) not in ' '.join(item['text'].split()):
            raise InvalidCalculation('Quote absent from cited source')
    if not literal_atoms(program):
        raise InvalidCalculation('At least one evidence-grounded operand is required')
    quotes=[x['quote'] for x in source_quotes]
    if not all_literals_grounded(program,quotes):
        raise InvalidCalculation('Every non-constant numeric operand must appear in cited quotations')
    value,trace=execute(program)
    return {'status':'verified_arithmetic','value':value,'program':program,'trace':trace,
            'citations':[{'id':x['id'],'quote':x['quote']} for x in source_quotes],
            'limitation':'Verifies arithmetic and literal presence in quotations, NOT whether the chosen cells answer the question or use consistent units.'}
