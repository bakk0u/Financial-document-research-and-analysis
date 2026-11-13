"""Shared session fixtures: one corpus, one index and one API client for the whole suite."""
from __future__ import annotations
import json
import pytest
from fastapi.testclient import TestClient
from financial_research.audit import Audit
from financial_research.corpus import DATA, load_finqa
from financial_research.retrieval import Engine


@pytest.fixture(scope='session')
def corpus():
    return load_finqa()


@pytest.fixture(scope='session')
def engine(corpus):
    c, _ = corpus
    return Engine(c, json.loads((DATA / 'reranker.json').read_text()))


@pytest.fixture(scope='session')
def client(tmp_path_factory, engine, corpus):
    from financial_research import app as module
    module.state.update(engine=engine, queries=corpus[1], audit=Audit(tmp_path_factory.mktemp('runtime') / 'audit.db'))
    module.RUNTIME = tmp_path_factory.mktemp('uploads')
    # Avoid repeated startup indexing; initialize production dependencies explicitly.
    return TestClient(module.app)
