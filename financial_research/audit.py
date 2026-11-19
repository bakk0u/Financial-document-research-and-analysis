from __future__ import annotations
import hashlib
import json
import sqlite3
import threading
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path


class Audit:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, payload TEXT NOT NULL, previous_hash TEXT NOT NULL, hash TEXT NOT NULL)')
        path.chmod(0o600)

    def append(self, payload: dict):
        event = {'id': str(uuid.uuid4()), 'at': datetime.now(timezone.utc).isoformat(), **payload}
        encoded = json.dumps(event, sort_keys=True, separators=(',', ':'))
        with self.lock, sqlite3.connect(self.path) as db:
            db.execute('BEGIN IMMEDIATE')
            last = db.execute('SELECT hash FROM events ORDER BY seq DESC LIMIT 1').fetchone()
            previous = last[0] if last else '0' * 64
            digest = hashlib.sha256((previous + encoded).encode()).hexdigest()
            db.execute('INSERT INTO events (payload, previous_hash, hash) VALUES (?, ?, ?)', (encoded, previous, digest))
        return event['id']

    def read(self, limit=100):
        if limit < 0:
            raise ValueError('Audit limit must be nonnegative')
        recent = deque(maxlen=limit)
        previous = '0' * 64
        valid = True
        total = 0
        with sqlite3.connect(self.path) as db:
            for payload, parent, digest in db.execute(
                'SELECT payload, previous_hash, hash FROM events ORDER BY seq'
            ):
                expected = hashlib.sha256((parent + payload).encode()).hexdigest()
                valid = valid and parent == previous and digest == expected
                previous = digest
                total += 1
                recent.append((payload, digest))
        return {
            'chain_valid': valid,
            'events': [{**json.loads(payload), 'hash': digest} for payload, digest in reversed(recent)],
            'total': total,
            'notice': 'Local hash chain detects modification relative to its retained history. It cannot prevent deletion or rewriting by the database owner.',
        }
