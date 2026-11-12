from __future__ import annotations
import hashlib
import json
import sqlite3
import threading
import uuid
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
        with sqlite3.connect(self.path) as db:
            rows = db.execute('SELECT payload, previous_hash, hash FROM events ORDER BY seq').fetchall()
        previous, valid = '0' * 64, True
        for payload, prev, digest in rows:
            valid &= prev == previous and hashlib.sha256((prev + payload).encode()).hexdigest() == digest
            previous = digest
        return {'chain_valid': bool(valid), 'events': [{**json.loads(p), 'hash': h} for p, _, h in rows[-limit:]][::-1],
                'total': len(rows), 'notice': 'Local hash chain detects modification relative to its retained history. It cannot prevent deletion or rewriting by the database owner.'}
