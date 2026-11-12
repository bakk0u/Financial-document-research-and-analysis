"""Restore the exact licensed public data snapshot; fail on upstream drift."""
import hashlib
import json
import sys
from pathlib import Path
import httpx
root=Path(__file__).resolve().parents[1]
provenance=json.loads((root/'data/provenance.json').read_text())
with httpx.Client(timeout=60,follow_redirects=True) as client:
    response=client.get(provenance['source_url'])
    response.raise_for_status()
    raw=response.content
blob=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
if blob!=provenance['upstream_blob_sha']:
    sys.exit('Upstream content changed; refusing to overwrite the pinned snapshot.')
canonical=json.dumps(json.loads(raw)).encode()
if hashlib.sha256(canonical).hexdigest()!=provenance['snapshot_sha256']:
    sys.exit('Canonical snapshot hash mismatch.')
tmp=root/'data/finqa-dev.tmp'
tmp.write_bytes(canonical)
tmp.replace(root/'data/finqa-dev.json')
print('Verified and restored FinQA snapshot.')
