"""Check the exact packaged files against sha256 hashes in MANIFEST.json."""
from pathlib import Path
import hashlib
import json
import sys

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'MANIFEST.json').read_text())
failed=[]
for name,expect in manifest.items():
    path=root/name
    actual=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    if actual != expect:failed.append(name)
if failed:
    print('Integrity verification failed:',', '.join(failed),file=sys.stderr)
    raise SystemExit(1)
print(f'Verified {len(manifest)} source/data/report files against SHA-256 manifest.')
