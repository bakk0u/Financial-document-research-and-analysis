"""Rewrite MANIFEST.json with SHA-256 hashes of every tracked file (run before committing a release)."""
import hashlib
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
names = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=root, check=True,
                       capture_output=True, text=True).stdout.splitlines()
manifest = {n: hashlib.sha256((root / n).read_bytes()).hexdigest()
            for n in sorted(names) if n != 'MANIFEST.json' and (root / n).is_file()}
(root / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f'{len(manifest)} files hashed')
