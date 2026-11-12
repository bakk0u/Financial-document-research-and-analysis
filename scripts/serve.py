"""Preview adapter; understands managed preview's Vite-style launch arguments."""
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import uvicorn
parser = argparse.ArgumentParser()
parser.add_argument('--host', default='127.0.0.1')
parser.add_argument('--port', type=int, default=8765)
parser.add_argument('--strictPort', action='store_true')
args = parser.parse_args()
uvicorn.run('financial_research.app:app', host=args.host, port=args.port)
