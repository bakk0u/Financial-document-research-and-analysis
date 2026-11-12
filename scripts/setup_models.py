"""Download the pinned public base model once; then verify the bundled adapter."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from financial_research.sentiment import ensure_base,predict
print('Preparing pinned DistilBERT base model (first download about 268 MB).',flush=True)
ensure_base()
result=predict(['Operating profit increased as revenue grew.'])
print('Model ready:',result['results'][0]['label'])
