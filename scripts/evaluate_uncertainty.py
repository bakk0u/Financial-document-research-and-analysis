"""Run a statistical audit on frozen test predictions (no training)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from financial_research.corpus import ROOT
from financial_research.uncertainty import generate
if __name__ == '__main__':
    result=generate(ROOT)
    print('Wrote reports/uncertainty.json and reports/UNCERTAINTY.md')
    for item in result['retrieval']:
        if item['scope']=='collection' and item['metric']=='recall_at_5' and item['contrast']=='reranked minus bm25':
            print(item)
