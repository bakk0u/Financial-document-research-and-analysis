"""Cross-platform local launcher. Run from the project root after installation."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import uvicorn
if __name__ == '__main__':
    print('Financial Document Research & Analysis: http://127.0.0.1:8765 (wait for index startup)')
    uvicorn.run('financial_research.app:app', host='127.0.0.1', port=8765)
