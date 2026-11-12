FROM python:3.12-slim
WORKDIR /app
COPY requirements.lock.txt .
RUN pip install --no-cache-dir -r requirements.lock.txt
COPY . .
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
EXPOSE 8765
CMD ["python", "-m", "uvicorn", "financial_research.app:app", "--host", "0.0.0.0", "--port", "8765"]
