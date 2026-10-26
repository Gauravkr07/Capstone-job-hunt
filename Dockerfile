FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    # Install CPU-only torch first, from PyTorch's dedicated CPU wheel index --
    # this container has no GPU, but sentence-transformers otherwise pulls in
    # the default CUDA build (~2GB of unused nvidia-* packages, slower and
    # more failure-prone to download).
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image so containers don't download it on
# every start (cached as its own layer; only re-fetched when deps change).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
ENV HF_HUB_OFFLINE=1

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
