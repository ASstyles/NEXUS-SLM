# NEXUS-SLM Production Hardened Container Image
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    API_PORT=8000 \
    API_HOST=0.0.0.0 \
    ACTIVE_MODEL_VERSION=nexus-qlora-v1

WORKDIR /app

# Install minimal OS dependencies & curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies using CPU-only PyTorch to minimize image footprint
COPY requirements.txt requirements-ci.txt requirements-training.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "torch>=2.2.0" --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir llama-cpp-python>=0.2.80 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu && \
    pip install --no-cache-dir -r requirements-ci.txt && \
    pip install --no-cache-dir --no-deps -e .

# Copy application source code, models, configs, and assets
COPY src/ ./src/
COPY api/ ./api/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY monitoring/ ./monitoring/
COPY data/ ./data/
COPY models/ ./models/
COPY experiments/ ./experiments/

# Create non-root system user for security
RUN useradd -m -u 10001 nexususer && \
    chown -R nexususer:nexususer /app

USER nexususer

EXPOSE 8000

STOPSIGNAL SIGTERM

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "scripts/deployment/run_api.py"]
