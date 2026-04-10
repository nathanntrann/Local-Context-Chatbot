FROM python:3.13-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/

# Install CPU-only PyTorch first (much smaller than default CUDA build),
# then install the project and remaining deps
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir .

# --- Runtime stage ---
FROM python:3.13-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/ src/
COPY knowledge/ knowledge/

# Create data directory for SQLite DB, vectorstore, and reports
RUN mkdir -p data/images data/reports data/vectorstore

# Cross-encoder model is downloaded on first use; pre-cache it
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" 2>/dev/null || true

EXPOSE 8000

CMD ["python", "-m", "inspect_assist"]
