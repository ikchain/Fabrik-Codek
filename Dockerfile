FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FABRIK_DATA_DIR=/app/data \
    FABRIK_DATALAKE_PATH=/app/data \
    FABRIK_API_HOST=0.0.0.0 \
    FABRIK_API_PORT=8420

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY docs ./docs
COPY prompts ./prompts
COPY scripts ./scripts
COPY skills ./skills
COPY utils ./utils
COPY .env.example ./

RUN pip install --upgrade pip \
    && pip install -e .

RUN mkdir -p \
    /app/data/01-raw/interactions \
    /app/data/01-raw/outcomes \
    /app/data/01-raw/code-changes \
    /app/data/02-processed/training-pairs \
    /app/data/03-metadata/decisions \
    /app/data/03-metadata/learnings \
    /app/data/profile \
    /app/data/raw/interactions \
    /app/data/raw/training_pairs \
    /app/data/processed \
    /app/data/embeddings

EXPOSE 8420

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${FABRIK_API_PORT}/health || exit 1

CMD ["fabrik", "serve", "--host", "0.0.0.0", "--port", "8420"]
