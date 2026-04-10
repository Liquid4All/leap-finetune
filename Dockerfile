FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip git curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app
COPY pyproject.toml uv.lock ./

RUN uv export --frozen --no-dev --no-emit-project --no-hashes > requirements.txt \
    && grep -v flash-attn requirements.txt > requirements-filtered.txt \
    && uv pip install --system --python python3.12 -r requirements-filtered.txt

COPY src/ src/

ENV PYTHONPATH=/app/src \
    LEAP_FINETUNE_DIR=/app \
    RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
