FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH=/opt/venv/bin:/root/.local/bin:${PATH}
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3-pip \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m pip install --no-cache-dir uv

WORKDIR /workspace

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY job_configs ./job_configs
COPY scripts ./scripts

RUN uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "leap-finetune"]
