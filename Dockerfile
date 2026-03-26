FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY docs ./docs
COPY data ./data
COPY examples ./examples
COPY notebooks ./notebooks
COPY experiments ./experiments

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -e . && \
    python -m pip install pymupdf pillow reportlab

ENTRYPOINT ["top-papers-graph"]
CMD ["env"]
