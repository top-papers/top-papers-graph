FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Runtime tools used by docker entrypoint (health/wait) + HTTPS
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python package (optionally with extras)
COPY pyproject.toml README.md LICENSE CHANGELOG.md /app/
COPY src /app/src
COPY configs /app/configs
COPY docs /app/docs
COPY scripts /app/scripts

ARG INSTALL_EXTRAS=""

RUN python -m pip install --upgrade pip \
    && if [ -n "$INSTALL_EXTRAS" ]; then \
        python -m pip install --no-cache-dir ".[${INSTALL_EXTRAS}]"; \
      else \
        python -m pip install --no-cache-dir .; \
      fi

# Add a non-root user for better security
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/runs /app/.cache /app/results \
    && chown -R appuser:appuser /app

COPY docker /app/docker

USER appuser

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["sleep", "infinity"]
