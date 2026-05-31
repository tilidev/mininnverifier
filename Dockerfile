FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
RUN uv sync --frozen --no-dev

ENV NUMBA_CACHE_DIR=/tmp/numba-cache
RUN mkdir -p /tmp/numba-cache

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["python", "-m", "cli"]
