FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy FASTEMBED_CACHE_PATH=/models

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY rag ./rag
RUN uv sync --frozen --no-dev
# bake local models into the image so containers start without downloading
RUN uv run python -c "from rag.models import dense, sparse, reranker; dense(); sparse(); reranker()"

RUN useradd -m app && mkdir -p /app/logs && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"
CMD ["uv", "run", "--no-sync", "uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
