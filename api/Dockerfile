FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY api ./api
COPY configs ./configs
COPY data/processed ./data/processed
COPY outputs ./outputs
COPY report ./report

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -e ".[api,ml]"

ENV STATVOCAB_CONFIG=configs/core.yaml
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
