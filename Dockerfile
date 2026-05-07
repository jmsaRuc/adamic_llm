FROM ghcr.io/astral-sh/uv:0.11-python3.13-trixie-slim AS uv

# -----------------------------------
# STAGE 1: prod stage
# Only install main dependencies
# -----------------------------------
FROM python:3.13-slim-trixie AS prod
RUN apt-get update && apt-get install -y \
  gcc \
  && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/usr/local
ENV UV_PYTHON_DOWNLOADS=never
ENV UV_NO_MANAGED_PYTHON=1

WORKDIR /app/src

RUN --mount=from=uv,source=/usr/local/bin/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-group dev-full

COPY . .

RUN --mount=from=uv,source=/usr/local/bin/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-group dev-full 

CMD ["/usr/local/bin/python", "-m", "adamic_llm"]

# -----------------------------------
# STAGE 3: development build
# Includes dev dependencies
# -----------------------------------
FROM prod AS dev

RUN --mount=from=uv,source=/usr/local/bin/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --group testing-lint
