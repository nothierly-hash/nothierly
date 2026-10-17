# Retail Gateway's OWN image -- shared by its two real processes (the API and the email worker,
# same "one image, different CMD per service" convention the repo root's Dockerfile already uses
# for its own 9 services), but never shared WITH those trading-core services.
#
# **Build context is the repo root** (docker-compose.retail.yml's `build: {context: .,
# dockerfile: retail_gateway/Dockerfile}`), not this directory alone -- needed so `COPY
# retail_gateway/` below preserves the `retail_gateway.` package path `uvicorn`/`celery` import
# below. This does NOT reintroduce the air-gap this package's own docstrings insist on: only the
# `retail_gateway/` subtree is ever copied (see the one COPY line below) -- no `core/`, `src/`,
# torch, ccxt, or any other trading dependency ever enters this image's filesystem. The two
# images share a build *context* on disk, never a layer, a process, or an event loop.
#
# Same multi-stage/--user rationale as the repo root's own Dockerfile (its own comment explains
# the real reason: no compiler toolchain shipped in the final image) -- not proven with a real
# `docker build` on this dev machine (Docker itself isn't installed here), same disclosed
# limitation that Dockerfile's own header already states.
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY retail_gateway/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY retail_gateway/ ./retail_gateway/

# Overridden per service in docker-compose.retail.yml (API vs. Celery worker command).
CMD ["uvicorn", "retail_gateway.main:app", "--host", "0.0.0.0", "--port", "8100"]
