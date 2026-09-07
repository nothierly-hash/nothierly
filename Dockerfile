# Shared image for the `nothierly-agents` (trading agents), `nothierly-
# gateway` (FastAPI B2B order-ingestion API), and `nothierly-dashboard`
# (Streamlit control room) Docker Compose services -- same dependencies,
# same source tree, different CMD per service (see docker-compose.yml).
#
# 3.12, not 3.11 or the 3.14 this project was developed against: every
# pinned dependency in requirements.txt (torch, pyzmq, xgboost, ...) has
# solid prebuilt-wheel coverage for 3.12 specifically -- see
# requirements.txt's own note on why pyzmq's lower bound was widened
# after 3.14 turned out to have no prebuilt wheel at all for the
# originally pinned range.
FROM python:3.12-slim

WORKDIR /app

# build-essential: covers the rare pinned dependency that falls back to
# a source build on this platform/architecture even with the wheel
# ranges already chosen for broad prebuilt-wheel coverage. Removed from
# the image in the same layer it's installed in, via apt's own cache
# cleanup, to keep the final image lean.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dependencies installed before the source tree is copied, so an
# ordinary code change doesn't invalidate this (slow, torch-sized) layer
# on every rebuild -- only a requirements.txt change does.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# This project's real runtime code lives across seven top-level packages,
# not just `src/` -- verified directly against this repo's actual layout
# (a prior version of this image silently omitted api_middleware/, risk/,
# network/, and pricing/, which would only surface as an ImportError the
# first time the gateway/cross-margin/pre-trade-oracle code paths actually
# ran inside a built container, not at build time).
COPY core/ ./core/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY api_middleware/ ./api_middleware/
COPY risk/ ./risk/
COPY network/ ./network/
COPY pricing/ ./pricing/
COPY pyproject.toml .

# Overridden per service in docker-compose.yml (`nothierly-agents` runs
# scripts/run_backend.py; `nothierly-gateway` runs uvicorn against
# api_middleware.gateway:app; `nothierly-dashboard` runs `streamlit run
# src/dashboard/nothierly_ems.py`) -- this is just a sane default for
# `docker run` without an explicit command.
CMD ["python", "scripts/run_backend.py"]
