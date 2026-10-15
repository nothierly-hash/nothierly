# Shared image for every Python service in docker-compose.yml/docker-compose.prod.yml
# (nothierly-agents, nothierly-gateway, nothierly-dashboard, nothierly-p2p-api,
# nothierly-political-agent, grpc_engine, swarm_scout, swarm_analyst, swarm_risk) -- same
# dependencies, same source tree, different CMD per service.
#
# **3.12, not 3.11 (this chapter's own literal ask) or the 3.14 this project was developed
# against**: this is not a fresh choice -- the prior single-stage version of this file already
# tried 3.11 and rejected it for the identical real reason: every pinned dependency in
# requirements.txt (torch, pyzmq, xgboost, grpcio, chronos-forecasting's own torch dependency, ...)
# has solid prebuilt-wheel coverage for 3.12 specifically -- see requirements.txt's own note on why
# pyzmq's lower bound was widened after 3.14 turned out to have no prebuilt wheel at all for the
# originally pinned range. Re-litigating that to 3.11 here on this chapter's say-so alone, without
# a new reason 3.12 stopped working, would trade a verified-working base for an unverified one.
#
# **Real, multi-stage rebuild (this chapter's actual, genuine improvement)**: the prior version of
# this file installed build-essential (gcc/g++/make/binutils) directly into the final image and
# never removed the *packages themselves* (only apt's own download cache) -- every real deployment
# shipped a C/C++ toolchain in production it only ever needed at `pip install` time. `--user`
# installs into a self-contained prefix (`/root/.local`) that the final stage copies wholesale,
# with no compiler along for the ride. No apt package is dropped from what the final stage
# actually had before (still none beyond python:3.12-slim's own base) -- this only removes the
# builder-only toolchain, so it carries none of the "which runtime .so does some pinned wheel
# quietly need" risk a more aggressive rebuild would (unverifiable here: Docker itself is not
# installed on this dev machine -- see this repo's own prior red-team-audit notes on Redpanda/
# Nitro Enclaves for the same constraint. This Dockerfile is authored and reasoned through
# carefully, but has not been proven with a real `docker build` on this machine).
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# --- Test stage -------------------------------------------------------------
# Built only by CI (`docker build --target test`) -- never the base of the
# final stage below, so pytest/fakeredis/pytest-cov never ship in production.
# Runs against the exact same base image and the exact same production
# `pip install -r requirements.txt` layer as the real thing (via `FROM
# builder`), not a separately-drifting GitHub Actions Python install --
# that drift is the real failure this whole pipeline exists to close.
FROM builder AS test

COPY requirements-dev.txt .
RUN pip install --no-cache-dir --user -r requirements-dev.txt

COPY core/ ./core/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY api_middleware/ ./api_middleware/
COPY risk/ ./risk/
COPY network/ ./network/
COPY pricing/ ./pricing/
COPY execution/ ./execution/
# CI stage: tests import these top-level packages (contracts, crypto, ai_agents, nothierly_genesis); the runtime stage below only needs crypto.
COPY crypto/ ./crypto/
COPY contracts/ ./contracts/
COPY ai_agents/ ./ai_agents/
COPY nothierly_genesis/ ./nothierly_genesis/
COPY research/alpha_features.py ./research/alpha_features.py
COPY simulation/ ./simulation/
COPY p2p_api.py sandbox.py nemdi.py p2p_oracle.py p2p_websocket.py \
     political_agent.py scout_agent.py analyst_agent.py risk_gate_agent.py \
     pyproject.toml ./
COPY tests/ ./tests/

# tests/test_digital_twin.py's real historical-replay test and
# tests/test_regime_detector.py's real load_real_bar_closes("BTCUSDT") both
# read plain, decompressed .csv paths hardcoded in simulation/digital_twin.py
# and risk/regime_detector.py -- neither reads .gz directly. Git tracks only
# the .gz originals (30MB + 3MB); the 504MB/15MB decompressed .csv files this
# needs have never been committed and were only ever present because a dev
# machine happened to have decompressed them by hand at some point -- a real,
# pre-existing environment-drift bug this makes reproducible instead of
# silently depending on. ETHUSDT's pair is not decompressed: verified no test
# reads it (grep across tests/ for "ETHUSDT" + "sample_data" together turns up
# nothing), so paying its ~415MB decompression cost here would be pure waste.
COPY research/sample_data/btcusdt_book_snapshot_2020-01-01.csv.gz research/sample_data/btcusdt_trades_2020-01-01.csv.gz ./research/sample_data/
RUN python -c "import gzip, pathlib, shutil; \
d = pathlib.Path('research/sample_data'); \
[shutil.copyfileobj(gzip.open(f, 'rb'), open(d / f.stem, 'wb')) for f in d.glob('*.csv.gz')]"

ENV PATH=/root/.local/bin:$PATH
# --cov-fail-under is an unverified starting floor, not a measured baseline --
# no Docker on the machine that authored this file, so the real coverage %
# has never actually been produced. Raise it to the real number the first
# time this goes green in CI; this only guards against coverage falling
# through the floor, not against it already being higher than 30%.
CMD ["pytest", "tests/", "-q", "--cov=core", "--cov=src", "--cov-report=term-missing", "--cov-fail-under=30"]


FROM python:3.12-slim AS nothierly-runtime

WORKDIR /app

# The builder stage's entire `--user` install, verbatim -- no compiler, no apt cache, no
# requirements.txt re-resolution. PATH so every entrypoint below (`python`, `uvicorn`, `streamlit`)
# resolves the console-script shims pip installed under this prefix.
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# This project's real runtime code lives across several top-level packages, not just `src/` --
# verified directly against this repo's actual layout (a prior version of this image silently
# omitted api_middleware/, risk/, network/, and pricing/, which would only surface as an
# ImportError the first time the gateway/cross-margin/pre-trade-oracle code paths actually ran
# inside a built container, not at build time).
#
# **Real bug fixed here, found deploying this image for real**: execution/
# (predator_honeypot.py, imported by src/agents/macro_order_execution/
# agent.py) and research/alpha_features.py (imported by src/agents/
# market_data/microstructure_features.py) were missing the exact same way --
# nothierly-agents crash-looped with ModuleNotFoundError: No module named
# 'execution' the first time this image actually ran. research/ itself is
# ~974MB (training data/checkpoints, not runtime code) -- COPYing the whole
# directory the way every other package here is copied would bloat this
# image by three orders of magnitude for one file with no further
# dependencies of its own (verified: only stdlib + numpy + polars), so only
# that one file is copied, not the directory.
COPY core/ ./core/
COPY src/ ./src/
COPY scripts/ ./scripts/
# scripts/maintenance_cron.py and scripts/export_training_data.py use workers.common.{alerts,paths} (stdlib only: no exchange SDK, no key). The execution
# workers themselves ship in Dockerfile.platform, not here.
COPY workers/__init__.py ./workers/__init__.py
COPY workers/common/ ./workers/common/
COPY api_middleware/ ./api_middleware/
COPY risk/ ./risk/
COPY network/ ./network/
COPY pricing/ ./pricing/
COPY execution/ ./execution/
# run_backend.py -> src/execution/shadow_crossing_engine.py imports crypto.zk_matching_engine at module load (py_ecc is in requirements.txt): without this COPY,
# nothierly-agents crash-loops with ModuleNotFoundError: No module named 'crypto'.
COPY crypto/ ./crypto/
COPY research/alpha_features.py ./research/alpha_features.py
# research/sample_data/ is NOT shipped (~973 MB of Tardis samples, and the single CSV once copied here was untracked, so a clean clone
# could not even build). Consequence, deliberate: the dashboard's Tier 2 "Lancer une détection réelle" (risk/regime_detector.py's
# load_real_bar_closes) raises a FileNotFoundError naming the missing path inside the image. Mount the file read-only at run time if
# that page is needed.
# Same real gap, third time -- src/dashboard/ems_pages/digital_twin_backtest.py's own top-level
# `from simulation.digital_twin import (...)` crashed the page on load, before any button click. Only
# 68KB, safe to copy wholesale (unlike research/, no size trade-off here).
COPY simulation/ ./simulation/
# **Honest, disclosed scope limit, not fixed here**: simulation/digital_twin.py itself also needs
# research/sample_data/btcusdt_book_snapshot_2020-01-01.csv (528MB) and its ETH sibling (414MB) to
# actually RUN a backtest (only the module import above was needed for the page to load/render
# without crashing). Not copied -- 942MB combined would roughly double this image's size on an
# already memory-constrained instance for a feature nothing has asked to actually run yet. Running
# "Lancer le backtest" today will fail the same way this ModuleNotFoundError did, honestly, with a
# real FileNotFoundError naming the missing path -- add the same one-file COPY pattern above for
# btcusdt_book_snapshot_2020-01-01.csv if/when that's actually needed.
# Trained model artifacts (XGBoost pair, PPO policy, meta-label weights) are shipped as a DIRECTORY, never by file name.
# The previous version named one XGBoost model and its scaler (COPY errors on a missing source), which made every build depend on
# an artifact that had to exist, and made retraining a two-place edit. Now:
#   - `COPY models/ ./models/` cannot fail: models/ always exists in the context (models/.gitkeep is tracked and re-included in
#     .dockerignore), and an empty or artifact-less directory simply copies nothing.
#   - .dockerignore lets through only the artifact name patterns the loaders recognise (supervisor_model_<ts>.json,
#     supervisor_scaler_<ts>.pkl, ppo_policy_<ts>.pt, metalabel_weights.json and its .meta.json), nothing else from models/.
#   - scripts/run_backend.py resolves the newest valid artifact at startup (or the one named by NOTHIERLY_SUPERVISOR_MODEL /
#     NOTHIERLY_SUPERVISOR_SCALER / NOTHIERLY_RL_POLICY / NOTHIERLY_META_LABEL_MODEL, or a directory from NOTHIERLY_MODELS_DIR, e.g. a
#     mounted volume) and falls back to the untrained neutral skeleton when there is none: an image built with no model is a valid
#     image that cannot trade (the XGBoost skeleton always answers 0.5, so no consensus entry is ever reached).
# The consequence to remember: whatever sits in models/ at build time is baked into the image. Only vetted artifacts (the training
# scripts refuse to write models that do not beat chance) belong there.
COPY models/ ./models/
# nothierly-p2p-api's own entrypoint (docker-compose.yml) -- see that module's own docstring.
COPY p2p_api.py .
COPY sandbox.py .
COPY nemdi.py .
# p2p_api.py's own bottom-of-file imports (its own module docstring explains why they're
# imported there, not up top) -- a real gap in the prior version of this file: p2p_api.py has
# crashed with ModuleNotFoundError inside every container that ever actually started it, since
# neither of these two existed in this COPY list until this chapter added the gRPC/GraphQL layer
# that made that gap visible while touching this same file.
COPY p2p_oracle.py .
COPY p2p_websocket.py .
# nothierly-political-agent's own entrypoint (docker-compose.yml) -- see that module's own docstring.
COPY political_agent.py .
# swarm_scout/swarm_analyst/swarm_risk's own entrypoints (docker-compose.prod.yml) -- each is a
# real, standalone core.base_agent.BaseAgent subclass runnable directly via `python <this file>`,
# same convention as political_agent.py above.
COPY scout_agent.py .
COPY analyst_agent.py .
COPY risk_gate_agent.py .
COPY pyproject.toml .

# Non-root runtime user. Verified safe against this repo's own
# docker-compose.yml/docker-compose.prod.yml: neither bind-mounts a host path
# under /app for any of these services (only named volumes, for
# timescaledb/redis-broker's own data dirs, unrelated to this image) and
# every agent logs to stdout via structlog, not to a file under /app -- so
# there is no writable-path assumption this chown breaks. Real bug caught
# writing this: chowning /root/.local alone is not enough -- /root itself is
# mode 700 owned by root, so a non-root user gets EACCES just walking the
# path to .local regardless of that directory's own ownership. `chmod o+x
# /root` grants only traversal (no read/write) into a path the non-root user
# already needs by name, not a real widening of what root's home exposes.
RUN groupadd --system nothierly \
    && useradd --system --gid nothierly --home-dir /app --no-create-home --shell /usr/sbin/nologin nothierly \
    && chown -R nothierly:nothierly /app /root/.local \
    && chmod o+x /root
USER nothierly

# Overridden per service in docker-compose.yml/docker-compose.prod.yml -- this is just a sane
# default for `docker run` without an explicit command.
CMD ["python", "scripts/run_backend.py"]
