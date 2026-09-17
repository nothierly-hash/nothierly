-- Retail Gateway schema -- applied idempotently on API startup (see db.py::init_schema).
-- Deliberately no migration framework (alembic, etc.): one table, one schema version so far.
-- Add a real migrations/ directory the day this needs to evolve under live data rather than
-- "drop and recreate" during development -- disclosed future work, not built preemptively.

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY,
    email TEXT NOT NULL,
    -- Case-insensitive uniqueness: "User@x.com" and "user@x.com" are the same real account.
    -- Enforced on this normalized column, not on `email` itself, so the user's original
    -- capitalization is still preserved for display/the welcome email.
    email_normalized TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_users_email_normalized ON users (email_normalized);

-- Read-only public performance cache -- the ONLY table the B2C dashboard's public endpoint
-- reads from. Populated exclusively by a one-way export job that runs on the trading-core
-- side (scripts/export_performance_snapshot.py, in the main repo, not here) and writes INTO
-- this database over the network -- this service itself never reads FROM the trading core in
-- either direction. See retail_gateway/main.py's module docstring for the full data-flow
-- argument.
CREATE TABLE IF NOT EXISTS public_performance_snapshot (
    id SERIAL PRIMARY KEY,
    metric_name TEXT NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    as_of TIMESTAMPTZ NOT NULL,
    published_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_perf_snapshot_metric_as_of ON public_performance_snapshot (metric_name, as_of DESC);
