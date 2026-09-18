-- Retail Gateway schema -- applied idempotently on API startup (see db.py::init_schema).
-- Deliberately no migration framework (alembic, etc.): one table, one schema version so far.
-- Add a real migrations/ directory the day this needs to evolve under live data rather than
-- "drop and recreate" during development -- disclosed future work, not built preemptively.

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY,
    email TEXT,
    -- Case-insensitive uniqueness: "User@x.com" and "user@x.com" are the same real account.
    -- Enforced on this normalized column, not on `email` itself, so the user's original
    -- capitalization is still preserved for display/the welcome email. Nullable (a phone-only
    -- OTP account has no email at all) -- the UNIQUE constraint on a nullable column still
    -- correctly allows any number of NULLs while still forbidding two rows with the same
    -- non-NULL value, exactly the semantics wanted here.
    email_normalized TEXT UNIQUE,
    phone_number TEXT UNIQUE,
    -- Nullable: a passwordless account created via OTP verification (see repository.py's
    -- get_or_create_by_identifier) genuinely has no password to hash. A row must still have at
    -- least one of email/phone/password -- enforced by the CHECK below, not just by convention.
    password_hash TEXT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT users_has_an_identity CHECK (email_normalized IS NOT NULL OR phone_number IS NOT NULL)
);

-- Real, idempotent schema evolution for a table that may already exist from before the Identity
-- Gateway mission (email_normalized/password_hash used to be NOT NULL) -- see this file's own
-- top comment on why ALTER here, not a migrations/ framework, at this project's current stage.
-- Postgres accepts DROP NOT NULL / ADD COLUMN IF NOT EXISTS as safe no-ops when already applied.
ALTER TABLE users ALTER COLUMN email DROP NOT NULL;
ALTER TABLE users ALTER COLUMN password_hash DROP NOT NULL;
ALTER TABLE users ADD COLUMN IF NOT EXISTS phone_number TEXT UNIQUE;

CREATE INDEX IF NOT EXISTS idx_users_email_normalized ON users (email_normalized);
CREATE INDEX IF NOT EXISTS idx_users_phone_number ON users (phone_number);

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
