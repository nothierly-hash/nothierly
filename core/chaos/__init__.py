"""Opt-in chaos-engineering instrumentation for Shadow Run load/fault testing.

Every module under this package is inert unless explicitly wired in by a
``scripts/chaos_*`` launcher -- nothing here is imported by any production
agent module (``src/agents/**``) or by ``scripts/run_backend.py``. That
separation is deliberate: chaos instrumentation (event-loop lag probes,
tracemalloc snapshots, fault-injecting exchange wrappers) has no business
running during a real LIVE deployment, and keeping it in its own package
makes that true by construction, not by convention.
"""
