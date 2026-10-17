"""Shared execution-layer infrastructure: broker abstraction, order slicing, secrets, data normalization.

Distinct from ``src/agents/`` (standalone agent processes) and ``core/``
(generic cross-cutting infra with no trading-domain logic): this package
holds trading-domain code meant to be imported by *multiple* consumers
(the API gateway, a future order-execution agent, the dashboard) rather
than owned by one agent's subpackage.
"""

from __future__ import annotations
