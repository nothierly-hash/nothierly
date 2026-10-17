"""Cognitive Router API -- a real ZMQ ROUTER-DEALER multi-agent LLM task hypervisor.

See :mod:`hypervisor.main_hypervisor`'s own module docstring for the full design and its honest,
disclosed scope: this project has no real OpenAI/Anthropic/Vast.ai credentials configured (see
that module's own docstring) -- the routing, pricing, DAG-planning, ZMQ transport, and Redis
persistence below are all real; the four "workers'" actual text generation honestly falls back to
clearly-labeled placeholder content when no real API key is configured, exactly the same
established pattern as :mod:`src.agents.guardian_ai`'s own ``llm_provider="none"`` fallback.
"""

from __future__ import annotations
