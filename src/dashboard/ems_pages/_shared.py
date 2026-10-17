"""Shared bootstrap + real-data helpers for every EMS page.

Each page under ``ems_pages/`` is executed by ``st.Page`` as its own
script -- ``sys.path`` needs the project root added the same way
``src/dashboard/app.py`` and ``src/dashboard/nothierly_ems.py`` already
do, and every page needs the same cached Redis client and small
"read safely, never crash a page on a Redis hiccup" helpers, so it
lives here once rather than nine times.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import json
from typing import Any

import redis
import streamlit as st

from core.config import get_settings
from core.iam import IamStore
from src.execution.security_vault import SecurityVault


@st.cache_resource
def get_redis_client() -> redis.Redis[str]:
    """The dashboard's own synchronous Redis client -- see src/dashboard/app.py's identical helper."""
    settings = get_settings().redis
    return redis.Redis(
        host=settings.host, port=settings.port, db=settings.db,
        password=settings.password.get_secret_value() if settings.password is not None else None,
        ssl=settings.use_tls, decode_responses=True,
    )


@st.cache_resource
def get_iam_store() -> IamStore:
    """The dashboard's real user/session/2FA store -- see core/iam.py's module docstring.

    Cached like :func:`get_redis_client`: one instance per Streamlit
    server process, reused across every page's rerun and every visitor's
    session (the store itself is stateless besides its Redis handle --
    per-user state lives in Redis, not on this object).
    """
    return IamStore(get_redis_client(), SecurityVault(get_settings().vault.master_key.get_secret_value()))


def safe_get(r: redis.Redis[str], key: str) -> str | None:
    try:
        value = r.get(key)
        return str(value) if value is not None else None
    except redis.RedisError:
        return None


def safe_get_json(r: redis.Redis[str], key: str) -> dict[str, Any] | None:
    raw = safe_get(r, key)
    if raw is None:
        return None
    try:
        result: dict[str, Any] = json.loads(raw)
        return result
    except (json.JSONDecodeError, TypeError):
        return None


def safe_lrange(r: redis.Redis[str], key: str, count: int) -> list[str]:
    try:
        return list(r.lrange(key, 0, count - 1))
    except redis.RedisError:
        return []


def safe_keys(r: redis.Redis[str], pattern: str) -> list[str]:
    try:
        return list(r.keys(pattern))
    except redis.RedisError:
        return []


def not_yet_implemented(feature: str, *, reason: str) -> None:
    """Honest, consistent placeholder -- never a fabricated chart/number in its place."""
    st.info(f"**{feature}** -- pas encore implémenté.\n\n{reason}")


def demo_mode_toggle(*, key: str, label: str = "🎭 Mode Démo (données simulées)") -> bool:
    """One consistent, explicit, off-by-default toggle for showing mock data instead of real (possibly empty) state.

    Same discipline established in ``ultimate_terminal.py``'s
    ``generate_5b_demo_data``: mock data is fine for a B2B sales demo,
    but only behind an explicit opt-in with an unmissable banner while
    active -- never silently substituted for real data by default. Every
    page using this returns to real, honest data (including honest
    empty states) the moment it's switched back off.
    """
    enabled = st.checkbox(label, value=False, key=key, help="Active des données simulées pour la démonstration -- jamais affichées sans cette bannière.")
    if enabled:
        st.warning("🎭 **MODE DÉMO ACTIF** -- les données ci-dessous sont simulées, pas les données réelles du système.")
    return enabled
