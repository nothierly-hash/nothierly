"""``users`` table access -- plain ``asyncpg``, no ORM (same project-wide call as
``core/timescale_client.py``'s own docstring makes: this service's queries are a handful of
straightforward single-row lookups/inserts, exactly what raw ``asyncpg`` already handles well).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

import asyncpg


class EmailAlreadyRegisteredError(Exception):
    pass


@dataclass
class UserRecord:
    id: uuid.UUID
    email: str
    password_hash: str
    is_active: bool
    created_at: datetime


class UserRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def create_user(self, *, email: str, password_hash: str) -> UserRecord:
        user_id = uuid.uuid4()
        normalized = email.strip().lower()
        try:
            row = await self._pool.fetchrow(
                """
                INSERT INTO users (id, email, email_normalized, password_hash)
                VALUES ($1, $2, $3, $4)
                RETURNING id, email, password_hash, is_active, created_at
                """,
                user_id,
                email.strip(),
                normalized,
                password_hash,
            )
        except asyncpg.UniqueViolationError as exc:
            raise EmailAlreadyRegisteredError(email) from exc
        assert row is not None  # INSERT ... RETURNING always returns exactly one row on success
        return UserRecord(**dict(row))

    async def get_by_email(self, email: str) -> UserRecord | None:
        normalized = email.strip().lower()
        row = await self._pool.fetchrow(
            """
            SELECT id, email, password_hash, is_active, created_at
            FROM users
            WHERE email_normalized = $1
            """,
            normalized,
        )
        return UserRecord(**dict(row)) if row is not None else None

    async def get_by_id(self, user_id: uuid.UUID) -> UserRecord | None:
        row = await self._pool.fetchrow(
            """
            SELECT id, email, password_hash, is_active, created_at
            FROM users
            WHERE id = $1
            """,
            user_id,
        )
        return UserRecord(**dict(row)) if row is not None else None

    async def list_recent_performance_metrics(self, *, limit: int = 50) -> list[asyncpg.Record]:
        """Read-only feed for the public dashboard -- see ``schema.sql``'s own comment on the
        one-way write path this table has (an external export job, never this API).
        """
        # `# type: ignore[no-any-return]`: asyncpg ships no stubs (pyproject.toml's own
        # ignore_missing_imports=true for asyncpg.*) -- `Pool.fetch` resolves to `Any`, which
        # mypy strict still flags against this method's concrete `list[asyncpg.Record]` return
        # annotation even though `asyncpg.Record` itself is also unresolved. Same real gap this
        # project already lives with everywhere else it types an asyncpg call precisely.
        return await self._pool.fetch(  # type: ignore[no-any-return]
            """
            SELECT DISTINCT ON (metric_name) metric_name, metric_value, as_of
            FROM public_performance_snapshot
            ORDER BY metric_name, as_of DESC
            LIMIT $1
            """,
            limit,
        )
