"""``users`` table access -- plain ``asyncpg``, no ORM (same project-wide call as
``core/timescale_client.py``'s own docstring makes: this service's queries are a handful of
straightforward single-row lookups/inserts, exactly what raw ``asyncpg`` already handles well).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import asyncpg


class EmailAlreadyRegisteredError(Exception):
    pass


@dataclass
class UserRecord:
    id: uuid.UUID
    email: str | None
    password_hash: str | None
    is_active: bool
    created_at: datetime
    phone_number: str | None = None


class UserRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    _SELECT_COLUMNS = "id, email, phone_number, password_hash, is_active, created_at"

    async def create_user(self, *, email: str, password_hash: str) -> UserRecord:
        user_id = uuid.uuid4()
        normalized = email.strip().lower()
        try:
            row = await self._pool.fetchrow(
                f"""
                INSERT INTO users (id, email, email_normalized, password_hash)
                VALUES ($1, $2, $3, $4)
                RETURNING {self._SELECT_COLUMNS}
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
            f"SELECT {self._SELECT_COLUMNS} FROM users WHERE email_normalized = $1",
            normalized,
        )
        return UserRecord(**dict(row)) if row is not None else None

    async def get_by_phone(self, phone_number: str) -> UserRecord | None:
        row = await self._pool.fetchrow(
            f"SELECT {self._SELECT_COLUMNS} FROM users WHERE phone_number = $1",
            phone_number,
        )
        return UserRecord(**dict(row)) if row is not None else None

    async def get_by_id(self, user_id: uuid.UUID) -> UserRecord | None:
        row = await self._pool.fetchrow(
            f"SELECT {self._SELECT_COLUMNS} FROM users WHERE id = $1",
            user_id,
        )
        return UserRecord(**dict(row)) if row is not None else None

    async def get_or_create_by_identifier(self, *, channel: Literal["email", "sms"], identifier: str) -> UserRecord:
        """The real account-provisioning step behind a successful OTP verification.

        A first-ever OTP verify for a given email/phone *is* signup here -- there is no separate
        "create account" step in the passwordless flow the brief describes (unlike the existing
        password-based ``POST /api/v1/signup``, which stays a distinct, explicit endpoint). No
        password is set (``password_hash`` stays NULL -- see ``schema.sql``'s own
        ``users_has_an_identity`` check for why that is safe: the row still has a real email or
        phone identity). A race between two concurrent first-verifies for the same identifier is
        resolved by ``ON CONFLICT DO UPDATE ... RETURNING``, a single atomic round-trip rather
        than a real check-then-insert race.
        """
        user_id = uuid.uuid4()
        if channel == "email":
            normalized = identifier.strip().lower()
            row = await self._pool.fetchrow(
                f"""
                INSERT INTO users (id, email, email_normalized)
                VALUES ($1, $2, $3)
                ON CONFLICT (email_normalized) DO UPDATE SET email_normalized = EXCLUDED.email_normalized
                RETURNING {self._SELECT_COLUMNS}
                """,
                user_id, identifier.strip(), normalized,
            )
        else:
            row = await self._pool.fetchrow(
                f"""
                INSERT INTO users (id, phone_number)
                VALUES ($1, $2)
                ON CONFLICT (phone_number) DO UPDATE SET phone_number = EXCLUDED.phone_number
                RETURNING {self._SELECT_COLUMNS}
                """,
                user_id, identifier,
            )
        assert row is not None
        return UserRecord(**dict(row))

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
