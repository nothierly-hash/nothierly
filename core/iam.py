"""Chapter 13, Étape 5.9: real Identity & Access Management backend.

**Real gap this closes**: before this module, "authentication" in this
project was exactly one thing -- :func:`src.dashboard.app.check_password`
(now archived), a single shared ``DASHBOARD_USER``/``DASHBOARD_PASSWORD``
pair compared with ``secrets.compare_digest``. There was no per-user
account, no role, no session/device history, no 2FA, and -- more
urgently -- the *current* live router, ``src/dashboard/nothierly_ems.py``,
calls no login gate at all: it is reachable with zero authentication.
Every feature the brief asks for (password change with re-auth, TOTP
2FA, backup codes, session history, IP whitelisting, admin user
management, global session timeout) requires a real multi-user store
that simply did not exist. This module is that store, plus the door
that was missing.

**Storage**: Redis-backed (matching every other dashboard-facing piece
of this project -- there is no Postgres/Timescale dependency for
dashboard-only concerns, and Timescale is reserved for durable
per-client billing fills, a different concern). One JSON hash per user
under ``nothierly:iam:user:{username}``, a capped session-history list
per user, a global session-timeout string, and a per-user IP allowlist
set.

**Password hashing**: `argon2-cffi <https://pypi.org/project/argon2-cffi/>`_
(``argon2.PasswordHasher``, Argon2id) -- verified installed and working
on this machine before being adopted (see the module's own test suite).
Argon2id is the PHC-winning, GPU/ASIC-resistant KDF the brief asked for
by name, not bcrypt.

**TOTP secrets** are never stored in plaintext: each is encrypted at
rest with :class:`src.execution.security_vault.SecurityVault`, the same
Fernet-based vault already protecting broker credentials, decrypted
only in-process at verification time.

**"Terminate all other sessions" -- an honest mechanism, not a fake
button**: this is a stateless multi-session Streamlit app with no
process-wide registry of open browser connections to forcibly
disconnect (there is no per-connection RPC channel to another
person's already-open tab). What *is* real and enforceable is a
per-user **set of live session tokens** in Redis: each real login
mints a fresh random token (:meth:`IamStore.register_session`) held in
that browser's ``st.session_state`` and checked against the set on
every rerun (:meth:`IamStore.is_session_still_valid`) -- every
Streamlit interaction is a full script rerun, so this check runs on
every click/navigation, not just once. Because each session has its
own distinct token, :meth:`IamStore.terminate_all_other_sessions` can
revoke every *other* token while sparing the caller's own -- a real
login-timestamp-only scheme cannot make that distinction when two
sessions happen to start close together. A genuine credential change
(password change, forced reset, account lock) instead calls
:meth:`IamStore.terminate_all_sessions`, revoking the caller's own
session too -- deliberately, since a session established under a
password that just changed should not keep trusting that login.

**IP whitelisting -- real storage, honest scope**: this module stores
the allowlist and answers "is this IP allowed" for the app to check at
its own login gate. It cannot make that check unbypassable on its own
-- a determined attacker who can reach the process at all bypasses any
application-level IP check; real enforcement belongs at a reverse
proxy or firewall (nginx `allow`/`deny`, a cloud security group). This
module gives the admin a real toggle and a real list to manage; it is
not represented as a substitute for network-level enforcement.
"""

from __future__ import annotations

import base64
import io
import json
import secrets
import time
from dataclasses import dataclass, field
from typing import Literal

import pyotp
import qrcode
import redis
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

from src.execution.security_vault import SecurityVault, VaultDecryptionError

Role = Literal["admin", "trader"]

_USER_KEY_PREFIX = "nothierly:iam:user:"
_SESSIONS_KEY_PREFIX = "nothierly:iam:sessions:"
_IP_WHITELIST_KEY_PREFIX = "nothierly:iam:ip_whitelist:"
_ACTIVE_SESSIONS_KEY_PREFIX = "nothierly:iam:active_sessions:"
_GLOBAL_TIMEOUT_KEY = "nothierly:iam:global_session_timeout_minutes"
_BACKUP_CODES_KEY_PREFIX = "nothierly:iam:backup_codes:"

_MAX_SESSION_HISTORY = 50
_DEFAULT_GLOBAL_TIMEOUT_MINUTES = 15
_BACKUP_CODE_COUNT = 10

_hasher = PasswordHasher()


def _decode(value: str | bytes | None) -> str | None:
    """Defensive str/bytes normalization for Redis read results.

    Real ``redis-py`` with ``decode_responses=True`` always returns
    ``str``. The ``fakeredis`` version this project's test suite pins
    does not fully honor that flag for every command (``get``/``keys``
    in particular -- see ``tests/test_alert_agent.py``'s identical
    finding), returning raw ``bytes`` instead. This module's own
    control flow (string equality checks, key-prefix slicing) breaks
    silently -- not loudly -- on an unexpected ``bytes`` value, so it
    is normalized once here rather than trusted at every call site.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


class IamError(Exception):
    """Base class for IAM operation failures the caller is expected to handle (never a raw KeyError/etc.)."""


class UserNotFoundError(IamError):
    pass


class InvalidCredentialsError(IamError):
    pass


class UserLockedError(IamError):
    pass


@dataclass
class IamUser:
    """One account. ``totp_secret_encrypted`` is Fernet ciphertext, never plaintext -- see module docstring."""

    username: str
    password_hash: str
    role: Role
    locked: bool = False
    totp_enabled: bool = False
    totp_secret_encrypted: str | None = None
    must_change_password: bool = False
    created_at: float = field(default_factory=lambda: time.time())

    def to_json(self) -> str:
        return json.dumps(
            {
                "username": self.username,
                "password_hash": self.password_hash,
                "role": self.role,
                "locked": self.locked,
                "totp_enabled": self.totp_enabled,
                "totp_secret_encrypted": self.totp_secret_encrypted,
                "must_change_password": self.must_change_password,
                "created_at": self.created_at,
            }
        )

    @staticmethod
    def from_json(raw: str) -> "IamUser":
        data = json.loads(raw)
        return IamUser(**data)


class IamStore:
    """Redis-backed user store. Deliberately takes a plain sync ``redis.Redis`` client (Streamlit pages are sync).

    Deliberately does **not** import ``streamlit``: the caller (a
    dashboard page) is responsible for pulling ``st.context.ip_address``
    / the ``User-Agent`` header and passing them in as plain strings --
    that keeps this module unit-testable against ``fakeredis`` with no
    real HTTP request in play, and keeps IAM logic reusable outside the
    dashboard process (e.g. a future API-level login endpoint).
    """

    def __init__(self, redis_client: "redis.Redis[str]", vault: SecurityVault) -> None:
        self._r = redis_client
        self._vault = vault

    # -- bootstrap ---------------------------------------------------

    def ensure_bootstrap_admin(self, *, username: str, password: str) -> None:
        """Idempotently migrate the legacy single shared login into a real IAM admin account.

        Solves the real chicken-and-egg problem a brand-new IAM store
        has (no users exist yet, so nobody could ever log in): if
        ``username`` doesn't already exist, it is created as
        ``role="admin"`` with an Argon2 hash of ``password`` -- the same
        credential operators already have configured via
        ``DASHBOARD_USER``/``DASHBOARD_PASSWORD``. A no-op once that
        account exists, so it is safe to call on every dashboard start.
        """
        if self.get_user(username) is not None:
            return
        self.create_user(username=username, password=password, role="admin")

    # -- user CRUD ---------------------------------------------------

    def create_user(self, *, username: str, password: str, role: Role) -> IamUser:
        if self.get_user(username) is not None:
            raise IamError(f"User {username!r} already exists.")
        user = IamUser(username=username, password_hash=_hasher.hash(password), role=role)
        self._save(user)
        return user

    def get_user(self, username: str) -> IamUser | None:
        raw = _decode(self._r.get(_USER_KEY_PREFIX + username))
        return IamUser.from_json(raw) if raw is not None else None

    def list_users(self) -> list[IamUser]:
        keys = self._r.keys(_USER_KEY_PREFIX + "*")
        usernames = [key[len(_USER_KEY_PREFIX) :] for key in (_decode(k) for k in keys) if key is not None]
        users = [self.get_user(username) for username in usernames]
        return sorted((u for u in users if u is not None), key=lambda u: u.username)

    def _save(self, user: IamUser) -> None:
        self._r.set(_USER_KEY_PREFIX + user.username, user.to_json())

    # -- authentication ---------------------------------------------

    def authenticate(self, *, username: str, password: str, ip: str, user_agent: str) -> IamUser:
        """Verify credentials and log the attempt. Raises rather than returning a falsy sentinel.

        A failed attempt (unknown user, wrong password, locked account)
        is still recorded to the session-history log with its real
        outcome -- an audit trail must see failed logins, not only
        successful ones.
        """
        user = self.get_user(username)
        if user is None:
            self._record_event(username, "login_failed_unknown_user", ip=ip, user_agent=user_agent)
            raise InvalidCredentialsError("Nom d'utilisateur ou mot de passe incorrect.")
        if user.locked:
            self._record_event(username, "login_failed_locked", ip=ip, user_agent=user_agent)
            raise UserLockedError(f"Le compte {username!r} est verrouillé.")
        try:
            _hasher.verify(user.password_hash, password)
        except VerifyMismatchError:
            self._record_event(username, "login_failed_bad_password", ip=ip, user_agent=user_agent)
            raise InvalidCredentialsError("Nom d'utilisateur ou mot de passe incorrect.") from None
        if _hasher.check_needs_rehash(user.password_hash):
            user.password_hash = _hasher.hash(password)
            self._save(user)
        self._record_event(username, "login_succeeded", ip=ip, user_agent=user_agent)
        return user

    def verify_current_password(self, *, username: str, password: str) -> bool:
        """Real Argon2 check against the stored hash, with no side effect -- the Session Confirmation primitive.

        Distinct from :meth:`change_password`: a caller that only needs
        to re-confirm identity before a security-relevant action (e.g.
        regenerating backup codes) must not reuse ``change_password``
        as a verification trick -- that method revokes every session
        (see :meth:`terminate_all_sessions`), which would silently log
        the caller out for an action that never intended to change the
        password at all.
        """
        user = self._require_user(username)
        try:
            _hasher.verify(user.password_hash, password)
        except VerifyMismatchError:
            return False
        return True

    def change_password(self, *, username: str, current_password: str, new_password: str) -> None:
        """Session Confirmation rule: every call requires the real current password, verified via Argon2."""
        user = self._require_user(username)
        try:
            _hasher.verify(user.password_hash, current_password)
        except VerifyMismatchError:
            raise InvalidCredentialsError("Mot de passe actuel incorrect.") from None
        user.password_hash = _hasher.hash(new_password)
        user.must_change_password = False
        self._save(user)
        self.terminate_all_sessions(username)

    def force_password_reset(self, username: str) -> None:
        """Admin action: flags the account so the next login must change its password."""
        user = self._require_user(username)
        user.must_change_password = True
        self._save(user)
        self.terminate_all_sessions(username)

    # -- TOTP / 2FA ---------------------------------------------------

    def begin_totp_enrollment(self, username: str) -> tuple[str, str, bytes]:
        """Generate a fresh TOTP secret + QR PNG. NOT persisted yet -- see :meth:`confirm_totp_enrollment`.

        Returns ``(secret, provisioning_uri, qr_png_bytes)``. The
        caller (the dashboard page) is responsible for holding
        ``secret`` in ``st.session_state`` only until the user proves
        possession by typing back one real code -- standard TOTP
        enrollment flow, prevents locking an account to a secret the
        user's authenticator app never actually received (e.g. a
        mis-scanned QR).
        """
        self._require_user(username)
        secret = pyotp.random_base32()
        totp = pyotp.TOTP(secret)
        uri = totp.provisioning_uri(name=username, issuer_name="Nothierly")
        qr_png = _render_qr_png(uri)
        return secret, uri, qr_png

    def confirm_totp_enrollment(self, *, username: str, pending_secret: str, code: str) -> None:
        """Verify one real code against ``pending_secret`` before persisting it (encrypted) and enabling 2FA."""
        user = self._require_user(username)
        if not pyotp.TOTP(pending_secret).verify(code, valid_window=1):
            raise InvalidCredentialsError("Code invalide -- vérifiez l'heure de votre appareil et réessayez.")
        user.totp_secret_encrypted = self._vault.encrypt(pending_secret)
        user.totp_enabled = True
        self._save(user)

    def disable_totp(self, *, username: str, current_password: str) -> None:
        user = self._require_user(username)
        try:
            _hasher.verify(user.password_hash, current_password)
        except VerifyMismatchError:
            raise InvalidCredentialsError("Mot de passe actuel incorrect.") from None
        user.totp_enabled = False
        user.totp_secret_encrypted = None
        self._save(user)
        self._r.delete(_BACKUP_CODES_KEY_PREFIX + username)

    def verify_totp(self, *, username: str, code: str) -> bool:
        user = self._require_user(username)
        if not user.totp_enabled or user.totp_secret_encrypted is None:
            return False
        try:
            secret = self._vault.decrypt(user.totp_secret_encrypted)
        except VaultDecryptionError:
            return False
        return bool(pyotp.TOTP(secret).verify(code, valid_window=1))

    # -- backup codes ---------------------------------------------------

    def generate_backup_codes(self, username: str) -> list[str]:
        """Generate 10 fresh one-time codes, store only their Argon2 hashes, return plaintext ONCE.

        Regenerating discards every previously issued code (each call
        replaces the whole set) -- matches the brief's "one-time
        display" requirement: there is no way to see them again later,
        by design, only to invalidate and re-generate.
        """
        self._require_user(username)
        codes = [secrets.token_hex(4) for _ in range(_BACKUP_CODE_COUNT)]
        hashed = [_hasher.hash(code) for code in codes]
        self._r.delete(_BACKUP_CODES_KEY_PREFIX + username)
        if hashed:
            self._r.rpush(_BACKUP_CODES_KEY_PREFIX + username, *hashed)
        return codes

    def consume_backup_code(self, *, username: str, code: str) -> bool:
        """Verify + burn one backup code. Each code works exactly once."""
        key = _BACKUP_CODES_KEY_PREFIX + username
        hashed_codes = self._r.lrange(key, 0, -1)
        for hashed in hashed_codes:
            try:
                _hasher.verify(hashed, code)
            except VerifyMismatchError:
                continue
            self._r.lrem(key, 1, hashed)
            return True
        return False

    def remaining_backup_code_count(self, username: str) -> int:
        return self._r.llen(_BACKUP_CODES_KEY_PREFIX + username)

    # -- session history / revocation ---------------------------------------------------

    def _record_event(self, username: str, event_type: str, *, ip: str, user_agent: str) -> None:
        entry = json.dumps({"timestamp": time.time(), "event_type": event_type, "ip": ip, "user_agent": user_agent})
        key = _SESSIONS_KEY_PREFIX + username
        self._r.lpush(key, entry)
        self._r.ltrim(key, 0, _MAX_SESSION_HISTORY - 1)

    def list_sessions(self, username: str, *, limit: int = 20) -> list[dict[str, object]]:
        raw_entries = self._r.lrange(_SESSIONS_KEY_PREFIX + username, 0, limit - 1)
        return [json.loads(raw) for raw in raw_entries]

    def register_session(self, username: str) -> str:
        """Mint and store a fresh, unique session token for one real login. Returns the token.

        Real per-session identity, not a shared login timestamp: a
        login-time-based epoch cannot tell "this browser's own session"
        apart from "every other session that happened to log in around
        the same moment" -- it cannot spare the caller's own session
        when revoking others. A Redis set of live tokens can: see
        :meth:`terminate_all_other_sessions`.
        """
        token = secrets.token_hex(16)
        self._r.sadd(_ACTIVE_SESSIONS_KEY_PREFIX + username, token)
        return token

    def is_session_still_valid(self, *, username: str, session_token: str) -> bool:
        return bool(self._r.sismember(_ACTIVE_SESSIONS_KEY_PREFIX + username, session_token))

    def terminate_session(self, username: str, session_token: str) -> None:
        """Revoke exactly one session -- a real, explicit logout (as opposed to idle-lock, which never revokes)."""
        self._r.srem(_ACTIVE_SESSIONS_KEY_PREFIX + username, session_token)

    def terminate_all_sessions(self, username: str) -> None:
        """Invalidate every session for this user, including the caller's own -- used by change_password/lock/reset.

        A genuine security-relevant change to the account (new
        password, forced reset, lock) revoking the very session that
        made the change is the expected, safe behavior: it forces a
        fresh login under the new credential rather than trusting a
        session established under the old one.
        """
        self._r.delete(_ACTIVE_SESSIONS_KEY_PREFIX + username)

    def terminate_all_other_sessions(self, username: str, *, keep_session_token: str) -> int:
        """Revoke every session for this user except ``keep_session_token``. Returns how many were revoked."""
        key = _ACTIVE_SESSIONS_KEY_PREFIX + username
        members = {m for m in (_decode(raw) for raw in self._r.smembers(key)) if m is not None}
        to_remove = members - {keep_session_token}
        if to_remove:
            self._r.srem(key, *to_remove)
        return len(to_remove)

    # -- IP whitelisting ---------------------------------------------------

    def set_ip_whitelist_enabled(self, username: str, enabled: bool) -> None:
        self._r.set(_IP_WHITELIST_KEY_PREFIX + username + ":enabled", "1" if enabled else "0")

    def is_ip_whitelist_enabled(self, username: str) -> bool:
        return _decode(self._r.get(_IP_WHITELIST_KEY_PREFIX + username + ":enabled")) == "1"

    def add_whitelisted_ip(self, username: str, ip: str) -> None:
        self._r.sadd(_IP_WHITELIST_KEY_PREFIX + username, ip)

    def remove_whitelisted_ip(self, username: str, ip: str) -> None:
        self._r.srem(_IP_WHITELIST_KEY_PREFIX + username, ip)

    def list_whitelisted_ips(self, username: str) -> list[str]:
        members = self._r.smembers(_IP_WHITELIST_KEY_PREFIX + username)
        return sorted(ip for ip in (_decode(m) for m in members) if ip is not None)

    def is_ip_allowed(self, username: str, ip: str) -> bool:
        """Best-effort application-level check -- see module docstring for why this is not network enforcement."""
        if not self.is_ip_whitelist_enabled(username):
            return True
        return ip in self.list_whitelisted_ips(username)

    # -- admin: roles / locking ---------------------------------------------------

    def set_role(self, username: str, role: Role) -> None:
        user = self._require_user(username)
        user.role = role
        self._save(user)

    def lock_user(self, username: str) -> None:
        user = self._require_user(username)
        user.locked = True
        self._save(user)
        self.terminate_all_sessions(username)

    def unlock_user(self, username: str) -> None:
        user = self._require_user(username)
        user.locked = False
        self._save(user)

    # -- admin: global session timeout ---------------------------------------------------

    def get_global_session_timeout_minutes(self) -> int:
        raw = _decode(self._r.get(_GLOBAL_TIMEOUT_KEY))
        return int(raw) if raw is not None else _DEFAULT_GLOBAL_TIMEOUT_MINUTES

    def set_global_session_timeout_minutes(self, minutes: int) -> None:
        if minutes < 1:
            raise ValueError("Session timeout must be at least 1 minute.")
        self._r.set(_GLOBAL_TIMEOUT_KEY, str(minutes))

    def _require_user(self, username: str) -> IamUser:
        user = self.get_user(username)
        if user is None:
            raise UserNotFoundError(f"User {username!r} not found.")
        return user


def _render_qr_png(data: str) -> bytes:
    img = qrcode.make(data)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def password_strength_score(password: str) -> tuple[int, str]:
    """A real, simple heuristic (length + character-class diversity), 0-4. Not a substitute for zxcvbn.

    Deliberately not claimed as anything more rigorous: this is the
    same class of estimate every "strength bar" widget uses without a
    dedicated library (which this project does not have installed) --
    length is weighted most heavily since it dominates real crack-time,
    then one point per character class present.
    """
    if not password:
        return 0, "Vide"
    score = 0
    if len(password) >= 8:
        score += 1
    if len(password) >= 12:
        score += 1
    class_count = sum(
        [
            any(c.islower() for c in password),
            any(c.isupper() for c in password),
            any(c.isdigit() for c in password),
            any(not c.isalnum() for c in password),
        ]
    )
    score += min(class_count, 2)
    score = min(score, 4)
    labels = {0: "Très faible", 1: "Faible", 2: "Moyen", 3: "Fort", 4: "Très fort"}
    return score, labels[score]
