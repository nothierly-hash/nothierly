"""Chapter 13, Étape 5.6, Pilier 4: symmetric encryption for broker secrets at rest.

**Real gap this closes**: before this module, every credential in this
project (OANDA token, Twilio auth token, LLM API keys, ...) lives
exclusively as a ``pydantic.SecretStr`` sourced from ``.env`` -- real
in-process redaction (never printed/logged), but never actually
*encrypted at rest*, and nothing in this codebase persists a secret to
Redis or any other store. That changes with Étape 5.5's per-exchange
broker credentials (see :mod:`src.execution.broker_gateway` and
``src/dashboard/ems_pages/liquidity_routing.py``'s credential form):
those need to be added at runtime, for exchanges not known at deploy
time, so a static ``.env``-only ``SecretStr`` field can't hold them --
they need a real place to live between the dashboard writing them and
a broker adapter reading them back, and that place must not be
plaintext Redis.

**Real, not "Vault-flavored theater"**: the brief's own wording offers
an explicit fallback ("HashiCorp Vault ... ou une simulation de
Vault"). Standing up an actual HashiCorp Vault server is real
infrastructure this project has no deployment story for (no Vault
service in ``docker-compose.yml``, no operational precedent anywhere
in this codebase) -- adding one for a single dashboard credential form
would be infrastructure the rest of the project doesn't match. What
*is* implemented here is real, working symmetric encryption
(``cryptography.fernet.Fernet`` -- AES-128-CBC + HMAC-SHA256,
authenticated, industry-standard, not homegrown crypto) with a real
key-derivation step, genuinely unreadable at rest without the master
key. It is a **local encryption-at-rest layer**, not Vault's dynamic
secrets, leasing, or audit-log feature set -- named ``SecurityVault``
for what it does (keep secrets encrypted, expose them only through
one audited choke point), not to claim it *is* HashiCorp Vault.
"""

from __future__ import annotations

import base64

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Fixed, not per-install-random: PBKDF2's salt exists to defeat rainbow
# tables across *many independent users sharing a hashing service*
# (e.g. a password database) -- irrelevant here, where there is exactly
# one master key, supplied directly by the operator via VAULT_MASTER_KEY,
# never transmitted or compared against a stored hash. A fixed,
# module-specific salt keeps key derivation deterministic (the same
# passphrase always derives the same Fernet key, so a service restart
# doesn't silently orphan every credential already encrypted with the
# previous run's random salt) without weakening the real security
# property this module provides -- secrecy of the master key itself.
_PBKDF2_SALT: bytes = b"nothierly-security-vault-v1"
_PBKDF2_ITERATIONS: int = 480_000  # OWASP 2023 minimum recommendation for PBKDF2-HMAC-SHA256


class VaultDecryptionError(Exception):
    """Raised when ciphertext fails to decrypt: wrong master key, or the data was tampered with/corrupted.

    Wraps :class:`cryptography.fernet.InvalidToken` so callers depend on
    one stable, project-owned exception type rather than reaching into
    ``cryptography``'s own exception hierarchy.
    """


def _derive_fernet_key(passphrase: str) -> bytes:
    """Derive a valid 32-byte url-safe-base64 Fernet key from an arbitrary operator passphrase.

    Fernet requires its key in one specific format (32 raw bytes,
    url-safe base64-encoded) -- this lets ``VAULT_MASTER_KEY`` in
    ``.env`` be any human-chosen passphrase (matching every other
    credential in this project's ``.env``) rather than requiring the
    operator to pre-generate a correctly-shaped Fernet key by hand.
    """
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=_PBKDF2_SALT, iterations=_PBKDF2_ITERATIONS)
    return base64.urlsafe_b64encode(kdf.derive(passphrase.encode("utf-8")))


class SecurityVault:
    """Encrypts/decrypts short secrets (API keys, API secrets) with one operator-supplied master key.

    Real, working AES-128-CBC + HMAC-SHA256 authenticated encryption via
    :class:`cryptography.fernet.Fernet` -- ciphertext is tamper-evident
    (:meth:`decrypt` raises rather than silently returning corrupted
    plaintext) and includes its own timestamp/nonce, so encrypting the
    same plaintext twice produces different ciphertext.
    """

    def __init__(self, master_key: str) -> None:
        """Construct a vault bound to one master key.

        Args:
            master_key: The operator's passphrase (from
                :class:`~core.config.VaultSettings`.master_key). Any
                non-empty string -- internally stretched into a real
                Fernet key via PBKDF2-HMAC-SHA256.

        Raises:
            ValueError: If ``master_key`` is empty.
        """
        if not master_key:
            raise ValueError("SecurityVault requires a non-empty master_key.")
        self._fernet = Fernet(_derive_fernet_key(master_key))

    def encrypt(self, plaintext: str) -> str:
        """Encrypt ``plaintext``, returning a url-safe ciphertext string safe to store in Redis/a DB."""
        return self._fernet.encrypt(plaintext.encode("utf-8")).decode("ascii")

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt a string previously produced by :meth:`encrypt`.

        Raises:
            VaultDecryptionError: If ``ciphertext`` doesn't decrypt
                under this vault's master key -- wrong key, corrupted
                data, or tampering. Never returns partial/garbage
                plaintext.
        """
        try:
            return self._fernet.decrypt(ciphertext.encode("ascii")).decode("utf-8")
        except InvalidToken as exc:
            raise VaultDecryptionError(
                "Ciphertext could not be decrypted -- wrong master key, or the data is corrupted/tampered."
            ) from exc
