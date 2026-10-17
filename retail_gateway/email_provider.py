"""Transactional email sending -- the synchronous side the Celery worker actually calls.

**Resend via stdlib ``urllib``, no new SDK dependency**: Resend's real API is one plain HTTPS
POST (``POST https://api.resend.com/emails``, Bearer auth, JSON body) -- ``urllib.request``
(already in every Python install) covers it completely, so no ``resend`` PyPI package is added
for what amounts to one endpoint. This mirrors this project's own established discipline of not
adding a dependency where the stdlib (or an already-required library) already does the job --
see requirements.txt's own note on ``core.http_client.ResilientHttpClient`` being reused instead
of adding a second async HTTP stack.

**AWS SES -- real, disclosed, not implemented**: the brief offered SES or Resend; Resend was
picked because it needed no new dependency and no AWS account to verify against in this
environment. Swapping in SES means adding a ``SesEmailProvider`` behind this same
:class:`EmailProvider` interface (``boto3`` -- already a dependency of the main trading repo,
not of this air-gapped one -- would need adding to ``retail_gateway/requirements.txt``
separately) and switching ``RETAIL_EMAIL_PROVIDER=ses``; real follow-up work, not silently
faked here.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Protocol

from retail_gateway.config import EmailSettings
from retail_gateway.logging_config import get_logger

_log = get_logger(__name__)

_RESEND_API_URL = "https://api.resend.com/emails"
_REQUEST_TIMEOUT_SECONDS = 10


class EmailProvider(Protocol):
    def send_welcome_email(self, *, to_address: str) -> None: ...
    def send_otp_email(self, *, to_address: str, otp_code: str) -> None: ...


class ConsoleEmailProvider:
    """Dev/unconfigured fallback -- logs instead of sending. Never raises. Never logs the real
    code at a level a real deployment would ship to a shared log aggregator by accident.
    """

    def send_welcome_email(self, *, to_address: str) -> None:
        _log.info("welcome_email_logged_not_sent", to=to_address, reason="RETAIL_EMAIL_PROVIDER=console")

    def send_otp_email(self, *, to_address: str, otp_code: str) -> None:
        # otp_code deliberately not logged -- logging_config.py's own module docstring states
        # this package never logs a secret anywhere; an OTP is one, even in the console fallback.
        _log.info("otp_email_logged_not_sent", to=to_address, reason="RETAIL_EMAIL_PROVIDER=console")


class ResendEmailProvider:
    def __init__(self, *, api_key: str, from_address: str) -> None:
        self._api_key = api_key
        self._from_address = from_address

    def send_welcome_email(self, *, to_address: str) -> None:
        self._post_email(
            to_address=to_address,
            subject="Bienvenue chez Nothierly",
            html="<p>Bienvenue -- votre compte a bien été créé.</p><p>Ceci est un email transactionnel automatique.</p>",
        )

    def send_otp_email(self, *, to_address: str, otp_code: str) -> None:
        self._post_email(
            to_address=to_address,
            subject="Votre code de vérification Nothierly",
            html=f"<p>Votre code de vérification est : <strong>{otp_code}</strong></p><p>Il expire dans 5 minutes.</p>",
        )

    def _post_email(self, *, to_address: str, subject: str, html: str) -> None:
        body = json.dumps({"from": self._from_address, "to": [to_address], "subject": subject, "html": html}).encode("utf-8")
        request = urllib.request.Request(
            _RESEND_API_URL,
            data=body,
            method="POST",
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
                if response.status >= 400:
                    raise RuntimeError(f"Resend returned HTTP {response.status}")
        except urllib.error.URLError as exc:
            # Re-raised so Celery's autoretry_for (see tasks.py) actually retries this delivery
            # instead of silently swallowing a real, transient send failure.
            raise RuntimeError(f"Resend request failed: {exc}") from exc


def build_email_provider(settings: EmailSettings) -> EmailProvider:
    if settings.provider == "resend":
        if settings.resend_api_key is None:
            _log.warning("resend_selected_without_api_key_falling_back_to_console")
            return ConsoleEmailProvider()
        return ResendEmailProvider(
            api_key=settings.resend_api_key.get_secret_value(),
            from_address=settings.from_address,
        )
    if settings.provider == "ses":
        _log.warning("ses_provider_not_implemented_falling_back_to_console")
        return ConsoleEmailProvider()
    return ConsoleEmailProvider()
