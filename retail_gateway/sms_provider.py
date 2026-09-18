"""SMS sending -- the synchronous side the Celery worker actually calls. Mirrors ``email_provider.py``'s own shape/reasoning exactly.

**Twilio via stdlib ``urllib``, no ``twilio`` SDK dependency**: Twilio's real Messages API is one
plain HTTPS POST (``POST https://api.twilio.com/2010-04-01/Accounts/{Sid}/Messages.json``, HTTP
Basic Auth, form-encoded body) -- ``urllib.request`` covers it completely, same reasoning
``email_provider.py``'s own docstring gives for Resend.
"""

from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
from base64 import b64encode
from typing import Protocol

from retail_gateway.config import SmsSettings
from retail_gateway.logging_config import get_logger

_log = get_logger(__name__)

_TWILIO_API_URL_TEMPLATE = "https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
_REQUEST_TIMEOUT_SECONDS = 10


class SmsProvider(Protocol):
    def send_otp_sms(self, *, to_number: str, otp_code: str) -> None: ...


class ConsoleSmsProvider:
    """Dev/unconfigured fallback -- logs instead of sending. Never raises. Never logs the real
    code at a level a real deployment would ship to a shared log aggregator by accident --
    ``info``, same level ``ConsoleEmailProvider`` already uses for the equivalent case.
    """

    def send_otp_sms(self, *, to_number: str, otp_code: str) -> None:
        _log.info("otp_sms_logged_not_sent", to=to_number, reason="RETAIL_SMS_PROVIDER=console")


class TwilioSmsProvider:
    def __init__(self, *, account_sid: str, auth_token: str, from_number: str) -> None:
        self._account_sid = account_sid
        self._auth_token = auth_token
        self._from_number = from_number

    def send_otp_sms(self, *, to_number: str, otp_code: str) -> None:
        body = urllib.parse.urlencode(
            {
                "To": to_number,
                "From": self._from_number,
                "Body": f"Nothierly: your verification code is {otp_code}. It expires in 5 minutes.",
            }
        ).encode("utf-8")
        basic_auth = b64encode(f"{self._account_sid}:{self._auth_token}".encode("utf-8")).decode("ascii")
        request = urllib.request.Request(
            _TWILIO_API_URL_TEMPLATE.format(account_sid=self._account_sid),
            data=body,
            method="POST",
            headers={
                "Authorization": f"Basic {basic_auth}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
                if response.status >= 400:
                    raise RuntimeError(f"Twilio returned HTTP {response.status}")
        except urllib.error.URLError as exc:
            # Re-raised so Celery's autoretry_for (see tasks.py) actually retries this delivery
            # instead of silently swallowing a real, transient send failure.
            raise RuntimeError(f"Twilio request failed: {exc}") from exc


def build_sms_provider(settings: SmsSettings) -> SmsProvider:
    if settings.provider == "twilio":
        if settings.twilio_account_sid is None or settings.twilio_auth_token is None or settings.twilio_from_number is None:
            _log.warning("twilio_selected_without_full_credentials_falling_back_to_console")
            return ConsoleSmsProvider()
        return TwilioSmsProvider(
            account_sid=settings.twilio_account_sid,
            auth_token=settings.twilio_auth_token.get_secret_value(),
            from_number=settings.twilio_from_number,
        )
    return ConsoleSmsProvider()
