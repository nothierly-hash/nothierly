"""Push notifications (Telegram) for critical, real-money events.

Same fail-soft contract as :func:`~src.agents.alert_agent.send_sms_alert`:
an unconfigured or unreachable Telegram bot must never crash the caller --
a notification is a best-effort convenience, not something a trading
agent's safety logic should ever depend on to actually behave safely.
:class:`TelegramNotifier` only ever *reports* that something critical
already happened (Kill Switch tripped, inventory limit breached, market
data went stale, a large fill executed) -- it is never itself part of the
decision to trip the Kill Switch, halt trading, or hedge a position. That
logic lives in :mod:`core.security` and the calling agent; this module
has no side effect on trading behavior if it silently fails.

Uses ``aiohttp`` directly against the Telegram Bot HTTP API (already a
project dependency via ``ccxt.pro`` -- no new SDK dependency for one
``POST`` call) rather than the ``python-telegram-bot`` package, which
pulls in a full polling/webhook framework this project has no use for
(one outbound "send a message" call, never receiving updates).
"""

from __future__ import annotations

from enum import Enum
from typing import Final

import aiohttp

from core.config import TelegramSettings
from core.logger import get_logger

_log = get_logger(__name__)

_TELEGRAM_API_BASE: Final[str] = "https://api.telegram.org"
_REQUEST_TIMEOUT_SECONDS: Final[float] = 10.0


class AlertSeverity(str, Enum):
    """How urgently a human should look at this. Purely cosmetic (an emoji prefix) -- never gates whether the message sends."""

    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


_SEVERITY_PREFIX: Final[dict[AlertSeverity, str]] = {
    AlertSeverity.CRITICAL: "🚨 CRITIQUE",
    AlertSeverity.WARNING: "⚠️ AVERTISSEMENT",
    AlertSeverity.INFO: "ℹ️ INFO",
}


class TelegramNotifier:
    """Sends a real Telegram message via the Bot API. Never raises -- see module docstring."""

    def __init__(self, settings: TelegramSettings) -> None:
        self._settings = settings

    @property
    def is_configured(self) -> bool:
        """``True`` only if both a bot token and a destination chat ID are set."""
        return self._settings.bot_token is not None and self._settings.chat_id is not None

    async def send(self, message: str, *, severity: AlertSeverity = AlertSeverity.INFO, source: str) -> bool:
        """Send one real push notification. Never raises.

        Args:
            message: Human-readable body. Kept as sent -- the caller is
                responsible for not including a secret in it (this
                function does not run it through
                :func:`core.logger._redact_secrets`; it is not a log
                line).
            severity: Cosmetic urgency prefix. Does not affect delivery.
            source: Which agent/component is sending this (e.g.
                ``"alpha-engine"``) -- included so one shared Telegram
                chat receiving alerts from multiple agents stays
                distinguishable.

        Returns:
            ``True`` if Telegram's API accepted the message (HTTP 200),
            ``False`` if unconfigured or the call failed for any reason.
            Never a delivery confirmation past Telegram's own API.
        """
        if not self.is_configured:
            _log.warning("telegram_not_configured_alert_not_sent", source=source, message_preview=message[:80])
            return False

        assert self._settings.bot_token is not None and self._settings.chat_id is not None  # narrowed by is_configured above
        url = f"{_TELEGRAM_API_BASE}/bot{self._settings.bot_token.get_secret_value()}/sendMessage"
        text = f"{_SEVERITY_PREFIX[severity]} -- Nothierly [{source}]\n\n{message}"

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_SECONDS)) as session:
                async with session.post(url, json={"chat_id": self._settings.chat_id, "text": text}) as response:
                    if response.status != 200:
                        body = await response.text()
                        _log.warning(
                            "telegram_send_failed",
                            source=source,
                            status=response.status,
                            # Telegram's own error body never contains the bot token (it's in the URL
                            # path, not echoed back) -- safe to log as-is.
                            detail=body[:300],
                        )
                        return False
        except Exception:
            _log.exception("telegram_send_raised", source=source)
            return False

        _log.info("telegram_alert_sent", source=source, severity=severity.value)
        return True
