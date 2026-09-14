"""Chapter 13, Étape 5.2: "The Red Phone" -- SMS circuit-breaker alert agent.

:class:`AlertAgent` watches three real danger conditions and, on each,
sends a rate-limited SMS via Twilio and trips the system's circuit
breaker (the real global Kill Switch, plus a fast broadcast telling
:class:`~src.agents.execution.agent.ExecutionAgent` to flatten every
open position right now).

**Real corrections to the brief, stated upfront**:

1. **No ZMQ "error stream" exists to subscribe to.** The brief frames
   this as "écoute passivement les flux d'erreurs (via un socket ZeroMQ
   SUB ou Redis Pub/Sub)" -- but nothing in this project publishes an
   error-event stream on any topic or channel; the existing bus topics
   (``TRADING_SIGNALS``, ``MARKET_DATA_L2``, ``CLIENT_ORDERS``,
   ``FEATURES``) are all normal operational traffic, not errors.
   Inventing a new error-publishing convention across every other agent
   was out of scope for this one module. Instead, this agent runs a
   short, regular tick (like every other :class:`~core.base_agent.
   BaseAgent`) and polls the real state each danger condition already
   produces -- see below. The Rule 3 anti-spam requirement applies
   identically either way: a condition detected on every tick while it
   persists still needs debouncing, regardless of whether the source
   was a message flood or a polled condition staying true.

2. **VPIN/TFI aren't wired into live production anywhere** (confirmed
   by direct inspection, same finding as ``src/agents/quant_copilot.py``
   -- VPIN and TFI both exist only in ``research/alpha_features.py``,
   offline research code). The real, live signal this agent can check
   is Order Book Imbalance, written by
   :class:`~src.agents.market_data.agent.MarketDataAgent` to
   ``market_metrics:{symbol}`` -- :data:`OBI_CATASTROPHIC_THRESHOLD`
   substitutes for the brief's VPIN/black-swan framing using that real
   signal instead of a nonexistent one.

3. **No per-client "Hard_Stop_Loss" exists in any contract anywhere**
   (checked ``api_middleware/gateway.py``'s ``MacroOrderRequest`` and
   the whole codebase). What *does* real exist is
   :class:`~src.agents.execution.agent.ExecutionAgent`'s own
   account-wide 24h/5% drawdown trigger, which already pushes a real
   ``GuardianAlert`` (``trigger="DRAWDOWN"``) onto the
   ``NOTHIERLY_GUARDIAN_ALERTS`` Redis list on every real trip. This
   agent watches that same real list rather than reimplementing a
   second, parallel (and likely inconsistent) drawdown calculation, or
   inventing a per-client concept that doesn't exist in this system yet
   -- a real per-client hard-stop-loss is a genuinely separate, larger
   feature (extending the gateway's order contract and per-client PnL
   tracking), not built here.

4. **"500ms" heartbeat staleness is not a real, calibratable number
   here, so it isn't used.** Real agent tick/poll cadences in this
   project are 1-5+ real seconds (``core/base_agent.py``'s own
   heartbeat TTL floors at 5.0s for exactly this reason); a fixed
   500ms staleness check would fire continuously under completely
   healthy operation. Redis's own TTL already encodes "hasn't
   heartbeated recently enough" per-agent, calibrated to each agent's
   real cadence -- this agent simply checks whether the
   ``heartbeat:{agent_id}`` key exists at all, requiring
   :data:`INFRA_MISSING_HEARTBEAT_STRIKES` consecutive misses (not
   one) before alerting, so a single unlucky poll boundary isn't a
   false positive.

5. **Twilio circuit-breaker's "cancel all orders" didn't exist either**
   -- neither ``ExecutionAgent`` nor ``OandaClient`` had a unified
   "flatten everything" method before this chapter (only
   single-instrument ``_close_position``/``cancel_all_orders`` for
   *pending* orders on one instrument). Added as
   :meth:`~src.agents.execution.agent.ExecutionAgent.cancel_all_positions`,
   reusing the existing, already-safe ``_close_position`` path for each
   open instrument, triggered by subscribing that agent to the new
   :data:`~core.schemas.EMERGENCY_HALT_TOPIC` this module publishes.
   The brief also names "l'Agent RL" as something that should cancel
   orders on receipt -- :class:`~src.agents.rl_signal.agent.RlSignalAgent`
   never places or holds orders at all (confirmed: it only ever
   publishes an opinion, :class:`~core.schemas.TradingSignal`); only
   :class:`~src.agents.execution.agent.ExecutionAgent` can actually act
   on this, so only it subscribes.

6. **The real global Kill Switch already exists** (``core.security``'s
   ``SYSTEM_STATUS`` Redis flag, ``trip_kill_switch``/``is_system_safe``,
   already gating every new order via ``@requires_system_active``).
   Rule 4's "EMERGENCY_HALT" broadcast is real and useful, but as a
   *complement* to that existing mechanism, not a replacement or a
   second, independent kill switch: this agent calls
   :func:`~core.security.trip_kill_switch` (blocks all *new* orders,
   system-wide, immediately) *and* publishes ``EMERGENCY_HALT`` (tells
   ExecutionAgent to proactively close *existing* ones) -- two real,
   distinct effects from one circuit-breaker trip, not a duplicate.

2026-09-14, "Surfusion" / liquidation-cascade brief -- real corrections:

7. **No live leverage/Open Interest feed exists for the numerator of the
   brief's proposed ratio.** The brief's ``LCR_t = (OI_t * %L_t) /
   absorption_capacity`` needs a real perpetual-futures Open Interest and
   long/short ratio; this codebase has neither (confirmed by direct
   search -- the only real Open Interest anywhere is
   :mod:`~src.agents.market_data.options_chain_client`'s *options*
   chain OI, an unrelated market/instrument, not perp futures leverage).
   :meth:`AlertAgent._check_liquidity_collapse_danger` below therefore
   does not compute the brief's full ratio. It builds the half that
   *is* real: a genuine "has this book's bid-side depth quietly
   collapsed relative to its own recent normal" detector, sourced from
   real, live L2 order-book history
   (:func:`~src.agents.market_data.l2_surface_packer.
   read_l2_surface_history_async`, the same real 5-level snapshots
   :class:`~src.agents.market_data.agent.MarketDataAgent` already
   persists on every book update -- previously read only by the
   dashboard's 3D depth topography, now reused here unchanged). This is
   a real, narrower, honestly-scoped signal, not a stand-in for the
   brief's full leverage-aware ratio.

8. **The brief's own normalization is self-referential and does not
   hold a stable "critical" threshold.** Its rolling min-max scaling
   renormalizes the danger index against its *own* recent history --
   a market that stays chronically fragile stretches that history to
   reabsorb the new normal, sliding the index back toward the middle
   of its own range even while the real underlying risk stays high.
   Replaced with this file's own established pattern instead: a fixed,
   live-Redis-overridable threshold on a stable, well-defined quantity
   (current bid depth vs. the *median* of recent bid depth, not
   min-max-rescaled), exactly mirroring :data:`OBI_CATASTROPHIC_THRESHOLD`
   / :data:`OBI_THRESHOLD_OVERRIDE_REDIS_KEY`'s own shape.

9. **The brief's bespoke ``tcp://*:5555`` ZMQ PUB socket and a raw
   ZMQ-polling thread inside Streamlit would each duplicate a working
   mechanism this project already has**, rather than fill a gap (unlike
   the OI/leverage point above, where nothing real existed yet): every
   agent, this one included, already publishes over the real
   broker-mediated ZMQ bus at this deployment's configured
   frontend/backend ports via :meth:`~core.base_agent.BaseAgent.publish`,
   and the dashboard's Red Phone page already reads
   :data:`_ALERT_HISTORY_REDIS_KEY` fresh on every rerun. This check
   reuses both unchanged -- no new socket, no new topic, no new
   dashboard read path -- by calling the exact same
   :meth:`_raise_alert` (category ``MICROSTRUCTURE``) the OBI check
   above already uses.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Final, Literal

import numpy as np
from twilio.rest import Client as TwilioClient

from core.base_agent import AgentControlState, BaseAgent
from core.config import AppSettings, TwilioSettings, get_settings
from core.logger import get_logger
from core.redis_client import RedisUnavailableError
from core.schemas import EMERGENCY_HALT_TOPIC, HEARTBEAT_REDIS_KEY_PREFIX, MARKET_METRICS_REDIS_KEY_PREFIX
from core.security import trip_kill_switch
from src.agents.market_data.l2_surface_packer import UnpackedL2Surface, read_l2_surface_history_async

_log = get_logger(__name__)

_AGENT_ID: Final[str] = "alert-agent"
TICK_INTERVAL_SECONDS: Final[float] = 2.0

DangerCategory = Literal["MICROSTRUCTURE", "ECONOMIC", "INFRASTRUCTURE"]

# -- Rule 1, Danger Microstructure (real OBI substitute for VPIN/TFI -- see module docstring) --
_MONITORED_SYMBOLS: Final[tuple[str, ...]] = ("BTC/USDT", "EUR_USD", "XAU_USD")
OBI_CATASTROPHIC_THRESHOLD: Final[float] = 0.9  # |OBI| this close to +-1 = one side of the book nearly empty

# Redis-backed live override for OBI_CATASTROPHIC_THRESHOLD, read fresh on
# every tick -- the same "dashboard writes, agent reads live, no restart"
# pattern ExecutionAgent's _get_dynamic_risk_pct/_RISK_PCT_REDIS_KEY
# already establishes. Written by src/dashboard/ems_pages/red_phone.py's
# sensitivity slider. Clamped at both ends so a corrupted/malicious Redis
# value can never disable the check (ceiling) or make it fire on normal
# book noise (floor).
OBI_THRESHOLD_OVERRIDE_REDIS_KEY: Final[str] = "ALERT_OBI_THRESHOLD_OVERRIDE"
_OBI_THRESHOLD_MIN: Final[float] = 0.5
_OBI_THRESHOLD_MAX: Final[float] = 0.99

# -- Rule 1, Danger Microstructure (bis): real bid-depth collapse, real L2 history -- see module docstring point 7-9 --
LIQUIDITY_COLLAPSE_THRESHOLD: Final[float] = 0.70  # current bid depth this far BELOW its own recent median = danger
LIQUIDITY_COLLAPSE_THRESHOLD_OVERRIDE_REDIS_KEY: Final[str] = "ALERT_LIQUIDITY_COLLAPSE_THRESHOLD_OVERRIDE"
_LIQUIDITY_COLLAPSE_THRESHOLD_MIN: Final[float] = 0.30
_LIQUIDITY_COLLAPSE_THRESHOLD_MAX: Final[float] = 0.95
# Reference-window frames + the current frame -- below this, "recent normal" isn't measured yet (avoids a
# false positive off a single early snapshot right after this agent or MarketDataAgent (re)starts).
_LIQUIDITY_COLLAPSE_MIN_REFERENCE_FRAMES: Final[int] = 5

# -- Rule 1, Danger Économique (real account-wide Guardian DRAWDOWN trigger -- see module docstring) --
_GUARDIAN_ALERTS_REDIS_KEY: Final[str] = "NOTHIERLY_GUARDIAN_ALERTS"
_GUARDIAN_ALERTS_POLL_COUNT: Final[int] = 5  # newest N entries checked each tick -- LPUSH means index 0 is newest

# -- Rule 1, Danger Infrastructure (real heartbeat keys, key-presence not a fixed millisecond threshold -- see module docstring) --
_MONITORED_AGENT_IDS: Final[tuple[str, ...]] = ("supervisor-agent", "rl-signal-agent", "execution-agent")
INFRA_MISSING_HEARTBEAT_STRIKES: Final[int] = 2  # consecutive misses required before alerting -- avoids a single poll-boundary false positive

# -- Rule 3, anti-spam --
_DEBOUNCE_REDIS_PREFIX: Final[str] = "alert_debounce:"
DEBOUNCE_TTL_SECONDS: Final[int] = 300  # 1 alert / category / 5 minutes, exactly as specified

# Real alert history -- same LPUSH+LTRIM (newest-first) shape as
# ExecutionAgent's own NOTHIERLY_GUARDIAN_ALERTS list. Read by the
# dashboard's DEFCON Red Phone panel.
_ALERT_HISTORY_REDIS_KEY: Final[str] = "RED_PHONE_ALERT_LOG"
_ALERT_HISTORY_MAX_ENTRIES: Final[int] = 50

# Extra SMS recipients beyond TwilioSettings.admin_phone_number (the one
# .env-configured number) -- a Redis SET of E.164 numbers, managed live
# from src/dashboard/ems_pages/red_phone.py's recipient CRUD form. Real,
# not fabricated: an added number actually receives the next real SMS
# _raise_alert sends, exactly like OBI_THRESHOLD_OVERRIDE_REDIS_KEY above.
RECIPIENTS_REDIS_KEY: Final[str] = "RED_PHONE_RECIPIENTS"


async def send_sms_alert(twilio_settings: TwilioSettings, *, message: str, to: str | None = None) -> bool:
    """Send one SMS via the official Twilio SDK, off the event loop (Rule 2).

    The ``twilio`` SDK is synchronous (blocking HTTP under the hood) --
    ``asyncio.to_thread`` is what the brief itself specifies to keep
    this agent's event loop free, exactly as implemented here, not a
    deviation.

    Args:
        twilio_settings: Real Twilio credentials + sending number.
        message: The SMS body.
        to: Destination number. Defaults to
            ``twilio_settings.admin_phone_number`` (the original,
            single-recipient behavior) when omitted -- callers iterating
            :meth:`AlertAgent._get_alert_recipients` pass each recipient
            explicitly instead.

    Returns:
        ``True`` if the send call completed without raising (Twilio
        accepted the message for delivery -- not a delivery
        confirmation), ``False`` if unconfigured, no destination is
        resolvable, or the call failed. Never raises: a failed SMS must
        not crash the agent watching for the next real danger.
    """
    destination = to or twilio_settings.admin_phone_number
    if (
        twilio_settings.account_sid is None
        or twilio_settings.auth_token is None
        or twilio_settings.phone_number is None
        or destination is None
    ):
        _log.warning("twilio_not_configured_alert_not_sent", message_preview=message[:80])
        return False

    client = TwilioClient(twilio_settings.account_sid, twilio_settings.auth_token.get_secret_value())
    try:
        await asyncio.to_thread(
            client.messages.create,
            body=message,
            from_=twilio_settings.phone_number,
            to=destination,
        )
    except Exception:
        _log.exception("sms_alert_send_failed")
        return False

    _log.info("sms_alert_sent", message_preview=message[:80], to=destination)
    return True


def _build_sms_body(category: DangerCategory, detail: str) -> str:
    """A surgical, actionable SMS body, matching the brief's own example format."""
    return f"[URGENT - EMS ALGO] {category} : {detail}"


def _bid_depth_quote(frame: UnpackedL2Surface) -> float:
    """Real total bid-side depth in quote-currency terms (sum of price*volume) over one real L2 frame's captured levels.

    :func:`~src.agents.market_data.l2_surface_packer.pack_snapshot` pads
    a thinner-than-``ORDER_BOOK_TOP_N`` side with explicit ``(0.0, 0.0)``
    rows rather than shifting subsequent levels -- summing unconditionally
    is still correct here (a zero-price zero-volume row contributes 0.0),
    so no separate real-level filter is needed.
    """
    return float(np.sum(frame.bid_prices.astype(np.float64) * frame.bid_volumes.astype(np.float64)))


class AlertAgent(BaseAgent):
    """Polls three real danger conditions every tick; on each, debounces, SMS-alerts, and trips the circuit breaker.

    See the module docstring for exactly which real data source backs
    each of the three danger categories, and why each deviates from the
    brief's literal (partly unimplementable) framing.
    """

    def __init__(
        self,
        *,
        settings: AppSettings | None = None,
        manage_own_signals: bool = True,
        manage_own_redis: bool = True,
    ) -> None:
        super().__init__(
            _AGENT_ID,
            tick_interval_seconds=TICK_INTERVAL_SECONDS,
            settings=settings,
            manage_own_signals=manage_own_signals,
            manage_own_redis=manage_own_redis,
        )
        self._settings = settings or get_settings()
        self._last_seen_guardian_alert_timestamp: float = 0.0
        self._infra_missing_streaks: dict[str, int] = {agent_id: 0 for agent_id in _MONITORED_AGENT_IDS}

    async def on_tick(self) -> None:
        """Run all three real danger checks once. Any single check's failure never blocks the others."""
        if self._redis_client is None:
            self._log.warning("alert_agent_tick_skipped_no_redis", agent_id=self.agent_id)
            return

        for check in (
            self._check_microstructure_danger,
            self._check_liquidity_collapse_danger,
            self._check_economic_danger,
            self._check_infrastructure_danger,
        ):
            try:
                await check()
            except Exception:
                self._log.exception("alert_agent_check_failed", agent_id=self.agent_id, check=check.__name__)

    # -- Rule 1: Danger Microstructure -----------------------------------

    async def _get_dynamic_obi_threshold(self) -> float:
        """Read the live OBI catastrophic threshold from Redis, hard-clamped and fail-safe.

        Mirrors :meth:`~src.agents.execution.agent.ExecutionAgent.
        _get_dynamic_risk_pct`'s fail-safe shape exactly: any failure
        (no Redis, timeout, missing/unparseable key) falls back to the
        real default, :data:`OBI_CATASTROPHIC_THRESHOLD`; a value that
        parses but falls outside ``[_OBI_THRESHOLD_MIN,
        _OBI_THRESHOLD_MAX]`` is clamped, never trusted verbatim.
        """
        assert self._redis_client is not None
        try:
            raw = await asyncio.wait_for(
                self._redis_client.execute("get", OBI_THRESHOLD_OVERRIDE_REDIS_KEY), timeout=2.0
            )
        except (RedisUnavailableError, asyncio.TimeoutError):
            return OBI_CATASTROPHIC_THRESHOLD

        if raw is None:
            return OBI_CATASTROPHIC_THRESHOLD

        try:
            threshold = float(raw)
        except (TypeError, ValueError):
            return OBI_CATASTROPHIC_THRESHOLD

        return max(_OBI_THRESHOLD_MIN, min(_OBI_THRESHOLD_MAX, threshold))

    async def _check_microstructure_danger(self) -> None:
        """Real OBI extreme, per monitored symbol -- see module docstring for the VPIN/TFI substitution."""
        assert self._redis_client is not None
        threshold = await self._get_dynamic_obi_threshold()
        for symbol in _MONITORED_SYMBOLS:
            raw = await self._redis_client.execute("get", f"{MARKET_METRICS_REDIS_KEY_PREFIX}{symbol}")
            if raw is None:
                continue
            metrics: dict[str, Any] = json.loads(raw)
            obi = float(metrics["order_book_imbalance"])
            if abs(obi) >= threshold:
                side = "acheteur" if obi > 0 else "vendeur"
                await self._raise_alert(
                    category="MICROSTRUCTURE",
                    detail=(
                        f"Déséquilibre catastrophique détecté sur {symbol}. OBI à {obi:+.2f} "
                        f"(carnet quasi-vide côté {side}). Ordres Maker à risque de sélection adverse extrême."
                    ),
                )

    # -- Rule 1: Danger Microstructure (bis) -- real bid-depth collapse -----

    async def _get_dynamic_liquidity_collapse_threshold(self) -> float:
        """Live Redis-overridable liquidity-collapse threshold -- exact same fail-safe shape as :meth:`_get_dynamic_obi_threshold`."""
        assert self._redis_client is not None
        try:
            raw = await asyncio.wait_for(
                self._redis_client.execute("get", LIQUIDITY_COLLAPSE_THRESHOLD_OVERRIDE_REDIS_KEY), timeout=2.0
            )
        except (RedisUnavailableError, asyncio.TimeoutError):
            return LIQUIDITY_COLLAPSE_THRESHOLD

        if raw is None:
            return LIQUIDITY_COLLAPSE_THRESHOLD

        try:
            threshold = float(raw)
        except (TypeError, ValueError):
            return LIQUIDITY_COLLAPSE_THRESHOLD

        return max(_LIQUIDITY_COLLAPSE_THRESHOLD_MIN, min(_LIQUIDITY_COLLAPSE_THRESHOLD_MAX, threshold))

    async def _check_liquidity_collapse_danger(self) -> None:
        """Real bid-side depth collapse, per monitored symbol -- see module docstring points 7-9 for why this, not the brief's full leverage ratio.

        Reads this symbol's real, live L2 surface history (real exchange
        book levels, not a proxy) and compares the newest frame's total
        bid depth (sum of price*volume over the real captured levels)
        against the *median* depth of the reference frames before it --
        a stable, self-normalizing-but-not-self-referential measure of
        "how depleted is the book right now, versus what's normal for
        this book," independent of the asset's absolute price or
        typical liquidity level (so the same fixed threshold is
        meaningful across BTC/USDT, EUR_USD, and every other monitored
        symbol without per-asset recalibration).
        """
        assert self._redis_client is not None
        threshold = await self._get_dynamic_liquidity_collapse_threshold()
        for symbol in _MONITORED_SYMBOLS:
            frames = await read_l2_surface_history_async(self._redis_client, symbol)
            if len(frames) <= _LIQUIDITY_COLLAPSE_MIN_REFERENCE_FRAMES:
                continue  # not enough real history yet to know what "normal" depth looks like

            *reference_frames, current_frame = frames
            reference_depths = [_bid_depth_quote(frame) for frame in reference_frames]
            current_depth = _bid_depth_quote(current_frame)

            reference_median = float(np.median(reference_depths))
            if reference_median <= 0.0:
                continue  # the book never had real depth to begin with -- nothing to have collapsed from

            collapse_fraction = 1.0 - (current_depth / reference_median)
            if collapse_fraction >= threshold:
                await self._raise_alert(
                    category="MICROSTRUCTURE",
                    detail=(
                        f"Effondrement de la profondeur du carnet (achat) détecté sur {symbol}. "
                        f"Profondeur actuelle : {current_depth:,.2f}, contre une médiane récente de "
                        f"{reference_median:,.2f} ({collapse_fraction:+.0%}). Carnet fragilisé -- "
                        "un seul ordre de vente agressif suffirait désormais à déplacer le prix "
                        "significativement (risque de cascade en cas de levier concentré, non mesurable "
                        "ici faute de flux d'Open Interest réel -- voir docstring du module)."
                    ),
                )

    # -- Rule 1: Danger Économique -----------------------------------------

    async def _check_economic_danger(self) -> None:
        """Real, existing account-wide Guardian DRAWDOWN trigger -- see module docstring for why this, not a fabricated per-client stop-loss."""
        assert self._redis_client is not None
        raw_entries = await self._redis_client.execute(
            "lrange", _GUARDIAN_ALERTS_REDIS_KEY, 0, _GUARDIAN_ALERTS_POLL_COUNT - 1
        )
        newest_timestamp = self._last_seen_guardian_alert_timestamp
        for raw_entry in raw_entries:
            entry: dict[str, Any] = json.loads(raw_entry)
            if entry.get("trigger") != "DRAWDOWN":
                continue
            entry_timestamp = float(entry["timestamp"])
            if entry_timestamp <= self._last_seen_guardian_alert_timestamp:
                continue
            newest_timestamp = max(newest_timestamp, entry_timestamp)
            await self._raise_alert(category="ECONOMIC", detail=str(entry["summary"]))

        self._last_seen_guardian_alert_timestamp = newest_timestamp

    # -- Rule 1: Danger Infrastructure -------------------------------------

    async def _check_infrastructure_danger(self) -> None:
        """Real heartbeat-key presence, per monitored agent -- see module docstring for why not a fixed 500ms diff."""
        assert self._redis_client is not None
        for agent_id in _MONITORED_AGENT_IDS:
            raw = await self._redis_client.execute("get", f"{HEARTBEAT_REDIS_KEY_PREFIX}{agent_id}")
            if raw is not None:
                self._infra_missing_streaks[agent_id] = 0
                continue

            self._infra_missing_streaks[agent_id] += 1
            if self._infra_missing_streaks[agent_id] == INFRA_MISSING_HEARTBEAT_STRIKES:
                await self._raise_alert(
                    category="INFRASTRUCTURE",
                    detail=(
                        f"Agent {agent_id} hors ligne : aucun heartbeat Redis depuis "
                        f"{INFRA_MISSING_HEARTBEAT_STRIKES} cycles consécutifs "
                        f"(~{INFRA_MISSING_HEARTBEAT_STRIKES * TICK_INTERVAL_SECONDS:.0f}s). "
                        "Déconnexion exchange ou crash possible."
                    ),
                )

    # -- Rules 2-4: debounce, SMS, circuit breaker -------------------------

    async def _get_alert_recipients(self) -> list[str]:
        """Every real number a danger alert SMS goes to: the .env admin number + any dashboard-added extras.

        Deduplicated (a number added twice, or matching the admin
        number, is only ever texted once per alert).
        """
        assert self._redis_client is not None
        recipients: list[str] = []
        if self._settings.twilio.admin_phone_number is not None:
            recipients.append(self._settings.twilio.admin_phone_number)
        try:
            extra = await self._redis_client.execute("smembers", RECIPIENTS_REDIS_KEY)
        except RedisUnavailableError:
            extra = set()
        for number in extra:
            number_str = number.decode() if isinstance(number, bytes) else str(number)
            if number_str not in recipients:
                recipients.append(number_str)
        return recipients

    async def _try_acquire_debounce(self, category: DangerCategory) -> bool:
        """Atomic Redis ``SET NX EX``: exactly one caller per category per 5-minute window wins.

        Real, race-free under a flood of concurrent detections (Rule 3's
        own "même s'il reçoit 10 000 alertes internes" scenario) --
        ``SET ... NX`` only succeeds if the key does not already exist,
        checked and set in one atomic Redis operation, not a
        read-then-write race a local counter/dict would have.
        """
        assert self._redis_client is not None
        acquired = await self._redis_client.execute(
            "set", f"{_DEBOUNCE_REDIS_PREFIX}{category}", "1", nx=True, ex=DEBOUNCE_TTL_SECONDS
        )
        return bool(acquired)

    async def _raise_alert(self, *, category: DangerCategory, detail: str) -> None:
        """One real danger, detected: debounce-gate, then SMS + circuit breaker as one coalesced incident.

        The debounce gates the *entire* alert action (SMS, Kill Switch
        trip, and EMERGENCY_HALT broadcast) as a unit, not just the SMS
        -- Rule 3's own framing ("l'agent ne doit jamais envoyer plus de
        1 SMS... même s'il reçoit 10 000 alertes internes") treats a
        persisting danger condition as one incident to coalesce, not as
        independent decisions per side effect; re-tripping an
        already-tripped Kill Switch and re-broadcasting EMERGENCY_HALT
        on every tick while a condition persists would be exactly the
        spam Rule 3 exists to prevent, just moved from the SMS channel
        to the ZMQ bus and Redis instead.

        ``AgentControlState.OFF``/``SHADOW`` (real, operator-set --
        see :class:`~core.base_agent.AgentControlState`): ``OFF``
        returns before even acquiring the debounce, deliberately --
        consuming it here would silently suppress a *real* alert for
        up to :data:`DEBOUNCE_TTL_SECONDS` after an operator switches
        this agent back to ``LIVE`` while the danger still persists.
        ``SHADOW`` does acquire the debounce (it still performs one
        real, rate-limit-worthy action -- logging to
        :data:`_ALERT_HISTORY_REDIS_KEY`) but sends no SMS, trips no
        Kill Switch, and broadcasts no ``EMERGENCY_HALT``.
        """
        if self.control_state == AgentControlState.OFF:
            self._log.warning(
                "alert_agent_response_suppressed_agent_off", agent_id=self.agent_id, category=category, detail=detail
            )
            return

        acquired = await self._try_acquire_debounce(category)
        if not acquired:
            self._log.warning("alert_agent_danger_debounced", agent_id=self.agent_id, category=category, detail=detail)
            return

        self._log.critical("alert_agent_danger_detected", agent_id=self.agent_id, category=category, detail=detail)

        if self.control_state == AgentControlState.SHADOW:
            self._log.warning(
                "alert_agent_response_suppressed_agent_shadow",
                agent_id=self.agent_id, category=category, detail=detail,
            )
            await self._log_alert_history(category=category, detail=detail, sms_sent=False)
            return

        sms_body = _build_sms_body(category, detail)
        recipients = await self._get_alert_recipients()
        results = [await send_sms_alert(self._settings.twilio, message=sms_body, to=number) for number in recipients]
        sms_sent = any(results)
        await self._log_alert_history(category=category, detail=detail, sms_sent=sms_sent)

        reason = f"[{category}] {detail}"
        await trip_kill_switch(reason)
        await self.publish(EMERGENCY_HALT_TOPIC, {"reason": detail, "category": category})

    async def _log_alert_history(self, *, category: DangerCategory, detail: str, sms_sent: bool) -> None:
        """Persist one real alert onto :data:`_ALERT_HISTORY_REDIS_KEY` -- same LPUSH+LTRIM shape as Guardian's own alert list.

        Added specifically so a real UI (the Red Phone panel) has real
        data to display, rather than needing to fabricate a log of SMS
        sends that never happened.
        """
        assert self._redis_client is not None
        entry = {
            "timestamp": time.time(),
            "category": category,
            "detail": detail,
            "sms_sent": sms_sent,
        }
        await self._redis_client.execute("lpush", _ALERT_HISTORY_REDIS_KEY, json.dumps(entry))
        await self._redis_client.execute("ltrim", _ALERT_HISTORY_REDIS_KEY, 0, _ALERT_HISTORY_MAX_ENTRIES - 1)
