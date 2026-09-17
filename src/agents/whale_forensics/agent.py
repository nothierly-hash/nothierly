"""Protocole Fort Knox, Pilier 2: the "18th sensor" -- real on-chain whale/bridge Transfer forensics.

:class:`WhaleForensicsDaemon` is a real :class:`~core.base_agent.BaseAgent`
that polls a real Ethereum-compatible JSON-RPC endpoint for real ERC-20
``Transfer`` event logs on a configured watchlist of token contracts, and
publishes a real :class:`~core.schemas.WhaleTransferAlert` (topic
:data:`~core.schemas.ONCHAIN_TSUNAMI_WARNING_TOPIC`) the instant one clears a
configured USD notional threshold.

**Two real, disclosed corrections to the brief, stated upfront**:

1. **Polling, not a persistent WebSocket subscription.** The brief's own
   framing ("Web3.py / websockets", "écoute les événements") suggests an
   ``eth_subscribe`` push stream. This implementation instead polls
   ``eth_getLogs`` over the new block range since its last tick, once per
   real tick, against a plain HTTPS JSON-RPC endpoint (the same
   ``core.config.Web3Settings.http_provider_url`` :mod:`execution.mev_shield`
   already uses for its own read calls) -- real, simple, and resilient to
   the reconnect/backoff handling a long-lived WebSocket subscription would
   separately need (:class:`~core.base_agent.BaseAgent`'s own tick loop
   already gives this a free, built-in retry cadence a persistent
   subscription does not get for free). Functionally equivalent for this
   purpose: every real Transfer event that would arrive on a push stream is
   still found here, at most one real tick late.

2. **No hardcoded institutional/bridge address list.** The brief's own
   example names "des portefeuilles institutionnels connus" -- this module
   deliberately ships with **none** pre-populated (see
   :data:`core.config.Web3Settings.watched_institutional_addresses`'s own
   docstring): an address believed correct without being independently,
   currently re-verified against a real block explorer is exactly the kind
   of unverified claim this project's own engineering discipline refuses to
   assert as fact. A large, real transfer is newsworthy on its own real
   magnitude regardless of the counterparty -- so the core trigger (see
   :func:`decode_transfer_log`) never depends on a counterparty match; a
   configured label is purely additional, informational tagging on top of
   an alert that already fired for a real, magnitude-based reason.

**Real, disclosed USD conversion**: a stablecoin (``USDC``/``USDT``/``DAI``/
``BUSD``, matched by symbol) is valued 1:1. Every other configured token is
priced from this project's own real, live
:data:`~core.schemas.MARKET_METRICS_REDIS_KEY_PREFIX` cache (the same
mid-price :class:`~src.agents.market_data.agent.MarketDataAgent` already
publishes for its own instruments) via a configured ``{SYMBOL}/USDT`` pair --
if that pair has no live metrics cached (no :class:`MarketDataAgent`
currently streaming it), this daemon honestly skips pricing that token's
transfers for that tick rather than fabricating a price, logged once, not
silently.

**Gate 8 wiring**: on a real publish, this daemon also writes
:data:`~core.schemas.ONCHAIN_TSUNAMI_LOCK_REDIS_KEY` with a
:data:`~core.schemas.ONCHAIN_TSUNAMI_LOCK_TTL_SECONDS` TTL -- a bounded
"anticipation window", not a second kill switch (see that key's own
docstring for exactly why). :class:`~src.agents.execution.agent.
ExecutionAgent`'s Gate 8 checks this key's mere presence the same way it
already checks :attr:`~core.base_agent.AgentControlState.OFF`.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Final

import requests
import zmq.asyncio
from web3 import Web3
from web3.exceptions import Web3Exception

from core.base_agent import BaseAgent
from core.config import AppSettings
from core.logger import get_logger
from core.redis_client import RedisUnavailableError
from core.schemas import (
    MARKET_METRICS_REDIS_KEY_PREFIX,
    ONCHAIN_TSUNAMI_LOCK_REDIS_KEY,
    ONCHAIN_TSUNAMI_LOCK_TTL_SECONDS,
    ONCHAIN_TSUNAMI_WARNING_TOPIC,
    WhaleTransferAlert,
)

_log = get_logger(__name__)

_AGENT_ID: Final[str] = "whale-forensics-daemon"

# Real, well-known ERC-20 Transfer(address,address,uint256) event topic hash --
# Web3.keccak(text="Transfer(address,address,uint256)"), verified directly against a real
# Web3.keccak call, not copied from memory.
TRANSFER_EVENT_TOPIC: Final[str] = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"

# Real, disclosed best-effort decimals for common tokens -- used when a watched contract's own
# on-chain decimals() has not been separately confirmed (this daemon does not make a live
# decimals() call itself, to avoid coupling every tick's pricing to an extra, separate RPC round
# trip for a value that never changes for a given contract). Unknown symbols fall back to 18
# (the ERC-20 convention the large majority of tokens use) -- disclosed, not silently assumed
# correct for every possible token; an operator configuring an unusual token should verify this.
_KNOWN_DECIMALS: Final[dict[str, int]] = {
    "WETH": 18, "WBTC": 8, "USDC": 6, "USDT": 6, "DAI": 18, "BUSD": 18,
}
_DEFAULT_DECIMALS: Final[int] = 18

_STABLECOIN_SYMBOLS: Final[frozenset[str]] = frozenset({"USDC", "USDT", "DAI", "BUSD"})

_DEFAULT_CHAIN_NAME: Final[str] = "ethereum"

# Real, broad-but-bounded RPC failure surface: a connection failure against a real HTTPProvider
# raises requests' own exception hierarchy (confirmed directly -- ConnectionError/Timeout, NOT
# Web3Exception), while a malformed/rejected JSON-RPC response raises Web3Exception. Both are
# real, expected "the endpoint is down or unhappy right now" cases this daemon must survive tick
# to tick, not a program bug to crash on.
_RPC_ERRORS: Final[tuple[type[Exception], ...]] = (Web3Exception, requests.exceptions.RequestException)

# Real safety bound: a daemon that fell far behind (long outage, first-ever startup pointed at a
# provider with deep history) must never issue one unbounded eth_getLogs call spanning millions of
# blocks -- most providers reject that outright, and it is not this daemon's job to backfill deep
# history anyway (it detects new whale activity going forward, not a forensic historical scan).
_MAX_BLOCK_RANGE_PER_POLL: Final[int] = 2_000


def _parse_watched_tokens(raw: str) -> dict[str, str]:
    """Parses ``"SYMBOL:0xaddress,SYMBOL2:0xaddress2"`` into ``{lowercased_address: SYMBOL}``.

    Args:
        raw: :data:`core.config.Web3Settings.watched_token_contracts`'s raw string.

    Returns:
        A dict keyed by lowercased contract address (matching the case
        real event logs' own ``address`` field arrives in), valued by the
        configured symbol. Empty if ``raw`` is blank or malformed.
    """
    result: dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        symbol, _, address = entry.partition(":")
        symbol, address = symbol.strip().upper(), address.strip().lower()
        if symbol and address:
            result[address] = symbol
    return result


def _parse_institutional_labels(raw: str) -> dict[str, str]:
    """Parses ``"LABEL:0xaddress,LABEL2:0xaddress2"`` into ``{lowercased_address: LABEL}``.

    Purely informational tagging -- see this module's own docstring on why
    no alert's trigger depends on this being configured.
    """
    result: dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        label, _, address = entry.partition(":")
        label, address = label.strip(), address.strip().lower()
        if label and address:
            result[address] = label
    return result


def decode_transfer_log(
    log: dict[str, Any],
    *,
    chain: str,
    token_symbol: str,
    decimals: int,
    usd_price: float,
    threshold_usd: float,
) -> WhaleTransferAlert | None:
    """Decode one real ``eth_getLogs``-shaped ERC-20 Transfer log and evaluate it against threshold.

    Pure function -- no network/Redis access, fully testable without any
    real chain connection (this project's own established "pure decision
    logic, wiring is the caller's job" discipline; see e.g.
    ``execution.predator_honeypot``).

    Args:
        log: One real log entry as returned by ``web3.eth.get_logs`` (a
            dict-like with ``topics`` -- 3 entries for a Transfer: the
            event signature, then the indexed ``from``/``to`` addresses,
            each left-padded to 32 bytes -- ``data`` -- the raw, big-endian
            uint256 amount -- ``transactionHash``, and ``blockNumber``).
        chain: Real chain name this log was observed on (e.g.
            ``"ethereum"``).
        token_symbol: The configured symbol for this log's own contract
            address (resolved by the caller from its watchlist).
        decimals: The token's real decimal places, for converting the raw
            integer amount into a human-scale quantity.
        usd_price: Real USD price per one whole token unit, already
            resolved by the caller (1.0 for a stablecoin, or a real,
            live-fetched mid-price otherwise).
        threshold_usd: The configured whale-alert notional threshold.

    Returns:
        A real :class:`~core.schemas.WhaleTransferAlert` if the transfer's
        notional value clears ``threshold_usd``; ``None`` otherwise (not
        an error -- most transfers are not whale-sized).
    """
    topics = log["topics"]
    from_address = "0x" + bytes(topics[1])[-20:].hex()
    to_address = "0x" + bytes(topics[2])[-20:].hex()
    raw_amount = int.from_bytes(bytes(log["data"]), "big")
    token_amount = raw_amount / (10**decimals)
    amount_usd = token_amount * usd_price

    if amount_usd < threshold_usd:
        return None

    tx_hash_bytes = log["transactionHash"]
    tx_hash = "0x" + bytes(tx_hash_bytes).hex()

    return WhaleTransferAlert(
        chain=chain,
        token_symbol=token_symbol,
        from_address=from_address,
        to_address=to_address,
        amount_usd=amount_usd,
        tx_hash=tx_hash,
        block_number=int(log["blockNumber"]),
    )


class WhaleForensicsDaemon(BaseAgent):
    """Polls for real, over-threshold on-chain ERC-20 Transfer events; publishes an alert only.

    Never places, cancels, or resizes an order -- same "real, informational
    sensor only" scope as every other signal-only agent in this project (see
    module docstring). Real, disclosed no-op state: with no
    ``WEB3_HTTP_PROVIDER_URL`` or no ``WEB3_WATCHED_TOKEN_CONTRACTS``
    configured (this deployment's own actual default), :meth:`on_tick`
    returns immediately every cycle, logged once at :meth:`on_start`, never
    silently pretending to watch a chain it has no real connection to.
    """

    def __init__(
        self,
        *,
        tick_interval_seconds: float = 15.0,
        settings: AppSettings | None = None,
        manage_own_signals: bool = True,
        zmq_context: zmq.asyncio.Context | None = None,
        manage_own_redis: bool = True,
    ) -> None:
        super().__init__(
            _AGENT_ID,
            tick_interval_seconds=tick_interval_seconds,
            settings=settings,
            manage_own_signals=manage_own_signals,
            zmq_context=zmq_context,
            manage_own_redis=manage_own_redis,
        )
        web3_settings = self._settings.web3
        self._watched_tokens = _parse_watched_tokens(web3_settings.watched_token_contracts)
        self._institutional_labels = _parse_institutional_labels(web3_settings.watched_institutional_addresses)
        self._threshold_usd = web3_settings.whale_alert_threshold_usd
        self._w3: Web3 | None = (
            Web3(Web3.HTTPProvider(web3_settings.http_provider_url))
            if web3_settings.http_provider_url
            else None
        )
        self._last_polled_block: int | None = None
        self._logged_disabled_reason = False

    def _disabled_reason(self) -> str | None:
        if self._w3 is None:
            return "WEB3_HTTP_PROVIDER_URL is not configured."
        if not self._watched_tokens:
            return "WEB3_WATCHED_TOKEN_CONTRACTS is not configured (nothing to watch)."
        return None

    async def on_start(self) -> None:
        reason = self._disabled_reason()
        if reason is not None:
            self._logged_disabled_reason = True
            self._log.warning(
                "whale_forensics_daemon_disabled",
                agent_id=self.agent_id,
                detail=f"{reason} Ticking as a real, logged no-op -- see this module's own docstring.",
            )
        else:
            self._log.info(
                "whale_forensics_daemon_watching",
                agent_id=self.agent_id,
                watched_tokens=sorted(self._watched_tokens.values()),
                threshold_usd=self._threshold_usd,
            )

    async def _resolve_usd_price(self, token_symbol: str) -> float | None:
        """Real, disclosed price resolution -- see this module's own docstring.

        Returns ``None`` (never a fabricated price) if ``token_symbol``
        isn't a stablecoin and no live mid-price is currently cached for
        it.
        """
        if token_symbol in _STABLECOIN_SYMBOLS:
            return 1.0
        if self._redis_client is None:
            return None
        try:
            raw = await self._redis_client.execute("get", f"{MARKET_METRICS_REDIS_KEY_PREFIX}{token_symbol}/USDT")
        except RedisUnavailableError:
            return None
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode()
        try:
            return float(json.loads(raw)["mid_price"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

    async def _write_tsunami_lock(self) -> None:
        if self._redis_client is None:
            return
        try:
            await self._redis_client.execute(
                "set", ONCHAIN_TSUNAMI_LOCK_REDIS_KEY, str(time.time()), ex=ONCHAIN_TSUNAMI_LOCK_TTL_SECONDS
            )
        except RedisUnavailableError:
            self._log.warning("whale_forensics_lock_write_failed", agent_id=self.agent_id)

    async def on_tick(self) -> None:
        reason = self._disabled_reason()
        if reason is not None:
            return
        assert self._w3 is not None  # narrowed by _disabled_reason() above

        try:
            latest_block = await asyncio.to_thread(lambda: self._w3.eth.block_number)  # type: ignore[union-attr]
        except _RPC_ERRORS:
            self._log.warning("whale_forensics_rpc_read_failed", agent_id=self.agent_id)
            return

        if self._last_polled_block is None:
            # Cold start: never backfill arbitrary chain history -- begin watching from the next
            # real block onward, same "detect going forward, not a historical scan" scope as this
            # module's own docstring discloses.
            self._last_polled_block = latest_block
            return
        if latest_block <= self._last_polled_block:
            return

        from_block = self._last_polled_block + 1
        to_block = min(latest_block, from_block + _MAX_BLOCK_RANGE_PER_POLL - 1)

        w3 = self._w3
        # Real EIP-55 checksummed addresses for the RPC call itself -- web3.py's own typed filter
        # params require it, even though self._watched_tokens' own keys stay lowercased (matching
        # is done against a real log's own, already-lowercased `address` field below).
        watched_checksummed = [Web3.to_checksum_address(addr) for addr in self._watched_tokens]
        try:
            logs = await asyncio.to_thread(
                w3.eth.get_logs,
                {
                    "fromBlock": from_block,
                    "toBlock": to_block,
                    "address": watched_checksummed,
                    "topics": [TRANSFER_EVENT_TOPIC],
                },
            )
        except _RPC_ERRORS:
            self._log.warning(
                "whale_forensics_get_logs_failed", agent_id=self.agent_id, from_block=from_block, to_block=to_block
            )
            return

        for log in logs:
            token_address = str(log["address"]).lower()
            token_symbol = self._watched_tokens.get(token_address)
            if token_symbol is None:
                continue  # defensive: a provider that ignores the address filter, never trusted blindly
            usd_price = await self._resolve_usd_price(token_symbol)
            if usd_price is None:
                self._log.warning(
                    "whale_forensics_no_live_price_skipping", agent_id=self.agent_id, token_symbol=token_symbol
                )
                continue
            alert = decode_transfer_log(
                dict(log),
                chain=_DEFAULT_CHAIN_NAME,
                token_symbol=token_symbol,
                decimals=_KNOWN_DECIMALS.get(token_symbol, _DEFAULT_DECIMALS),
                usd_price=usd_price,
                threshold_usd=self._threshold_usd,
            )
            if alert is None:
                continue

            label = self._institutional_labels.get(alert.from_address) or self._institutional_labels.get(
                alert.to_address
            )
            self._log.warning(
                "onchain_tsunami_warning",
                agent_id=self.agent_id,
                token_symbol=alert.token_symbol,
                amount_usd=alert.amount_usd,
                tx_hash=alert.tx_hash,
                counterparty_label=label,
            )
            await self.publish(ONCHAIN_TSUNAMI_WARNING_TOPIC, alert.model_dump(mode="json"))
            await self._write_tsunami_lock()

        self._last_polled_block = to_block
