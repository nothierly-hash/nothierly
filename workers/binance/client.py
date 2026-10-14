"""Client REST Binance USDⓈ-M Futures du worker : requêtes signées (HMAC-SHA256), stdlib seulement.

Même choix que le worker Bybit (pas de SDK) : la signature est écrite ici, testée contre la formule documentée, et le
transport est injectable. Pas de ``binance-futures-connector`` : une dépendance de plus pour cinq appels.

**Signature (doc officielle).** ``HMAC_SHA256(secret, chaîne de requête)`` en hexadécimal, ajoutée en dernier paramètre
(``signature``) ; ``timestamp`` (ms) obligatoire, ``recvWindow`` optionnel (5000 par défaut) ; l'en-tête ``X-MBX-APIKEY``
porte la clé. Tout est envoyé dans la chaîne de requête, y compris pour ``POST``. Si l'horloge locale dérive, Binance
refuse avec ``-1021`` : le client se resynchronise sur ``GET /fapi/v1/time`` (décalage mémorisé) et retente UNE fois, ce
qui est sûr puisque la requête a été refusée avant tout traitement.

**Garde-fous de clé.**

- **Droit de retrait** : ``GET /sapi/v1/account/apiRestrictions`` (hôte spot ``api.binance.com``, pas l'hôte futures)
  renvoie ``enableWithdrawals``. Ce n'est PAS ``/fapi/v1/apiTradingStatus``, qui ne décrit que les règles quantitatives
  de trading (indicateurs et verrous) et ne dit rien des permissions de la clé. Sur mainnet, une clé dont les droits ne
  peuvent pas être lus est refusée (on ne peut pas prouver l'absence de retrait). Le réseau de test n'a pas cet endpoint :
  la vérification y est indisponible, ce que le démarrage journalise.
- **Mode de position** : ``GET /fapi/v1/positionSide/dual`` (``dualSidePosition``). Le worker ne gère que le mode one-way.

**Ordres.** ``POST /fapi/v1/order`` en ``LIMIT`` IOC (ou GTC) avec ``newOrderRespType=RESULT`` : la réponse porte le
statut final (``FILLED``/``PARTIALLY_FILLED``/``EXPIRED``) et ``executedQty``. Un refus de Binance (code d'erreur négatif,
par exemple marge insuffisante) est un résultat ``rejected`` ; une réponse illisible, un délai dépassé ou ``-1007``
(statut d'exécution inconnu) sont AMBIGUS et lèvent : le worker ne réessaie pas, le signal suivant relit la position.

**Non vérifié contre le vrai service.** Aucune clé n'a été utilisée. Les formes de réponse viennent de la documentation
officielle (``account`` V3, ``positionSide/dual``, ``apiRestrictions``, en-têtes et signature) et, pour les filtres de
``exchangeInfo`` (``PRICE_FILTER``, ``LOT_SIZE``, ``MIN_NOTIONAL``) et ``bookTicker``, de la connaissance de l'API : la page
de documentation consultée ne les détaillait pas. Un symbole dont un filtre indispensable manque est exclu (échec sûr).
**Réseau de test** : la documentation donne ``demo-fapi.binance.com`` (page générale) et, ailleurs,
``testnet.binancefuture.com`` ; l'adresse est donc surchargeable (``base_url``).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Final, Protocol

from workers.binance.planner import InstrumentSpec

_META_TTL_S: Final[float] = 600.0
_CLOCK_SKEW_CODE: Final[int] = -1021  # « Timestamp for this request is outside of the recvWindow »
_UNKNOWN_STATUS_CODE: Final[int] = -1007  # « Timeout waiting for response from backend server. Send status unknown »


class BinanceNetwork(str, Enum):
    TESTNET = "testnet"
    MAINNET = "mainnet"


_REST_URLS: Final[dict[BinanceNetwork, str]] = {
    BinanceNetwork.TESTNET: "https://demo-fapi.binance.com",
    BinanceNetwork.MAINNET: "https://fapi.binance.com",
}
SAPI_URL: Final[str] = "https://api.binance.com"
WS_URLS: Final[dict[BinanceNetwork, str]] = {
    BinanceNetwork.TESTNET: "wss://demo-fstream.binance.com",
    BinanceNetwork.MAINNET: "wss://fstream.binance.com",
}


class BinanceApiError(RuntimeError):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"Binance {code}: {message}")
        self.code = code
        self.message = message


class HedgeModeError(RuntimeError):
    """Une position hedge (``positionSide`` LONG/SHORT) existe : ce worker ne gère que le mode one-way."""


@dataclass(frozen=True)
class BinancePosition:
    size: Decimal  # signé : positif = long
    mark_price: Decimal


@dataclass(frozen=True)
class BinanceSnapshot:
    equity: Decimal  # totalMarginBalance : solde + PnL latent
    positions: Mapping[str, BinancePosition]


@dataclass(frozen=True)
class BinanceOrderRequest:
    symbol: str
    is_buy: bool
    size: Decimal
    limit_px: Decimal
    tif: str
    reduce_only: bool
    client_order_id: str


@dataclass(frozen=True)
class BinanceOrderOutcome:
    status: str  # filled | partial | unfilled | rejected
    order_id: str | None
    executed_qty: Decimal
    detail: str


@dataclass(frozen=True)
class KeyPermissions:
    """Droits de la clé API. ``verified=False`` : ils n'ont pas pu être lus (réseau de test, droit de lecture absent...)."""

    verified: bool
    can_withdraw: bool | None = None
    can_read: bool | None = None
    can_trade_futures: bool | None = None
    ip_restricted: bool | None = None
    reason: str = ""


def check_key_permissions(info: KeyPermissions, *, live: bool, mainnet: bool) -> list[str]:
    """Problèmes bloquants de la clé API (liste vide = acceptable)."""
    problems = []
    if info.verified and info.can_withdraw:
        problems.append("la clé a le droit de retrait (enableWithdrawals) : créer une clé sans retrait")
    if mainnet and not info.verified:
        problems.append(
            "les droits de la clé n'ont pas pu être lus (GET /sapi/v1/account/apiRestrictions"
            + (f" : {info.reason}" if info.reason else "")
            + ") : refusée, car l'absence de droit de retrait ne peut pas être prouvée (activer « Enable Reading »)"
        )
    if live and info.verified and info.can_trade_futures is False:
        problems.append("la clé n'a pas « Enable Futures » : impossible d'ordonner")
    return problems


class BinanceClient(Protocol):
    def instruments(self) -> Mapping[str, InstrumentSpec]: ...

    def snapshot(self) -> BinanceSnapshot: ...

    def mid(self, symbol: str) -> Decimal | None: ...

    def place_order(self, request: BinanceOrderRequest) -> BinanceOrderOutcome:
        """Envoie l'ordre. Lève si le résultat est ambigu ; un refus de Binance est un ``rejected``."""
        ...


# (méthode, url, en-têtes, corps, délai) -> (statut HTTP, corps de réponse)
Transport = Callable[[str, str, Mapping[str, str], bytes | None, float], tuple[int, bytes]]


def _urllib_transport(method: str, url: str, headers: Mapping[str, str], body: bytes | None, timeout: float) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 -- URLs https fixes (_REST_URLS, SAPI_URL) ou surcharge explicite
            return int(response.status), bytes(response.read())
    except urllib.error.HTTPError as exc:
        return int(exc.code), bytes(exc.read())


def _dec(value: Any, default: str = "0") -> Decimal:
    try:
        return Decimal(str(value)) if value not in (None, "") else Decimal(default)
    except InvalidOperation as exc:
        raise ValueError("nombre illisible dans la réponse de Binance") from exc


def _fmt(value: Decimal) -> str:
    return format(value.normalize(), "f")


class BinanceRestClient:
    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
        network: BinanceNetwork,
        base_url: str | None = None,
        sapi_url: str = SAPI_URL,
        recv_window_ms: int = 5000,
        timeout: float = 10.0,
        transport: Transport | None = None,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        if not api_key or not api_secret:
            raise ValueError("api_key et api_secret sont requis.")
        self._api_key = api_key
        self._secret = api_secret.encode("utf-8")
        self._network = network
        self._base = (base_url or _REST_URLS[network]).rstrip("/")
        self._sapi = sapi_url.rstrip("/")
        self._recv = str(recv_window_ms)
        self._timeout = timeout
        self._transport = transport or _urllib_transport
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self._offset_ms = 0
        self._specs: dict[str, InstrumentSpec] = {}
        self._specs_at = 0.0

    def __repr__(self) -> str:
        return f"BinanceRestClient(base={self._base})"  # jamais la clé ni le secret

    # --- requêtes ---------------------------------------------------------------------------

    def _decode(self, status: int, raw: bytes) -> Any:
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise BinanceApiError(status, f"réponse illisible : {raw[:200]!r}") from exc
        if isinstance(payload, dict) and isinstance(payload.get("code"), int) and payload["code"] < 0:
            raise BinanceApiError(int(payload["code"]), str(payload.get("msg", "")))
        if status >= 400:
            raise BinanceApiError(status, f"HTTP {status} : {raw[:200]!r}")
        return payload

    def _signed(self, method: str, path: str, params: Mapping[str, str] | None = None, *, base: str | None = None, _retry: bool = True) -> Any:
        query = urllib.parse.urlencode({**(params or {}), "recvWindow": self._recv, "timestamp": str(self._clock_ms() + self._offset_ms)})
        signature = hmac.new(self._secret, query.encode("utf-8"), hashlib.sha256).hexdigest()
        url = f"{base or self._base}{path}?{query}&signature={signature}"  # `signature` en dernier, comme exigé
        status, raw = self._transport(method, url, {"X-MBX-APIKEY": self._api_key}, b"" if method != "GET" else None, self._timeout)
        try:
            return self._decode(status, raw)
        except BinanceApiError as exc:
            if exc.code == _CLOCK_SKEW_CODE and _retry:
                self.sync_time()
                return self._signed(method, path, params, base=base, _retry=False)
            raise

    def _public(self, path: str, params: Mapping[str, str] | None = None) -> Any:
        query = urllib.parse.urlencode(params or {})
        status, raw = self._transport("GET", f"{self._base}{path}" + (f"?{query}" if query else ""), {}, None, self._timeout)
        return self._decode(status, raw)

    def _keyed(self, method: str, path: str) -> Any:
        """Point d'accès ``USER_STREAM`` : l'en-tête de clé suffit, ni signature ni horodatage."""
        status, raw = self._transport(method, f"{self._base}{path}", {"X-MBX-APIKEY": self._api_key}, b"" if method != "GET" else None, self._timeout)
        return self._decode(status, raw) if raw.strip() else {}

    def sync_time(self) -> int:
        """Mesure le décalage entre l'horloge locale et celle de Binance ; renvoie le décalage en ms."""
        before = self._clock_ms()
        server = int(self._public("/fapi/v1/time")["serverTime"])
        self._offset_ms = server - (before + self._clock_ms()) // 2
        return self._offset_ms

    # --- lectures ---------------------------------------------------------------------------

    def instruments(self) -> Mapping[str, InstrumentSpec]:
        if not self._specs or time.monotonic() - self._specs_at > _META_TTL_S:
            specs: dict[str, InstrumentSpec] = {}
            for item in self._public("/fapi/v1/exchangeInfo").get("symbols", []):
                if item.get("contractType") != "PERPETUAL" or item.get("quoteAsset") != "USDT" or item.get("status") != "TRADING":
                    continue
                filters = {str(f.get("filterType")): f for f in item.get("filters", [])}
                lot, price = filters.get("LOT_SIZE"), filters.get("PRICE_FILTER")
                if lot is None or price is None:  # sans pas de quantité ou de prix, impossible d'arrondir : symbole exclu
                    continue
                notional = filters.get("MIN_NOTIONAL", {})
                max_qty = _dec(lot.get("maxQty"))
                specs[str(item["symbol"])] = InstrumentSpec(
                    symbol=str(item["symbol"]),
                    qty_step=_dec(lot["stepSize"]),
                    min_qty=_dec(lot.get("minQty")),
                    max_qty=max_qty if max_qty > 0 else None,
                    min_notional=_dec(notional.get("notional")),
                    tick_size=_dec(price["tickSize"]),
                )
            self._specs, self._specs_at = specs, time.monotonic()
        return self._specs

    def snapshot(self) -> BinanceSnapshot:
        account = self._signed("GET", "/fapi/v3/account")
        positions: dict[str, BinancePosition] = {}
        for item in account.get("positions", []):
            size = _dec(item.get("positionAmt"))
            if size == 0:
                continue
            if str(item.get("positionSide", "BOTH")) != "BOTH":
                raise HedgeModeError(f"position hedge détectée sur {item.get('symbol')} (positionSide={item.get('positionSide')})")
            mark = abs(_dec(item.get("notional")) / size)  # notional = positionAmt x prix de marque (signé)
            positions[str(item["symbol"])] = BinancePosition(size, mark)
        return BinanceSnapshot(_dec(account.get("totalMarginBalance")), positions)

    def mid(self, symbol: str) -> Decimal | None:
        row = self._public("/fapi/v1/ticker/bookTicker", {"symbol": symbol})
        bid, ask = _dec(row.get("bidPrice")), _dec(row.get("askPrice"))
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        last = _dec(self._public("/fapi/v1/ticker/price", {"symbol": symbol}).get("price"))
        return last if last > 0 else None

    def is_hedge_mode(self) -> bool:
        """``True`` si le compte est en mode hedge (``dualSidePosition``)."""
        return bool(self._signed("GET", "/fapi/v1/positionSide/dual").get("dualSidePosition"))

    def key_permissions(self) -> KeyPermissions:
        if self._network is not BinanceNetwork.MAINNET:
            return KeyPermissions(False, reason="indisponible sur le réseau de test")
        try:
            info = self._signed("GET", "/sapi/v1/account/apiRestrictions", base=self._sapi)
        except (BinanceApiError, OSError, ValueError) as exc:
            return KeyPermissions(False, reason=type(exc).__name__)
        return KeyPermissions(
            True,
            can_withdraw=bool(info.get("enableWithdrawals")),
            can_read=bool(info.get("enableReading")),
            can_trade_futures=bool(info.get("enableFutures")),
            ip_restricted=bool(info.get("ipRestrict")),
        )

    # --- flux privé (listenKey) -------------------------------------------------------------

    def start_user_stream(self) -> str:
        """``POST /fapi/v1/listenKey`` : renvoie la clé du flux privé (la même si une est déjà active, validité prolongée)."""
        return str(self._keyed("POST", "/fapi/v1/listenKey")["listenKey"])

    def keepalive_user_stream(self) -> None:
        self._keyed("PUT", "/fapi/v1/listenKey")

    # --- ordre ------------------------------------------------------------------------------

    def place_order(self, request: BinanceOrderRequest) -> BinanceOrderOutcome:
        params = {
            "symbol": request.symbol,
            "side": "BUY" if request.is_buy else "SELL",
            "type": "LIMIT",
            "timeInForce": request.tif,
            "quantity": _fmt(request.size),
            "price": _fmt(request.limit_px),
            "newClientOrderId": request.client_order_id,
            "newOrderRespType": "RESULT",
        }
        if request.reduce_only:
            params["reduceOnly"] = "true"
        try:
            response = self._signed("POST", "/fapi/v1/order", params)
        except BinanceApiError as exc:
            if exc.code == _UNKNOWN_STATUS_CODE or exc.code > 0:  # statut inconnu ou réponse HTTP sans code Binance : ambigu
                raise
            return BinanceOrderOutcome("rejected", None, Decimal(0), f"{exc.code}: {exc.message}"[:300])
        executed = _dec(response.get("executedQty"))
        status = str(response.get("status", ""))
        order_id = None if response.get("orderId") is None else str(response["orderId"])
        if status == "FILLED":
            return BinanceOrderOutcome("filled", order_id, executed, "")
        if executed > 0:
            return BinanceOrderOutcome("partial", order_id, executed, status)
        return BinanceOrderOutcome("unfilled", order_id, executed, status)
