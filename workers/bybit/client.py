"""Client REST Bybit V5 du worker : requêtes signées (HMAC-SHA256), stdlib seulement.

Signature (doc officielle) : ``HMAC_SHA256(secret, timestamp + clé API + recv_window + (query | corps JSON))``,
en hexadécimal minuscule, dans l'en-tête ``X-BAPI-SIGN``, avec ``X-BAPI-API-KEY``, ``X-BAPI-TIMESTAMP`` et
``X-BAPI-RECV-WINDOW``. Le corps JSON est signé **tel qu'envoyé** (mêmes octets).

Garde-fou de clé : ``GET /v5/user/query-api`` expose les permissions de la clé. Le worker refuse (``check_key_info``)
toute clé qui porte ``Wallet: Withdraw`` : même principe que l'agent sans retrait côté Hyperliquid. Une clé de
trading doit avoir « Contract - Orders & Positions » (mandatoire pour le copy trading, doc V5).

Limites connues, volontairement traitées en échec sûr plutôt que devinées :

- **Compte de solde** : seul ``accountType=UNIFIED`` est documenté pour ``wallet-balance``. Où se trouvent les
  fonds d'un master trader n'est pas précisé : à vérifier sur testnet. Un solde lu sur le mauvais compte
  fausserait la taille des ordres ; ``max_equity_usd`` (config du worker) plafonne l'equity utilisée.
- **Mode de position** : une position hedge (``positionIdx`` 1 ou 2) lève ``HedgeModeError`` ; seul le mode
  one-way (``positionIdx`` 0) est géré. L'exemple copy trading de la doc utilise ``positionIdx: 1`` : à vérifier.
- **Réponse d'ordre** : Bybit n'accuse que la réception (« please use the websocket to confirm the order status »).
  Le client renvoie ``accepted``, jamais ``filled`` : le remplissage réel se lit dans la position au signal suivant.

Jamais exécuté contre Bybit (ni testnet ni mainnet) : aucune clé n'a été utilisée. Les tests vérifient la
signature contre la formule documentée et le décodage avec de fausses réponses.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Final, Protocol

from workers.bybit.planner import InstrumentSpec

_META_TTL_S: Final[float] = 600.0
_PAGE_GUARD: Final[int] = 50  # pagination bornée : jamais de boucle infinie sur un curseur défaillant


class BybitNetwork(str, Enum):
    TESTNET = "testnet"
    MAINNET = "mainnet"


_BASE_URLS: Final[dict[BybitNetwork, str]] = {
    BybitNetwork.TESTNET: "https://api-testnet.bybit.com",
    BybitNetwork.MAINNET: "https://api.bybit.com",
}


class BybitApiError(RuntimeError):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"Bybit {code}: {message}")
        self.code = code
        self.message = message


class HedgeModeError(RuntimeError):
    """Une position hedge (``positionIdx`` 1 ou 2) existe : ce worker ne gère que le mode one-way."""


@dataclass(frozen=True)
class BybitPosition:
    size: Decimal  # signé : positif = long
    mark_price: Decimal


@dataclass(frozen=True)
class BybitSnapshot:
    equity: Decimal
    positions: Mapping[str, BybitPosition]


@dataclass(frozen=True)
class BybitOrderRequest:
    symbol: str
    is_buy: bool
    size: Decimal
    limit_px: Decimal
    tif: str
    reduce_only: bool
    order_link_id: str


@dataclass(frozen=True)
class BybitOrderOutcome:
    status: str  # accepted | rejected
    order_id: str | None
    detail: str


@dataclass(frozen=True)
class KeyInfo:
    read_only: bool
    permissions: Mapping[str, Sequence[str]]

    @property
    def can_withdraw(self) -> bool:
        return "Withdraw" in self.permissions.get("Wallet", ())

    @property
    def can_trade_contracts(self) -> bool:
        granted = self.permissions.get("ContractTrade", ())
        return "Order" in granted and "Position" in granted


def check_key_info(info: KeyInfo, *, live: bool) -> list[str]:
    """Problèmes bloquants de la clé API (liste vide = acceptable)."""
    problems = []
    if info.can_withdraw:
        problems.append("la clé a le droit de retrait (Wallet: Withdraw) : créer une clé sans retrait")
    if live and info.read_only:
        problems.append("la clé est en lecture seule : impossible d'ordonner")
    if live and not info.can_trade_contracts:
        problems.append("la clé n'a pas « Contract - Orders & Positions » (ContractTrade: Order, Position)")
    return problems


class BybitClient(Protocol):
    def instruments(self) -> Mapping[str, InstrumentSpec]: ...

    def snapshot(self) -> BybitSnapshot: ...

    def mid(self, symbol: str) -> Decimal | None: ...

    def key_info(self) -> KeyInfo: ...

    def place_order(self, request: BybitOrderRequest) -> BybitOrderOutcome:
        """Envoie l'ordre. Lève en cas d'erreur réseau (résultat ambigu) ; un refus de Bybit est un ``rejected``."""
        ...


# (méthode, url, en-têtes, corps, délai) -> (statut HTTP, corps de réponse)
Transport = Callable[[str, str, Mapping[str, str], bytes | None, float], tuple[int, bytes]]


def _urllib_transport(method: str, url: str, headers: Mapping[str, str], body: bytes | None, timeout: float) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 -- URLs fixes https (_BASE_URLS)
            return int(response.status), bytes(response.read())
    except urllib.error.HTTPError as exc:
        return int(exc.code), bytes(exc.read())


def _dec(value: Any, default: str = "0") -> Decimal:
    return Decimal(str(value)) if value not in (None, "") else Decimal(default)


class BybitRestClient:
    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
        network: BybitNetwork,
        recv_window_ms: int = 5000,
        timeout: float = 10.0,
        transport: Transport | None = None,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        if not api_key or not api_secret:
            raise ValueError("api_key et api_secret sont requis.")
        self._api_key = api_key
        self._secret = api_secret.encode("utf-8")
        self._base = _BASE_URLS[network]
        self._recv = str(recv_window_ms)
        self._timeout = timeout
        self._transport = transport or _urllib_transport
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self._specs: dict[str, InstrumentSpec] = {}
        self._specs_at = 0.0

    def __repr__(self) -> str:
        return f"BybitRestClient(base={self._base})"

    # --- requête signée ---------------------------------------------------------------------

    def _signed_headers(self, payload: str) -> dict[str, str]:
        timestamp = str(self._clock_ms())
        to_sign = timestamp + self._api_key + self._recv + payload
        return {
            "X-BAPI-API-KEY": self._api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-SIGN": hmac.new(self._secret, to_sign.encode("utf-8"), hashlib.sha256).hexdigest(),
            "X-BAPI-RECV-WINDOW": self._recv,
        }

    def _call(self, method: str, path: str, *, params: Mapping[str, str] | None = None, body: Mapping[str, Any] | None = None) -> tuple[int, str, Any]:
        """Renvoie ``(retCode, retMsg, result)``. Lève ``BybitApiError`` si la réponse n'est pas du JSON Bybit."""
        if method == "GET":
            query = urllib.parse.urlencode(params or {})
            url = f"{self._base}{path}" + (f"?{query}" if query else "")
            headers, data = self._signed_headers(query), None
        else:
            text = json.dumps(body or {}, separators=(",", ":"))  # signé et envoyé : mêmes octets
            url, data = f"{self._base}{path}", text.encode("utf-8")
            headers = {**self._signed_headers(text), "Content-Type": "application/json"}
        status, raw = self._transport(method, url, headers, data, self._timeout)
        try:
            payload = json.loads(raw)
            return int(payload["retCode"]), str(payload.get("retMsg", "")), payload.get("result")
        except (ValueError, KeyError, TypeError) as exc:
            raise BybitApiError(status, f"réponse illisible : {raw[:200]!r}") from exc

    def _get(self, path: str, params: Mapping[str, str]) -> Any:
        code, message, result = self._call("GET", path, params=params)
        if code != 0:
            raise BybitApiError(code, message)
        return result

    def _pages(self, path: str, params: Mapping[str, str]) -> list[Any]:
        """Concatène les ``list`` de toutes les pages (curseur ``nextPageCursor``), de façon bornée."""
        items: list[Any] = []
        cursor = ""
        for _ in range(_PAGE_GUARD):
            result = self._get(path, {**params, **({"cursor": cursor} if cursor else {})})
            items.extend(result.get("list", []))
            cursor = str(result.get("nextPageCursor") or "")
            if not cursor:
                return items
        raise BybitApiError(-1, f"pagination sans fin sur {path}")

    # --- lectures ---------------------------------------------------------------------------

    def instruments(self) -> Mapping[str, InstrumentSpec]:
        if not self._specs or time.monotonic() - self._specs_at > _META_TTL_S:
            specs: dict[str, InstrumentSpec] = {}
            for item in self._pages("/v5/market/instruments-info", {"category": "linear", "status": "Trading", "limit": "1000"}):
                if item.get("settleCoin") != "USDT" or item.get("contractType") != "LinearPerpetual":
                    continue
                lots, price = item["lotSizeFilter"], item["priceFilter"]
                max_qty = _dec(lots.get("maxOrderQty"), "0")
                specs[str(item["symbol"])] = InstrumentSpec(
                    symbol=str(item["symbol"]),
                    qty_step=_dec(lots["qtyStep"]),
                    min_qty=_dec(lots.get("minOrderQty")),
                    max_qty=max_qty if max_qty > 0 else None,
                    min_notional=_dec(lots.get("minNotionalValue")),
                    tick_size=_dec(price["tickSize"]),
                    copy_trading=str(item.get("copyTrading", "unknown")),
                )
            self._specs, self._specs_at = specs, time.monotonic()
        return self._specs

    def snapshot(self) -> BybitSnapshot:
        wallet = self._get("/v5/account/wallet-balance", {"accountType": "UNIFIED"}).get("list", [])
        equity = _dec(wallet[0].get("totalEquity")) if wallet else Decimal(0)
        positions: dict[str, BybitPosition] = {}
        for item in self._pages("/v5/position/list", {"category": "linear", "settleCoin": "USDT", "limit": "200"}):
            size = _dec(item.get("size"))
            if size == 0:
                continue
            if int(item.get("positionIdx", 0)) != 0:
                raise HedgeModeError(f"position hedge détectée sur {item.get('symbol')} (positionIdx={item.get('positionIdx')})")
            signed = size if item.get("side") == "Buy" else -size
            positions[str(item["symbol"])] = BybitPosition(signed, _dec(item.get("markPrice")))
        return BybitSnapshot(equity, positions)

    def mid(self, symbol: str) -> Decimal | None:
        rows = self._get("/v5/market/tickers", {"category": "linear", "symbol": symbol}).get("list", [])
        if not rows:
            return None
        bid, ask, last = _dec(rows[0].get("bid1Price")), _dec(rows[0].get("ask1Price")), _dec(rows[0].get("lastPrice"))
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        return last if last > 0 else None

    def key_info(self) -> KeyInfo:
        result = self._get("/v5/user/query-api", {})
        permissions = {str(k): [str(p) for p in v] for k, v in (result.get("permissions") or {}).items()}
        return KeyInfo(read_only=int(result.get("readOnly", 0)) == 1, permissions=permissions)

    # --- ordre ------------------------------------------------------------------------------

    def place_order(self, request: BybitOrderRequest) -> BybitOrderOutcome:
        body = {
            "category": "linear",
            "symbol": request.symbol,
            "side": "Buy" if request.is_buy else "Sell",
            "orderType": "Limit",
            "qty": format(request.size.normalize(), "f"),
            "price": format(request.limit_px.normalize(), "f"),
            "timeInForce": request.tif,
            "reduceOnly": request.reduce_only,
            "orderLinkId": request.order_link_id,
            "positionIdx": 0,
        }
        code, message, result = self._call("POST", "/v5/order/create", body=body)
        if code != 0:
            return BybitOrderOutcome("rejected", None, f"{code}: {message}"[:300])
        return BybitOrderOutcome("accepted", str((result or {}).get("orderId")), "")
