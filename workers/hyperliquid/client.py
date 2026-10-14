"""Client Hyperliquid du worker : protocole minimal + adaptateur sur le SDK officiel.

Le worker ne parle à Hyperliquid que par ``HyperliquidClient``. L'adaptateur ``SdkHyperliquidClient``
utilise ``hyperliquid-python-sdk`` (``Info`` pour lire, ``Exchange`` pour ordonner). Ce SDK est
synchrone (``requests``) : le worker appelle ces méthodes via ``asyncio.to_thread``.

Clé : le worker ne doit jamais détenir la clé du compte principal. ``agent_secret`` est la clé d'un
**API wallet** (agent) approuvé par ce compte : il peut passer des ordres, il ne peut pas retirer de
fonds. Elle n'est lue qu'ici, jamais journalisée ni conservée en attribut. Sans ``agent_secret``,
le client est en **lecture seule** (dry-run) et ``place_order`` refuse.

Non vérifié contre le vrai service : le SDK n'est pas installé dans cet environnement et aucun appel
réseau n'a été fait. Les signatures utilisées viennent du code source officiel du SDK ; les tests
vérifient le câblage avec de faux modules. À valider sur testnet, en particulier le comportement d'un
agent qui trade pour un vault (``vault_address``).
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Final, Protocol

from workers.common.ratelimit import RateLimitError, RateLimitGate, rate_limit_status


class Network(str, Enum):
    TESTNET = "testnet"
    MAINNET = "mainnet"


_BASE_URLS: Final[dict[Network, str]] = {
    Network.TESTNET: "https://api.hyperliquid-testnet.xyz",
    Network.MAINNET: "https://api.hyperliquid.xyz",
}
_SECRET_RE: Final[re.Pattern[str]] = re.compile(r"^(0x)?[0-9a-fA-F]{64}$")
_ADDRESS_RE: Final[re.Pattern[str]] = re.compile(r"^0x[0-9a-fA-F]{40}$")
_META_TTL_S: Final[float] = 600.0


@dataclass(frozen=True)
class AccountSnapshot:
    equity: Decimal  # accountValue perp, en USD (les soldes spot ne sont pas comptés)
    positions: Mapping[str, Decimal]  # actif -> taille signée (positif = long)


@dataclass(frozen=True)
class OrderRequest:
    asset: str
    is_buy: bool
    size: Decimal
    limit_px: Decimal
    tif: str
    reduce_only: bool
    cloid: str


@dataclass(frozen=True)
class OrderOutcome:
    status: str  # filled | resting | rejected
    filled_size: Decimal
    avg_px: Decimal | None
    oid: int | None
    detail: str


class HyperliquidClient(Protocol):
    def asset_decimals(self) -> Mapping[str, int]:
        """``szDecimals`` de chaque actif perp listé."""
        ...

    def snapshot(self) -> AccountSnapshot: ...

    def mids(self) -> Mapping[str, Decimal]: ...

    def place_order(self, request: OrderRequest) -> OrderOutcome:
        """Envoie l'ordre. Lève si le client est en lecture seule ou si le réseau échoue."""
        ...


def parse_order_response(response: Any) -> OrderOutcome:
    """Interprète la réponse de ``Exchange.order`` : ``filled``, ``resting`` ou ``rejected``."""
    if not isinstance(response, Mapping) or response.get("status") != "ok":
        return OrderOutcome("rejected", Decimal(0), None, None, f"réponse inattendue : {str(response)[:300]}")
    try:
        status = response["response"]["data"]["statuses"][0]
    except (KeyError, IndexError, TypeError):
        return OrderOutcome("rejected", Decimal(0), None, None, f"statuts absents : {str(response)[:300]}")
    if isinstance(status, Mapping) and "filled" in status:
        filled = status["filled"]
        return OrderOutcome("filled", Decimal(str(filled["totalSz"])), Decimal(str(filled["avgPx"])), int(filled["oid"]), "")
    if isinstance(status, Mapping) and "resting" in status:
        return OrderOutcome("resting", Decimal(0), None, int(status["resting"]["oid"]), "")
    if isinstance(status, Mapping) and "error" in status:
        return OrderOutcome("rejected", Decimal(0), None, None, str(status["error"])[:300])
    return OrderOutcome("rejected", Decimal(0), None, None, f"statut inconnu : {str(status)[:300]}")


class SdkHyperliquidClient:
    """Adaptateur sur ``hyperliquid-python-sdk``. Imports différés : requis seulement à l'usage."""

    def __init__(
        self,
        *,
        network: Network,
        account_address: str,
        agent_secret: str | None = None,
        vault_address: str | None = None,
        timeout: float = 10.0,
        rate_gate: RateLimitGate | None = None,
    ) -> None:
        if not _ADDRESS_RE.match(account_address):
            raise ValueError("account_address doit être une adresse 0x de 40 caractères hexadécimaux.")
        if vault_address is not None and not _ADDRESS_RE.match(vault_address):
            raise ValueError("vault_address doit être une adresse 0x de 40 caractères hexadécimaux.")
        if agent_secret is not None and not _SECRET_RE.match(agent_secret):
            raise ValueError("agent_secret n'a pas le format d'une clé privée (64 caractères hexadécimaux).")
        try:
            from hyperliquid.exchange import Exchange  # SDK optionnel, sans stubs (voir pyproject : override mypy « hyperliquid.* »)
            from hyperliquid.info import Info
        except ImportError as exc:  # pragma: no cover - dépend de l'environnement
            raise RuntimeError("hyperliquid-python-sdk n'est pas installé : pip install hyperliquid-python-sdk") from exc
        base_url = _BASE_URLS[network]
        # Les soldes et positions sont ceux du vault s'il y en a un, sinon du compte principal.
        self._query_address = vault_address or account_address
        self._info: Any = Info(base_url, skip_ws=True, timeout=timeout)
        self._exchange: Any = None
        if agent_secret is not None:
            from eth_account import Account

            wallet = Account.from_key(agent_secret)  # la clé n'est pas conservée au-delà de cet appel
            self._exchange = Exchange(
                wallet, base_url, vault_address=vault_address, account_address=account_address, timeout=timeout
            )
        self._decimals: dict[str, int] = {}
        self._decimals_at = 0.0
        self.rate_gate = rate_gate or RateLimitGate()

    @property
    def read_only(self) -> bool:
        return self._exchange is None

    def __repr__(self) -> str:
        return f"SdkHyperliquidClient(read_only={self.read_only})"

    def _sdk(self, call: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Appel du SDK sous la porte de débit : refusé localement pendant une pause, et un 429/418 (``status_code`` + ``header`` de l'erreur du SDK)
        arme la pause. Toute autre erreur est relancée telle quelle."""
        self.rate_gate.check()
        try:
            result = call(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- les erreurs du SDK n'ont pas de classe commune typée ; relancée si ce n'est pas une limite de débit
            limited = rate_limit_status(exc)
            if limited is None:
                raise
            raise RateLimitError(limited[0], self.rate_gate.block(*limited)) from exc
        self.rate_gate.note_success()
        return result

    def asset_decimals(self) -> Mapping[str, int]:
        if not self._decimals or time.monotonic() - self._decimals_at > _META_TTL_S:
            universe = self._sdk(self._info.meta)["universe"]
            self._decimals = {str(a["name"]): int(a["szDecimals"]) for a in universe}
            self._decimals_at = time.monotonic()
        return self._decimals

    def snapshot(self) -> AccountSnapshot:
        state = self._sdk(self._info.user_state, self._query_address)
        positions = {
            str(p["position"]["coin"]): Decimal(str(p["position"]["szi"])) for p in state.get("assetPositions", [])
        }
        return AccountSnapshot(Decimal(str(state["marginSummary"]["accountValue"])), positions)

    def mids(self) -> Mapping[str, Decimal]:
        return {str(coin): Decimal(str(px)) for coin, px in self._sdk(self._info.all_mids).items()}

    def place_order(self, request: OrderRequest) -> OrderOutcome:
        if self._exchange is None:
            raise RuntimeError("Client en lecture seule : aucune clé d'agent, aucun ordre possible.")
        from hyperliquid.utils.types import Cloid  # SDK optionnel, sans stubs (override mypy « hyperliquid.* »)

        response = self._sdk(
            self._exchange.order,
            request.asset,
            request.is_buy,
            float(request.size),
            float(request.limit_px),
            {"limit": {"tif": request.tif}},
            reduce_only=request.reduce_only,
            cloid=Cloid.from_str(request.cloid),
        )
        return parse_order_response(response)
