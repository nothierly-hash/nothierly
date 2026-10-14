"""Client MetaTrader 5 du worker Darwinex Zero : protocole minimal + adaptateur sur le paquet Python ``MetaTrader5``.

Le paquet ``MetaTrader5`` n'existe que sous **Windows** et pilote un terminal MT5 installé et lancé : le worker Darwinex
tourne donc sur l'hôte Windows (ou une VM Windows), et reçoit ses signaux du bus ZMQ chiffré (``contracts.curve_transport``).

**Le terminal trade le compte sur lequel il est connecté**, quel qu'il soit : ``initialize()`` sans identifiants
reprend « le dernier compte » (doc MQL5). Un compte réel ouvert à la main dans le même terminal serait donc piloté par
erreur. Garde-fous, appliqués à CHAQUE lecture et avant CHAQUE ordre (pas seulement au démarrage) :

- le numéro de compte du terminal doit être exactement ``expected_login`` (``WrongAccountError`` sinon) ;
- un compte de type réel est refusé tant que ``allow_real_account`` n'est pas explicitement vrai (``RealAccountRefused``).

Autres limites traitées en échec sûr :

- **Plusieurs positions sur un symbole** (compte hedging) : ``MultiplePositionsError``, rien n'est envoyé. Une seule
  position nette par symbole est gérée.
- **Réponse d'ordre ambiguë** (``None``, délai, perte de connexion) : statut ``error``, jamais rejoué ; la position est
  relue au signal suivant.
- **Fils d'exécution** : l'API MT5 n'est pas documentée comme réentrante ; tous les appels passent par un verrou unique.

Jamais exécuté contre un vrai terminal : ``MetaTrader5`` n'est pas installé ici et aucun compte n'a été utilisé. Les tests
vérifient le câblage avec un faux module et les codes de retour de la doc officielle.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Final, Protocol

from workers.darwinex.planner import Mt5Position, Quote, SymbolSpec

_VALUATION_MOVE: Final[Decimal] = Decimal("0.001")  # 10 points de base du prix : assez grand pour que l'arrondi au centime soit négligeable
_RES_OK: Final[int] = 1  # RES_S_OK de last_error() : « Success »
_RETCODES: Final[dict[int, str]] = {
    10004: "REQUOTE", 10006: "REJECT", 10007: "CANCEL", 10008: "PLACED", 10009: "DONE", 10010: "DONE_PARTIAL",
    10011: "ERROR", 10012: "TIMEOUT", 10013: "INVALID", 10014: "INVALID_VOLUME", 10015: "INVALID_PRICE",
    10016: "INVALID_STOPS", 10017: "TRADE_DISABLED", 10018: "MARKET_CLOSED", 10019: "NO_MONEY", 10020: "PRICE_CHANGED",
    10021: "PRICE_OFF", 10024: "TOO_MANY_REQUESTS", 10026: "SERVER_DISABLES_AT", 10027: "CLIENT_DISABLES_AT",
    10030: "INVALID_FILL", 10031: "CONNECTION", 10032: "ONLY_REAL", 10036: "POSITION_CLOSED", 10038: "INVALID_CLOSE_VOLUME",
    10042: "LONG_ONLY", 10043: "SHORT_ONLY", 10044: "CLOSE_ONLY", 10045: "FIFO_CLOSE", 10046: "HEDGE_PROHIBITED",
}
_DONE, _PARTIAL, _PLACED = 10009, 10010, 10008
_AMBIGUOUS = frozenset({10012, 10031})  # délai / connexion perdue : l'ordre est peut-être parti


class Mt5Error(RuntimeError):
    """Terminal injoignable, ou lecture MT5 impossible."""


class UnsupportedPlatformError(Mt5Error):
    """Le paquet ``MetaTrader5`` n'existe que sous Windows : le worker ne peut pas tourner ici."""


class Mt5NotInstalledError(Mt5Error):
    """Le paquet ``MetaTrader5`` est absent (``pip install MetaTrader5``, Windows uniquement)."""


class WrongAccountError(Mt5Error):
    """Le terminal est connecté à un autre compte que celui que ce worker a le droit de piloter."""


class RealAccountRefused(Mt5Error):
    """Le compte est de type réel et ``allow_real_account`` n'a pas été demandé."""


class MultiplePositionsError(Mt5Error):
    """Plusieurs positions sur un même symbole (hedging) : non géré, rien n'est envoyé."""


@dataclass(frozen=True)
class Mt5Snapshot:
    equity: Decimal
    positions: Mapping[str, Mt5Position]


@dataclass(frozen=True)
class AccountInfo:
    login: int
    trade_mode: str  # DEMO | CONTEST | REAL | UNKNOWN
    margin_mode: str  # NETTING | EXCHANGE | HEDGING | UNKNOWN
    currency: str
    server: str
    trade_allowed: bool


@dataclass(frozen=True)
class Mt5OrderRequest:
    symbol: str
    is_buy: bool
    volume: Decimal
    price: Decimal | None  # ``None`` : omis (exécution « Market », où le prix n'est pas utilisé)
    deviation: int
    sl: Decimal | None
    filling: str  # IOC | FOK | RETURN
    position_ticket: int | None
    comment: str


@dataclass(frozen=True)
class Mt5OrderOutcome:
    status: str  # filled | partial | placed | rejected | error
    retcode: int | None
    detail: str


class Mt5Client(Protocol):
    def account_check(self) -> AccountInfo:
        """Vérifie le compte (numéro, type) ; lève ``WrongAccountError`` / ``RealAccountRefused``."""
        ...

    def symbol_spec(self, symbol: str) -> SymbolSpec | None: ...

    def quote(self, symbol: str) -> Quote | None: ...

    def snapshot(self) -> Mt5Snapshot: ...

    def notional_per_lot(self, symbol: str, price: Decimal) -> Decimal | None:
        """Valeur d'un lot en devise du compte au prix ``price``, ou ``None`` si indéterminable."""
        ...

    def send(self, request: Mt5OrderRequest) -> Mt5OrderOutcome:
        """Envoie l'ordre. Un refus du courtier est un ``rejected`` ; une réponse ambiguë est un ``error``."""
        ...


def interpret_result(result: Any) -> Mt5OrderOutcome:
    """Interprète le retour de ``order_send`` (codes de retour de la doc MQL5)."""
    if result is None:
        return Mt5OrderOutcome("error", None, "order_send a renvoyé None : résultat inconnu")
    code = int(result.retcode)
    name = _RETCODES.get(code, "UNKNOWN")
    detail = f"{code} {name}: {getattr(result, 'comment', '')}"[:300]
    if code == _DONE:
        return Mt5OrderOutcome("filled", code, detail)
    if code == _PARTIAL:
        return Mt5OrderOutcome("partial", code, detail)
    if code == _PLACED:
        return Mt5OrderOutcome("placed", code, detail)
    if code in _AMBIGUOUS:
        return Mt5OrderOutcome("error", code, detail)
    return Mt5OrderOutcome("rejected", code, detail)


class MetaTrader5Client:
    """Adaptateur ``MetaTrader5``. Import différé : requis seulement à l'usage, et seulement sous Windows."""

    def __init__(
        self,
        *,
        expected_login: int,
        magic: int,
        path: str | None = None,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        timeout_ms: int = 60_000,
        allow_real_account: bool = False,
        mt5_module: Any | None = None,
    ) -> None:
        if mt5_module is None:
            # Contrôle d'OS ici, et non à l'import du module : avec un faux module injecté (tests, CI Linux) rien ne l'exige.
            if sys.platform != "win32":
                raise UnsupportedPlatformError(
                    f"Le worker Darwinex exige Windows (le paquet MetaTrader5 n'existe pas sur {sys.platform}) : "
                    "lancer ce worker sur la machine Windows qui héberge le terminal MT5."
                )
            try:
                import MetaTrader5  # Windows seulement, sans stubs (voir pyproject : override mypy « MetaTrader5.* »)
            except ImportError as exc:
                raise Mt5NotInstalledError("MetaTrader5 n'est pas installé : pip install MetaTrader5 (Windows uniquement).") from exc
            mt5_module = MetaTrader5
        self._mt5: Any = mt5_module
        self._expected_login = expected_login
        self._magic = magic
        self._init_args: dict[str, Any] = {"timeout": timeout_ms}
        if login is not None:
            self._init_args.update(login=login, server=server)
            if password is not None:
                self._init_args["password"] = password
        self._path = path
        self._allow_real = allow_real_account
        self._lock = threading.RLock()

    def __repr__(self) -> str:
        return f"MetaTrader5Client(expected_login={self._expected_login})"  # jamais de mot de passe ni de serveur

    def close(self) -> None:
        with self._lock:
            self._mt5.shutdown()

    # --- connexion et compte ----------------------------------------------------------------

    def _ensure_connected(self) -> None:
        info = self._mt5.terminal_info()
        if info is not None and getattr(info, "connected", False):
            return
        args = [self._path] if self._path else []
        if not self._mt5.initialize(*args, **self._init_args):
            raise Mt5Error(f"initialize a échoué : {self._mt5.last_error()}")

    def _account(self) -> Any:
        self._ensure_connected()
        info = self._mt5.account_info()
        if info is None:
            raise Mt5Error(f"account_info a échoué : {self._mt5.last_error()}")
        if int(info.login) != self._expected_login:
            raise WrongAccountError(
                f"Le terminal est connecté au compte {info.login}, ce worker ne pilote que le compte {self._expected_login}."
            )
        if self._trade_mode_name(info.trade_mode) == "REAL" and not self._allow_real:
            raise RealAccountRefused("Compte de type réel : refusé sans --allow-real-account.")
        return info

    def _trade_mode_name(self, value: int) -> str:
        return _name_of(self._mt5, value, {"DEMO": "ACCOUNT_TRADE_MODE_DEMO", "CONTEST": "ACCOUNT_TRADE_MODE_CONTEST", "REAL": "ACCOUNT_TRADE_MODE_REAL"})

    def account_check(self) -> AccountInfo:
        with self._lock:
            info = self._account()
            margin = _name_of(self._mt5, info.margin_mode, {
                "NETTING": "ACCOUNT_MARGIN_MODE_RETAIL_NETTING", "EXCHANGE": "ACCOUNT_MARGIN_MODE_EXCHANGE",
                "HEDGING": "ACCOUNT_MARGIN_MODE_RETAIL_HEDGING"})
            return AccountInfo(int(info.login), self._trade_mode_name(info.trade_mode), margin, str(info.currency), str(info.server), bool(info.trade_allowed))

    # --- lectures ---------------------------------------------------------------------------

    def symbol_spec(self, symbol: str) -> SymbolSpec | None:
        with self._lock:
            self._ensure_connected()
            self._mt5.symbol_select(symbol, True)
            info = self._mt5.symbol_info(symbol)
            if info is None:
                return None
            mode = _name_of(self._mt5, info.trade_mode, {
                "FULL": "SYMBOL_TRADE_MODE_FULL", "LONGONLY": "SYMBOL_TRADE_MODE_LONGONLY", "SHORTONLY": "SYMBOL_TRADE_MODE_SHORTONLY",
                "CLOSEONLY": "SYMBOL_TRADE_MODE_CLOSEONLY", "DISABLED": "SYMBOL_TRADE_MODE_DISABLED"})
            execution = _name_of(self._mt5, info.trade_exemode, {
                "REQUEST": "SYMBOL_TRADE_EXECUTION_REQUEST", "INSTANT": "SYMBOL_TRADE_EXECUTION_INSTANT",
                "MARKET": "SYMBOL_TRADE_EXECUTION_MARKET", "EXCHANGE": "SYMBOL_TRADE_EXECUTION_EXCHANGE"})
            return SymbolSpec(
                symbol=symbol, digits=int(info.digits), point=Decimal(str(info.point)),
                volume_min=Decimal(str(info.volume_min)), volume_max=Decimal(str(info.volume_max)),
                volume_step=Decimal(str(info.volume_step)), stops_level=int(info.trade_stops_level),
                trade_mode=mode, execution_mode=execution, filling_flags=int(info.filling_mode),
            )

    def quote(self, symbol: str) -> Quote | None:
        with self._lock:
            self._ensure_connected()
            tick = self._mt5.symbol_info_tick(symbol)
            if tick is None or tick.bid <= 0 or tick.ask <= 0:
                return None
            return Quote(Decimal(str(tick.bid)), Decimal(str(tick.ask)))

    def snapshot(self) -> Mt5Snapshot:
        with self._lock:
            info = self._account()
            raw = self._mt5.positions_get()
            if raw is None:
                code = self._mt5.last_error()[0]
                if code != _RES_OK:
                    raise Mt5Error(f"positions_get a échoué : {self._mt5.last_error()}")
                raw = ()
            positions: dict[str, Mt5Position] = {}
            for p in raw:
                if p.symbol in positions:
                    raise MultiplePositionsError(f"plusieurs positions ouvertes sur {p.symbol}")
                volume = Decimal(str(p.volume))
                signed = volume if int(p.type) == int(self._mt5.POSITION_TYPE_BUY) else -volume
                positions[str(p.symbol)] = Mt5Position(int(p.ticket), signed, Decimal(str(p.price_current)))
            return Mt5Snapshot(Decimal(str(info.equity)), positions)

    def notional_per_lot(self, symbol: str, price: Decimal) -> Decimal | None:
        """Valeur d'un lot en devise du compte : gain d'un lot pour un petit mouvement de prix, ramené au prix.

        ``order_calc_profit`` convertit déjà la devise de cotation vers celle du compte ; on en déduit la valeur par
        unité de prix, puis la valeur du lot au prix courant (``contrat x prix x conversion``).
        """
        with self._lock:
            self._ensure_connected()
            move = price * _VALUATION_MOVE
            profit = self._mt5.order_calc_profit(self._mt5.ORDER_TYPE_BUY, symbol, 1.0, float(price), float(price + move))
            if profit is None or profit <= 0:
                return None
            return Decimal(str(profit)) / move * price

    # --- ordre ------------------------------------------------------------------------------

    def send(self, request: Mt5OrderRequest) -> Mt5OrderOutcome:
        with self._lock:
            self._account()  # le compte est revérifié juste avant d'ordonner
            mt5 = self._mt5
            body: dict[str, Any] = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": request.symbol,
                "volume": float(request.volume),
                "type": mt5.ORDER_TYPE_BUY if request.is_buy else mt5.ORDER_TYPE_SELL,
                "deviation": request.deviation,
                "magic": self._magic,
                "comment": request.comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": getattr(mt5, f"ORDER_FILLING_{request.filling}"),
            }
            if request.price is not None:
                body["price"] = float(request.price)
            if request.sl is not None:
                body["sl"] = float(request.sl)
            if request.position_ticket is not None:
                body["position"] = request.position_ticket
            return interpret_result(mt5.order_send(body))


def _name_of(mt5: Any, value: Any, names: Mapping[str, str]) -> str:
    for label, constant in names.items():
        if hasattr(mt5, constant) and int(getattr(mt5, constant)) == int(value):
            return label
    return "UNKNOWN"
