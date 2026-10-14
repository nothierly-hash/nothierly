"""Worker Darwinex Zero : amène la position réelle MT5 vers l'exposition cible des ``PlatformSignal`` signés.

Réception signée (et chiffrée si CURVE est configuré), file « dernier signal par actif », coupe-circuit de drawdown,
homme mort et boucle ZMQ : voir ``workers.common.base``. Ici : lecture du compte MT5, planification (``planner``) et
envoi des ordres au terminal.

**Doit tourner sous Windows**, sur la même machine que le terminal MetaTrader 5 (paquet ``MetaTrader5``). Le worker ne
pilote que le compte dont le numéro est ``--account-login`` : voir ``client`` (le terminal trade le compte connecté,
quel qu'il soit). Un compte de type réel est refusé sans ``--allow-real-account``.

Particularités :

- **Stop-loss serveur** : tout ordre d'ouverture porte le ``stop_loss_price`` du signal (exigé par le contrat).
- **Une inversion se fait en deux temps** (fermeture avec le ticket, puis ouverture au signal suivant) : sur un compte
  hedging, un ordre opposé plus gros n'inverse pas, il ouvre une position inverse sans fermer la première.
- **Marché fermé** (week-end) : l'ordre est refusé par le courtier (``MARKET_CLOSED``) ; l'homme mort réessaie à
  chaque intervalle, donc une position non fermable reste ouverte jusqu'à la réouverture : le risque de gap du week-end
  n'est pas couvert par ce worker.
- **Exécution « Market »** (selon le courtier) : le prix n'est pas utilisé, la déviation ne plafonne donc pas le
  glissement. Les ordres sont alors des ordres au marché.
- **Dry-run par défaut** : aucun ordre n'est envoyé. Jamais exécuté contre un vrai terminal.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import ClassVar

from contracts.curve_transport import CurveClientConfig, CurveConfigError, transport_refusal
from contracts.platform_signal import PlatformSignal, TargetVenue
from contracts.secure_signal_bus import load_secret_from_env
from workers.common.alerts import AlertConfigError, WebhookAlerter
from workers.common.base import EmergencyOrder, ExposureWorkerBase, ExposureWorkerConfig, clear_execution_halt, dead_man_timeout_problem
from workers.common.exposure import emergency_slippage_bps
from workers.common.logging_setup import configure_logging
from workers.common.state import ExecutionState
from workers.darwinex.client import (
    MetaTrader5Client,
    Mt5Client,
    Mt5Error,
    Mt5OrderRequest,
    Mt5Snapshot,
    MultiplePositionsError,
    RealAccountRefused,
    WrongAccountError,
)
from workers.darwinex.planner import (
    ZERO,
    Mt5OrderPlan,
    Quote,
    SymbolSpec,
    choose_filling,
    deviation_points,
    lots_for_pct,
    make_comment,
    mt5_symbol,
    plan_order,
)

_log = logging.getLogger(__name__)

__all__ = ["DarwinexWorker", "DarwinexWorkerConfig", "main"]


@dataclass(frozen=True)
class _Market:
    spec: SymbolSpec
    quote: Quote
    snapshot: Mt5Snapshot


@dataclass(frozen=True)
class DarwinexWorkerConfig(ExposureWorkerConfig):
    db_path: str | Path = "darwinex_worker.sqlite3"
    symbol_suffix: str = ""
    max_spread_bps: Decimal = Decimal("20")  # une ouverture au-delà est refusée (une fermeture ne l'est jamais)
    default_stop_loss_bps: Decimal | None = Decimal("100")  # stop posé par le worker si le signal n'en porte pas ; None = refuser


class DarwinexWorker(ExposureWorkerBase):
    venue: ClassVar[TargetVenue] = TargetVenue.DARWINEX_ZERO
    _tag: ClassVar[str] = "dwx"
    supports_dead_man: ClassVar[bool] = True

    def __init__(self, config: DarwinexWorkerConfig, client: Mt5Client, state: ExecutionState | None = None) -> None:
        super().__init__(config, state)
        self._client = client
        self._dwx_cfg = config

    def map_symbol(self, symbol: str) -> str | None:
        return mt5_symbol(symbol, self._dwx_cfg.symbol_suffix, self._cfg.symbol_overrides)

    async def _read_market(self, asset: str) -> _Market | str:
        """Lit symbole, cotation et compte ; renvoie un code d'erreur (``str``) si une lecture est impossible."""
        try:
            spec = await asyncio.to_thread(self._client.symbol_spec, asset)
            if spec is None:
                return "unknown_symbol"
            quote = await asyncio.to_thread(self._client.quote, asset)
            snapshot = await asyncio.to_thread(self._client.snapshot)
        except WrongAccountError as exc:
            _log.critical("dwx_wrong_account %s", exc)
            return "wrong_account"
        except RealAccountRefused as exc:
            _log.critical("dwx_real_account_refused %s", exc)
            return "real_account_refused"
        except MultiplePositionsError as exc:
            _log.error("dwx_multiple_positions %s", exc)
            return "hedge_positions_unsupported"
        except Exception:  # noqa: BLE001 -- lecture terminal : on n'ordonne pas à l'aveugle
            _log.exception("dwx_read_failed")
            return "read_error"
        if quote is None:
            return "no_quote"
        return _Market(spec, quote, snapshot)

    async def execute_signal(self, signal: PlatformSignal, asset: str, *, now_ms: int | None = None) -> str:
        """Lit le compte, planifie et (hors dry-run) envoie l'ordre. Renvoie un code court, toujours journalisé."""
        now = int(time.time() * 1000) if now_ms is None else now_ms
        if signal.is_expired(now) or now - signal.timestamp_ms > self._cfg.max_age_ms:
            self._record(signal, asset, "stale_at_execution", now, None, "")
            return "stale_at_execution"
        market = await self._read_market(asset)
        if isinstance(market, str):
            self._record(signal, asset, market, now, None, "")
            return market
        spec, quote, snapshot = market.spec, market.quote, market.snapshot
        mid = (quote.bid + quote.ask) / 2
        notional = await asyncio.to_thread(self._client.notional_per_lot, asset, mid)
        if notional is None:
            self._record(signal, asset, "no_valuation", now, None, "")
            return "no_valuation"
        others = ZERO
        for other, position in snapshot.positions.items():
            if other == asset:
                continue
            value = await asyncio.to_thread(self._client.notional_per_lot, other, position.price_current)
            if value is None:  # on ne laisse pas passer un plafond d'exposition brute calculé à l'aveugle
                self._record(signal, asset, "no_valuation", now, None, f"valeur inconnue pour {other}")
                return "no_valuation"
            others += abs(position.volume) * value

        halted = self._update_halt(snapshot.equity, now)
        target = lots_for_pct(snapshot.equity, notional, Decimal(str(signal.position_size_pct)), spec)
        plan = plan_order(
            signal, symbol=asset, target_abs_lots=target, equity=snapshot.equity, position=snapshot.positions.get(asset),
            quote=quote, spec=spec, notional_per_lot=notional, others_notional=others, limits=self._cfg.limits,
            halted=halted, max_spread_bps=self._dwx_cfg.max_spread_bps, default_stop_loss_bps=self._dwx_cfg.default_stop_loss_bps,
        )
        if not isinstance(plan, Mt5OrderPlan):
            self._record(signal, asset, f"skipped:{plan.reason}", now, None, "")
            return plan.reason
        if self._cfg.dry_run:
            self._record(signal, asset, "dry_run", now, plan, "stop par défaut" if plan.sl_defaulted else "")
            _log.info("dwx_dry_run %s %s lots=%s px=%s sl=%s close=%s", asset, "BUY" if plan.is_buy else "SELL", plan.volume, plan.price, plan.sl, plan.reduce_only)
            return "dry_run"
        request = Mt5OrderRequest(
            symbol=asset, is_buy=plan.is_buy, volume=plan.volume,
            price=None if spec.execution_mode == "MARKET" else plan.price,  # exécution Market : le prix n'est pas utilisé
            deviation=plan.deviation_points, sl=plan.sl, filling=plan.filling, position_ticket=plan.position_ticket, comment=plan.comment,
        )
        try:
            outcome = await asyncio.to_thread(self._client.send, request)
        except WrongAccountError as exc:
            _log.critical("dwx_wrong_account_at_send %s", exc)
            self._record(signal, asset, "wrong_account", now, plan, "")
            return "wrong_account"
        except Exception as exc:  # noqa: BLE001 -- ambigu : pas de nouvel essai, le prochain signal relit la position
            _log.exception("dwx_send_failed")
            self._record(signal, asset, "error", now, plan, repr(exc)[:200])
            return "error"
        self._record(signal, asset, outcome.status, now, plan, ("stop par défaut ; " if plan.sl_defaulted else "") + outcome.detail)
        return outcome.status

    async def emergency_flatten(self, asset: str, *, attempt: int, now_ms: int) -> str:
        """Ferme la position de ``asset`` avec son ticket, au marché avec une déviation qui double à chaque tentative."""
        market = await self._read_market(asset)
        if isinstance(market, str):
            self._record_emergency(asset, market, now_ms, None)
            return market
        spec, quote, snapshot = market.spec, market.quote, market.snapshot
        position = snapshot.positions.get(asset)
        if position is None or position.volume == 0:
            return "no_position"
        filling = choose_filling(spec)
        if filling is None:
            self._record_emergency(asset, "no_valid_filling", now_ms, None)
            return "no_valid_filling"
        is_buy = position.volume < 0  # une position vendeuse se ferme en achetant
        price = quote.ask if is_buy else quote.bid
        slippage = emergency_slippage_bps(self._cfg.emergency_slippage_bps, attempt)
        order = EmergencyOrder(is_buy, min(abs(position.volume), spec.volume_max), price)
        if self._cfg.dry_run:
            self._record_emergency(asset, "dry_run", now_ms, order)
            _log.warning("dwx_dead_man_dry_run %s %s lots=%s (aucun ordre envoyé)", asset, "BUY" if is_buy else "SELL", order.size)
            return "dry_run"
        request = Mt5OrderRequest(
            symbol=asset, is_buy=is_buy, volume=order.size,
            price=None if spec.execution_mode == "MARKET" else price,
            deviation=deviation_points(price, slippage, spec), sl=None, filling=filling, position_ticket=position.ticket,
            comment=make_comment(f"deadman:{asset}:{now_ms}:{attempt}"),
        )
        try:
            outcome = await asyncio.to_thread(self._client.send, request)
        except Exception as exc:  # noqa: BLE001 -- ambigu : la position est relue au prochain cycle, qui décide
            _log.exception("dwx_dead_man_send_failed")
            self._record_emergency(asset, "error", now_ms, order, repr(exc)[:200])
            return "error"
        self._record_emergency(asset, outcome.status, now_ms, order, outcome.detail)
        return outcome.status


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Worker Darwinex Zero (MetaTrader 5, Windows ; dry-run par défaut).")
    parser.add_argument("--address", default=os.environ.get("NOTHIERLY_ZMQ_ADDRESS"), help="endpoint ZMQ du publisher")
    parser.add_argument("--db", default="darwinex_worker.sqlite3")
    parser.add_argument("--account-login", type=int, default=int(os.environ.get("DWZ_ACCOUNT_LOGIN", "0")),
                        help="numéro du SEUL compte que ce worker a le droit de piloter (ou DWZ_ACCOUNT_LOGIN)")
    parser.add_argument("--live", action="store_true", help="envoie réellement les ordres ; sinon dry-run")
    parser.add_argument("--allow-real-account", action="store_true", help="autorise un compte de type réel")
    parser.add_argument("--symbol-suffix", default="", help="suffixe des symboles chez le courtier (ex. .r)")
    parser.add_argument("--magic", type=int, default=24100601, help="identifiant de stratégie MT5 des ordres")
    parser.add_argument("--max-spread-bps", type=Decimal, default=Decimal("20"))
    parser.add_argument("--default-stop-loss-bps", type=Decimal, default=Decimal("100"),
                        help="stop-loss posé par le worker à cet écart du prix d'entrée si le signal n'en porte pas (100 = 1 %% ; 0 = refuser)")
    parser.add_argument("--clear-halt", action="store_true", help="lève les verrous (drawdown et homme mort) puis quitte")
    parser.add_argument("--dead-man-timeout", type=float, default=300.0, metavar="SECONDES",
                        help="ferme toutes les positions puis se verrouille après ce délai sans signal valide (0 = désactivé)")
    parser.add_argument("--allow-plaintext-remote", action="store_true", help="autorise un endpoint non local SANS chiffrement CURVE")
    parser.add_argument("--log-dir", default=None, help="dossier d'un fichier de journal tournant (10 Mo x 5) en plus de stdout ; défaut : NOTHIERLY_LOG_DIR")
    args = parser.parse_args(argv)
    if args.clear_halt:  # ne touche que la base locale : ni terminal MT5, ni compte, ni secret du bus
        configure_logging("darwinex-worker", log_dir=args.log_dir)
        clear_execution_halt(ExecutionState(args.db))
        _log.info("dwx_halt_cleared")
        return 0
    if args.dead_man_timeout < 0:
        parser.error("--dead-man-timeout doit être >= 0")
    if (timeout_problem := dead_man_timeout_problem(args.dead_man_timeout)) is not None:
        parser.error(timeout_problem)
    if args.default_stop_loss_bps < 0:
        parser.error("--default-stop-loss-bps doit être >= 0")
    if not args.address:
        parser.error("--address ou NOTHIERLY_ZMQ_ADDRESS requis")
    if args.account_login <= 0:
        parser.error("--account-login (ou DWZ_ACCOUNT_LOGIN) requis : le terminal trade le compte connecté, il faut épingler le bon")
    try:
        curve = CurveClientConfig.from_env()
    except CurveConfigError as exc:
        parser.error(str(exc))
    refusal = transport_refusal(args.address, curve_enabled=curve is not None, allow_plaintext_remote=args.allow_plaintext_remote)
    if refusal:
        parser.error(refusal)
    if args.live and args.allow_real_account and args.dead_man_timeout == 0:
        parser.error("--live sur un compte réel exige l'homme mort : --dead-man-timeout doit être > 0")
    try:
        alerter = WebhookAlerter.from_env(source="darwinex-worker")
    except AlertConfigError as exc:
        parser.error(str(exc))
    configure_logging("darwinex-worker", log_dir=args.log_dir)
    login_env = os.environ.get("MT5_LOGIN")
    try:
        client = MetaTrader5Client(
            expected_login=args.account_login, magic=args.magic, path=os.environ.get("MT5_PATH"),
            login=int(login_env) if login_env else None, password=os.environ.get("MT5_PASSWORD"), server=os.environ.get("MT5_SERVER"),
            allow_real_account=args.allow_real_account,
        )
        info = client.account_check()
    except Mt5Error as exc:  # mauvais OS, paquet absent, terminal injoignable, mauvais compte : un message, pas une trace
        _log.error("dwx_cannot_start %s", exc)
        print(f"Erreur : {exc}", file=sys.stderr)
        return 2
    _log.info("dwx_account login=%s mode=%s margin=%s currency=%s server=%s", info.login, info.trade_mode, info.margin_mode, info.currency, info.server)
    config = DarwinexWorkerConfig(
        address=args.address, secret_key=load_secret_from_env(), db_path=args.db, dry_run=not args.live,
        symbol_suffix=args.symbol_suffix, max_spread_bps=args.max_spread_bps,
        default_stop_loss_bps=args.default_stop_loss_bps or None,
        dead_man_timeout_s=args.dead_man_timeout or None, curve=curve, alerter=alerter,
    )
    worker = DarwinexWorker(config, client)
    _log.info("dwx_worker_start mode=%s", "LIVE" if args.live else "DRY-RUN")
    try:
        # Boucle Selector explicite : sous Windows, la boucle Proactor par défaut est incompatible avec zmq.asyncio.
        asyncio.run(worker.run(), loop_factory=asyncio.SelectorEventLoop)
    except KeyboardInterrupt:
        _log.info("dwx_worker_stopped")
    finally:
        client.close()
        if alerter is not None:
            alerter.close(2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
