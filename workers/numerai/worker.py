"""Worker Numerai Signals : écoute les ``PlatformSignal`` signés, accumule un score par ticker, soumet un CSV.

Flux : bus ZMQ -> ``open_envelope`` (signature, plateforme, âge, rejeu) -> conversion en score ->
SQLite -> à l'heure prévue, ``generate_submission_csv`` -> client Numerai.

Écarts volontaires avec le brief, à connaître :

1. **Le score dépend de la direction.** Le brief demandait ``signal = confidence_score``. Or la confiance
   est une magnitude : un SHORT à 0.9 et un LONG à 0.9 donneraient le même 0.9, et Numerai lirait le
   short comme une forte conviction haussière. Le score soumis est donc ``0.5 + sens * confiance / 2``
   (sens = +1 LONG, -1 SHORT, 0 FLAT), borné à [0.01, 0.99] : LONG 0.9 -> 0.95, SHORT 0.9 -> 0.05.
2. **Le format de ``numerai_ticker`` n'est pas supposé.** La doc Numerai n'en donne aucun exemple ;
   ``"AAPL US"`` (symbole + pays) est le format par défaut mais n'est pas confirmé. Quand un client réel
   fournit l'univers du jour, tout ticker absent de cet univers est écarté avant l'envoi. Seuls les
   symboles ``EQUITY_US_*`` sont traduits automatiquement ; les autres pays (codes différents) et les
   symboles à point (``BRK.B``, non représentables dans ``PlatformSignal``) passent par ``symbol_overrides``.
3. **Minimum de 100 tickers.** Numerai exige au moins 100 tickers valides pour une soumission live. Sous
   ce seuil le worker ne soumet rien (statut ``insufficient``) et réessaie plus tard : un moteur qui ne
   publie que quelques actions ne peut pas alimenter Numerai.
4. **Calendrier.** Les rounds Signals ont lieu du mardi au samedi, avec une fenêtre d'environ une heure
   en semaine et un horaire d'ouverture variable. Le défaut est donc mardi-vendredi à 13h15 UTC, pas
   lundi-vendredi ; il est configurable (``SubmissionSchedule``) et doit être recoupé avec le round
   réellement ouvert. Une soumission tardive est acceptée par Numerai mais ne permet pas de staker.
5. **Rien n'est envoyé par défaut.** Sans client explicite, ``DryRunClient`` journalise sans soumettre.

Anti-rejeu : un message authentique rejoué ne doit pas écraser le score du jour. Le worker garde donc un
âge maximal, un ``ReplayGuard`` rechargé depuis SQLite au démarrage, et n'accepte un score que s'il est
plus récent que celui en place.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import zmq
import zmq.asyncio

from contracts.curve_transport import CurveClientConfig, CurveConfigError, apply_client, is_local_endpoint, transport_refusal
from contracts.platform_signal import SignalDirection, TargetVenue
from contracts.secure_signal_bus import DEFAULT_TOPIC, ReplayGuard, load_secret_from_env, open_envelope, verify_signature
from workers.numerai.client import DryRunClient, NumeraiClient, NumerapiClient
from workers.numerai.store import SignalStore

_log = logging.getLogger(__name__)

SIGNAL_MIN: Final[float] = 0.01
SIGNAL_MAX: Final[float] = 0.99
MIN_TICKERS: Final[int] = 100
DEFAULT_MAX_AGE_MS: Final[int] = 24 * 3600 * 1000
_DONE_STATUSES: Final[frozenset[str]] = frozenset({"submitted", "dry_run"})


class SubmissionNotReady(Exception):
    """Aucun fichier valide ne peut être produit pour l'instant."""


class InsufficientSignals(SubmissionNotReady):
    """Moins de ``min_tickers`` scores valides et récents."""


def signal_value(
    direction: SignalDirection, confidence: float, low: float = SIGNAL_MIN, high: float = SIGNAL_MAX
) -> float:
    """Score Numerai dans [low, high] : 0.5 est neutre, LONG au-dessus, SHORT en dessous."""
    sign = {SignalDirection.LONG: 1.0, SignalDirection.SHORT: -1.0, SignalDirection.FLAT: 0.0}[direction]
    return min(max(0.5 + sign * confidence / 2.0, low), high)


def to_numerai_ticker(
    symbol: str, *, ticker_format: str = "{symbol} {country}", overrides: Mapping[str, str] | None = None
) -> str | None:
    """``EQUITY_US_AAPL`` -> ``"AAPL US"``. ``None`` si le symbole n'est pas traduisible sans table explicite."""
    if overrides and symbol in overrides:
        return overrides[symbol]
    parts = symbol.split("_")
    if len(parts) != 3 or parts[0] != "EQUITY" or parts[1] != "US":
        return None
    return ticker_format.format(symbol=parts[2], country=parts[1])


def _as_utc(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now doit être un datetime avec fuseau horaire (UTC).")
    return now.astimezone(timezone.utc)


@dataclass(frozen=True)
class SubmissionSchedule:
    """Jours (lundi = 0) et heure UTC à partir desquels la soumission du jour est due.

    Une fois l'heure passée, la soumission reste due jusqu'à la fin de la journée UTC : un worker
    redémarré à 14h soumet tout de suite au lieu d'attendre le lendemain.
    """

    hour_utc: int = 13
    minute_utc: int = 15
    weekdays: frozenset[int] = frozenset({1, 2, 3, 4})

    def __post_init__(self) -> None:
        if not (0 <= self.hour_utc <= 23 and 0 <= self.minute_utc <= 59):
            raise ValueError("heure/minute UTC invalides.")
        if not self.weekdays or not self.weekdays <= frozenset(range(7)):
            raise ValueError("weekdays doit être un sous-ensemble non vide de 0..6.")

    def is_due(self, now: datetime) -> bool:
        now = _as_utc(now)
        return now.weekday() in self.weekdays and (now.hour, now.minute) >= (self.hour_utc, self.minute_utc)

    @staticmethod
    def day_key(now: datetime) -> str:
        return _as_utc(now).strftime("%Y%m%d")


@dataclass(frozen=True)
class NumeraiWorkerConfig:
    address: str
    secret_key: str | bytes
    db_path: str | Path = "numerai_worker.sqlite3"
    output_dir: Path = Path("numerai_submissions")
    topic: bytes = DEFAULT_TOPIC
    schedule: SubmissionSchedule = SubmissionSchedule()
    min_tickers: int = MIN_TICKERS
    max_age_ms: int = DEFAULT_MAX_AGE_MS
    ticker_format: str = "{symbol} {country}"
    symbol_overrides: Mapping[str, str] = field(default_factory=dict)
    retry_after_s: float = 300.0
    max_attempts_per_day: int = 5
    poll_ms: int = 500
    schedule_check_interval_s: float = 15.0
    curve: CurveClientConfig | None = None  # chiffrement et authentification du canal ; ``None`` = en clair


class NumeraiWorker:
    def __init__(
        self, config: NumeraiWorkerConfig, client: NumeraiClient | None = None, store: SignalStore | None = None
    ) -> None:
        verify_signature({}, config.secret_key)  # lève WeakSecretError dès le démarrage si la clé est faible
        self._cfg = config
        self._client: NumeraiClient = client if client is not None else DryRunClient()
        self._store = store if store is not None else SignalStore(config.db_path)
        self._replay_guard = ReplayGuard(capacity=100_000)
        now_ms = int(time.time() * 1000)
        for signal_id in self._store.recent_signal_ids(now_ms - config.max_age_ms):
            self._replay_guard.check_and_add(uuid.UUID(signal_id))  # un redémarrage n'ouvre pas de fenêtre de rejeu
        self._store.prune_seen_before(now_ms - 2 * config.max_age_ms)

    @property
    def store(self) -> SignalStore:
        return self._store

    # --- réception --------------------------------------------------------------------------

    def handle_frames(self, frames: Sequence[bytes], *, now_ms: int | None = None) -> str:
        """Traite un message multipart ``[topic, enveloppe]``. Renvoie un code : ``stored`` ou la raison du rejet."""
        if len(frames) != 2 or frames[0] != self._cfg.topic:
            return "bad_frames"
        now = int(time.time() * 1000) if now_ms is None else now_ms
        verdict = open_envelope(
            frames[1],
            self._cfg.secret_key,
            venue=TargetVenue.NUMERAI_SIGNALS,
            max_age_ms=self._cfg.max_age_ms,
            replay_guard=self._replay_guard,
            now_ms=now,
        )
        signal = verdict.signal
        if signal is None:
            return verdict.reject_reason or "rejected"
        self._store.remember_signal(str(signal.signal_id), now)
        if signal.confidence_score is None:  # exclu par le contrat pour Numerai ; garde-fou pour mypy et défense en profondeur
            return "no_confidence"
        ticker = to_numerai_ticker(
            signal.symbol, ticker_format=self._cfg.ticker_format, overrides=self._cfg.symbol_overrides
        )
        if ticker is None:
            _log.warning("numerai_unmappable_symbol symbol=%s", signal.symbol)
            return "unmappable_symbol"
        value = signal_value(signal.direction, signal.confidence_score)
        stored = self._store.upsert_score(ticker, value, str(signal.signal_id), signal.timestamp_ms)
        return "stored" if stored else "ignored_older"

    # --- fichier de soumission --------------------------------------------------------------

    def _generate(self, now: datetime) -> tuple[Path, int, int]:
        now_ms = int(now.timestamp() * 1000)
        rows = self._store.scores_since(now_ms - self._cfg.max_age_ms)
        universe = self._client.ticker_universe()
        if universe is not None:
            unknown = [r.numerai_ticker for r in rows if r.numerai_ticker not in universe]
            if unknown:
                _log.warning("numerai_tickers_outside_universe count=%d sample=%s", len(unknown), unknown[:5])
            rows = [r for r in rows if r.numerai_ticker in universe]
        if len(rows) < self._cfg.min_tickers:
            raise InsufficientSignals(f"{len(rows)} ticker(s) valide(s), {self._cfg.min_tickers} requis par Numerai.")
        self._cfg.output_dir.mkdir(parents=True, exist_ok=True)
        path = self._cfg.output_dir / f"submission_{now:%Y%m%d}.csv"
        tmp = path.with_suffix(".csv.tmp")
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["numerai_ticker", "signal"])
            for row in rows:
                writer.writerow([row.numerai_ticker, f"{row.signal:.6f}"])
        os.replace(tmp, path)  # jamais de fichier à moitié écrit sous le nom final
        return path, len(rows), now_ms

    def generate_submission_csv(self, now: datetime | None = None) -> Path:
        """Écrit ``submission_YYYYMMDD.csv`` (``numerai_ticker,signal``). Lève ``InsufficientSignals`` si < 100 tickers."""
        return self._generate(_as_utc(now))[0]

    def submit_if_due(self, now: datetime | None = None) -> str:
        """Soumet une fois par jour dû. Codes : ``not_due``, ``already_submitted``, ``gave_up``, ``waiting_retry``,
        ``insufficient``, ``failed``, ``submitted`` (ou ``dry_run`` avec le client sans réseau)."""
        now = _as_utc(now)
        now_ms = int(now.timestamp() * 1000)
        schedule = self._cfg.schedule
        if not schedule.is_due(now):
            return "not_due"
        day = schedule.day_key(now)
        record = self._store.submission(day)
        attempts = record.attempts if record else 0
        if record and record.status in _DONE_STATUSES:
            return "already_submitted"
        if attempts >= self._cfg.max_attempts_per_day:
            return "gave_up"
        if record and now_ms - record.last_attempt_ms < self._cfg.retry_after_s * 1000:
            return "waiting_retry"
        try:
            path, n_rows, snapshot_ms = self._generate(now)
        except InsufficientSignals as exc:
            self._store.record_submission(day, "insufficient", attempts, now_ms, 0, str(exc))
            _log.warning("numerai_submission_insufficient %s", exc)
            return "insufficient"
        except Exception as exc:  # noqa: BLE001 -- l'univers vient du réseau ; tout échec est réessayé
            self._store.record_submission(day, "failed", attempts + 1, now_ms, 0, f"generate: {exc}")
            _log.exception("numerai_generate_failed")
            return "failed"
        try:
            submission_id = self._client.submit(path)
        except Exception as exc:  # noqa: BLE001 -- l'API Numerai peut lever n'importe quoi
            self._store.record_submission(day, "failed", attempts + 1, now_ms, n_rows, f"submit: {exc}")
            _log.exception("numerai_submit_failed")
            return "failed"
        status = "dry_run" if submission_id.startswith("dry-run:") else "submitted"
        self._store.record_submission(day, status, attempts + 1, now_ms, n_rows, submission_id)
        self._store.delete_scores_up_to(snapshot_ms)
        _log.info("numerai_submission_done status=%s rows=%d id=%s", status, n_rows, submission_id)
        return status

    # --- boucle -----------------------------------------------------------------------------

    async def run(self, stop: asyncio.Event | None = None) -> None:
        """Écoute le bus sans bloquer et déclenche la soumission quand elle est due, jusqu'à ``stop``.

        Sous Windows, à lancer sur une boucle ``asyncio.SelectorEventLoop`` (``zmq.asyncio`` ne
        fonctionne pas avec la boucle Proactor par défaut, sauf si ``tornado`` est installé).
        """
        stop = stop if stop is not None else asyncio.Event()
        context = zmq.asyncio.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SUBSCRIBE, self._cfg.topic)
        if self._cfg.curve is not None:
            apply_client(socket, self._cfg.curve)
        elif not is_local_endpoint(self._cfg.address):
            _log.warning("numerai_plaintext_remote address=%s : canal NON chiffré", self._cfg.address)
        socket.connect(self._cfg.address)
        next_check = 0.0
        try:
            while not stop.is_set():
                if await socket.poll(self._cfg.poll_ms):
                    frames = await socket.recv_multipart()
                    try:
                        outcome = self.handle_frames(frames)
                        _log.debug("numerai_frame outcome=%s", outcome)
                    except Exception:  # noqa: BLE001 -- un message ne doit jamais arrêter le worker
                        _log.exception("numerai_handle_frames_failed")
                if time.monotonic() >= next_check:
                    next_check = time.monotonic() + self._cfg.schedule_check_interval_s
                    try:
                        await asyncio.to_thread(self.submit_if_due)  # appel réseau bloquant hors de la boucle
                    except Exception:  # noqa: BLE001
                        _log.exception("numerai_submit_if_due_failed")
        finally:
            socket.close(linger=0)
            context.term()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Worker Numerai Signals (dry-run par défaut).")
    parser.add_argument("--address", default=os.environ.get("NOTHIERLY_ZMQ_ADDRESS"), help="endpoint ZMQ du publisher")
    parser.add_argument("--db", default="numerai_worker.sqlite3")
    parser.add_argument("--out", default="numerai_submissions")
    parser.add_argument("--live", action="store_true", help="soumet réellement (NUMERAI_* requis) ; sinon dry-run")
    parser.add_argument("--allow-plaintext-remote", action="store_true", help="autorise un endpoint non local SANS chiffrement CURVE")
    args = parser.parse_args(argv)
    if not args.address:
        parser.error("--address ou NOTHIERLY_ZMQ_ADDRESS requis")
    try:
        curve = CurveClientConfig.from_env()
    except CurveConfigError as exc:
        parser.error(str(exc))
    refusal = transport_refusal(args.address, curve_enabled=curve is not None, allow_plaintext_remote=args.allow_plaintext_remote)
    if refusal:
        parser.error(refusal)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    client: NumeraiClient = NumerapiClient.from_env() if args.live else DryRunClient()
    config = NumeraiWorkerConfig(
        address=args.address, secret_key=load_secret_from_env(), db_path=args.db, output_dir=Path(args.out), curve=curve
    )
    try:
        # Boucle Selector explicite : sous Windows, la boucle Proactor par défaut est incompatible avec zmq.asyncio.
        asyncio.run(NumeraiWorker(config, client=client).run(), loop_factory=asyncio.SelectorEventLoop)
    except KeyboardInterrupt:
        _log.info("numerai_worker_stopped")


if __name__ == "__main__":
    main()
