"""Reconstitue un historique « VPIN + barres de prix » à partir des trades agrégés publics de Binance (futures USDⓈ-M, ``aggTrades``).

    python scripts/backfill_vpin.py --symbol BTCUSDT --files data/BTCUSDT-aggTrades-2026-10-01.zip data/BTCUSDT-aggTrades-2026-10-02.zip \
        --out-db data/vpin_backfill.sqlite3 --out-bars data/bars_BTCUSDT.npz
    python scripts/backfill_vpin.py --symbol BTCUSDT --start 2026-10-01 --end 2026-10-03 --list              # URLs + tailles, RIEN n'est téléchargé
    python scripts/backfill_vpin.py --symbol BTCUSDT --start 2026-10-01 --end 2026-10-03 --download-dir data --yes-download

**Même méthode que le tracker en direct** (``workers/market_data/vpin_tracker.py``) : la classe ``VolumeBucketVpin`` elle-même, mêmes seaux (``--bucket-size``,
100 000 USDT par défaut) et même fenêtre (``--window``, 50), côté du PRENEUR = ``not is_buyer_maker`` (comme ``m == false`` dans le flux en direct), identifiants de
trade déjà vus ignorés, fenêtre remise à zéro après un trou de plus de ``--max-gap-s`` (30 s). Les seaux fermés sont écrits dans une archive SQLite de MÊME
format que celle du tracker (``VpinArchive``), à part de l'archive en direct : l'historique sert à entraîner, les données du Shadow Run (postérieures) à tester.

**Barres de prix** (``--out-bars``) : une barre de 60 s par minute où il y a eu des trades (prix du dernier trade, notionnel acheteur et vendeur). Une minute sans
trade n'existe pas : un trou de collecte reste visible, il n'est jamais comblé (voir ``src/rl/historical_features.py``).

**Ce que ce n'est pas** : ni le spread, ni le carnet, ni la macro, ni le sentiment (aucun historique de trades ne les contient). Un modèle entraîné là-dessus
n'a pas le contrat de la politique de production : voir ``feature_set`` dans ses métadonnées.

**Téléchargement.** Les fichiers viennent de ``data.binance.vision`` (publics, sans clé). Ils sont volumineux (de quelques dizaines à plusieurs centaines de Mo
par jour pour BTCUSDT) : ``--list`` donne les tailles d'abord, ``--download-dir`` exige ``--yes-download`` et vérifie chaque fichier contre son ``.CHECKSUM``
SHA-256 publié. Format des fichiers non vérifié contre le vrai site par ce code (voir ``parse_row`` : colonnes de la documentation, en-tête facultatif).

Mono-passe et mémoire constante : les trades ne sont jamais tous chargés ; relancer écrase les deux sorties.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
import time
import urllib.request
import zipfile
from collections.abc import Iterable, Iterator, Sequence
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Final

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workers.market_data.vpin import VolumeBucketVpin  # noqa: E402
from workers.market_data.vpin_archive import BucketRecord, VpinArchive  # noqa: E402

BASE_URL: Final[str] = "https://data.binance.vision/data/futures/um/daily/aggTrades"
DEFAULT_BUCKET_SIZE: Final[Decimal] = Decimal(100_000)
DEFAULT_WINDOW: Final[int] = 50
DEFAULT_MAX_GAP_S: Final[float] = 30.0
BAR_S: Final[float] = 60.0
_COLUMNS: Final[int] = 7  # agg_trade_id, price, quantity, first_trade_id, last_trade_id, transact_time, is_buyer_maker


def day_url(symbol: str, day: date) -> str:
    return f"{BASE_URL}/{symbol}/{symbol}-aggTrades-{day.isoformat()}.zip"


def days(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("--end doit être >= --start")
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def parse_row(row: Sequence[str]) -> tuple[int, Decimal, Decimal, int, bool] | None:
    """``(id, prix, quantité, temps_ms, preneur_a_acheté)`` ; ``None`` pour la ligne d'en-tête ; lève ``ValueError`` pour une ligne malformée."""
    if len(row) < _COLUMNS:
        raise ValueError(f"ligne à {len(row)} colonnes (attendu {_COLUMNS})")
    if not row[0].strip().lstrip("-").isdigit():
        return None  # en-tête (les fichiers récents en ont un, les anciens non)
    maker = row[6].strip().lower()
    if maker not in ("true", "false"):
        raise ValueError(f"is_buyer_maker illisible : {row[6][:10]!r}")
    try:
        return int(row[0]), Decimal(row[1]), Decimal(row[2]), int(row[5]), maker == "false"  # acheteur = teneur de marché -> le preneur a VENDU
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("nombre illisible") from exc


def iter_zip_rows(path: Path) -> Iterator[Sequence[str]]:
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"{path.name} : un seul CSV attendu, {len(names)} trouvés")
        with archive.open(names[0]) as raw:
            yield from csv.reader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))


class StreamingBars:
    """Agrège des trades en barres de ``BAR_S`` secondes au fil de l'eau (même résultat que ``build_time_bars`` sur l'ensemble)."""

    def __init__(self, bar_s: float = BAR_S) -> None:
        self._bar_ms = int(round(bar_s * 1000))
        self._index: int | None = None
        self._close = 0.0
        self._buy = 0.0
        self._sell = 0.0
        self._ends: list[float] = []
        self._closes: list[float] = []
        self._buys: list[float] = []
        self._sells: list[float] = []

    def add(self, time_ms: int, price: Decimal, qty: Decimal, is_buy: bool) -> None:
        index = time_ms // self._bar_ms
        if self._index is not None and index != self._index:
            self._flush()
        self._index = index
        self._close = float(price)
        notional = float(price * qty)
        if is_buy:
            self._buy += notional
        else:
            self._sell += notional

    def _flush(self) -> None:
        assert self._index is not None
        self._ends.append((self._index + 1) * self._bar_ms / 1000.0)
        self._closes.append(self._close)
        self._buys.append(self._buy)
        self._sells.append(self._sell)
        self._buy = self._sell = 0.0

    def result(self) -> dict[str, np.ndarray]:
        if self._index is not None:
            self._flush()
            self._index = None
        return {
            "bar_end_s": np.array(self._ends, dtype=np.float64), "close": np.array(self._closes, dtype=np.float64),
            "buy_notional": np.array(self._buys, dtype=np.float64), "sell_notional": np.array(self._sells, dtype=np.float64),
        }


def backfill(
    rows: Iterable[Sequence[str]],
    *,
    symbol: str,
    archive: VpinArchive,
    bucket_size: Decimal = DEFAULT_BUCKET_SIZE,
    window: int = DEFAULT_WINDOW,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
    bars: StreamingBars | None = None,
    state: dict[str, int] | None = None,
) -> dict[str, int]:
    """Passe les lignes de trades (dans l'ordre) dans le calcul VPIN et les barres ; renvoie des compteurs. ``state`` permet d'enchaîner plusieurs fichiers."""
    archive.start()  # le thread d'ecriture doit tourner : sans lui la file se remplit et la passe attend indefiniment (ou perdait les seaux avant l'attente)
    calc = VolumeBucketVpin(bucket_size, window)
    counters = state if state is not None else {}
    for key in ("trades", "skipped_ids", "bad_rows", "gaps", "buckets"):
        counters.setdefault(key, 0)
    last_id = counters.get("last_id", -1)
    last_time = counters.get("last_time_ms", -1)
    for row in rows:
        try:
            parsed = parse_row(row)
        except ValueError:
            counters["bad_rows"] += 1
            continue
        if parsed is None:
            continue
        trade_id, price, qty, time_ms, is_buy = parsed
        if trade_id <= last_id:
            counters["skipped_ids"] += 1
            continue
        if last_time >= 0 and (time_ms - last_time) / 1000.0 > max_gap_s:
            calc.reset()
            counters["gaps"] += 1
        last_id, last_time = trade_id, time_ms
        if price <= 0 or qty <= 0:
            counters["bad_rows"] += 1
            continue
        counters["trades"] += 1
        if bars is not None:
            bars.add(time_ms, price, qty, is_buy)
        for reading in calc.add_trade(price, qty, is_buy, time_ms):
            counters["buckets"] += 1
            record = BucketRecord(symbol, reading.closed_at_ms, reading.vpin, reading.buy_volume, reading.sell_volume, bucket_size, window)
            # Contrairement au tracker en direct (qui ne doit jamais bloquer sa boucle et abandonne si la file est pleine), l'historique ATTEND : sans cette attente,
            # une passe rapide perdrait presque tous ses seaux (file de 50 000) sans le dire.
            while not archive.submit(record):
                archive.dropped -= 1  # ce refus n'est pas une perte : on réessaie
                time.sleep(0.001)
    counters["last_id"], counters["last_time_ms"] = last_id, last_time
    return counters


def run(
    files: Sequence[Path], *, symbol: str, out_db: Path, out_bars: Path, bucket_size: Decimal, window: int, max_gap_s: float,
) -> dict[str, int]:
    """Traite les fichiers dans l'ordre comme UN SEUL flux continu (le seau et la fenêtre en cours passent d'un jour au suivant)."""
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    out_db.parent.mkdir(parents=True, exist_ok=True)
    out_bars.parent.mkdir(parents=True, exist_ok=True)
    if out_db.exists():
        out_db.unlink()  # relancer écrase (la doc le dit) : jamais de mélange avec une ancienne passe
    archive = VpinArchive(out_db)
    bars = StreamingBars()

    def all_rows() -> Iterator[Sequence[str]]:
        for path in files:
            yield from iter_zip_rows(path)

    counters = backfill(all_rows(), symbol=symbol, archive=archive, bucket_size=bucket_size, window=window, max_gap_s=max_gap_s, bars=bars)
    lost = archive.close()
    if lost or archive.dropped:
        raise RuntimeError(f"{lost + archive.dropped} seaux n'ont pas pu être écrits dans l'archive : sortie incomplète, relancer.")
    result = bars.result()
    arrays: dict[str, Any] = {"symbol": np.array(symbol), "bar_s": np.array(BAR_S), **result}
    np.savez_compressed(out_bars, **arrays)
    counters["bars"] = len(result["close"])
    return counters


# -- Téléchargement (explicite) ------------------------------------------------------------------


def remote_size(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length is not None else None
    except OSError:
        return None


def download_verified(url: str, dest: Path) -> None:
    """Télécharge ``url`` vers ``dest`` puis vérifie son SHA-256 contre ``url + '.CHECKSUM'`` ; supprime le fichier si la somme ne correspond pas."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response, dest.open("wb") as handle:
        digest = hashlib.sha256()
        while chunk := response.read(1 << 20):
            handle.write(chunk)
            digest.update(chunk)
    with urllib.request.urlopen(url + ".CHECKSUM", timeout=30) as response:
        expected = response.read().decode("utf-8", errors="replace").split()[0].lower()
    if digest.hexdigest().lower() != expected:
        dest.unlink(missing_ok=True)
        raise ValueError(f"somme de contrôle invalide pour {dest.name} : fichier supprimé")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VPIN + barres de prix historiques à partir des aggTrades publics de Binance (même méthode que le tracker).")
    parser.add_argument("--symbol", required=True, help="ex. BTCUSDT")
    parser.add_argument("--files", nargs="+", type=Path, default=None, help="fichiers .zip d'aggTrades, dans l'ordre chronologique")
    parser.add_argument("--start", type=date.fromisoformat, default=None)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument("--list", action="store_true", help="affiche les URLs et leurs tailles, ne télécharge rien")
    parser.add_argument("--download-dir", type=Path, default=None)
    parser.add_argument("--yes-download", action="store_true", help="confirme le téléchargement (volumineux)")
    parser.add_argument("--out-db", type=Path, default=Path("data/vpin_backfill.sqlite3"))
    parser.add_argument("--out-bars", type=Path, default=Path("data/bars.npz"))
    parser.add_argument("--bucket-size", type=Decimal, default=DEFAULT_BUCKET_SIZE)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--max-gap-s", type=float, default=DEFAULT_MAX_GAP_S)
    args = parser.parse_args(argv)
    symbol = args.symbol.upper()
    if args.bucket_size <= 0 or args.window < 1 or args.max_gap_s <= 0:
        parser.error("--bucket-size, --window et --max-gap-s doivent être positifs")

    files: list[Path] | None = args.files
    if args.list or args.download_dir is not None:
        if args.start is None or args.end is None:
            parser.error("--start et --end sont requis avec --list ou --download-dir")
        urls = [day_url(symbol, d) for d in days(args.start, args.end)]
        if args.list:
            total = 0
            for url in urls:
                size = remote_size(url)
                total += size or 0
                print(f"{url}  {'?' if size is None else f'{size / 1e6:.1f} Mo'}")
            print(f"total annoncé : {total / 1e6:.1f} Mo ({len(urls)} fichiers) -- rien n'a été téléchargé")
            return 0
        if not args.yes_download:
            print(f"{len(urls)} fichiers à télécharger depuis data.binance.vision : relancer avec --yes-download (utiliser --list pour les tailles).", file=sys.stderr)
            return 2
        files = []
        for url in urls:
            dest = args.download_dir / url.rsplit("/", 1)[1]
            if not dest.is_file():
                print(f"téléchargement {url}")
                download_verified(url, dest)
            files.append(dest)
    if not files:
        parser.error("indiquer --files, ou --start/--end avec --download-dir")

    counters = run(files, symbol=symbol, out_db=args.out_db, out_bars=args.out_bars, bucket_size=args.bucket_size, window=args.window, max_gap_s=args.max_gap_s)
    print(
        f"{symbol} : {counters['trades']} trades, {counters['buckets']} seaux, {counters['bars']} barres de {BAR_S:.0f} s ; "
        f"{counters['gaps']} trous > {args.max_gap_s:.0f} s, {counters['skipped_ids']} identifiants rejoués ignorés, {counters['bad_rows']} lignes illisibles."
    )
    print(f"archive VPIN : {args.out_db}\nbarres : {args.out_bars}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
