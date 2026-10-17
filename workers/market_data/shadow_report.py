"""Rapport en lecture seule sur l'archive VPIN du Shadow Run : combien de données, sur quelle période, avec quels trous, et quelle distribution.

    python -m workers.market_data.shadow_report [--db /data/vpin_history.sqlite3] [--toxic 0.7] [--hours 24]

Ouvre la base en LECTURE SEULE (mode=ro) : peut tourner pendant que le tracker écrit, ne modifie rien. Stdlib seule. Ne dit rien de la rentabilité : une
archive VPIN décrit la microstructure du marché, pas un résultat de trading. Elle sert à juger si l'on a assez de données propres pour entraîner.

``--hours N`` restreint le rapport aux N dernières heures (``--hours 24`` : le bilan des dernières 24 h) et ajoute le décompte de seaux par heure UTC, où une heure
vide trahit un trou de collecte. Un VPIN dont plus de 85 % des seaux dépassent le seuil toxique est signalé « saturé » : le seau est alors trop petit pour ce marché.
"""

from __future__ import annotations

import argparse
import sqlite3
import statistics
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from workers.market_data.vpin_archive import DEFAULT_FILENAME, TABLE, default_archive_path

_GAP_S = 600.0  # au-delà : trou de collecte (le tracker supprime sa clé après 60 s sans message ; l'entraînement rejette au-delà de 600 s)
_SATURATION_PCT = 85.0
_HOUR_MS = 3_600_000


def _utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _percentile(sorted_values: list[float], q: float) -> float:
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


def _share(sorted_values: list[float], level: float) -> float:
    return 100.0 * sum(v > level for v in sorted_values) / len(sorted_values)


def _sizes(values: list[float]) -> str:
    return ", ".join(f"{x:,.0f}" for x in sorted(values))


def report(db: Path, toxic: float, *, hours: float | None = None, now_ms: int | None = None) -> str:
    """``hours`` : ne regarde que les ``hours`` dernières heures (jusqu'à ``now_ms``, défaut l'heure actuelle) et ajoute le décompte de seaux par heure UTC."""
    uri = f"file:{db.as_posix()}?mode=ro"
    cutoff = 0 if hours is None else int(((time.time() * 1000) if now_ms is None else now_ms) - hours * _HOUR_MS)
    with sqlite3.connect(uri, uri=True, timeout=10) as conn:
        symbols = [row[0] for row in conn.execute(f"SELECT DISTINCT symbol FROM {TABLE} WHERE timestamp_ms >= ? ORDER BY symbol", (cutoff,))]
        window = "toute l'archive" if hours is None else f"les {hours:g} dernières heures"
        lines = [f"Archive : {db} ({db.stat().st_size / 1e6:.1f} Mo) -- fenêtre : {window}", f"Seuil « toxique » : VPIN > {toxic}", ""]
        if not symbols:
            return "\n".join(lines + ["Aucune ligne dans cette fenêtre : rien n'a encore été collecté (ou le collecteur est arrêté)."])
        for symbol in symbols:
            rows = conn.execute(
                f"SELECT timestamp_ms, vpin_value FROM {TABLE} WHERE symbol = ? AND timestamp_ms >= ? ORDER BY timestamp_ms", (symbol, cutoff)
            ).fetchall()
            sizes = [r[0] for r in conn.execute(f"SELECT DISTINCT bucket_size FROM {TABLE} WHERE symbol = ? AND timestamp_ms >= ?", (symbol, cutoff))]
            times = [r[0] for r in rows]
            values = sorted(r[1] for r in rows if r[1] is not None)
            span_h = (times[-1] - times[0]) / _HOUR_MS
            gaps = [(b - a) / 1000 for a, b in zip(times, times[1:])]
            long_gaps = [g for g in gaps if g > _GAP_S]
            lines.append(f"== {symbol}")
            lines.append(f"  seaux : {len(rows)} dont {len(values)} avec un VPIN (fenêtre pleine) ; période {_utc(times[0])} -> {_utc(times[-1])} UTC ({span_h:.1f} h)")
            if span_h > 0:
                lines.append(
                    f"  cadence : {len(rows) / span_h:.1f} seaux/h ; plus long intervalle entre deux seaux : {max(gaps, default=0.0):.0f} s ; trous > {_GAP_S:.0f} s : {len(long_gaps)}"
                )
            if values:
                lines.append(f"  VPIN : moyenne {statistics.fmean(values):.3f}, médiane {_percentile(values, 0.5):.3f}, p95 {_percentile(values, 0.95):.3f}, max {values[-1]:.3f}")
                lines.append(f"  part des seaux : > 0,5 : {_share(values, 0.5):.1f} % ; > {toxic} : {_share(values, toxic):.1f} % ; > 0,9 : {_share(values, 0.9):.1f} %")
                if len(values) >= 20 and _share(values, toxic) > _SATURATION_PCT:
                    lines.append(
                        f"  ⚠ VPIN SATURÉ : {_share(values, toxic):.0f} % des seaux au-dessus de {toxic}. Seau de {_sizes(sizes)} : probablement trop petit pour ce marché ; "
                        "l'indicateur ne sépare plus les régimes (voir scripts/backfill_vpin.py pour mesurer d'autres tailles)."
                    )
            else:
                lines.append("  aucun VPIN calculé (la fenêtre n'est pas encore pleine)")
            if sizes:
                lines.append(f"  taille de seau : {_sizes(sizes)}")
            if hours is not None:
                per_hour: dict[int, int] = {}
                for t in times:
                    per_hour[t // _HOUR_MS] = per_hour.get(t // _HOUR_MS, 0) + 1
                first, last = times[0] // _HOUR_MS, times[-1] // _HOUR_MS  # du premier au dernier seau : les heures vides au milieu sont les trous
                cells = [f"{datetime.fromtimestamp(h * 3600, tz=timezone.utc):%H}h:{per_hour.get(h, 0)}" for h in range(first, last + 1)]
                empty = sum(1 for h in range(first, last + 1) if per_hour.get(h, 0) == 0)
                late_h = (times[0] - cutoff) / _HOUR_MS
                if late_h > 1.0:
                    lines.append(f"  ⚠ premier seau de la fenêtre {late_h:.1f} h après son début : collecte arrêtée ou démarrée plus tard")
                lines.append(f"  seaux par heure UTC : {' '.join(cells)}" + (f"  ⚠ {empty} heure(s) sans aucun seau" if empty else ""))
            lines.append("")
    lines.append("Lecture : ce rapport mesure la COLLECTE (volume, continuité, régimes de microstructure). Il ne mesure aucune rentabilité.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rapport en lecture seule sur l'archive VPIN du Shadow Run.")
    parser.add_argument("--db", type=Path, default=None, help=f"défaut : <NOTHIERLY_DATA_DIR ou data/>/{DEFAULT_FILENAME}")
    parser.add_argument("--toxic", type=float, default=0.7, help="seuil de VPIN considéré toxique")
    parser.add_argument("--hours", type=float, default=None, help="ne regarde que les N dernières heures (ex. 24 pour un bilan de 24 h)")
    args = parser.parse_args(argv)
    if args.hours is not None and args.hours <= 0:
        parser.error("--hours doit être > 0")
    db = args.db or default_archive_path()
    if not db.is_file():
        print(f"[rapport] archive introuvable : {db}", file=sys.stderr)
        return 1
    print(report(db, args.toxic, hours=args.hours))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
