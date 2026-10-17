"""Rapport en lecture seule sur l'archive VPIN du Shadow Run : combien de données, sur quelle période, avec quels trous, et quelle distribution.

    python -m workers.market_data.shadow_report [--db /data/vpin_history.sqlite3] [--toxic 0.7]

Ouvre la base en LECTURE SEULE (mode=ro) : peut tourner pendant que le tracker écrit, ne modifie rien. Stdlib seule. Ne dit rien de la rentabilité : une
archive VPIN décrit la microstructure du marché, pas un résultat de trading. Elle sert à juger si l'on a assez de données propres pour entraîner.
"""

from __future__ import annotations

import argparse
import sqlite3
import statistics
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from workers.market_data.vpin_archive import DEFAULT_FILENAME, TABLE, default_archive_path

_GAP_S = 600.0  # au-delà : trou de collecte (le tracker supprime sa clé après 60 s sans message ; l'entraînement rejette au-delà de 600 s)


def _utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _percentile(sorted_values: list[float], q: float) -> float:
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


def _share(sorted_values: list[float], level: float) -> float:
    return 100.0 * sum(v > level for v in sorted_values) / len(sorted_values)


def report(db: Path, toxic: float) -> str:
    uri = f"file:{db.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=10) as conn:
        symbols = [row[0] for row in conn.execute(f"SELECT DISTINCT symbol FROM {TABLE} ORDER BY symbol")]
        lines = [f"Archive : {db} ({db.stat().st_size / 1e6:.1f} Mo)", f"Seuil « toxique » : VPIN > {toxic}", ""]
        if not symbols:
            return "\n".join(lines + ["Aucune ligne dans l'archive : rien n'a encore été collecté."])
        for symbol in symbols:
            rows = conn.execute(f"SELECT timestamp_ms, vpin_value FROM {TABLE} WHERE symbol = ? ORDER BY timestamp_ms", (symbol,)).fetchall()
            times = [r[0] for r in rows]
            values = sorted(r[1] for r in rows if r[1] is not None)
            span_h = (times[-1] - times[0]) / 3_600_000
            gaps = [(b - a) / 1000 for a, b in zip(times, times[1:])]
            long_gaps = [g for g in gaps if g > _GAP_S]
            lines.append(f"== {symbol}")
            lines.append(f"  seaux : {len(rows)} dont {len(values)} avec un VPIN (fenêtre pleine) ; période {_utc(times[0])} -> {_utc(times[-1])} UTC ({span_h:.1f} h)")
            if span_h > 0:
                lines.append(f"  cadence : {len(rows) / span_h:.1f} seaux/h ; plus long intervalle entre deux seaux : {max(gaps, default=0.0):.0f} s ; trous > {_GAP_S:.0f} s : {len(long_gaps)}")
            if values:
                lines.append(
                    f"  VPIN : moyenne {statistics.fmean(values):.3f}, médiane {_percentile(values, 0.5):.3f}, p95 {_percentile(values, 0.95):.3f}, max {values[-1]:.3f}"
                )
                lines.append(f"  part des seaux : > 0,5 : {_share(values, 0.5):.1f} % ; > {toxic} : {_share(values, toxic):.1f} % ; > 0,9 : {_share(values, 0.9):.1f} %")
            else:
                lines.append("  aucun VPIN calculé (la fenêtre n'est pas encore pleine)")
            lines.append("")
    lines.append("Lecture : ce rapport mesure la COLLECTE (volume, continuité, régimes de microstructure). Il ne mesure aucune rentabilité.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rapport en lecture seule sur l'archive VPIN du Shadow Run.")
    parser.add_argument("--db", type=Path, default=None, help=f"défaut : <NOTHIERLY_DATA_DIR ou data/>/{DEFAULT_FILENAME}")
    parser.add_argument("--toxic", type=float, default=0.7, help="seuil de VPIN considéré toxique")
    args = parser.parse_args(argv)
    db = args.db or default_archive_path()
    if not db.is_file():
        print(f"[rapport] archive introuvable : {db}", file=sys.stderr)
        return 1
    print(report(db, args.toxic))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
