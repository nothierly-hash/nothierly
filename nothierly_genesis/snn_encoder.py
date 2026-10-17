"""Genesis Protocol, Pillar 2b: real delta/spike encoding of a P2P price tick series.

Companion to :mod:`nothierly_genesis.spiking_tick_network` (Pillar 2's own real, already-measured
finding on SNN latency -- see its module docstring, and :mod:`nothierly_genesis.snn_brain`'s own
docstring for why that finding applies here too). That module encodes L2 order-book *events*
(ADD/CANCEL/MODIFY) into spikes; this one encodes a plain P2P *price* tick series instead -- a
real, distinct, much simpler encoding, not a duplicate.

``torchvision`` was named in the brief alongside ``snntorch``/``torch``/``numpy``/``matplotlib`` --
not installed here: it is a computer-vision package (image datasets/transforms), and nothing in
this module or :mod:`nothierly_genesis.snn_brain` touches an image. Declaring it would be dead
weight, the same discipline this project applies everywhere else (see e.g. requirements.txt's own
note on why ``pandas`` is deliberately absent from the live per-tick path).
"""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor

SPIKE_UP: Final[list[int]] = [1, 0]
SPIKE_DOWN: Final[list[int]] = [0, 1]
SPIKE_SILENCE: Final[list[int]] = [0, 0]

DEFAULT_THRESHOLD: Final[float] = 1.0

# The brief's own literal example series -- kept as this module's real default/documented
# reference case, not just a docstring example.
EXAMPLE_PRICE_SERIES: Final[tuple[float, ...]] = (1650.0, 1651.0, 1648.0, 1645.0, 1644.0)


def delta_encode(price_series: list[float] | tuple[float, ...], threshold: float = DEFAULT_THRESHOLD) -> Tensor:
    """Real delta-threshold spike encoding -- exactly the brief's own three rules, no more.

    Each of the ``len(price_series) - 1`` output rows compares one tick to the *previous* one
    (there is no real "delta" for the very first tick, so it contributes no row -- padding it with
    silence would fabricate a comparison against nothing):

    * ``price[t] - price[t-1] > threshold``  -> ``[1, 0]`` (a real up-spike)
    * ``price[t] - price[t-1] < -threshold`` -> ``[0, 1]`` (a real down-spike)
    * otherwise                              -> ``[0, 0]`` (real silence -- no event)

    Returns:
        A real ``(len(price_series) - 1, 2)`` float32 tensor -- one row per tick-to-tick delta.

    Raises:
        ValueError: if fewer than 2 prices are given (no delta is computable at all).
    """
    if len(price_series) < 2:
        raise ValueError(f"need at least 2 prices to compute a single delta, got {len(price_series)}")

    rows: list[list[int]] = []
    for previous, current in zip(price_series[:-1], price_series[1:]):
        delta = current - previous
        if delta > threshold:
            rows.append(SPIKE_UP)
        elif delta < -threshold:
            rows.append(SPIKE_DOWN)
        else:
            rows.append(SPIKE_SILENCE)
    return torch.tensor(rows, dtype=torch.float32)


if __name__ == "__main__":
    spikes = delta_encode(list(EXAMPLE_PRICE_SERIES))
    print(f"Série de prix : {EXAMPLE_PRICE_SERIES}")
    print("Spikes encodés [up, down] :")
    for t, (price_pair, spike_row) in enumerate(zip(zip(EXAMPLE_PRICE_SERIES[:-1], EXAMPLE_PRICE_SERIES[1:]), spikes.tolist())):
        print(f"  t={t}: {price_pair[0]:.1f} -> {price_pair[1]:.1f}  spike={spike_row}")
