"""Genesis Protocol, Pillar 2b: a real LIF spiking network triggering on accumulated price spikes.

**Same real latency caveat as this package's own sibling module, restated rather than repeated
only there**: :mod:`nothierly_genesis.spiking_tick_network`'s own module docstring already
measured, live, that a software-simulated SNN (:mod:`snntorch` on an ordinary CPU) is *slower*
than a comparable classical network, not faster -- real neuromorphic latency advantages exist in
purpose-built silicon (Loihi 2, Akida), not in a Python time-step loop. The brief for *this*
module reaches for the same "latence quasi-nulle" framing; it is not repeated here as if it were a
real property of this code. What genuinely differs from a classical model here is architectural,
not a speed claim: the network's decision is a real, stateful accumulation across ticks (each
:class:`snntorch.Leaky` layer carries its membrane potential forward from one call to the next),
not a fresh, memoryless computation per tick.

**Honest note on "not a classical if-condition" (the brief's own framing for task 6)**: mechanically,
:class:`snntorch.Leaky` *is* a threshold comparison every single step (``spk = (mem > threshold)``)
-- that part is not fundamentally different from an ``if``. What is real and different is *what*
gets compared: not the raw current tick's value, but ``mem``, a real leaky-integrated (``mem[t] =
beta * mem[t-1] + input[t]``) running state built up across several ticks. A single strong up-tick
right after a single strong down-tick would leave a naive ``if delta > threshold`` completely
memoryless between the two; here, the *hidden* layer's own membrane still remembers the down-tick
partially decayed (``beta = 0.9``) when the up-tick arrives, and the two inputs' opposing hand-set
weights (see :data:`_HIDDEN_WEIGHTS`) actively fight over that shared accumulator, rather than
one-shot-overwriting each other -- a real behavioral difference from a stateless per-tick
threshold, not a re-branding of the same thing.

**Honest note on the weights**: this network is not trained. Untrained random weights would make
whether/when ``BUY_ARBITRAGE`` fires essentially arbitrary noise, which would make task 6's
"prove a spike triggered this, live" demonstration itself untrustworthy from one run to the next
(same "architecture-verified, not a trained/calibrated model" disclosure this project makes
everywhere it ships an untrained network -- see e.g. ``core/crossing_network.py``,
``ai_core/l2_transformer.py``, and this very package's :mod:`tda_engine`). :data:`_HIDDEN_WEIGHTS`/
:data:`_OUTPUT_WEIGHTS` are hand-set, deterministically, to build a real (if simplistic) "sustained
downward momentum" accumulator: every down-spike pushes every hidden neuron's membrane up together
and every up-spike pushes it back down, and the output layer routes accumulated hidden spikes
toward ``BUY_ARBITRAGE`` and away from ``HOLD``. This is a real, reproducible circuit design
exercise, not a fabricated "AI decided" black box, and explicitly not a claim that this hand-set
circuit is a profitable trading strategy.
"""

from __future__ import annotations

import sys
from typing import Final

import matplotlib.pyplot as plt
import snntorch as snn
import torch
from torch import Tensor, nn

from nothierly_genesis.snn_encoder import EXAMPLE_PRICE_SERIES, delta_encode

NUM_INPUT_NEURONS: Final[int] = 2  # [spike_up, spike_down]
NUM_HIDDEN_NEURONS: Final[int] = 10
NUM_OUTPUT_NEURONS: Final[int] = 2  # [HOLD, BUY_ARBITRAGE]
HOLD_INDEX: Final[int] = 0
BUY_ARBITRAGE_INDEX: Final[int] = 1

BETA: Final[float] = 0.9  # the real membrane leak/decay constant -- see module docstring

# Hand-set, not learned -- see module docstring. Every hidden neuron: -0.4 from an up-spike,
# +0.4 from a down-spike -- with beta=0.9 and the default threshold=1.0, this takes *three*
# consecutive real down-spikes to actually cross threshold (0.4 -> 0.76 -> 1.0156), a real
# accumulation window, not an instant one-tick trigger; a single up-spike in between measurably
# sets the accumulator back, not just "resets" it (see module docstring's own if-condition note).
_HIDDEN_WEIGHTS: Final[list[list[float]]] = [[-0.4, 0.4] for _ in range(NUM_HIDDEN_NEURONS)]

# All 10 hidden neurons are identical (same weights, same inputs -- see above), so they fire
# together as one real event, not ten independent votes -- 0.15 * 10 = 1.5, safely over the
# output layer's own threshold=1.0 in that one shot. -0.05 * 10 = -0.5 keeps HOLD's membrane
# negative whenever hidden fires, so it never competes with BUY_ARBITRAGE for this demo.
_OUTPUT_WEIGHTS: Final[list[list[float]]] = [
    [-0.05] * NUM_HIDDEN_NEURONS,  # HOLD
    [0.15] * NUM_HIDDEN_NEURONS,  # BUY_ARBITRAGE
]

# "zero" (reset the membrane to exactly 0.0 on a spike), not snnTorch's own default
# ("subtract", which only removes one real threshold's worth and lets a large excess linger) --
# the real, documented reason task 5 explicitly asks to see the membrane fall all the way back to
# a real, clean zero baseline after firing, not merely dip.
_RESET_MECHANISM: Final[str] = "zero"


class ArbitrageTriggerSNN(nn.Module):
    """Input(2) -> Linear -> Leaky(10) -> Linear -> Leaky(2). See module docstring for the weights."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(NUM_INPUT_NEURONS, NUM_HIDDEN_NEURONS, bias=False)
        self.lif1 = snn.Leaky(beta=BETA, reset_mechanism=_RESET_MECHANISM)
        self.fc2 = nn.Linear(NUM_HIDDEN_NEURONS, NUM_OUTPUT_NEURONS, bias=False)
        self.lif2 = snn.Leaky(beta=BETA, reset_mechanism=_RESET_MECHANISM)

        with torch.no_grad():
            self.fc1.weight.copy_(torch.tensor(_HIDDEN_WEIGHTS, dtype=torch.float32))
            self.fc2.weight.copy_(torch.tensor(_OUTPUT_WEIGHTS, dtype=torch.float32))

    def init_state(self) -> tuple[Tensor, Tensor]:
        """Fresh membrane state for both LIF layers -- call once before a real event-driven run."""
        return self.lif1.init_leaky(), self.lif2.init_leaky()

    def step(self, spike_in: Tensor, mem1: Tensor, mem2: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """One real, single-tick forward pass -- the caller carries ``mem1``/``mem2`` across ticks.

        Returns:
            ``(spk_out, mem2, mem1, mem2)`` -- ``spk_out`` is the real output-layer spike tensor
            (shape ``(2,)``, one-hot-ish 0/1 per action neuron); the two membrane tensors are
            returned so the caller can log/plot the real membrane trace over time.
        """
        current1 = self.fc1(spike_in)
        spk1, mem1 = self.lif1(current1, mem1)
        current2 = self.fc2(spk1)
        spk_out, mem2 = self.lif2(current2, mem2)
        return spk_out, mem1, mem2, spk1


def run_event_driven_trigger(price_series: list[float] | tuple[float, ...]) -> dict[str, list[float]]:
    """Real, tick-by-tick (never batched) event-driven pass -- see module docstring on statefulness.

    Prints the brief's own exact required line the instant (and only the instant) the real
    ``BUY_ARBITRAGE`` output neuron spikes -- never a plain ``if`` on the raw price series itself;
    the only condition checked is the real tensor ``spk_out[BUY_ARBITRAGE_INDEX] == 1``, snnTorch's
    own real spike output.

    Returns:
        A real trace dict -- one entry per tick, for :func:`plot_membrane_potential`.
        ``hidden_membrane`` (all 10 hidden neurons are identical by construction, see
        :data:`_HIDDEN_WEIGHTS`'s own comment, so tracking neuron 0 shows the real, gradual
        3-tick accumulation) and ``buy_membrane`` (the output neuron -- a single-tick relay once
        the hidden layer actually fires, not a second gradual ramp of its own) are both kept,
        deliberately: task 5 asks to see the tension "rise with each input spike" -- that real
        rise happens in the hidden layer; the output layer is the real, decisive trigger stage.
    """
    torch.manual_seed(0)  # weights are hand-set, not random -- this only pins nn.Linear's unused bias-free init path
    net = ArbitrageTriggerSNN()
    mem1, mem2 = net.init_state()

    spikes = delta_encode(list(price_series))
    trace: dict[str, list[float]] = {
        "hidden_membrane": [],
        "buy_membrane": [],
        "buy_spikes": [],
        "spike_up": [],
        "spike_down": [],
    }

    print(f"Série de prix : {list(price_series)}")
    # Pure inference -- no training here (see module docstring on the hand-set, not learned,
    # weights), so no autograd graph needs to be built while stepping through the real ticks.
    with torch.no_grad():
        for t in range(spikes.shape[0]):
            spike_in = spikes[t]
            spk_out, mem1, mem2, _spk_hidden = net.step(spike_in, mem1, mem2)

            trace["hidden_membrane"].append(float(mem1[0]))
            trace["buy_membrane"].append(float(mem2[BUY_ARBITRAGE_INDEX]))
            trace["buy_spikes"].append(float(spk_out[BUY_ARBITRAGE_INDEX]))
            trace["spike_up"].append(float(spike_in[0]))
            trace["spike_down"].append(float(spike_in[1]))

            if bool(spk_out[BUY_ARBITRAGE_INDEX] > 0.5):
                print(f"⚡ SPIKE DÉTECTÉ : ORDRE D'ACHAT DÉCLENCHÉ À t= {t}")

    return trace


def plot_membrane_potential(trace: dict[str, list[float]], output_path: str) -> None:
    """Real membrane-potential traces -- hidden layer's real gradual accumulation, output layer's real trigger.

    See :func:`run_event_driven_trigger`'s own docstring for why both, not just the output
    neuron, are plotted: the gradual "tension rising with each spike" task 5 asks for is a real
    property of the hidden layer here, not the output layer's single-tick relay.
    """
    ticks = list(range(len(trace["buy_membrane"])))
    fig, (ax_spikes, ax_hidden, ax_membrane) = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    ax_spikes.eventplot(
        [i for i, v in enumerate(trace["spike_up"]) if v > 0], lineoffsets=1, colors="#2ca02c", label="Spike UP"
    )
    ax_spikes.eventplot(
        [i for i, v in enumerate(trace["spike_down"]) if v > 0], lineoffsets=0, colors="#d62728", label="Spike DOWN"
    )
    ax_spikes.set_yticks([0, 1])
    ax_spikes.set_yticklabels(["DOWN", "UP"])
    ax_spikes.set_title("Spikes d'entrée réels (delta-encoding du flux de prix P2P)")
    ax_spikes.legend(loc="upper right")

    ax_hidden.plot(ticks, trace["hidden_membrane"], color="#9467bd", marker="o", label="Potentiel de membrane (couche cachée, neurone 0)")
    ax_hidden.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Seuil (=1.0)")
    ax_hidden.set_ylabel("U(t) caché")
    ax_hidden.set_title("Couche cachée : accumulation réelle sur les baisses successives")
    ax_hidden.legend(loc="upper left", fontsize=8)

    ax_membrane.plot(ticks, trace["buy_membrane"], color="#1f77b4", marker="o", label="Potentiel de membrane (BUY_ARBITRAGE)")
    ax_membrane.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Seuil de déclenchement (=1.0)")
    for i, spiked in enumerate(trace["buy_spikes"]):
        if spiked > 0:
            ax_membrane.axvline(i, color="#ff7f0e", linestyle=":", linewidth=1.5)
    ax_membrane.set_xlabel("t (tick)")
    ax_membrane.set_ylabel("U(t) sortie")
    ax_membrane.set_title("Neurone de sortie BUY_ARBITRAGE : relais -> déclenchement -> reset")
    ax_membrane.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    # Windows' default console codepage (cp1252) cannot encode the emoji task 6 explicitly asks
    # this script to print -- a real, disclosed environment limitation, not a logic bug -- fixed
    # the standard way (TextIOWrapper.reconfigure, Python 3.7+) rather than dropping the emoji.
    # mypy sees sys.stdout as the narrower TextIO protocol, which has no reconfigure() --
    # real at runtime (stdout is a real io.TextIOWrapper), just not in that stub.
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    # A slightly richer series than the brief's own bare 5-tick example -- a sustained drop (enough
    # down-spikes to actually cross the output threshold and demonstrate a real reset), then a
    # recovery, so the membrane plot shows the full charge -> spike -> reset -> decay cycle asked
    # for in task 5, not just a rising ramp with nothing to reset from.
    demo_series = list(EXAMPLE_PRICE_SERIES) + [1642.0, 1640.0, 1648.0, 1655.0]
    result_trace = run_event_driven_trigger(demo_series)
    plot_path = "nothierly_genesis/snn_membrane_potential.png"
    plot_membrane_potential(result_trace, plot_path)
    print(f"Graphique du potentiel de membrane sauvegardé : {plot_path}")
