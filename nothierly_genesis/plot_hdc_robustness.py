"""Genesis Protocol: real, measured robustness curves for hdc_engine.py -- see that module's own
docstring for why raw cosine similarity and nearest-neighbor classification accuracy are two real,
different metrics, and why only the second is what "the system still works under data loss"
actually depends on.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nothierly_genesis.hdc_engine import (
    MarketItemMemory,
    corrupt,
    cosine_similarity,
    encode_market_snapshot,
    measure_classification_accuracy,
)

_OUTPUT_PATH: Path = Path(__file__).resolve().parent / "hdc_robustness_curve.png"

_CANDIDATE_STATES = {
    "calme_ngn_ars_try": [
        {"country_fiat": "NGN", "premium_percentage": 5.2},
        {"country_fiat": "ARS", "premium_percentage": 8.7},
        {"country_fiat": "TRY", "premium_percentage": 1.1},
    ],
    "crise_ngn": [
        {"country_fiat": "NGN", "premium_percentage": 13.5},
        {"country_fiat": "ARS", "premium_percentage": 2.0},
        {"country_fiat": "TRY", "premium_percentage": 0.5},
    ],
    "crise_ars": [
        {"country_fiat": "NGN", "premium_percentage": 1.0},
        {"country_fiat": "ARS", "premium_percentage": 14.0},
        {"country_fiat": "TRY", "premium_percentage": 0.8},
    ],
    "marche_plat": [
        {"country_fiat": "NGN", "premium_percentage": 0.2},
        {"country_fiat": "ARS", "premium_percentage": 0.3},
        {"country_fiat": "TRY", "premium_percentage": 0.1},
    ],
}


def generate_robustness_plot(output_path: Path = _OUTPUT_PATH) -> Path:
    memory = MarketItemMemory()
    clean_states = {name: encode_market_snapshot(snap, memory) for name, snap in _CANDIDATE_STATES.items()}
    reference_state = clean_states["calme_ngn_ars_try"]

    fractions = np.linspace(0.0, 0.9, 19)
    # x100: cosine_similarity returns a real value in [-1, 1] -- rescaled here only so it shares
    # the same 0-100 axis as the accuracy percentage below; the real, underlying value plotted is
    # exactly `100 * cosine_similarity`, not a different quantity.
    similarities = [cosine_similarity(reference_state, corrupt(reference_state, float(f))) * 100.0 for f in fractions]
    accuracies = [measure_classification_accuracy(clean_states, float(f), trials=500) * 100.0 for f in fractions]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(fractions * 100, similarities, marker="o", color="#d62728", label="Similarité cosinus brute x100 (état corrompu vs état propre)")
    ax.plot(fractions * 100, accuracies, marker="s", color="#2ca02c", label=f"Précision de classification réelle ({len(clean_states)} états candidats)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Corruption simulée des données (%)")
    ax.set_ylabel("Similarité (x100) / Précision (%)")
    ax.set_title("HDC Engine -- robustesse réelle mesurée, pas affirmée")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    path = generate_robustness_plot()
    print(f"Graphique de robustesse HDC sauvegardé : {path}")
