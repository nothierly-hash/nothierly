"""P2P Premium Scanner -- real-time view of nothierly-p2p-api's own real Crypto/Forex arbitrage scan.

**Real data only, read from a real running service, not recomputed here**: this page is a
thin, read-only client of the ``nothierly-p2p-api`` container (see ``p2p_api.py``'s own
module docstring for the full real data sourcing -- Binance P2P, open.er-api.com,
CoinGecko). Reached over the Compose network's internal DNS
(``http://nothierly-p2p-api:8000``, the service name as its own hostname) -- never the
host-published port, which is for the operator's own ``curl``/browser access, not
container-to-container traffic. Honest empty state if that service is unreachable or has
found nothing this cycle -- never a fabricated premium.

**Cross-country opportunities, scoped honestly**: :data:`p2p_api.CrossCountryOpportunity`
is a real price signal, not a claim the fiat leg is actually moveable between the two
countries -- shown here with the exact same caveat that class's own docstring states, not
silently dropped.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from src.dashboard.theme import CRIMSON, MINT, STEEL_GRAY

_P2P_API_URL = "http://nothierly-p2p-api:8000/api/v1/p2p-premiums"
_REQUEST_TIMEOUT_SECONDS = 8.0
# Real, load-bearing since p2p_api.py's own production endpoint now requires a valid X-API-Key
# (SOC 2 hardening -- see that module's own require_production_api_key): this dashboard page is
# itself one of the "internal" callers core/config.py's P2PApiSettings.internal_key docstring
# describes -- its raw value lives only in this container's own env (docker-compose.yml), and
# only its SHA-256 hash is ever configured server-side (P2P_API_KEY_SHA256_HASHES). Unset (a
# fresh/misconfigured deployment) degrades to the same honest empty state _fetch_scan already had
# before this feature existed -- never a crash.
_P2P_API_INTERNAL_KEY = os.environ.get("P2P_API_INTERNAL_KEY")

st.title("🌍 P2P Premium Scanner")
st.caption(
    "Primes d'arbitrage P2P Crypto/Forex réelles, en direct -- voir p2p_api.py. Prix P2P (Binance), "
    "taux de change officiels (open.er-api.com) et prix spot USDT (CoinGecko) tous réels, aucune donnée "
    "simulée. Les opportunités inter-pays sont un signal de prix réel, pas une promesse d'exécution -- "
    "voir la mise en garde dans le tableau correspondant plus bas."
)

refresh_col, _ = st.columns([1, 3])
with refresh_col:
    refresh_seconds = st.select_slider(
        "Rafraîchissement auto", options=[10, 15, 30, 60], value=30, format_func=lambda s: f"{s}s"
    )
st_autorefresh(interval=refresh_seconds * 1000, key="p2p_scanner_autorefresh")


def _fetch_scan() -> tuple[dict[str, Any] | None, str | None]:
    """Real, synchronous GET against the real p2p-api service. Returns ``(body, error)`` -- never both set."""
    headers = {"Accept": "application/json"}
    if _P2P_API_INTERNAL_KEY:
        headers["X-API-Key"] = _P2P_API_INTERNAL_KEY
    try:
        request = urllib.request.Request(_P2P_API_URL, headers=headers)
        with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
            return json.loads(response.read()), None
    except urllib.error.HTTPError as exc:
        # Real, common cause now that the production endpoint requires auth: a missing/misrotated
        # P2P_API_INTERNAL_KEY surfaces as a real 401, not a generic network error -- worth its
        # own honest message rather than being folded into URLError's generic string below.
        return None, f"HTTP {exc.code}: {exc.reason}"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return None, str(exc)


body, error = _fetch_scan()

if error is not None:
    st.error(
        f"**Service nothierly-p2p-api injoignable** -- {error}\n\n"
        "Vérifiez que le conteneur tourne (`docker compose ps nothierly-p2p-api`) -- aucune donnée "
        "simulée n'est affichée à sa place."
    )
    st.stop()

assert body is not None
candidates: list[dict[str, Any]] = body.get("top_candidates", [])
opportunities: list[dict[str, Any]] = body.get("cross_country_opportunities", [])
excluded: dict[str, str] = body.get("excluded", {})

st.caption(
    f"Dernier scan réel : {body.get('timestamp', '?')} "
    + (f"(cache : {body['cache_age_seconds']:.0f}s)" if body.get("cache_age_seconds") else "(scan frais)")
)

st.markdown("---")
kpi1, kpi2, kpi3 = st.columns(3)
if candidates:
    best = candidates[0]
    kpi1.metric("Meilleure prime réelle", f"{best['country_fiat']} +{best['premium_percentage']:.2f}%")
else:
    kpi1.metric("Meilleure prime réelle", "N/A")
if opportunities:
    best_opp = opportunities[0]
    kpi2.metric(
        "Meilleur écart inter-pays",
        f"{best_opp['buy_country_fiat']}→{best_opp['sell_country_fiat']} +{best_opp['spread_percentage']:.2f}%",
    )
else:
    kpi2.metric("Meilleur écart inter-pays", "N/A")
kpi3.metric("Devises exclues (données réelles indisponibles)", len(excluded))

if not candidates:
    st.info(
        "**Aucune prime réelle disponible pour l'instant** -- soit Binance P2P n'a réellement aucune "
        "annonce pour ces devises en ce moment, soit le fournisseur forex ne les couvre pas. Voir le détail des "
        "exclusions ci-dessous. Rien n'est affiché à leur place."
    )
    if excluded:
        st.dataframe(
            [{"Devise": fiat, "Raison réelle": reason} for fiat, reason in excluded.items()],
            width="stretch", hide_index=True,
        )
    st.stop()

st.markdown("---")
st.subheader("Primes réelles par pays")
st.caption(
    "Prime = ( Prix P2P réel / Taux de change officiel réel − Prix spot USDT réel ) × 100. "
    "Profondeur et taux de complétion marchand sont des champs réels déjà présents dans les annonces "
    "Binance -- une prime élevée adossée à peu de liquidité réelle est un signal plus faible qu'une "
    "prime plus modeste sur un vrai volume important."
)

candidate_rows = [
    {
        "Devise": c["country_fiat"],
        "Prime réelle (%)": c["premium_percentage"],
        "Prix P2P réel": c["p2p_price"],
        "Taux officiel réel": c["forex_rate"],
        "Équivalent USD réel": c["usd_equivalent_price"],
        "Liquidité réelle (USDT)": c["avg_liquidity_usdt"],
        "Complétion marchand réelle": f"{c['avg_merchant_completion_rate'] * 100:.1f}%",
        "Annonces échantillonnées": c["p2p_ads_sampled"],
    }
    for c in candidates
]
st.dataframe(candidate_rows, width="stretch", hide_index=True)

fig_colors = [MINT if c["premium_percentage"] >= 0 else CRIMSON for c in candidates]
try:
    import plotly.graph_objects as go

    fig = go.Figure(
        data=[
            go.Bar(
                x=[c["country_fiat"] for c in candidates],
                y=[c["premium_percentage"] for c in candidates],
                marker_color=fig_colors,
                text=[f"{c['premium_percentage']:.2f}%" for c in candidates],
                textposition="outside",
            )
        ]
    )
    fig.add_hline(y=0, line_dash="dash", line_color=STEEL_GRAY)
    fig.update_layout(
        height=360, margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Prime réelle (%)", showlegend=False,
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
except ImportError:
    pass

if excluded:
    with st.expander(f"⚠️ {len(excluded)} devise(s) exclue(s) -- données réelles indisponibles"):
        st.dataframe(
            [{"Devise": fiat, "Raison réelle": reason} for fiat, reason in excluded.items()],
            width="stretch", hide_index=True,
        )

st.markdown("---")
st.subheader("Opportunités inter-pays (signal de prix réel)")
st.warning(
    "⚠️ Ceci compare des prix réels normalisés en équivalent USD -- ce n'est PAS une garantie "
    "d'exécution. Réaliser l'écart exige un accès de paiement réel (compte bancaire/e-wallet) dans "
    "les deux pays, ou une contrepartie qui accepte de faire ce transfert -- ce service n'a aucun "
    "moyen de savoir si c'est le cas pour vous."
)

if not opportunities:
    st.info("Aucune opportunité inter-pays réelle détectée sur ce scan (moins de 2 devises avec des données réelles disponibles).")
else:
    st.dataframe(
        [
            {
                "Acheter (pays)": o["buy_country_fiat"],
                "Vendre (pays)": o["sell_country_fiat"],
                "Équivalent USD achat": o["buy_usd_equivalent"],
                "Équivalent USD vente": o["sell_usd_equivalent"],
                "Écart réel (%)": o["spread_percentage"],
            }
            for o in opportunities
        ],
        width="stretch", hide_index=True,
    )

st.markdown("---")
st.subheader("📈 NEMDI -- tendance réelle de l'indice de détresse")
st.caption(
    "Indice public gratuit (voir nemdi.py) reconstruit à partir de l'historique réel accumulé par "
    "ce service -- GET /api/v1/nemdi/history. Vide/court sur un déploiement récent, jamais interpolé "
    "ou rempli artificiellement : chaque point correspond à un vrai cycle de scan passé."
)


def _fetch_nemdi_history(hours: int) -> tuple[dict[str, Any] | None, str | None]:
    url = f"http://nothierly-p2p-api:8000/api/v1/nemdi/history?hours={hours}"
    headers = {"Accept": "application/json"}
    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
            return json.loads(response.read()), None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return None, str(exc)


nemdi_hours = st.select_slider("Fenêtre réelle", options=[6, 24, 72, 168, 720], value=24, format_func=lambda h: f"{h}h")
nemdi_body, nemdi_error = _fetch_nemdi_history(nemdi_hours)

if nemdi_error is not None:
    st.warning(f"NEMDI /history injoignable : {nemdi_error}")
elif not nemdi_body or not nemdi_body.get("points"):
    st.info(
        "Aucun point d'historique réel pour l'instant sur cette fenêtre -- ce déploiement n'a pas "
        "encore accumulé assez de cycles de scan. Ce n'est pas une erreur, juste un déploiement jeune."
    )
else:
    nemdi_points = nemdi_body["points"]
    try:
        import plotly.graph_objects as go

        fig_nemdi = go.Figure(
            data=[
                go.Scatter(
                    x=[p["timestamp"] for p in nemdi_points],
                    y=[p["nemdi_score"] for p in nemdi_points],
                    mode="lines+markers",
                    line=dict(color=MINT, width=2),
                    name="NEMDI",
                )
            ]
        )
        fig_nemdi.add_hline(y=50, line_dash="dash", line_color=STEEL_GRAY, annotation_text="Neutre (50)")
        fig_nemdi.update_layout(
            height=320, margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="NEMDI (0-100)", yaxis_range=[0, 100], showlegend=False,
        )
        st.plotly_chart(fig_nemdi, width="stretch", config={"displayModeBar": False})
    except ImportError:
        pass
    st.caption(f"{len(nemdi_points)} point(s) réel(s) sur {nemdi_hours}h -- dernier score : {nemdi_points[-1]['nemdi_score']}.")
