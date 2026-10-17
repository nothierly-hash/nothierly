"""Liquidity Routing & API Gateway.

**Chapter 13, Étape 5.6 update**: this page now has a real multi-broker
layer to show -- :mod:`src.execution.broker_gateway` (real
``BaseBroker`` ABC + ``BrokerFactory``, wrapping the real, pre-existing
``AsyncExchangeClient``/``OandaClient``). What follows is real: pinging
a broker below makes a genuine ``connect()`` call and times it; adding
a credential encrypts it with :class:`~src.execution.security_vault.
SecurityVault` before it ever touches Redis, never stored plaintext.

**Smart Routing -- corrected, a real router now, not a stale gap-note**:
this section previously said no live router consulted the "Smart
Routing" toggle to split an order across brokers, and that no order
was actually routed by capital-allocation logic. Both were true when
first written; fixed by :mod:`~src.execution.smart_order_router` (real
fee-inclusive water-filling across every connected candidate exchange's
real, currently-quoted order book -- see that module's own docstring
for the full design and its safety-machinery integration). Still
honest about scope: this page calls that router directly, on demand,
from a dashboard button -- there is still no ``CLIENT_ORDERS_TOPIC``
consumer automatically invoking it for every macro order (see
``api_middleware/gateway.py``'s own docstring on that separate gap).

API keys added through the credential form are never displayed again
after submission -- only *whether* one is configured, per exchange.
"""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal

import streamlit as st

from core.config import get_settings
from core.nothierly_ontology import AssetClass, OrderSide, UniversalAsset
from core.redis_client import RedisClient
from src.dashboard.ems_pages._shared import get_redis_client, safe_get
from src.dashboard.theme import status_badge
from src.execution.broker_gateway import BrokerFactory, CcxtBrokerAdapter, store_broker_credentials
from src.execution.security_vault import SecurityVault
from src.execution.smart_order_router import CANDIDATE_EXCHANGES, AllocationPlan, build_allocation_plan, execute_allocation_plan

_SMART_ROUTING_REDIS_KEY = "SMART_ROUTING_ENABLED"

st.title("🔌 Liquidity Routing & API Gateway")
st.caption("Gestion des connexions broker et routage de liquidité.")

settings = get_settings()

st.subheader("Broker connecté : OANDA (forex)")
# OandaSettings.api_token/account_id are required fields with no
# default (see core/config.py) -- if they were genuinely unset,
# get_settings() itself would already have failed the whole app before
# this page could render. So "configured" here really means "a value
# is present" -- whether it's a *working* credential is only knowable
# by actually calling the real API, which the button below does.
st.markdown(f"Statut : {status_badge('SAFE', f'COMPTE {settings.oanda.account_id} -- ENV {settings.oanda.environment.upper()}')}", unsafe_allow_html=True)

if st.button("Vérifier la connexion (appel API réel)"):
    from src.agents.supervisor.broker_client import OandaClient

    async def _check() -> tuple[float | None, str | None]:
        try:
            async with OandaClient(
                api_token=settings.oanda.api_token.get_secret_value(),
                account_id=settings.oanda.account_id,
                environment=settings.oanda.environment,
            ) as client:
                balance = await client.get_account_balance()
                return balance, None
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            return None, str(exc)

    balance, error = asyncio.run(_check())
    if error is not None:
        st.error(f"Échec de connexion réelle à OANDA : {error}")
    else:
        st.success(f"Connexion réelle réussie -- solde de compte réel : ${balance:,.2f}")

st.markdown("---")
st.subheader("Brokers crypto (Binance / Bybit / OKX / Deribit)")
st.caption(
    "Un seul adaptateur réel (CcxtBrokerAdapter) couvre les quatre -- ccxt unifie déjà leur protocole, "
    "voir src/execution/broker_gateway.py. Le ping ci-dessous est un vrai appel connect(), pas une simulation."
)

r = get_redis_client()

for exchange_id in CANDIDATE_EXCHANGES:
    col_name, col_status, col_action = st.columns([2, 2, 2])
    col_name.markdown(f"**{exchange_id.upper()}**")

    ping_key = f"liquidity_routing_ping_{exchange_id}"
    if ping_key in st.session_state:
        latency_ms, error = st.session_state[ping_key]
        if error is not None:
            col_status.markdown(status_badge("PANIC", "ÉCHEC"), unsafe_allow_html=True)
        else:
            col_status.markdown(status_badge("SAFE", f"{latency_ms:.0f} ms"), unsafe_allow_html=True)
    else:
        col_status.markdown(status_badge("PENDING", "NON TESTÉ"), unsafe_allow_html=True)

    if col_action.button("Tester la connexion (réel)", key=f"ping_{exchange_id}"):

        async def _ping(exchange_id: str = exchange_id) -> tuple[float | None, str | None]:
            redis_client = await RedisClient.get_instance(settings.redis)
            try:
                broker = await BrokerFactory.create(exchange_id, settings=settings, redis_client=redis_client)
                start = time.monotonic()
                async with broker:
                    pass
                return (time.monotonic() - start) * 1000.0, None
            except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
                return None, str(exc)

        latency_ms, error = asyncio.run(_ping())
        st.session_state[ping_key] = (latency_ms, error)
        if error is not None:
            st.error(f"Échec de connexion réelle à {exchange_id.upper()} : {error}")
        st.rerun()

st.markdown("---")
st.subheader("Interactive Brokers (TWS / IB Gateway)")
st.caption(
    "Intégration distincte de ccxt -- IBKR ne s'authentifie pas par clé API mais via l'application "
    "TWS/IB Gateway elle-même, en local. Configure IBKR_HOST/IBKR_PORT/IBKR_CLIENT_ID si besoin "
    f"(actuellement {settings.ibkr.host}:{settings.ibkr.port})."
)
if "liquidity_routing_ping_ibkr" in st.session_state:
    ibkr_latency_ms, ibkr_error = st.session_state["liquidity_routing_ping_ibkr"]
    if ibkr_error is not None:
        st.markdown(status_badge("PANIC", "ÉCHEC"), unsafe_allow_html=True)
    else:
        st.markdown(status_badge("SAFE", f"{ibkr_latency_ms:.0f} ms"), unsafe_allow_html=True)
else:
    st.markdown(status_badge("PENDING", "NON TESTÉ"), unsafe_allow_html=True)

if st.button("Tester la connexion à TWS/IB Gateway (réel)"):

    async def _ping_ibkr() -> tuple[float | None, str | None]:
        try:
            broker = await BrokerFactory.create("ibkr", settings=settings)
            start = time.monotonic()
            async with broker:
                pass
            return (time.monotonic() - start) * 1000.0, None
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            return None, str(exc)

    ibkr_latency_ms, ibkr_error = asyncio.run(_ping_ibkr())
    st.session_state["liquidity_routing_ping_ibkr"] = (ibkr_latency_ms, ibkr_error)
    if ibkr_error is not None:
        st.error(
            f"Échec de connexion réelle à TWS/IB Gateway : {ibkr_error}\n\n"
            "Nécessite TWS ou IB Gateway lancé en local avec l'API activée -- non démarré dans cet environnement."
        )
    st.rerun()

st.markdown("---")
st.subheader("Ajouter des identifiants (chiffrés, jamais en clair)")
st.caption("Chiffré via src/execution/security_vault.py (Fernet) avant d'être écrit dans Redis -- jamais stocké en clair.")

with st.form("add_broker_credentials_form", clear_on_submit=True):
    cred_exchange = st.selectbox("Exchange", CANDIDATE_EXCHANGES)
    cred_api_key = st.text_input("API Key", type="password")
    cred_api_secret = st.text_input("API Secret", type="password")
    if st.form_submit_button("Chiffrer et enregistrer") and cred_api_key and cred_api_secret:

        async def _store(exchange_id: str, api_key: str, api_secret: str) -> None:
            redis_client = await RedisClient.get_instance(settings.redis)
            vault = SecurityVault(settings.vault.master_key.get_secret_value())
            await store_broker_credentials(exchange_id, api_key=api_key, api_secret=api_secret, redis_client=redis_client, vault=vault)

        asyncio.run(_store(cred_exchange, cred_api_key, cred_api_secret))
        st.toast(f"Identifiants {cred_exchange.upper()} chiffrés et enregistrés.", icon="🔒")

st.markdown("---")
st.subheader("Smart Routing")
smart_routing_on = safe_get(r, _SMART_ROUTING_REDIS_KEY) == "1"
new_smart_routing = st.toggle(
    "Activer le fractionnement multi-broker des ordres macro",
    value=smart_routing_on,
    help="Un vrai flag Redis persistant -- voir la note honnête ci-dessous sur ce qu'il ne fait pas encore.",
)
if new_smart_routing != smart_routing_on:
    r.set(_SMART_ROUTING_REDIS_KEY, "1" if new_smart_routing else "0")
    st.rerun()
st.caption(
    "Réel : ce flag est réellement lu/écrit dans Redis, et contrôle désormais le vrai routeur ci-dessous -- "
    "voir src/execution/smart_order_router.py."
)

st.markdown("---")
st.subheader("Allocation de capital par route de liquidité")

if not smart_routing_on:
    st.info("Activez le \"Smart Routing\" ci-dessus pour router un ordre entre plusieurs brokers crypto.")
else:
    st.caption(
        "Water-filling réel par prix effectif (frais inclus) sur les carnets d'ordres réels des brokers "
        "sélectionnés -- voir src/execution/smart_order_router.py pour la conception complète (dont "
        "l'intégration Kill Switch / MOCK_MODE / UniversalOrder détaillée dans son docstring)."
    )
    mock_mode_active = settings.mock_mode
    st.markdown(
        status_badge("PENDING" if mock_mode_active else "SAFE", "MOCK_MODE" if mock_mode_active else "RÉEL"),
        unsafe_allow_html=True,
    )

    with st.form("route_order_form"):
        route_symbol = st.text_input("Symbole", value="BTC/USDT")
        route_side = st.selectbox("Sens", ["BUY", "SELL"])
        route_quantity = st.number_input("Quantité", min_value=0.0, value=1.0, step=0.1)
        route_venues = st.multiselect("Venues candidates", CANDIDATE_EXCHANGES, default=list(CANDIDATE_EXCHANGES))
        col_plan, col_execute = st.columns(2)
        build_only = col_plan.form_submit_button("Calculer le plan (réel, aucun ordre)", width="stretch")
        execute_now = col_execute.form_submit_button(
            "Calculer ET exécuter" + (" (simulé -- MOCK_MODE)" if mock_mode_active else " (⚠️ ordre RÉEL)"),
            width="stretch",
        )

    if (build_only or execute_now) and route_venues:

        async def _route_order() -> tuple[AllocationPlan, list[str]]:
            connected: list[CcxtBrokerAdapter] = []
            connect_errors: list[str] = []
            redis_client = await RedisClient.get_instance(settings.redis)
            for exchange_id in route_venues:
                broker = await BrokerFactory.create(exchange_id, settings=settings, redis_client=redis_client)
                # CANDIDATE_EXCHANGES only ever names real ccxt.pro exchange
                # ids (never "oanda"/"ibkr"), so BrokerFactory.create always
                # really returns a CcxtBrokerAdapter here -- asserted, not
                # just assumed, since the SOR needs get_taker_fee_rate,
                # which isn't part of the BaseBroker ABC (see that method's
                # own docstring).
                assert isinstance(broker, CcxtBrokerAdapter)
                try:
                    await broker.connect()
                    connected.append(broker)
                except Exception as exc:  # noqa: BLE001 -- one venue's connect failure must not block the others
                    connect_errors.append(f"{exchange_id.upper()}: {exc}")
            try:
                plan = await build_allocation_plan(connected, route_symbol, OrderSide(route_side), Decimal(str(route_quantity)))
                if execute_now:
                    base, _, quote = route_symbol.partition("/")
                    asset = UniversalAsset(
                        symbol=route_symbol, asset_class=AssetClass.CRYPTO_SPOT,
                        base_currency=base or route_symbol, quote_currency=quote or "USDT",
                        tick_size=Decimal("0.01"), lot_size=Decimal("0.00000001"),
                    )
                    result = await execute_allocation_plan(
                        plan, connected, route_symbol, asset=asset, algo_id="MANUAL_SOR_DASHBOARD",
                        regulatory_client_shortcode="DASHBOARD-OPERATOR", settings=settings,
                        redis_client=redis_client,
                    )
                    st.session_state["sor_last_result"] = result
                else:
                    st.session_state.pop("sor_last_result", None)
                return plan, connect_errors
            finally:
                await asyncio.gather(*(broker.close() for broker in connected), return_exceptions=True)

        plan, connect_errors = asyncio.run(_route_order())
        st.session_state["sor_last_plan"] = plan
        for error in connect_errors:
            st.warning(f"Connexion échouée, venue ignorée : {error}")

    last_plan: AllocationPlan | None = st.session_state.get("sor_last_plan")
    if last_plan is not None:
        if not last_plan.allocations:
            st.error("Aucune venue n'a répondu avec un carnet d'ordres exploitable -- plan vide.")
        else:
            st.table(
                [
                    {"Venue": a.broker_id.upper(), "Quantité allouée": round(a.quantity, 8), "Prix effectif (frais inclus)": round(a.effective_price, 2)}
                    for a in last_plan.allocations
                ]
            )
            st.caption(
                f"Alloué : {last_plan.allocated_quantity} / {last_plan.requested_quantity} demandé -- "
                + ("entièrement rempli." if last_plan.fully_filled else "rempli partiellement (profondeur réelle insuffisante sur les venues sélectionnées).")
            )

        last_result = st.session_state.get("sor_last_result")
        if last_result is not None:
            if last_result.orders:
                st.table(
                    [
                        {"Venue": o.broker_id.upper(), "Statut": o.status.value, "Quantité exécutée": float(o.filled_quantity), "Prix moyen": float(o.avg_fill_price) if o.avg_fill_price is not None else None}
                        for o in last_result.orders
                    ]
                )
            for broker_id, reason in last_result.failed_venues:
                st.error(f"{broker_id.upper()} : échec de l'exécution -- {reason}")
