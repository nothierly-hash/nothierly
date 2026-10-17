"""Cloud Infrastructure & FinOps -- real multi-instance burn rate, real R2 bucket health, real host RAM/VRAM.

**Real, now aggregated across instances**: :func:`~src.mlops.vast_client.
summarize_instance_costs` closes this page's own previously-disclosed
gap ("no code sums real spend across instances or projects it forward")
using :meth:`~src.mlops.vast_client.VastAIClient.list_instances` (new --
the real Vast.ai ``GET /instances/`` endpoint, with no instance ID,
returns every instance on the account) -- real burn rate (sum of
``dph_total`` across currently-``running`` instances) and real accrued
cost (``dph_total`` x hours elapsed since each instance's own real
``start_date``), never a projected or estimated figure.

**Master switch**: the toggle below writes
:data:`~core.schemas.VAST_AI_ENABLED_REDIS_KEY` directly via this
process's own Redis connection -- read live, on every call, by
:func:`~src.mlops.vast_client.is_vast_ai_enabled`, so both
``nothierly-agents`` (:mod:`scripts.run_remote_training`) and this
dashboard process see a flip immediately, with no container restart.
Gates :meth:`~src.mlops.vast_client.VastAIClient.search_gpu`/
``search_offers``/``rent_instance``/``start_instance`` (exposure-
creating calls) -- deliberately **not** the "Mettre en pause"/"Détruire"
buttons further down, or the cost-aggregation section above: a cost-
control switch that also blocked you from seeing or stopping what is
already running would defeat its own purpose the moment you needed it
most.

**R2 "cluster health", precisely what it is and isn't**:
:func:`~src.data_pipeline.data_lake_sync.check_bucket_health` is a real
``HEAD`` connectivity ping plus a real, capped object-count/size walk
of the configured bucket. It is **not** a usage-against-quota
percentage -- Cloudflare's S3-compatible API (the only credential type
:class:`~core.config.R2Settings` holds) doesn't expose one; that would
need a separate Cloudflare account-level API token this project
doesn't have configured. Shown as real connectivity + real usage, not
relabeled as "cluster health %".

**"RAM/VRAM", honestly scoped to this host**: :func:`~src.mlops.
host_metrics.get_host_memory_snapshot` reports *this dashboard's own
host machine* -- there is no remote hardware-telemetry channel to a
rented Vast.ai instance anywhere in this project (see that module's
own docstring). Real and useful as this project's own operating
environment, but not the remote training instance's utilization.
"""

from __future__ import annotations

import asyncio

import streamlit as st

from core.config import get_settings
from core.schemas import VAST_AI_ENABLED_REDIS_KEY
from src.dashboard.ems_pages._shared import get_redis_client, safe_get
from src.dashboard.theme import status_badge
from src.mlops.host_metrics import get_host_memory_snapshot
from src.mlops.vast_client import summarize_instance_costs

st.title("☁️ Cloud Infrastructure & FinOps")
st.caption("Suivi réel de l'infrastructure Vast.ai, R2, et de cet hôte -- pas de chiffre inventé.")

settings = get_settings()
r = get_redis_client()

st.subheader("Vast.ai -- recherche d'instance réelle")
vast_configured = settings.vastai.api_key is not None
st.markdown(status_badge("SAFE" if vast_configured else "PENDING", "CLÉ CONFIGURÉE" if vast_configured else "NON CONFIGURÉ"), unsafe_allow_html=True)

st.markdown("---")
st.subheader("🔌 Master Switch -- module Vast.ai")
_vast_enabled_raw = safe_get(r, VAST_AI_ENABLED_REDIS_KEY)
_vast_enabled_live = (
    _vast_enabled_raw.strip().lower() == "true" if _vast_enabled_raw is not None else settings.vastai.enabled
)
_vast_toggle = st.toggle(
    "Activer le module Vast.ai (recherche/location/redémarrage d'instances)",
    value=_vast_enabled_live,
    key="vast_ai_master_switch",
    help=(
        "OFF (par défaut) bloque la recherche, la location, et le redémarrage d'instance -- "
        "'Mettre en pause' et 'Détruire' restent toujours disponibles ci-dessous, y compris OFF, "
        "pour ne jamais empêcher de couper une facturation déjà en cours."
    ),
)
if _vast_toggle != _vast_enabled_live:
    r.set(VAST_AI_ENABLED_REDIS_KEY, "true" if _vast_toggle else "false")
    st.toast(
        f"Module Vast.ai {'activé' if _vast_toggle else 'désactivé'} -- effectif immédiatement, "
        "aucun redémarrage de conteneur nécessaire.",
        icon="🔌",
    )
    st.rerun()
st.markdown(
    status_badge("SAFE" if _vast_toggle else "PANIC", "MODULE ACTIVÉ" if _vast_toggle else "MODULE DÉSACTIVÉ"),
    unsafe_allow_html=True,
)

if vast_configured:
    st.markdown("---")
    st.subheader("🆘 Actions d'urgence -- toutes les instances réelles")
    st.caption(
        "Ces deux boutons ne sont jamais bloqués par le Master Switch ci-dessus -- couper une "
        "facturation déjà en cours doit toujours rester possible."
    )
    emergency_col1, emergency_col2 = st.columns(2)

    with emergency_col1:
        if st.button("⏸️ Mettre toutes les instances en pause", width="stretch"):
            from src.mlops.vast_client import VastAIClient

            async def _pause_all() -> tuple[list[tuple[int, str | None]], str | None]:
                results: list[tuple[int, str | None]] = []
                try:
                    async with VastAIClient(api_key=settings.vastai.api_key.get_secret_value()) as client:  # type: ignore[union-attr]
                        instances = await client.list_instances()
                        for instance in instances:
                            instance_id = int(instance["id"])
                            try:
                                await client.stop_instance(instance_id)
                                results.append((instance_id, None))
                            except Exception as exc:  # noqa: BLE001
                                results.append((instance_id, str(exc)))
                    return results, None
                except Exception as exc:  # noqa: BLE001
                    return results, str(exc)

            with st.spinner("Pause réelle de chaque instance en cours..."):
                pause_results, fatal_error = asyncio.run(_pause_all())
            if fatal_error is not None:
                st.error(f"Échec réel : {fatal_error}")
            elif not pause_results:
                st.info("Aucune instance réelle à mettre en pause.")
            else:
                for instance_id, error in pause_results:
                    if error is None:
                        st.success(f"Instance {instance_id} : pause réelle envoyée.")
                    else:
                        st.error(f"Instance {instance_id} : échec réel -- {error}")

    with emergency_col2:
        _confirm_destroy_key = "confirm_destroy_all_vast_instances"
        if _confirm_destroy_key not in st.session_state:
            st.session_state[_confirm_destroy_key] = False

        if not st.session_state[_confirm_destroy_key]:
            if st.button("🗑️ Détruire toutes les instances", type="primary", width="stretch"):
                st.session_state[_confirm_destroy_key] = True
                st.rerun()
        else:
            st.warning("Confirmez : ceci détruit réellement CHAQUE instance Vast.ai sur ce compte, irréversible.")
            col_confirm, col_cancel = st.columns(2)
            with col_confirm:
                if st.button("✅ CONFIRMER -- tout détruire", type="primary", width="stretch"):
                    from src.mlops.vast_client import VastAIClient

                    async def _destroy_all() -> tuple[list[tuple[int, str | None]], str | None]:
                        results: list[tuple[int, str | None]] = []
                        try:
                            async with VastAIClient(api_key=settings.vastai.api_key.get_secret_value()) as client:  # type: ignore[union-attr]
                                instances = await client.list_instances()
                                for instance in instances:
                                    instance_id = int(instance["id"])
                                    try:
                                        await client.destroy_instance(instance_id)
                                        results.append((instance_id, None))
                                    except Exception as exc:  # noqa: BLE001
                                        results.append((instance_id, str(exc)))
                            return results, None
                        except Exception as exc:  # noqa: BLE001
                            return results, str(exc)

                    with st.spinner("Destruction réelle de chaque instance en cours..."):
                        destroy_results, fatal_error = asyncio.run(_destroy_all())
                    st.session_state[_confirm_destroy_key] = False
                    if fatal_error is not None:
                        st.error(f"Échec réel : {fatal_error}")
                    elif not destroy_results:
                        st.info("Aucune instance réelle à détruire.")
                    else:
                        for instance_id, error in destroy_results:
                            if error is None:
                                st.success(f"Instance {instance_id} : détruite réellement.")
                            else:
                                st.error(f"Instance {instance_id} : échec réel -- {error}. VÉRIFIEZ MANUELLEMENT sur console.vast.ai.")
            with col_cancel:
                if st.button("Annuler", width="stretch", key="cancel_destroy_all"):
                    st.session_state[_confirm_destroy_key] = False
                    st.rerun()

if vast_configured:
    instance_id = st.number_input("ID d'instance Vast.ai à interroger", min_value=0, step=1, value=0)
    if st.button("Interroger l'instance (appel API réel)") and instance_id:
        from src.mlops.vast_client import VastAIClient

        async def _lookup() -> tuple[dict[str, object] | None, str | None]:
            try:
                async with VastAIClient(api_key=settings.vastai.api_key.get_secret_value()) as client:  # type: ignore[union-attr]
                    details = await client.get_instance_details(int(instance_id))
                    return details, None
            except Exception as exc:  # noqa: BLE001
                return None, str(exc)

        details, error = asyncio.run(_lookup())
        if error is not None:
            st.error(f"Échec de l'appel réel à Vast.ai : {error}")
        else:
            st.json(details)

    st.markdown("---")
    st.subheader("Burn Rate agrégé (réel, toutes instances)")
    if st.button("🔥 Lister toutes les instances et agréger les coûts (réel)"):
        from src.mlops.vast_client import VastAIClient

        async def _list_and_summarize() -> tuple[list[dict[str, object]] | None, str | None]:
            try:
                async with VastAIClient(api_key=settings.vastai.api_key.get_secret_value()) as client:  # type: ignore[union-attr]
                    instances = await client.list_instances()
                    return instances, None
            except Exception as exc:  # noqa: BLE001
                return None, str(exc)

        instances, error = asyncio.run(_list_and_summarize())
        if error is not None:
            st.error(f"Échec de l'appel réel à Vast.ai : {error}")
        elif not instances:
            st.info("Aucune instance réelle actuellement louée sur ce compte Vast.ai.")
        else:
            summary = summarize_instance_costs(instances)
            c1, c2, c3 = st.columns(3)
            c1.metric("Instances en cours d'exécution", summary.running_instance_count)
            c2.metric("Burn Rate réel ($/h)", f"${summary.total_burn_rate_usd_per_hour:,.2f}")
            c3.metric("Coût cumulé réel (accrued)", f"${summary.total_accrued_cost_usd:,.2f}")
            st.dataframe(
                [
                    {
                        "ID": row.instance_id, "GPU": row.gpu_name or "--", "Statut": row.actual_status,
                        "$/heure": round(row.dph_total, 4), "Coût cumulé réel ($)": row.accrued_cost_usd,
                    }
                    for row in summary.per_instance
                ],
                width="stretch", hide_index=True,
            )
            st.caption(
                "Détails bruts (réels) par instance, tels que renvoyés par l'API Vast.ai -- utilisation GPU/RAM "
                "éventuellement présente si l'API la rapporte pour cette instance."
            )
            for instance in instances:
                with st.expander(f"Instance {instance.get('id', '?')} -- JSON brut réel"):
                    st.json(instance)
else:
    st.info("Configure VASTAI_API_KEY pour activer ce module.")

st.markdown("---")
st.subheader("Cloudflare R2")
r2_configured = settings.r2.account_id is not None and settings.r2.bucket_name is not None
st.markdown(status_badge("SAFE" if r2_configured else "PENDING", f"BUCKET: {settings.r2.bucket_name}" if r2_configured else "NON CONFIGURÉ"), unsafe_allow_html=True)

if r2_configured:
    if st.button("🪣 Vérifier la santé du bucket (appel S3 réel)"):
        from src.data_pipeline.data_lake_sync import check_bucket_health

        with st.spinner("Ping + énumération réelle du bucket R2 en cours..."):
            health = check_bucket_health()

        if not health.reachable:
            st.error(f"Bucket injoignable (réel) : {health.error}")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Latence réelle", f"{health.latency_ms:.0f} ms" if health.latency_ms is not None else "--")
            c2.metric("Objets réels", f"{health.object_count:,}" if health.object_count is not None else "échec d'énumération")
            c3.metric(
                "Taille réelle totale",
                f"{health.total_size_bytes / (1024 ** 3):.2f} GB" if health.total_size_bytes is not None else "--",
            )
            if health.truncated:
                st.caption("⚠️ Énumération plafonnée (réel) -- le bucket contient probablement plus d'objets que ce compte.")
            if health.error is not None:
                st.warning(f"Bucket joignable, mais échec réel de l'énumération des objets : {health.error}")
    st.caption(
        "Pas de pourcentage \"quota utilisé\" -- l'API S3-compatible de R2 n'expose pas cette donnée "
        "(nécessiterait un jeton API Cloudflare distinct, non détenu par ce projet)."
    )
else:
    st.info("Configure R2_ACCOUNT_ID / R2_BUCKET_NAME / identifiants R2 pour activer ce module.")

st.markdown("---")
st.subheader("Hôte local (RAM/VRAM) -- cette machine, pas l'instance Vast.ai distante")
snapshot = get_host_memory_snapshot()
c1, c2, c3 = st.columns(3)
c1.metric("RAM utilisée (réel)", f"{snapshot.ram_percent:.1f}%")
c2.metric("RAM totale (réel)", f"{snapshot.ram_total_bytes / (1024 ** 3):.1f} GB")
if snapshot.gpu_available and snapshot.vram_total_bytes is not None and snapshot.vram_used_bytes is not None:
    vram_pct = snapshot.vram_used_bytes / snapshot.vram_total_bytes * 100.0
    c3.metric(f"VRAM réelle ({snapshot.gpu_name})", f"{vram_pct:.1f}%")
else:
    c3.metric("VRAM", "Aucun GPU CUDA réel détecté")
st.caption(
    "Réel, mais scope honnête : ceci mesure la machine qui exécute ce dashboard, jamais l'instance Vast.ai "
    "distante (aucun canal de télémétrie matérielle distante n'existe dans ce projet)."
)
