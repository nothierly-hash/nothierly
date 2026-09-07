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
from src.dashboard.theme import status_badge
from src.mlops.host_metrics import get_host_memory_snapshot
from src.mlops.vast_client import summarize_instance_costs

st.title("☁️ Cloud Infrastructure & FinOps")
st.caption("Suivi réel de l'infrastructure Vast.ai, R2, et de cet hôte -- pas de chiffre inventé.")

settings = get_settings()

st.subheader("Vast.ai -- recherche d'instance réelle")
vast_configured = settings.vastai.api_key is not None
st.markdown(status_badge("SAFE" if vast_configured else "PENDING", "CLÉ CONFIGURÉE" if vast_configured else "NON CONFIGURÉ"), unsafe_allow_html=True)

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
