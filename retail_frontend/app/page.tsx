import { fetchPublicPerformance, type PerformanceMetric } from "@/lib/api";

// Real cached-data-only dashboard -- see retail_gateway/main.py's module docstring for the
// one-way export job (scripts/export_performance_snapshot.py, trading-core side) that populates
// what this reads. This page has zero ability to act on Nothierly's own Daemons -- it only ever
// renders rows GET /api/v1/public/performance returns.
export const revalidate = 60; // re-fetch this cached snapshot at most once a minute, server-side

export default async function DashboardPage() {
  let metrics: PerformanceMetric[] = [];
  let loadError: string | null = null;
  try {
    metrics = await fetchPublicPerformance();
  } catch {
    loadError = "Impossible de charger les données pour le moment.";
  }

  return (
    <section>
      <h1>Performance Nothierly</h1>
      <p style={{ color: "#9aa4b2" }}>
        Données en cache, en lecture seule -- ce tableau de bord n&apos;a aucun accès d&apos;action sur le moteur de
        trading.
      </p>

      {loadError && <p style={{ color: "#e08080" }}>{loadError}</p>}
      {!loadError && metrics.length === 0 && <p style={{ color: "#9aa4b2" }}>Aucune donnée publiée pour le moment.</p>}

      {metrics.length > 0 && (
        <table style={{ width: "100%", borderCollapse: "collapse", marginTop: "1rem" }}>
          <thead>
            <tr style={{ textAlign: "left", borderBottom: "1px solid #1f232b" }}>
              <th style={{ padding: "0.5rem 0" }}>Métrique</th>
              <th style={{ padding: "0.5rem 0" }}>Valeur</th>
              <th style={{ padding: "0.5rem 0" }}>Au</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map((metric) => (
              <tr key={metric.metric_name} style={{ borderBottom: "1px solid #1f232b" }}>
                <td style={{ padding: "0.5rem 0" }}>{metric.metric_name}</td>
                <td style={{ padding: "0.5rem 0" }}>{metric.metric_value.toLocaleString("fr-FR")}</td>
                <td style={{ padding: "0.5rem 0", color: "#9aa4b2" }}>{new Date(metric.as_of).toLocaleString("fr-FR")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
