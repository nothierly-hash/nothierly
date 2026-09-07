"""Chapter 13, Étape 5.7: "THE BILLING JUSTIFIER" -- monthly PDF client reports.

**Real, load-bearing corrections to the brief, stated upfront:**

1. **The brief assumes a queryable Postgres/Polars database of
   per-client transactions already exists** ("interroge la base de
   données... pour extraire toutes les transactions d'un client_id").
   Verified by direct inspection before writing any of this: it did
   not. :class:`~src.agents.archivist.agent.ArchivistAgent` wrote every
   bus message into one undifferentiated ``bus_messages`` table with no
   ``client_id`` column, and no fill-recording path anywhere carried
   client attribution at all. :mod:`src.execution.billing_schema`'s new
   ``client_fills`` hypertable, and the ``client_id``/``fee_usd``/
   ``slippage_saved_usd`` fields now threaded through
   ``api_middleware/gateway.py``'s ``_write_shadow_fill``, are what make
   this module's Rule 1 possible at all -- built this same session, not
   assumed to already exist.

2. **"Gross PnL"/"Net PnL" don't fit this business model, stated
   honestly rather than faked.** Chapter 13's own founding pivot
   (``api_middleware/gateway.py``'s module docstring) is
   Execution-as-a-Service: Nothierly never takes directional positions
   on behalf of a B2B client, so there is no client-side PnL for this
   system to have generated -- a client's PnL depends entirely on their
   own trading decisions and market moves, which this system doesn't
   see or control. What Nothierly *does* generate, and can honestly
   measure, is execution quality -- the "Nothierly Edge"
   (slippage avoided). :class:`ClientMonthlyMetrics` reports Gross/Net
   PnL and Sharpe/Max-Drawdown as ``None`` with an explicit note when a
   client has no priced position history to compute them from (true for
   every client today, since none exists) rather than fabricating
   numbers for a business model that doesn't produce them.

3. **Memory-safe for "hundreds of clients"**: not by streaming raw rows
   into Python, but by doing the actual aggregation (``SUM``/``COUNT``/
   ``AVG``) server-side in one SQL statement per client and only ever
   holding one client's small, already-aggregated result in memory at a
   time (:func:`generate_monthly_reports`) -- the real way to keep
   per-client memory bounded regardless of how many raw fills that
   client has, not an implementation detail left to chance.

4. **LLM summary**: reuses the same real direct-REST calling
   convention :mod:`src.agents.guardian_ai`/:mod:`src.agents.
   quant_copilot` already established (no LangChain, no provider SDK)
   via a new, genuinely single-shot helper -- :func:`chat_with_nothierly`/
   :func:`run_copilot_turn` are conversation/tool-calling-shaped and the
   wrong fit for "summarize this data in 3 lines" (see this module's own
   investigation before writing it).

5. **PDF via ReportLab, not WeasyPrint**: the brief's own suggestion,
   WeasyPrint, requires a system-wide GTK3/Pango/Cairo native runtime
   that is not installed on this Windows machine and fails at *import*
   time -- verified directly, not assumed, before switching. ReportLab
   has no native dependency. Charts are rendered via ``plotly`` (already
   a real dependency here) + ``kaleido`` for static PNG export, not a
   new ``matplotlib`` dependency for a capability this project already has.
"""

from __future__ import annotations

import asyncio
import io
import random
import smtplib
from datetime import date, datetime, timedelta, timezone
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator, Final

import asyncpg
import jinja2
import plotly.graph_objects as go
from pydantic import BaseModel, ConfigDict, Field
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from core.config import AppSettings, EmailSettings, GuardianSettings, get_settings
from core.http_client import ResilientHttpClient
from core.logger import get_logger
from core.timescale_client import connect_with_backoff
from src.execution.billing_schema import CLIENT_FILLS_TABLE, get_client_monthly_pnl

if TYPE_CHECKING:
    # apscheduler is only imported for real inside schedule_monthly_reports
    # (not at module level) -- this module is usable without apscheduler
    # installed by anything that doesn't call that one function.
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

_log = get_logger(__name__)

# -- Brand: reuses src/dashboard/theme.py's real, already-established
# Nothierly palette verbatim -- one visual identity across the dashboard
# and this report, not a second one invented for PDFs.
_MINT: Final[str] = "#00E676"
_CRIMSON: Final[str] = "#FF1744"
_INDIGO: Final[str] = "#2962FF"
_CYAN: Final[str] = "#00E5FF"
_VOID_BLACK: Final[str] = "#0a0f18"
_TEXT_DIM: Final[str] = "#5B6B7C"

def _format_signed_usd(value: float) -> str:
    """``$1,234.56`` / ``-$1,234.56`` -- sign before the currency symbol, not ``$-1,234.56``."""
    return f"-${abs(value):,.2f}" if value < 0 else f"${value:,.2f}"


_LLM_TIMEOUT_SECONDS: Final[float] = 20.0
_LLM_MAX_TOKENS: Final[int] = 300
_DEFAULT_MODEL_BY_PROVIDER: Final[dict[str, str]] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-20241022",
}
_SUMMARY_SYSTEM_PROMPT: Final[str] = (
    "Tu es l'IA d'Exécution institutionnelle de Nothierly. On te donne les métriques réelles "
    "d'un client pour le mois écoulé. Écris exactement 3 phrases professionnelles en français, "
    "à destination du client, résumant le mois : le contexte de marché si pertinent, ce que "
    "Nothierly a concrètement fait pour lui (exécutions, edge/slippage évité en dollars), et "
    "un ton institutionnel sobre. Ne jamais inventer un chiffre qui n'est pas dans les données "
    "fournies -- si une métrique est indisponible, ne la mentionne simplement pas. Si un résultat "
    "déclaré par le client (gross PnL) et un pourcentage d'edge sont fournis, tu peux les relier "
    "honnêtement (ex: \"votre résultat du mois reflète en partie une perte de marché indépendante "
    "de notre exécution ; l'edge Nothierly représente X% de ce résultat\") -- mais ne prétends "
    "jamais connaître ou avoir simulé ce que le résultat total du client aurait été sans Nothierly : "
    "tu ne mesures que la composante d'exécution, pas l'ensemble de sa stratégie."
)


class ClientMonthlyMetrics(BaseModel):
    """One client's real, aggregated metrics for one month -- see module docstring on why some fields are honestly ``None``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    client_id: str
    period_start: date
    period_end: date
    is_demo: bool = False

    fill_count: int
    notional_executed_usd: float
    nothierly_edge_usd: float
    fees_paid_usd: float
    exposure_by_instrument: dict[str, float] = Field(
        default_factory=dict, description="Real fill count per instrument, for the exposure chart."
    )
    daily_cumulative_edge_usd: list[tuple[date, float]] = Field(
        default_factory=list, description="Real cumulative Nothierly Edge saved, by day -- the report's PnL-style curve."
    )

    gross_pnl_usd: float | None = Field(
        default=None,
        description=(
            "None unless the client themselves reported it via POST /api/v1/clients/{id}/monthly-pnl -- "
            "this system never takes directional positions for a client and has no other way to know it."
        ),
    )
    net_pnl_usd: float | None = Field(
        default=None, description="gross_pnl_usd - fees_paid_usd, only when gross_pnl_usd is real (client-reported)."
    )
    edge_as_pct_of_pnl: float | None = Field(
        default=None,
        description=(
            "nothierly_edge_usd as a % of |gross_pnl_usd|, only when the client reported one. The real, "
            "defensible churn-reduction number: not a claim about the client's full alternate-universe "
            "result (this system doesn't simulate that), only the real execution-slippage delta it "
            "actually controls and measures -- see module docstring."
        ),
    )
    max_drawdown_pct: float | None = None
    sharpe_ratio: float | None = None
    data_availability_note: str = Field(
        default=(
            "Gross/Net PnL nécessitent que le client déclare son propre résultat (POST "
            "/api/v1/clients/{id}/monthly-pnl) -- Nothierly ne prend jamais de position directionnelle "
            "pour le compte du client et n'a aucun autre moyen de le connaître. Max Drawdown et Sharpe "
            "Ratio nécessiteraient en plus un historique de position prix par prix, indisponible même "
            "avec un PnL déclaré."
        )
    )


# =============================================================================
# Rule 1: real, memory-bounded aggregation against client_fills
# =============================================================================


async def _list_active_clients(pool: asyncpg.Pool, *, period_start: datetime, period_end: datetime) -> list[str]:
    """Real, small result set: which distinct clients have any real fill in the period -- never all raw rows."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"SELECT DISTINCT client_id FROM {CLIENT_FILLS_TABLE} WHERE time >= $1 AND time < $2",
            period_start, period_end,
        )
    return [str(row["client_id"]) for row in rows]


async def aggregate_client_month(
    pool: asyncpg.Pool, client_id: str, *, year: int, month: int
) -> ClientMonthlyMetrics:
    """Real, server-side aggregation for one client's one calendar month -- never loads raw fills into Python.

    Args:
        pool: A connected pool (:func:`core.timescale_client.connect_with_backoff`).
        client_id: Which client.
        year: Calendar year.
        month: Calendar month (1-12).

    Returns:
        Real metrics computed from whatever ``client_fills`` rows exist
        for this client in the period -- honestly all-zero if none do,
        never fabricated.
    """
    period_start = datetime(year, month, 1, tzinfo=timezone.utc)
    period_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) if month == 12 else datetime(year, month + 1, 1, tzinfo=timezone.utc)

    async with pool.acquire() as conn:
        summary = await conn.fetchrow(
            f"""
            SELECT
                COUNT(*)                                   AS fill_count,
                COALESCE(SUM(quantity * price), 0)          AS notional_executed_usd,
                COALESCE(SUM(slippage_saved_usd), 0)        AS nothierly_edge_usd,
                COALESCE(SUM(fee_usd), 0)                   AS fees_paid_usd
            FROM {CLIENT_FILLS_TABLE}
            WHERE client_id = $1 AND time >= $2 AND time < $3
            """,
            client_id, period_start, period_end,
        )
        exposure_rows = await conn.fetch(
            f"""
            SELECT instrument, COUNT(*) AS n
            FROM {CLIENT_FILLS_TABLE}
            WHERE client_id = $1 AND time >= $2 AND time < $3
            GROUP BY instrument
            """,
            client_id, period_start, period_end,
        )
        daily_rows = await conn.fetch(
            f"""
            SELECT time::date AS day, SUM(slippage_saved_usd) AS edge
            FROM {CLIENT_FILLS_TABLE}
            WHERE client_id = $1 AND time >= $2 AND time < $3
            GROUP BY time::date ORDER BY day ASC
            """,
            client_id, period_start, period_end,
        )

    cumulative = 0.0
    daily_cumulative: list[tuple[date, float]] = []
    for row in daily_rows:
        cumulative += float(row["edge"] or 0.0)
        daily_cumulative.append((row["day"], round(cumulative, 2)))

    fees_paid_usd = round(float(summary["fees_paid_usd"]), 2) if summary else 0.0
    nothierly_edge_usd = round(float(summary["nothierly_edge_usd"]), 2) if summary else 0.0

    # Real, only when the client themselves reported it -- see
    # ClientMonthlyMetrics' own docstring/field descriptions for why this
    # is the one honest way this system can ever know a client's outcome.
    gross_pnl_usd = await get_client_monthly_pnl(pool, client_id=client_id, year=year, month=month)
    net_pnl_usd: float | None = None
    edge_as_pct_of_pnl: float | None = None
    if gross_pnl_usd is not None:
        net_pnl_usd = round(gross_pnl_usd - fees_paid_usd, 2)
        if gross_pnl_usd != 0:
            edge_as_pct_of_pnl = round(nothierly_edge_usd / abs(gross_pnl_usd) * 100.0, 1)

    data_availability_note = ClientMonthlyMetrics.model_fields["data_availability_note"].default
    if gross_pnl_usd is not None:
        data_availability_note = (
            "Gross/Net PnL basés sur le résultat déclaré par le client lui-même -- Nothierly ne prend "
            "jamais de position directionnelle et ne peut pas calculer ce chiffre de façon indépendante. "
            "Max Drawdown et Sharpe Ratio resteraient indisponibles même avec un PnL déclaré (nécessitent "
            "un historique de position prix par prix)."
        )

    return ClientMonthlyMetrics(
        client_id=client_id,
        period_start=period_start.date(),
        period_end=(period_end - timedelta(days=1)).date(),
        fill_count=int(summary["fill_count"]) if summary else 0,
        notional_executed_usd=round(float(summary["notional_executed_usd"]), 2) if summary else 0.0,
        nothierly_edge_usd=nothierly_edge_usd,
        fees_paid_usd=fees_paid_usd,
        exposure_by_instrument={str(r["instrument"]): int(r["n"]) for r in exposure_rows},
        daily_cumulative_edge_usd=daily_cumulative,
        gross_pnl_usd=gross_pnl_usd,
        net_pnl_usd=net_pnl_usd,
        edge_as_pct_of_pnl=edge_as_pct_of_pnl,
        data_availability_note=data_availability_note,
    )


async def generate_monthly_reports(
    pool: asyncpg.Pool, *, year: int, month: int
) -> AsyncIterator[ClientMonthlyMetrics]:
    """Yield one real, aggregated report per active client -- one client's small result in memory at a time.

    This is what actually satisfies "gérer des centaines de clients en
    parallèle sans bloquer la mémoire": the heavy lifting (summing
    potentially many raw fills) happens inside each ``SELECT``, in
    Postgres, not in this process -- what this generator ever holds is
    one client ID, then one already-aggregated
    :class:`ClientMonthlyMetrics` at a time, discarded before the next.
    """
    period_start = datetime(year, month, 1, tzinfo=timezone.utc)
    period_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) if month == 12 else datetime(year, month + 1, 1, tzinfo=timezone.utc)
    client_ids = await _list_active_clients(pool, period_start=period_start, period_end=period_end)
    for client_id in client_ids:
        yield await aggregate_client_month(pool, client_id, year=year, month=month)


def generate_demo_metrics(client_id: str, *, year: int, month: int) -> ClientMonthlyMetrics:
    """A clearly-labeled (``is_demo=True``) illustrative report for B2B sales demos -- never silently real.

    Same discipline as ``src/dashboard/ultimate_terminal.py``'s
    ``generate_5b_demo_data``: every consumer of this (the PDF renderer,
    the email summary) must check ``is_demo`` and visibly label the
    output as a demonstration, not present it as this client's real month.
    """
    rng = random.Random(hash(client_id) & 0xFFFFFFFF)
    period_start = date(year, month, 1)
    period_end = ((datetime(year, month, 1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)).date()
    days_in_month = (period_end - period_start).days + 1

    daily_cumulative: list[tuple[date, float]] = []
    cumulative = 0.0
    for day_offset in range(days_in_month):
        cumulative += max(0.0, rng.gauss(410.0, 180.0))
        daily_cumulative.append((period_start + timedelta(days=day_offset), round(cumulative, 2)))

    return ClientMonthlyMetrics(
        client_id=client_id, period_start=period_start, period_end=period_end, is_demo=True,
        fill_count=rng.randint(800, 2400),
        notional_executed_usd=round(rng.uniform(1_500_000, 6_000_000), 2),
        nothierly_edge_usd=round(cumulative, 2),
        fees_paid_usd=round(rng.uniform(400, 1800), 2),
        exposure_by_instrument={"BTC/USDT": rng.randint(300, 900), "ETH/USDT": rng.randint(150, 500), "EUR_USD": rng.randint(50, 300)},
        daily_cumulative_edge_usd=daily_cumulative,
        data_availability_note="Rapport de démonstration -- données entièrement simulées, aucune activité réelle.",
    )


# =============================================================================
# Rule 2: charts (plotly + kaleido, real data or an honest empty state)
# =============================================================================


def render_cumulative_edge_chart(metrics: ClientMonthlyMetrics) -> bytes:
    """Real cumulative Nothierly Edge (slippage saved) over the month -- this report's PnL-style curve, honestly labeled for what it is.

    Not a PnL curve (see module docstring on why this system has no
    client PnL to chart) -- the cumulative dollar value of execution
    quality delivered, which is the real thing this business measures.
    """
    if not metrics.daily_cumulative_edge_usd:
        fig = go.Figure()
        fig.add_annotation(text="Aucune exécution ce mois-ci", showarrow=False, font=dict(size=16, color=_TEXT_DIM))
    else:
        dates = [d for d, _ in metrics.daily_cumulative_edge_usd]
        values = [v for _, v in metrics.daily_cumulative_edge_usd]
        fig = go.Figure(
            data=[go.Scatter(x=dates, y=values, mode="lines", line=dict(color=_CYAN, width=3), fill="tozeroy", fillcolor="rgba(0,229,255,0.12)")]
        )
    fig.update_layout(
        title="Nothierly Edge cumulé ($) -- slippage évité", paper_bgcolor=_VOID_BLACK, plot_bgcolor=_VOID_BLACK,
        font=dict(color="#E8EAED"), width=620, height=340, margin=dict(l=50, r=30, t=50, b=40),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"), yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
    )
    result: bytes = fig.to_image(format="png")
    return result


def render_exposure_pie_chart(metrics: ClientMonthlyMetrics) -> bytes:
    """Real per-instrument execution-count exposure -- computed from real ``client_fills`` rows, or an honest empty state."""
    if not metrics.exposure_by_instrument:
        fig = go.Figure()
        fig.add_annotation(text="Aucune exécution ce mois-ci", showarrow=False, font=dict(size=16, color=_TEXT_DIM))
    else:
        labels = list(metrics.exposure_by_instrument.keys())
        values = list(metrics.exposure_by_instrument.values())
        palette = [_CYAN, _MINT, _INDIGO, _CRIMSON, "#8892a0"]
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=palette), hole=0.45)])
    fig.update_layout(
        title="Répartition des exécutions par actif", paper_bgcolor=_VOID_BLACK,
        font=dict(color="#E8EAED"), width=620, height=340, margin=dict(l=30, r=30, t=50, b=30),
    )
    result: bytes = fig.to_image(format="png")
    return result


# =============================================================================
# Rule 2: institutional PDF (ReportLab Platypus -- see module docstring on why not WeasyPrint)
# =============================================================================


def render_report_pdf(metrics: ClientMonthlyMetrics, *, summary_text: str) -> bytes:
    """Render one real, white-label institutional PDF report -- Nothierly's real brand palette (src/dashboard/theme.py), not a new one.

    Charts are embedded directly from in-memory PNG bytes
    (:class:`reportlab.platypus.Image` accepts a file-like object) --
    never written to a temporary file on disk, which the brief's own
    "images png temporaires" phrasing implies but this avoids entirely:
    no cleanup step, no risk of an orphaned temp file on a crashed run.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, topMargin=20 * mm, bottomMargin=20 * mm, leftMargin=20 * mm, rightMargin=20 * mm,
        title=f"Nothierly -- Rapport {metrics.client_id} {metrics.period_start:%Y-%m}",
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("NothierlyTitle", parent=styles["Title"], textColor=colors.HexColor(_CYAN), fontName="Helvetica-Bold", fontSize=22)
    heading_style = ParagraphStyle("NothierlyHeading", parent=styles["Heading2"], textColor=colors.HexColor(_INDIGO), spaceAfter=4)
    body_style = ParagraphStyle("NothierlyBody", parent=styles["BodyText"], textColor=colors.HexColor("#1a1f28"), leading=15)
    dim_style = ParagraphStyle("NothierlyDim", parent=styles["BodyText"], textColor=colors.HexColor(_TEXT_DIM), fontSize=8, leading=11)
    demo_style = ParagraphStyle("NothierlyDemo", parent=styles["BodyText"], textColor=colors.HexColor(_CRIMSON), fontName="Helvetica-Bold", fontSize=12)

    elements: list[object] = []
    if metrics.is_demo:
        elements.append(Paragraph("RAPPORT DE DÉMONSTRATION — DONNÉES ENTIÈREMENT SIMULÉES", demo_style))
        elements.append(Spacer(1, 8 * mm))

    elements.append(Paragraph("NOTHIERLY", title_style))
    elements.append(Paragraph(f"Rapport Mensuel d'Exécution — Client {metrics.client_id}", heading_style))
    elements.append(Paragraph(f"Période : {metrics.period_start:%d %B %Y} — {metrics.period_end:%d %B %Y}", dim_style))
    elements.append(Spacer(1, 6 * mm))
    elements.append(Paragraph(summary_text.replace("\n", "<br/>"), body_style))
    elements.append(Spacer(1, 6 * mm))

    table_data = [
        ["Métrique", "Valeur"],
        ["Nombre d'exécutions", str(metrics.fill_count)],
        ["Notionnel exécuté", f"${metrics.notional_executed_usd:,.2f}"],
        ["Nothierly Edge (slippage évité)", f"${metrics.nothierly_edge_usd:,.2f}"],
        ["Frais payés", f"${metrics.fees_paid_usd:,.2f}"],
    ]
    if metrics.gross_pnl_usd is not None:
        table_data.append(["Résultat déclaré par le client (Gross PnL)", _format_signed_usd(metrics.gross_pnl_usd)])
    if metrics.net_pnl_usd is not None:
        table_data.append(["Résultat net (après frais Nothierly)", _format_signed_usd(metrics.net_pnl_usd)])
    if metrics.edge_as_pct_of_pnl is not None:
        table_data.append(["Nothierly Edge en % du résultat déclaré", f"{metrics.edge_as_pct_of_pnl:.1f}%"])
    table = Table(table_data, colWidths=[100 * mm, 60 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(_VOID_BLACK)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(_CYAN)),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7fa")]),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    elements.append(table)
    elements.append(Spacer(1, 8 * mm))

    elements.append(Image(io.BytesIO(render_cumulative_edge_chart(metrics)), width=160 * mm, height=88 * mm))
    elements.append(Spacer(1, 4 * mm))
    elements.append(Image(io.BytesIO(render_exposure_pie_chart(metrics)), width=160 * mm, height=88 * mm))
    elements.append(Spacer(1, 8 * mm))
    elements.append(Paragraph(metrics.data_availability_note, dim_style))

    def _draw_footer(canvas: Canvas, doc: SimpleDocTemplate) -> None:
        """"Généré par Nothierly AI" on every page -- the real "preuve de travail" branding, not just body text once.

        A real reader forwarding this PDF to another fund manager sees
        it on whichever page they screenshot or print, not only if they
        happen to keep page 1.
        """
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor(_TEXT_DIM))
        canvas.drawString(20 * mm, 12 * mm, "Généré par Nothierly AI — Execution-as-a-Service")
        canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, f"Page {doc.page}")
        canvas.restoreState()

    doc.build(elements, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    return buffer.getvalue()


# =============================================================================
# Rule 3: real, single-shot LLM summary -- see module docstring for why not
# run_copilot_turn/chat_with_nothierly (both conversation/tool-shaped, wrong fit)
# =============================================================================


def _fallback_summary_text(metrics: ClientMonthlyMetrics) -> str:
    """Honest, deterministic stand-in when no LLM provider is configured -- same convention as quant_copilot._fallback_reply."""
    pnl_sentence = ""
    if metrics.gross_pnl_usd is not None and metrics.edge_as_pct_of_pnl is not None:
        outcome_word = "perte" if metrics.gross_pnl_usd < 0 else "gain"
        pnl_sentence = (
            f" Votre résultat déclaré ce mois-ci était une {outcome_word} de ${abs(metrics.gross_pnl_usd):,.2f} ; "
            f"l'edge d'exécution Nothierly représente {metrics.edge_as_pct_of_pnl:.1f}% de ce montant."
        )
    return (
        f"Cher client, ci-joint votre rapport d'exécution pour la période du {metrics.period_start:%d/%m/%Y} "
        f"au {metrics.period_end:%d/%m/%Y}. Nothierly a traité {metrics.fill_count} exécutions réelles pour un "
        f"notionnel de ${metrics.notional_executed_usd:,.2f}, avec un edge d'exécution (slippage évité) de "
        f"${metrics.nothierly_edge_usd:,.2f}.{pnl_sentence} [Résumé généré automatiquement -- aucun LLM configuré "
        "(GUARDIAN_LLM_PROVIDER/GUARDIAN_API_KEY).]"
    )


async def _call_openai_summary(http: ResilientHttpClient, api_key: str, model: str, data_description: str) -> str:
    body = await http.post_json(
        "https://api.openai.com/v1/chat/completions",
        json_body={
            "model": model,
            "messages": [{"role": "system", "content": _SUMMARY_SYSTEM_PROMPT}, {"role": "user", "content": data_description}],
            "max_tokens": _LLM_MAX_TOKENS, "temperature": 0.3,
        },
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    return str(body["choices"][0]["message"]["content"]).strip()


async def _call_anthropic_summary(http: ResilientHttpClient, api_key: str, model: str, data_description: str) -> str:
    body = await http.post_json(
        "https://api.anthropic.com/v1/messages",
        json_body={
            "model": model, "max_tokens": _LLM_MAX_TOKENS, "system": _SUMMARY_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": data_description}],
        },
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
    )
    text_parts = [block["text"] for block in body["content"] if block["type"] == "text"]
    return "\n".join(text_parts).strip()


async def generate_report_summary_text(metrics: ClientMonthlyMetrics, *, settings: AppSettings | None = None) -> str:
    """A real, single-shot LLM summary of ``metrics`` -- never a conversation turn, never tool-calling machinery.

    Falls back to :func:`_fallback_summary_text` (never fabricates a
    number the LLM might hallucinate) when no provider is configured,
    or the real call fails for any reason.
    """
    settings = settings or get_settings()
    guardian_settings: GuardianSettings = settings.guardian
    if guardian_settings.llm_provider == "none" or guardian_settings.api_key is None:
        return _fallback_summary_text(metrics)

    data_description = (
        f"Client: {metrics.client_id}\n"
        f"Période: {metrics.period_start} au {metrics.period_end}\n"
        f"Nombre d'exécutions réelles: {metrics.fill_count}\n"
        f"Notionnel exécuté: ${metrics.notional_executed_usd:,.2f}\n"
        f"Nothierly Edge (slippage évité en $): ${metrics.nothierly_edge_usd:,.2f}\n"
        f"Frais payés: ${metrics.fees_paid_usd:,.2f}\n"
        + (
            f"Résultat déclaré par le client (gross PnL, sa propre donnée, pas la nôtre): {_format_signed_usd(metrics.gross_pnl_usd)}\n"
            f"Edge Nothierly en % de ce résultat: {metrics.edge_as_pct_of_pnl:.1f}%\n"
            if metrics.gross_pnl_usd is not None and metrics.edge_as_pct_of_pnl is not None
            else ""
        )
        + ("(Ceci est un rapport de démonstration avec des données simulées.)\n" if metrics.is_demo else "")
    )
    api_key = guardian_settings.api_key.get_secret_value()
    model = guardian_settings.model or _DEFAULT_MODEL_BY_PROVIDER[guardian_settings.llm_provider]

    try:
        async with ResilientHttpClient(timeout_seconds=_LLM_TIMEOUT_SECONDS) as http:
            if guardian_settings.llm_provider == "openai":
                return await _call_openai_summary(http, api_key, model, data_description)
            return await _call_anthropic_summary(http, api_key, model, data_description)
    except Exception:
        _log.exception("report_summary_llm_call_failed_using_fallback", client_id=metrics.client_id)
        return _fallback_summary_text(metrics)


# =============================================================================
# Rule 3: distribution -- email via smtplib (see module docstring for why not
# a provider SDK), HTML body via Jinja2 (the PDF itself uses ReportLab, not HTML)
# =============================================================================

_EMAIL_HTML_TEMPLATE: Final[str] = """
<html><body style="font-family: Arial, Helvetica, sans-serif; background:#0a0f18; color:#E8EAED; padding:24px;">
  <h2 style="color:#00E5FF; letter-spacing:0.05em;">NOTHIERLY</h2>
  {% if is_demo %}
  <p style="color:#FF1744; font-weight:bold;">RAPPORT DE DÉMONSTRATION — DONNÉES SIMULÉES</p>
  {% endif %}
  <p style="line-height:1.6;">{{ summary_text }}</p>
  <p style="color:#5B6B7C; font-size:12px; margin-top:24px;">Rapport complet en pièce jointe (PDF).</p>
</body></html>
"""


def render_email_html(metrics: ClientMonthlyMetrics, *, summary_text: str) -> str:
    """The report email's HTML body -- Jinja2, autoescaped (``summary_text`` may come from an LLM, never trusted raw)."""
    template = jinja2.Environment(autoescape=True).from_string(_EMAIL_HTML_TEMPLATE)
    return template.render(is_demo=metrics.is_demo, summary_text=summary_text)


async def send_report_email(
    *, to_address: str, metrics: ClientMonthlyMetrics, pdf_bytes: bytes, summary_text: str, settings: AppSettings | None = None
) -> bool:
    """Send one real report email via SMTP, PDF attached. Fails soft -- same contract as ``send_sms_alert``.

    Returns:
        ``True`` if the send call completed without raising, ``False``
        if unconfigured or the call failed. Never raises.
    """
    settings = settings or get_settings()
    email_settings: EmailSettings = settings.email
    if email_settings.smtp_host is None or email_settings.from_address is None:
        _log.warning("email_not_configured_report_not_sent", client_id=metrics.client_id)
        return False

    message = MIMEMultipart("mixed")
    message["Subject"] = f"Nothierly -- Rapport mensuel {metrics.period_start:%B %Y}" + (" (DÉMO)" if metrics.is_demo else "")
    message["From"] = email_settings.from_address
    message["To"] = to_address
    message.attach(MIMEText(render_email_html(metrics, summary_text=summary_text), "html"))
    attachment = MIMEApplication(pdf_bytes, _subtype="pdf")
    attachment.add_header(
        "Content-Disposition", "attachment",
        filename=f"nothierly_report_{metrics.client_id}_{metrics.period_start:%Y_%m}.pdf",
    )
    message.attach(attachment)

    def _send_sync() -> None:
        with smtplib.SMTP(email_settings.smtp_host, email_settings.smtp_port, timeout=15) as server:  # type: ignore[arg-type]
            server.starttls()
            if email_settings.smtp_user is not None and email_settings.smtp_password is not None:
                server.login(email_settings.smtp_user, email_settings.smtp_password.get_secret_value())
            server.send_message(message)

    try:
        # smtplib is synchronous (blocking socket I/O) -- asyncio.to_thread
        # keeps this coroutine's event loop free, same reasoning
        # send_sms_alert already applies to the synchronous Twilio SDK.
        await asyncio.to_thread(_send_sync)
    except Exception:
        _log.exception("report_email_send_failed", client_id=metrics.client_id)
        return False

    _log.info("report_email_sent", client_id=metrics.client_id, to=to_address)
    return True


# =============================================================================
# Rule 3: automation -- APScheduler cron (real, already-used dependency --
# src/data_pipeline/collectors/crypto_collector.py -- not a new Celery dependency)
# =============================================================================

_REPORTS_ARCHIVE_DIR: Final[Path] = Path("reports")


async def run_monthly_reports_job(
    *, year: int | None = None, month: int | None = None, client_email_directory: dict[str, str] | None = None
) -> None:
    """The real cron job body: aggregate, render, archive, and (if a recipient is known) email every active client's report.

    **A real, honest gap, stated plainly**: there is no client contact
    directory anywhere in this project (``MacroOrderRequest.client_id``
    is an opaque string, nothing maps it to an email address) --
    ``client_email_directory`` must be supplied by the caller (e.g. from
    a future client-management table) until one exists. A client with
    no known address still gets a real PDF generated and archived to
    :data:`_REPORTS_ARCHIVE_DIR`, just not emailed -- logged, not
    silently dropped.

    Args:
        year: Defaults to the previous calendar month's year (this job
            is meant to fire on the 1st of the month, when the prior
            month just closed).
        month: Defaults likewise.
        client_email_directory: ``{client_id: email_address}``. Clients
            missing from it are archived but not emailed.
    """
    settings = get_settings()
    if year is None or month is None:
        last_month_end = datetime.now(timezone.utc).replace(day=1) - timedelta(days=1)
        year, month = last_month_end.year, last_month_end.month
    client_email_directory = client_email_directory or {}

    pool = await connect_with_backoff(settings.timescale, max_attempts=3)
    try:
        from src.execution.billing_schema import ensure_schema as ensure_billing_schema

        await ensure_billing_schema(pool)
        processed = 0
        async for metrics in generate_monthly_reports(pool, year=year, month=month):
            await _process_one_client_report(metrics, settings=settings, client_email_directory=client_email_directory)
            processed += 1
        _log.info("monthly_reports_job_completed", year=year, month=month, clients_processed=processed)
    finally:
        await pool.close()


async def _process_one_client_report(
    metrics: ClientMonthlyMetrics, *, settings: AppSettings, client_email_directory: dict[str, str]
) -> None:
    """One client's full Rule-2/Rule-3 pipeline: summarize, render, archive, email if possible."""
    summary_text = await generate_report_summary_text(metrics, settings=settings)
    pdf_bytes = render_report_pdf(metrics, summary_text=summary_text)

    _REPORTS_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = _REPORTS_ARCHIVE_DIR / f"nothierly_report_{metrics.client_id}_{metrics.period_start:%Y_%m}.pdf"
    archive_path.write_bytes(pdf_bytes)

    recipient = client_email_directory.get(metrics.client_id)
    if recipient is None:
        _log.warning("no_email_on_file_report_archived_only", client_id=metrics.client_id, archive_path=str(archive_path))
        return

    await send_report_email(to_address=recipient, metrics=metrics, pdf_bytes=pdf_bytes, summary_text=summary_text, settings=settings)


def schedule_monthly_reports(scheduler: AsyncIOScheduler) -> None:
    """Register the real cron trigger: the 1st of every month at 00:01, exactly per the brief.

    Args:
        scheduler: An already-constructed ``apscheduler.schedulers.
            asyncio.AsyncIOScheduler`` -- this function only registers
            the job, it does not own the scheduler's lifecycle (start/
            shutdown is the caller's responsibility, matching
            :mod:`src.data_pipeline.collectors.crypto_collector`'s own
            established pattern for its own scheduled jobs).
    """
    from apscheduler.triggers.cron import CronTrigger

    scheduler.add_job(
        run_monthly_reports_job, trigger=CronTrigger(day=1, hour=0, minute=1), id="nothierly_monthly_reports", replace_existing=True,
    )
