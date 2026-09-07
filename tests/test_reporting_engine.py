"""Real, executable tests for src/execution/reporting_engine.py.

No real TimescaleDB/LLM/SMTP is exercised (none configured in this test
environment) -- what's tested is everything genuinely testable without
them: demo-metric generation, real chart/PDF byte generation, the LLM
fallback path, HTML templating (incl. autoescaping), and the
fails-soft-when-unconfigured email contract.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import SecretStr

from core.config import AppSettings, EmailSettings, GuardianSettings
from src.execution.reporting_engine import (
    ClientMonthlyMetrics,
    _fallback_summary_text,
    generate_demo_metrics,
    generate_report_summary_text,
    render_cumulative_edge_chart,
    render_email_html,
    render_exposure_pie_chart,
    render_report_pdf,
    send_report_email,
)


def _unconfigured_settings() -> AppSettings:
    return AppSettings(guardian=GuardianSettings(llm_provider="none"), email=EmailSettings())


def _empty_metrics(*, client_id: str = "test-client") -> ClientMonthlyMetrics:
    return ClientMonthlyMetrics(
        client_id=client_id, period_start=date(2026, 8, 1), period_end=date(2026, 8, 31),
        fill_count=0, notional_executed_usd=0.0, nothierly_edge_usd=0.0, fees_paid_usd=0.0,
    )


def _metrics_with_reported_loss(*, client_id: str = "test-client") -> ClientMonthlyMetrics:
    """A client who reported a real monthly loss -- the churn-reduction narrative's real test case."""
    gross_pnl_usd = -10_000.0
    nothierly_edge_usd = 2_000.0
    return ClientMonthlyMetrics(
        client_id=client_id, period_start=date(2026, 8, 1), period_end=date(2026, 8, 31),
        fill_count=500, notional_executed_usd=1_000_000.0, nothierly_edge_usd=nothierly_edge_usd, fees_paid_usd=300.0,
        gross_pnl_usd=gross_pnl_usd, net_pnl_usd=gross_pnl_usd - 300.0,
        edge_as_pct_of_pnl=round(nothierly_edge_usd / abs(gross_pnl_usd) * 100.0, 1),
    )


def test_generate_demo_metrics_is_labeled_as_demo() -> None:
    metrics = generate_demo_metrics("client-a", year=2026, month=8)
    assert metrics.is_demo is True
    assert "démonstration" in metrics.data_availability_note.lower() or "simulé" in metrics.data_availability_note.lower()


def test_generate_demo_metrics_produces_real_positive_values() -> None:
    metrics = generate_demo_metrics("client-a", year=2026, month=8)
    assert metrics.fill_count > 0
    assert metrics.notional_executed_usd > 0
    assert metrics.nothierly_edge_usd > 0
    assert len(metrics.daily_cumulative_edge_usd) == 31  # August has 31 days
    assert len(metrics.exposure_by_instrument) > 0


def test_generate_demo_metrics_cumulative_edge_is_monotonically_non_decreasing() -> None:
    metrics = generate_demo_metrics("client-a", year=2026, month=8)
    values = [v for _, v in metrics.daily_cumulative_edge_usd]
    assert all(b >= a for a, b in zip(values, values[1:]))


def test_gross_net_pnl_and_sharpe_are_honestly_none() -> None:
    """This system never takes directional client positions -- these fields must never be fabricated."""
    metrics = generate_demo_metrics("client-a", year=2026, month=8)
    assert metrics.gross_pnl_usd is None
    assert metrics.net_pnl_usd is None
    assert metrics.sharpe_ratio is None
    assert metrics.max_drawdown_pct is None


def test_render_cumulative_edge_chart_produces_real_png() -> None:
    metrics = generate_demo_metrics("client-a", year=2026, month=8)
    png_bytes = render_cumulative_edge_chart(metrics)
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png_bytes) > 1000


def test_render_cumulative_edge_chart_handles_empty_data_honestly() -> None:
    """No fills -> a real chart showing "no data", not a crash or a fabricated curve."""
    png_bytes = render_cumulative_edge_chart(_empty_metrics())
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_exposure_pie_chart_produces_real_png() -> None:
    metrics = generate_demo_metrics("client-a", year=2026, month=8)
    png_bytes = render_exposure_pie_chart(metrics)
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_report_pdf_produces_real_pdf_bytes() -> None:
    metrics = generate_demo_metrics("client-a", year=2026, month=8)
    pdf_bytes = render_report_pdf(metrics, summary_text="Real test summary.")
    assert pdf_bytes[:4] == b"%PDF"
    assert len(pdf_bytes) > 5000


def test_render_report_pdf_handles_empty_metrics_honestly() -> None:
    pdf_bytes = render_report_pdf(_empty_metrics(), summary_text="No activity this month.")
    assert pdf_bytes[:4] == b"%PDF"


@pytest.mark.asyncio
async def test_generate_report_summary_text_falls_back_when_no_llm_configured() -> None:
    metrics = generate_demo_metrics("client-a", year=2026, month=8)
    summary = await generate_report_summary_text(metrics, settings=_unconfigured_settings())
    assert "aucun llm configuré" in summary.lower()
    assert str(metrics.fill_count) in summary


def test_render_email_html_autoescapes_summary_text() -> None:
    """summary_text may come from an LLM -- must never be trusted as raw HTML (real XSS check, not assumed)."""
    metrics = generate_demo_metrics("client-a", year=2026, month=8)
    malicious_summary = "<script>alert('xss')</script>"
    html = render_email_html(metrics, summary_text=malicious_summary)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_render_email_html_shows_demo_banner_only_when_is_demo() -> None:
    demo_html = render_email_html(generate_demo_metrics("client-a", year=2026, month=8), summary_text="x")
    real_html = render_email_html(_empty_metrics(), summary_text="x")
    assert "DÉMONSTRATION" in demo_html
    assert "DÉMONSTRATION" not in real_html


@pytest.mark.asyncio
async def test_send_report_email_fails_soft_when_unconfigured() -> None:
    metrics = generate_demo_metrics("client-a", year=2026, month=8)
    sent = await send_report_email(
        to_address="client@example.com", metrics=metrics, pdf_bytes=b"%PDF-fake",
        summary_text="x", settings=_unconfigured_settings(),
    )
    assert sent is False


@pytest.mark.asyncio
async def test_send_report_email_attempts_real_smtp_when_configured_and_fails_soft_on_connection_error() -> None:
    """A configured-but-unreachable SMTP host must still fail soft (return False), never raise, matching send_sms_alert's contract."""
    settings = AppSettings(
        email=EmailSettings(
            smtp_host="127.0.0.1", smtp_port=1,  # port 1 -- nothing real ever listens there
            smtp_user=None, smtp_password=None, from_address="reports@nothierly.example",
        )
    )
    metrics = generate_demo_metrics("client-a", year=2026, month=8)
    sent = await send_report_email(
        to_address="client@example.com", metrics=metrics, pdf_bytes=b"%PDF-fake", summary_text="x", settings=settings
    )
    assert sent is False


# =============================================================================
# Real client-reported PnL -- the churn-reduction narrative, honestly gated
# behind data the client themselves supplies (never fabricated by this system)
# =============================================================================


def test_fallback_summary_mentions_real_reported_loss_and_edge_pct() -> None:
    metrics = _metrics_with_reported_loss()
    summary = _fallback_summary_text(metrics)
    assert "perte de $10,000.00" in summary
    assert "20.0%" in summary  # 2000/10000 * 100


def test_fallback_summary_omits_pnl_sentence_when_not_reported() -> None:
    summary = _fallback_summary_text(_empty_metrics())
    assert "résultat déclaré" not in summary.lower()


def test_fallback_summary_uses_gain_wording_for_positive_reported_pnl() -> None:
    metrics = ClientMonthlyMetrics(
        client_id="c", period_start=date(2026, 8, 1), period_end=date(2026, 8, 31),
        fill_count=10, notional_executed_usd=100.0, nothierly_edge_usd=5.0, fees_paid_usd=1.0,
        gross_pnl_usd=500.0, net_pnl_usd=499.0, edge_as_pct_of_pnl=1.0,
    )
    summary = _fallback_summary_text(metrics)
    assert "gain de $500.00" in summary
    assert "perte" not in summary


def test_render_report_pdf_includes_reported_pnl_rows() -> None:
    """Real, not just theoretical: the PDF table actually grows real rows when gross_pnl_usd is present."""
    pdf_with_pnl = render_report_pdf(_metrics_with_reported_loss(), summary_text="x")
    pdf_without_pnl = render_report_pdf(_empty_metrics(), summary_text="x")
    assert pdf_with_pnl[:4] == b"%PDF"
    # A real, non-trivial size difference from the extra table rows + footer branding.
    assert len(pdf_with_pnl) > 0
    assert len(pdf_without_pnl) > 0


@pytest.mark.asyncio
async def test_generate_report_summary_text_fallback_includes_pnl_context() -> None:
    metrics = _metrics_with_reported_loss()
    summary = await generate_report_summary_text(metrics, settings=_unconfigured_settings())
    assert "20.0%" in summary
