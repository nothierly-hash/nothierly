"""run_backend: the forex (OANDA) and stock (Alpaca) price feeds are opt-in, read-only, and fail fast on bad or fake credentials (never a silent no-op, never an echoed secret)."""

from __future__ import annotations

import pytest

from scripts import run_backend as rb
from src.agents.market_data.quote_agent import AlpacaQuoteSource, OandaQuoteSource

REAL_OANDA = {"oanda_account_id": "101-001-1234567-001", "oanda_token": "a" * 32 + "-" + "b" * 32}


def test_nothing_starts_by_default_even_with_credentials_present() -> None:
    assert rb._resolve_quote_sources({}, **REAL_OANDA) == []
    assert rb._resolve_quote_sources({"ALPACA_API_KEY": "k" * 20, "ALPACA_API_SECRET": "s" * 40}) == []  # les cles seules ne demarrent rien : il faut choisir le flux


def test_the_oanda_feed_starts_on_the_forex_instruments_with_real_credentials() -> None:
    (source,) = rb._resolve_quote_sources({"NOTHIERLY_FOREX_FEED": "OANDA"}, **REAL_OANDA, oanda_environment="practice")
    assert isinstance(source, OandaQuoteSource)
    assert source._params == {"instruments": "EUR_USD,XAU_USD"} and "fxpractice" in source._url and "101-001-1234567-001" in source._url


@pytest.mark.parametrize("account,token", [("", "token"), ("acc", ""), ("shadow-run-placeholder", "x" * 30), ("101-001-1-1", "shadow-run-placeholder"), ("test-placeholder", "test-placeholder")])
def test_the_oanda_feed_refuses_empty_or_placeholder_credentials(account: str, token: str) -> None:
    with pytest.raises(ValueError, match="OANDA_ACCOUNT_ID"):
        rb._resolve_quote_sources({"NOTHIERLY_FOREX_FEED": "oanda"}, oanda_account_id=account, oanda_token=token)


def test_the_alpaca_feed_uses_defaults_or_the_given_symbols_and_never_echoes_secrets() -> None:
    env = {"NOTHIERLY_STOCK_FEED": "alpaca", "ALPACA_API_KEY": "PKREALKEY12345678", "ALPACA_API_SECRET": "REALSECRET" + "x" * 30}
    (default,) = rb._resolve_quote_sources(env)
    assert isinstance(default, AlpacaQuoteSource) and default._subscribe == '{"action": "subscribe", "quotes": ["AAPL", "MSFT", "SPY"]}'
    (custom,) = rb._resolve_quote_sources({**env, "NOTHIERLY_STOCK_SYMBOLS": "nvda, tsla ,brk.b"})
    assert isinstance(custom, AlpacaQuoteSource)
    assert '"NVDA", "TSLA", "BRK.B"' in custom._subscribe
    with pytest.raises(ValueError) as caught:
        rb._resolve_quote_sources({**env, "ALPACA_API_SECRET": "shadow-run-placeholder"})
    assert "REALSECRET" not in str(caught.value) and "PKREALKEY" not in str(caught.value)


@pytest.mark.parametrize("symbols", ["AAPL, 12", "A$PL", "TOOLONGSYMBOL"])
def test_unreadable_stock_symbols_are_refused(symbols: str) -> None:
    env = {"NOTHIERLY_STOCK_FEED": "alpaca", "ALPACA_API_KEY": "k" * 20, "ALPACA_API_SECRET": "s" * 40, "NOTHIERLY_STOCK_SYMBOLS": symbols}
    with pytest.raises(ValueError, match="NOTHIERLY_STOCK_SYMBOLS"):
        rb._resolve_quote_sources(env)


def test_unknown_feed_names_and_missing_alpaca_keys_are_refused() -> None:
    with pytest.raises(ValueError, match="seule la valeur 'oanda'"):
        rb._resolve_quote_sources({"NOTHIERLY_FOREX_FEED": "fxcm"}, **REAL_OANDA)
    with pytest.raises(ValueError, match="seule la valeur 'alpaca'"):
        rb._resolve_quote_sources({"NOTHIERLY_STOCK_FEED": "polygon"})
    with pytest.raises(ValueError, match="ALPACA_API_KEY"):
        rb._resolve_quote_sources({"NOTHIERLY_STOCK_FEED": "alpaca"})


def test_both_feeds_can_run_together() -> None:
    env = {"NOTHIERLY_FOREX_FEED": "oanda", "NOTHIERLY_STOCK_FEED": "alpaca", "ALPACA_API_KEY": "k" * 20, "ALPACA_API_SECRET": "s" * 40}
    sources = rb._resolve_quote_sources(env, **REAL_OANDA)
    assert [type(s).__name__ for s in sources] == ["OandaQuoteSource", "AlpacaQuoteSource"]
