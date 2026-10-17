"""Reusable, exchange-agnostic network resilience primitives (reconnect/backoff, backpressure).

Nothing under this package talks to a specific broker/exchange API --
see ``src/agents/market_data/exchange_client.py`` for the real,
ccxt.pro-based Binance client. This package exists for infrastructure
that any WebSocket consumer in this project could reuse.
"""
