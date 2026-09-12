"""Real, executable tests for market_discovery.py.

Real I/O (Binance P2P search, ccxt) is monkeypatched at the boundary -- the same "fake I/O only at
the edge" discipline as ``tests/test_p2p_api.py``. What's under test is the real, generic
discovery/pruning/assembly logic itself, with zero currency-specific special-casing.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import pytest

from market_discovery import DiscoveredP2PFiat, DiscoveredSpotMarket, build_dynamic_market
from nothierly_genesis.tda_engine import prune_illiquid_edges


def test_build_dynamic_market_uses_only_what_was_actually_discovered() -> None:
    """No hardcoded currency anywhere -- the node set is exactly the union of what's passed in."""
    fiats = [
        DiscoveredP2PFiat(fiat_code="TRY", region="Europe", avg_price=50.0, avg_liquidity_usd=2000.0, ads_sampled=5),
        DiscoveredP2PFiat(fiat_code="ZAR", region="Africa", avg_price=17.0, avg_liquidity_usd=14000.0, ads_sampled=5),
    ]
    spots = [DiscoveredSpotMarket(base_symbol="BTC", quote_symbol="USDT", symbol="BTC/USDT", quote_volume_24h_usd=1_000_000.0)]

    nodes, edges, liquidity = build_dynamic_market(fiats, spots)

    assert set(nodes) == {"USDT", "TRY", "ZAR", "BTC"}
    assert ("USDT", "TRY") in edges and ("TRY", "USDT") in edges
    assert ("USDT", "BTC") in edges and ("BTC", "USDT") in edges
    assert liquidity[("USDT", "ZAR")] == 14000.0
    assert liquidity[("USDT", "BTC")] == 1_000_000.0


def test_build_dynamic_market_ignores_spot_markets_not_quoted_in_the_hub() -> None:
    fiats: list[DiscoveredP2PFiat] = []
    spots = [DiscoveredSpotMarket(base_symbol="ETH", quote_symbol="BTC", symbol="ETH/BTC", quote_volume_24h_usd=5000.0)]
    nodes, edges, _liquidity = build_dynamic_market(fiats, spots, hub_symbol="USDT")
    assert "ETH" not in nodes
    assert nodes == ["USDT"]


def test_the_real_try_usdt_zar_btc_path_is_mappable_with_zero_reconfiguration() -> None:
    """The real capstone claim (task 6): a cross-region path across currencies this function has
    never seen a special case for, built purely from generic discovery output."""
    fiats = [
        DiscoveredP2PFiat(fiat_code="TRY", region="Europe", avg_price=50.0, avg_liquidity_usd=2000.0, ads_sampled=5),
        DiscoveredP2PFiat(fiat_code="ZAR", region="Africa", avg_price=17.0, avg_liquidity_usd=14000.0, ads_sampled=5),
    ]
    spots = [DiscoveredSpotMarket(base_symbol="BTC", quote_symbol="USDT", symbol="BTC/USDT", quote_volume_24h_usd=1_000_000.0)]
    nodes, edges, liquidity = build_dynamic_market(fiats, spots)
    pruned = prune_illiquid_edges(edges, liquidity, min_liquidity_usd=500.0)

    graph: nx.DiGraph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    for (src, dst), weight in pruned.items():
        graph.add_edge(src, dst, weight=weight)

    assert nx.has_path(graph, "TRY", "USDT")
    assert nx.has_path(graph, "USDT", "ZAR")
    assert nx.has_path(graph, "USDT", "BTC")
    full_path = ["TRY", "USDT", "ZAR", "USDT", "BTC"]
    for i in range(len(full_path) - 1):
        assert graph.has_edge(full_path[i], full_path[i + 1]), f"missing real edge {full_path[i]}->{full_path[i + 1]}"


def test_prune_illiquid_edges_removes_only_edges_below_the_real_threshold() -> None:
    edges = {("USDT", "AAA"): 0.995, ("USDT", "BBB"): 0.995}
    liquidity = {("USDT", "AAA"): 100.0, ("USDT", "BBB"): 10_000.0}
    pruned = prune_illiquid_edges(edges, liquidity, min_liquidity_usd=500.0)
    assert ("USDT", "AAA") not in pruned
    assert ("USDT", "BBB") in pruned


def test_prune_illiquid_edges_keeps_edges_with_no_known_liquidity_data() -> None:
    """A real, disclosed default (see prune_illiquid_edges's own docstring): no data means kept,
    not silently dropped -- this module's own synthetic demo edges rely on exactly this."""
    edges = {("USDT", "SYNTHETIC"): 1.02}
    pruned = prune_illiquid_edges(edges, liquidity_by_edge={}, min_liquidity_usd=500.0)
    assert ("USDT", "SYNTHETIC") in pruned


@pytest.mark.asyncio
async def test_discover_p2p_fiats_filters_out_illiquid_and_nonexistent_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    import market_discovery as md

    class _FakeResult:
        def __init__(self, avg_price: float, avg_liquidity_usdt: float, ads_sampled: int) -> None:
            self.avg_price = avg_price
            self.avg_liquidity_usdt = avg_liquidity_usdt
            self.ads_sampled = ads_sampled

    async def fake_fetch(http: Any, fiat: str) -> _FakeResult | None:
        if fiat == "REAL":
            return _FakeResult(avg_price=10.0, avg_liquidity_usdt=1000.0, ads_sampled=5)
        if fiat == "THIN":
            return _FakeResult(avg_price=10.0, avg_liquidity_usdt=10.0, ads_sampled=5)
        return None  # DEAD

    monkeypatch.setattr(md, "_fetch_best_p2p_price", fake_fetch)

    discovered = await md.discover_p2p_fiats(("REAL", "THIN", "DEAD"), min_liquidity_usd=500.0)

    assert [f.fiat_code for f in discovered] == ["REAL"]
