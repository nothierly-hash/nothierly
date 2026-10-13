"""Tests du contrat ``contracts.trading_signal.TradingSignal`` (moteur -> workers multi-plateformes).

Chaque règle de validation a son test, y compris les deux cas donnés dans la spécification
d'origine (``LIMIT`` sans ``limit_price`` ; Darwinex sans ``stop_loss_price``). Le fichier
``trading_signal.schema.json`` est comparé au modèle pour détecter une dérive.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from typing import Any

import pytest
from pydantic import ValidationError

from contracts.trading_signal import (
    SCHEMA_PATH,
    AssetClass,
    OrderType,
    SignalDirection,
    TargetVenue,
    TradingSignal,
    json_schema,
)


def _hyperliquid(**overrides: Any) -> TradingSignal:
    base: dict[str, Any] = dict(
        target_venue=TargetVenue.HYPERLIQUID,
        asset_class=AssetClass.CRYPTO,
        symbol="CRYPTO_BTC_USD",
        direction=SignalDirection.LONG,
        confidence_score=0.85,
        position_size_pct=5.0,
        order_type=OrderType.MARKET,
    )
    base.update(overrides)
    return TradingSignal(**base)


def _darwinex(**overrides: Any) -> TradingSignal:
    base: dict[str, Any] = dict(
        target_venue=TargetVenue.DARWINEX,
        asset_class=AssetClass.FOREX,
        symbol="FOREX_EUR_USD",
        direction=SignalDirection.SHORT,
        confidence_score=0.6,
        position_size_pct=2.0,
        order_type=OrderType.MARKET,
        metadata={"stop_loss_price": 1.12},
    )
    base.update(overrides)
    return TradingSignal(**base)


def _numerai(**overrides: Any) -> TradingSignal:
    base: dict[str, Any] = dict(
        target_venue=TargetVenue.NUMERAI,
        asset_class=AssetClass.EQUITY,
        symbol="EQUITY_US_AAPL",
        direction=SignalDirection.LONG,
        confidence_score=0.7,
        position_size_pct=0.0,
        order_type=OrderType.PREDICTION_ONLY,
    )
    base.update(overrides)
    return TradingSignal(**base)


# --- Cas valides et sérialisation -------------------------------------------------------------


def test_valid_signals_for_each_venue_are_accepted() -> None:
    assert _hyperliquid().target_venue is TargetVenue.HYPERLIQUID
    assert _darwinex().target_venue is TargetVenue.DARWINEX
    assert _numerai().order_type is OrderType.PREDICTION_ONLY


def test_defaults_generate_a_real_uuid_and_a_current_millisecond_timestamp() -> None:
    before = int(time.time() * 1000)
    signal = _hyperliquid()
    after = int(time.time() * 1000)
    assert isinstance(signal.signal_id, uuid.UUID)
    assert before <= signal.timestamp_ms <= after
    assert _hyperliquid().signal_id != signal.signal_id  # deux signaux, deux identifiants


def test_zmq_payload_roundtrips_to_an_equal_signal() -> None:
    original = _darwinex(metadata={"stop_loss_price": 1.12, "note": "x"})
    restored = TradingSignal.from_zmq_payload(original.to_zmq_payload())
    assert restored == original


def test_zmq_payload_is_plain_json_with_the_specified_fields() -> None:
    body = json.loads(_hyperliquid().to_zmq_payload())
    assert set(body) == {
        "signal_id", "timestamp_ms", "target_venue", "asset_class", "symbol", "direction",
        "confidence_score", "position_size_pct", "order_type", "metadata", "ttl_ms",
    }
    assert body["target_venue"] == "HYPERLIQUID"
    assert body["order_type"] == "MARKET"
    uuid.UUID(body["signal_id"])  # une chaîne UUID valide


def test_from_zmq_payload_accepts_bytes_as_received_from_the_socket() -> None:
    original = _hyperliquid()
    assert TradingSignal.from_zmq_payload(original.to_zmq_payload().encode("utf-8")) == original


def test_from_zmq_payload_rejects_a_corrupted_message() -> None:
    body = json.loads(_hyperliquid().to_zmq_payload())
    body["confidence_score"] = 7.0
    with pytest.raises(ValidationError):
        TradingSignal.from_zmq_payload(json.dumps(body))


# --- LIMIT / limit_price ----------------------------------------------------------------------


def test_limit_order_without_limit_price_is_rejected() -> None:
    with pytest.raises(ValidationError, match="limit_price"):
        _hyperliquid(order_type=OrderType.LIMIT)


def test_limit_order_with_limit_price_is_accepted() -> None:
    assert _hyperliquid(order_type=OrderType.LIMIT, metadata={"limit_price": 86000.0}).metadata["limit_price"] == 86000.0


@pytest.mark.parametrize("bad", [0, -5.0, math.nan, math.inf, "86000", True, None])
def test_limit_price_must_be_a_positive_finite_number(bad: Any) -> None:
    with pytest.raises(ValidationError, match="limit_price"):
        _hyperliquid(order_type=OrderType.LIMIT, metadata={"limit_price": bad})


# --- Numerai ----------------------------------------------------------------------------------


@pytest.mark.parametrize("order_type", [OrderType.MARKET, OrderType.LIMIT])
def test_numerai_only_accepts_prediction_only(order_type: OrderType) -> None:
    with pytest.raises(ValidationError, match="PREDICTION_ONLY"):
        _numerai(order_type=order_type, position_size_pct=1.0, metadata={"limit_price": 190.0})


def test_numerai_rejects_a_non_equity_class() -> None:
    with pytest.raises(ValidationError, match="ne traite pas la classe"):
        _numerai(asset_class=AssetClass.CRYPTO, symbol="CRYPTO_BTC_USD")


# --- Darwinex ---------------------------------------------------------------------------------


def test_darwinex_executable_order_without_stop_loss_is_rejected() -> None:
    with pytest.raises(ValidationError, match="stop_loss_price"):
        _darwinex(metadata={})


def test_darwinex_prediction_only_does_not_need_a_stop_loss() -> None:
    signal = _darwinex(order_type=OrderType.PREDICTION_ONLY, metadata={}, position_size_pct=0.0)
    assert signal.order_type is OrderType.PREDICTION_ONLY


def test_darwinex_rejects_a_non_forex_class() -> None:
    with pytest.raises(ValidationError, match="ne traite pas la classe"):
        _darwinex(asset_class=AssetClass.CRYPTO, symbol="CRYPTO_BTC_USD")


# --- Hyperliquid et ALL -----------------------------------------------------------------------


def test_hyperliquid_rejects_a_non_crypto_class() -> None:
    with pytest.raises(ValidationError, match="ne traite pas la classe"):
        _hyperliquid(asset_class=AssetClass.FOREX, symbol="FOREX_EUR_USD")


def test_all_routes_to_the_venue_of_the_asset_class_and_applies_its_rules() -> None:
    assert _hyperliquid(target_venue=TargetVenue.ALL).target_venue is TargetVenue.ALL  # crypto : aucune règle de plus
    with pytest.raises(ValidationError, match="PREDICTION_ONLY"):  # equity -> Numerai
        _numerai(target_venue=TargetVenue.ALL, order_type=OrderType.MARKET, position_size_pct=1.0)
    with pytest.raises(ValidationError, match="stop_loss_price"):  # forex -> Darwinex
        _darwinex(target_venue=TargetVenue.ALL, metadata={})


# --- Bornes et cohérence direction / taille ---------------------------------------------------


@pytest.mark.parametrize("bad", [-0.01, 1.01, math.nan, math.inf])
def test_confidence_score_must_stay_in_unit_interval(bad: float) -> None:
    with pytest.raises(ValidationError):
        _hyperliquid(confidence_score=bad)


@pytest.mark.parametrize("ok", [0.0, 1.0])
def test_confidence_score_bounds_are_inclusive(ok: float) -> None:
    assert _hyperliquid(confidence_score=ok).confidence_score == ok


@pytest.mark.parametrize("bad", [-0.01, 100.01, math.nan])
def test_position_size_pct_must_stay_in_zero_to_one_hundred(bad: float) -> None:
    with pytest.raises(ValidationError):
        _hyperliquid(position_size_pct=bad)


def test_position_size_pct_upper_bound_is_inclusive() -> None:
    assert _hyperliquid(position_size_pct=100.0).position_size_pct == 100.0


def test_flat_signal_must_have_zero_size() -> None:
    assert _hyperliquid(direction=SignalDirection.FLAT, position_size_pct=0.0).direction is SignalDirection.FLAT
    with pytest.raises(ValidationError, match="FLAT"):
        _hyperliquid(direction=SignalDirection.FLAT, position_size_pct=3.0)


def test_executable_long_or_short_needs_a_positive_size() -> None:
    with pytest.raises(ValidationError, match="position_size_pct > 0"):
        _hyperliquid(position_size_pct=0.0)


# --- Symbole ----------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["btc_usd", "CRYPTO", "CRYPTO BTC", "CRYPTO_btc", ""])
def test_symbol_must_be_an_uppercase_underscore_ticker(bad: str) -> None:
    with pytest.raises(ValidationError):
        _hyperliquid(symbol=bad)


def test_symbol_prefix_must_match_the_asset_class() -> None:
    with pytest.raises(ValidationError, match="doit commencer par 'CRYPTO_'"):
        _hyperliquid(symbol="FOREX_EUR_USD")


# --- Stop-loss du bon côté --------------------------------------------------------------------


def test_long_stop_must_be_below_the_limit_price() -> None:
    ok = _darwinex(
        direction=SignalDirection.LONG, order_type=OrderType.LIMIT, metadata={"limit_price": 1.10, "stop_loss_price": 1.08}
    )
    assert ok.metadata["stop_loss_price"] == 1.08
    with pytest.raises(ValidationError, match="LONG"):
        _darwinex(
            direction=SignalDirection.LONG, order_type=OrderType.LIMIT, metadata={"limit_price": 1.10, "stop_loss_price": 1.12}
        )


def test_short_stop_must_be_above_the_limit_price() -> None:
    ok = _darwinex(
        direction=SignalDirection.SHORT, order_type=OrderType.LIMIT, metadata={"limit_price": 1.10, "stop_loss_price": 1.12}
    )
    assert ok.metadata["stop_loss_price"] == 1.12
    with pytest.raises(ValidationError, match="SHORT"):
        _darwinex(
            direction=SignalDirection.SHORT, order_type=OrderType.LIMIT, metadata={"limit_price": 1.10, "stop_loss_price": 1.08}
        )


@pytest.mark.parametrize("bad", [0, -1.0, math.nan, "1.1", False])
def test_stop_loss_price_must_be_a_positive_finite_number(bad: Any) -> None:
    with pytest.raises(ValidationError, match="stop_loss_price"):
        _darwinex(metadata={"stop_loss_price": bad})


# --- Immuabilité, champs inconnus, expiration -------------------------------------------------


def test_unknown_fields_are_refused() -> None:
    body = json.loads(_hyperliquid().to_zmq_payload())
    body["leverage"] = 50
    with pytest.raises(ValidationError):
        TradingSignal.from_zmq_payload(json.dumps(body))


def test_signal_is_immutable() -> None:
    signal = _hyperliquid()
    with pytest.raises(ValidationError):
        signal.position_size_pct = 99.0  # type: ignore[misc]


def test_no_ttl_means_never_expired() -> None:
    assert _numerai().is_expired(now_ms=10**15) is False


def test_expiration_boundary_is_exact() -> None:
    signal = _hyperliquid(timestamp_ms=1_000_000, ttl_ms=1500)
    assert signal.is_expired(now_ms=1_001_500) is False  # age == ttl : encore valide
    assert signal.is_expired(now_ms=1_001_501) is True  # age == ttl + 1 : expiré


# --- JSON Schema publié -----------------------------------------------------------------------


def test_published_json_schema_matches_the_model() -> None:
    on_disk = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert on_disk == json_schema(), "trading_signal.schema.json est périmé : python -m contracts.trading_signal"


def test_json_schema_pins_the_documented_bounds_and_enums() -> None:
    props = json_schema()["properties"]
    assert props["confidence_score"]["minimum"] == 0.0 and props["confidence_score"]["maximum"] == 1.0
    assert props["position_size_pct"]["minimum"] == 0.0 and props["position_size_pct"]["maximum"] == 100.0
    assert json_schema()["additionalProperties"] is False
    assert json_schema()["$defs"]["TargetVenue"]["enum"] == ["NUMERAI", "HYPERLIQUID", "DARWINEX", "ALL"]
