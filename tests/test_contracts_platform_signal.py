"""Tests du contrat ``contracts.platform_signal.PlatformSignal`` (moteur -> workers multi-plateformes).

Chaque règle de validation a son test, y compris les deux cas donnés dans la spécification
d'origine (``LIMIT`` sans ``limit_price`` ; Darwinex sans ``stop_loss_price``). Le fichier
``platform_signal.schema.json`` est comparé au modèle pour détecter une dérive.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from typing import Any

import pytest
from pydantic import ValidationError

from contracts.platform_signal import (
    SCHEMA_PATH,
    AssetClass,
    OrderType,
    SignalDirection,
    TargetVenue,
    PlatformSignal,
    json_schema,
)


def _hyperliquid(**overrides: Any) -> PlatformSignal:
    base: dict[str, Any] = dict(
        target_venue=TargetVenue.HYPERLIQUID,
        asset_class=AssetClass.CRYPTO,
        symbol="CRYPTO_BTC_USD",
        direction=SignalDirection.LONG,
        position_size_pct=5.0,
        order_type=OrderType.MARKET,
    )
    base.update(overrides)
    return PlatformSignal(**base)


def _darwinex(**overrides: Any) -> PlatformSignal:
    base: dict[str, Any] = dict(
        target_venue=TargetVenue.DARWINEX_ZERO,
        asset_class=AssetClass.FOREX,
        symbol="FOREX_EUR_USD",
        direction=SignalDirection.SHORT,
        position_size_pct=2.0,
        order_type=OrderType.MARKET,
        metadata={"stop_loss_price": 1.12},
    )
    base.update(overrides)
    return PlatformSignal(**base)


def _numerai(**overrides: Any) -> PlatformSignal:
    base: dict[str, Any] = dict(
        target_venue=TargetVenue.NUMERAI_SIGNALS,
        asset_class=AssetClass.EQUITY,
        symbol="EQUITY_US_AAPL",
        direction=SignalDirection.LONG,
        confidence_score=0.7,
        position_size_pct=0.0,
        order_type=OrderType.PREDICTION_ONLY,
    )
    base.update(overrides)
    return PlatformSignal(**base)


# --- Cas valides et sérialisation -------------------------------------------------------------


def test_valid_signals_for_each_venue_are_accepted() -> None:
    assert _hyperliquid().target_venue is TargetVenue.HYPERLIQUID
    assert _darwinex().target_venue is TargetVenue.DARWINEX_ZERO
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
    restored = PlatformSignal.from_zmq_payload(original.to_zmq_payload())
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
    assert PlatformSignal.from_zmq_payload(original.to_zmq_payload().encode("utf-8")) == original


def test_from_zmq_payload_rejects_a_corrupted_message() -> None:
    body = json.loads(_hyperliquid().to_zmq_payload())
    body["confidence_score"] = 7.0
    with pytest.raises(ValidationError):
        PlatformSignal.from_zmq_payload(json.dumps(body))


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
        _numerai(confidence_score=bad)


@pytest.mark.parametrize("ok", [0.0, 1.0])
def test_confidence_score_bounds_are_inclusive(ok: float) -> None:
    assert _numerai(confidence_score=ok).confidence_score == ok


# --- confidence_score : exigé pour Numerai, interdit ailleurs (fuite d'IP) ---------------------


def test_numerai_requires_a_confidence_score() -> None:
    with pytest.raises(ValidationError, match="Numerai exige confidence_score"):
        _numerai(confidence_score=None)


@pytest.mark.parametrize("builder", [_hyperliquid, _darwinex])
def test_visible_flow_venues_refuse_a_confidence_score(builder: Any) -> None:
    with pytest.raises(ValidationError, match="confidence_score est interdit"):
        builder(confidence_score=0.85)


def test_confidence_rule_follows_the_asset_class_when_target_is_all() -> None:
    with pytest.raises(ValidationError, match="confidence_score est interdit"):
        _hyperliquid(target_venue=TargetVenue.ALL, confidence_score=0.5)  # crypto -> Hyperliquid
    with pytest.raises(ValidationError, match="Numerai exige confidence_score"):
        _numerai(target_venue=TargetVenue.ALL, confidence_score=None)  # equity -> Numerai


def test_realtime_signals_serialize_confidence_as_null() -> None:
    assert json.loads(_hyperliquid().to_zmq_payload())["confidence_score"] is None


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
        PlatformSignal.from_zmq_payload(json.dumps(body))


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
    assert on_disk == json_schema(), "platform_signal.schema.json est périmé : python -m contracts.platform_signal"


def test_json_schema_pins_the_documented_bounds_and_enums() -> None:
    props = json_schema()["properties"]
    number, null = props["confidence_score"]["anyOf"]  # Optional[float] : nombre borné ou null
    assert number["minimum"] == 0.0 and number["maximum"] == 1.0 and null == {"type": "null"}
    assert "confidence_score" not in json_schema()["required"]
    assert props["position_size_pct"]["minimum"] == 0.0 and props["position_size_pct"]["maximum"] == 100.0
    assert json_schema()["additionalProperties"] is False
    assert json_schema()["$defs"]["TargetVenue"]["enum"] == ["HYPERLIQUID", "BINANCE", "BYBIT", "ZIGNALY", "ENZYME", "NUMERAI_SIGNALS", "AXI_SELECT", "DARWINEX_ZERO", "ALL"]


# --- Les 8 plateformes ------------------------------------------------------------------------

_CRYPTO_VENUES = [TargetVenue.HYPERLIQUID, TargetVenue.BINANCE, TargetVenue.BYBIT, TargetVenue.ZIGNALY, TargetVenue.ENZYME]
_FOREX_VENUES = [TargetVenue.AXI_SELECT, TargetVenue.DARWINEX_ZERO]
_NON_NUMERAI = _CRYPTO_VENUES + _FOREX_VENUES


def _for_venue(venue: TargetVenue, **overrides: Any) -> PlatformSignal:
    """Un signal valide pour ``venue`` (classe d'actif, ordre et métadonnées propres à sa plateforme)."""
    if venue is TargetVenue.NUMERAI_SIGNALS:
        return _numerai(**overrides)
    if venue in _FOREX_VENUES:
        return _darwinex(target_venue=venue, **overrides)
    return _hyperliquid(target_venue=venue, **overrides)


def test_target_venue_is_exactly_the_eight_platforms_plus_all() -> None:
    assert {v.value for v in TargetVenue} == {
        "HYPERLIQUID", "BINANCE", "BYBIT", "ZIGNALY", "ENZYME", "NUMERAI_SIGNALS", "AXI_SELECT", "DARWINEX_ZERO", "ALL",
    }


def test_every_platform_belongs_to_exactly_one_asset_class() -> None:
    from contracts.platform_signal import venues_for

    assert venues_for(AssetClass.CRYPTO) == frozenset(_CRYPTO_VENUES)
    assert venues_for(AssetClass.FOREX) == frozenset(_FOREX_VENUES)
    assert venues_for(AssetClass.EQUITY) == frozenset({TargetVenue.NUMERAI_SIGNALS})
    all_venues = set().union(*(venues_for(c) for c in AssetClass))
    assert all_venues == set(TargetVenue) - {TargetVenue.ALL}


@pytest.mark.parametrize("venue", [v for v in TargetVenue if v is not TargetVenue.ALL])
def test_a_valid_signal_exists_for_every_platform(venue: TargetVenue) -> None:
    assert _for_venue(venue).target_venue is venue


@pytest.mark.parametrize(
    ("venue", "wrong_class", "symbol"),
    [
        (TargetVenue.BINANCE, AssetClass.EQUITY, "EQUITY_US_AAPL"),
        (TargetVenue.BYBIT, AssetClass.FOREX, "FOREX_EUR_USD"),
        (TargetVenue.ZIGNALY, AssetClass.FOREX, "FOREX_EUR_USD"),
        (TargetVenue.ENZYME, AssetClass.EQUITY, "EQUITY_US_AAPL"),
        (TargetVenue.AXI_SELECT, AssetClass.CRYPTO, "CRYPTO_BTC_USD"),
        (TargetVenue.DARWINEX_ZERO, AssetClass.CRYPTO, "CRYPTO_BTC_USD"),
        (TargetVenue.NUMERAI_SIGNALS, AssetClass.FOREX, "FOREX_EUR_USD"),
    ],
)
def test_each_platform_refuses_the_wrong_asset_class(venue: TargetVenue, wrong_class: AssetClass, symbol: str) -> None:
    with pytest.raises(ValidationError, match="ne traite pas la classe"):
        PlatformSignal(
            target_venue=venue, asset_class=wrong_class, symbol=symbol, direction=SignalDirection.FLAT,
            position_size_pct=0.0, order_type=OrderType.PREDICTION_ONLY, confidence_score=0.5,
        )


@pytest.mark.parametrize("venue", _NON_NUMERAI)
def test_confidence_score_is_forbidden_on_every_platform_except_numerai(venue: TargetVenue) -> None:
    assert _for_venue(venue).confidence_score is None
    with pytest.raises(ValidationError, match="confidence_score est interdit"):
        _for_venue(venue, confidence_score=0.9)


def test_confidence_score_is_required_only_for_numerai_signals() -> None:
    assert _numerai().confidence_score == 0.7
    with pytest.raises(ValidationError, match="Numerai exige confidence_score"):
        _numerai(confidence_score=None)


def test_all_expands_to_every_platform_of_the_asset_class() -> None:
    crypto = _hyperliquid(target_venue=TargetVenue.ALL)
    assert {v for v in TargetVenue if v is not TargetVenue.ALL and crypto.targets(v)} == set(_CRYPTO_VENUES)
    forex = _darwinex(target_venue=TargetVenue.ALL)
    assert {v for v in TargetVenue if v is not TargetVenue.ALL and forex.targets(v)} == set(_FOREX_VENUES)
    equity = _numerai(target_venue=TargetVenue.ALL)
    assert {v for v in TargetVenue if v is not TargetVenue.ALL and equity.targets(v)} == {TargetVenue.NUMERAI_SIGNALS}


def test_a_targeted_platform_is_the_only_one_that_matches() -> None:
    binance = _hyperliquid(target_venue=TargetVenue.BINANCE)
    assert binance.targets(TargetVenue.BINANCE)
    assert not any(binance.targets(v) for v in TargetVenue if v is not TargetVenue.BINANCE)


def test_stop_loss_is_required_for_darwinex_zero_but_no_rule_is_invented_for_axi_select() -> None:
    with pytest.raises(ValidationError, match="stop_loss_price"):
        _darwinex(target_venue=TargetVenue.DARWINEX_ZERO, metadata={})
    assert _darwinex(target_venue=TargetVenue.AXI_SELECT, metadata={}).target_venue is TargetVenue.AXI_SELECT
    with pytest.raises(ValidationError, match="stop_loss_price"):  # ALL couvre aussi Darwinex Zero
        _darwinex(target_venue=TargetVenue.ALL, metadata={})
