"""Tests du transport signé ``contracts.secure_signal_bus`` (publisher du moteur, vérification worker).

Couvre : le vecteur HMAC-SHA256 publié (RFC 4231), la falsification (payload, signature, mauvaise
clé), le rejeu, l'expiration, l'aiguillage par plateforme, et un vrai trajet PUB -> SUB sur TCP
local, dont l'injection d'un faux signal par un émetteur qui ne connaît pas la clé.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
import zmq
from pydantic import ValidationError

from contracts.platform_signal import (
    AssetClass,
    OrderType,
    PlatformSignal,
    SignalDirection,
    TargetVenue,
)
from contracts.secure_signal_bus import (
    DEFAULT_TOPIC,
    SECRET_ENV_VAR,
    ReplayGuard,
    SecureSignalPublisher,
    WeakSecretError,
    _hmac_hex,
    build_envelope,
    load_secret_from_env,
    open_envelope,
    sign_payload,
    verify_signature,
)

SECRET = "test-secret-0123456789abcdef-0123456789abcdef"  # 45 octets, valeur de test uniquement
OTHER_SECRET = "another-secret-9876543210fedcba-9876543210fedcba"


def _crypto(**overrides: Any) -> PlatformSignal:
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


def _forex(**overrides: Any) -> PlatformSignal:
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


# --- HMAC ------------------------------------------------------------------------------------


def test_hmac_matches_the_published_rfc_4231_vector() -> None:
    """RFC 4231, cas de test 2 : HMAC-SHA256, clé 'Jefe', message 'what do ya want for nothing?'."""
    assert (
        _hmac_hex(b"Jefe", "what do ya want for nothing?")
        == "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
    )


def test_envelope_carries_exactly_signature_and_payload() -> None:
    signal = _crypto()
    envelope = build_envelope(signal, SECRET)
    assert set(envelope) == {"signature", "payload"}
    assert envelope["payload"] == signal.to_zmq_payload()
    assert envelope["signature"] == sign_payload(signal.to_zmq_payload(), SECRET)
    assert len(envelope["signature"]) == 64  # SHA-256 en hexadécimal


# --- Clé -------------------------------------------------------------------------------------


@pytest.mark.parametrize("weak", ["", "short", "x" * 31, b"y" * 31])
def test_weak_secrets_are_refused_everywhere(weak: str | bytes) -> None:
    with pytest.raises(WeakSecretError):
        sign_payload("{}", weak)
    with pytest.raises(WeakSecretError):
        verify_signature({"signature": "", "payload": "{}"}, weak)
    with pytest.raises(WeakSecretError):
        SecureSignalPublisher("tcp://127.0.0.1:*", weak)


def test_secret_of_exactly_the_minimum_length_is_accepted() -> None:
    assert len(sign_payload("{}", "k" * 32)) == 64


def test_load_secret_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SECRET_ENV_VAR, SECRET)
    assert load_secret_from_env() == SECRET.encode("utf-8")
    monkeypatch.delenv(SECRET_ENV_VAR)
    with pytest.raises(WeakSecretError, match=SECRET_ENV_VAR):
        load_secret_from_env()
    monkeypatch.setenv(SECRET_ENV_VAR, "trop-court")
    with pytest.raises(WeakSecretError):
        load_secret_from_env()


# --- verify_signature ------------------------------------------------------------------------


def test_verify_accepts_dict_json_string_and_bytes() -> None:
    envelope = build_envelope(_crypto(), SECRET)
    assert verify_signature(envelope, SECRET) is True
    assert verify_signature(json.dumps(envelope), SECRET) is True
    assert verify_signature(json.dumps(envelope).encode("utf-8"), SECRET) is True


def test_verify_rejects_a_wrong_key() -> None:
    assert verify_signature(build_envelope(_crypto(), SECRET), OTHER_SECRET) is False


def test_verify_rejects_a_tampered_payload() -> None:
    envelope = build_envelope(_crypto(position_size_pct=5.0), SECRET)
    forged = {**envelope, "payload": envelope["payload"].replace('"position_size_pct":5.0', '"position_size_pct":90.0')}
    assert forged["payload"] != envelope["payload"]
    assert verify_signature(forged, SECRET) is False


def test_verify_rejects_a_tampered_or_swapped_signature() -> None:
    envelope = build_envelope(_crypto(), SECRET)
    flipped = envelope["signature"][:-1] + ("0" if envelope["signature"][-1] != "0" else "1")
    assert verify_signature({**envelope, "signature": flipped}, SECRET) is False
    assert verify_signature({**envelope, "signature": envelope["signature"].upper()}, SECRET) is False
    other = build_envelope(_forex(), SECRET)
    assert verify_signature({"signature": other["signature"], "payload": envelope["payload"]}, SECRET) is False


@pytest.mark.parametrize(
    "garbage",
    [
        {},
        {"signature": "abc"},
        {"payload": "{}"},
        {"signature": 123, "payload": "{}"},
        {"signature": "abc", "payload": {"a": 1}},
        "not json at all",
        b"\xff\xfe\x00",
        "[1, 2, 3]",
        "null",
        None,
        42,
    ],
)
def test_verify_never_raises_on_a_malformed_envelope(garbage: Any) -> None:
    assert verify_signature(garbage, SECRET) is False


# --- open_envelope : acceptation et rejets ---------------------------------------------------


def test_open_envelope_accepts_a_good_signal_for_its_venue() -> None:
    signal = _crypto()
    verdict = open_envelope(build_envelope(signal, SECRET), SECRET, venue=TargetVenue.HYPERLIQUID)
    assert verdict.accepted and verdict.reject_reason is None
    assert verdict.signal == signal


def test_open_envelope_rejects_a_forged_signature_and_logs_only_the_reason(caplog: pytest.LogCaptureFixture) -> None:
    envelope = build_envelope(_crypto(), OTHER_SECRET)
    with caplog.at_level(logging.WARNING, logger="contracts.secure_signal_bus"):
        verdict = open_envelope(envelope, SECRET, venue=TargetVenue.HYPERLIQUID)
    assert verdict.signal is None and verdict.reject_reason == "bad_signature"
    assert "bad_signature" in caplog.text
    assert envelope["signature"] not in caplog.text and "CRYPTO_BTC_USD" not in caplog.text


def test_open_envelope_reports_a_malformed_envelope() -> None:
    assert open_envelope("{{{", SECRET, venue=TargetVenue.HYPERLIQUID).reject_reason == "malformed_envelope"


def test_a_validly_signed_but_incoherent_payload_is_rejected() -> None:
    """Le moteur (ou quelqu'un qui a sa clé) signe un contenu que le contrat refuse : rejeté quand même."""
    body = json.loads(_crypto().to_zmq_payload())
    body["position_size_pct"] = 500.0
    payload = json.dumps(body)
    envelope = {"signature": sign_payload(payload, SECRET), "payload": payload}
    assert verify_signature(envelope, SECRET) is True
    assert open_envelope(envelope, SECRET, venue=TargetVenue.HYPERLIQUID).reject_reason == "invalid_payload"


def test_a_valid_signal_replayed_to_the_wrong_venue_worker_is_refused() -> None:
    envelope = build_envelope(_crypto(), SECRET)
    assert open_envelope(envelope, SECRET, venue=TargetVenue.DARWINEX_ZERO).reject_reason == "wrong_venue"
    assert open_envelope(envelope, SECRET, venue=TargetVenue.NUMERAI_SIGNALS).reject_reason == "wrong_venue"


def test_target_all_is_routed_by_asset_class() -> None:
    envelope = build_envelope(_crypto(target_venue=TargetVenue.ALL), SECRET)
    assert open_envelope(envelope, SECRET, venue=TargetVenue.HYPERLIQUID).accepted
    assert open_envelope(envelope, SECRET, venue=TargetVenue.DARWINEX_ZERO).reject_reason == "wrong_venue"


def test_worker_venue_cannot_be_all() -> None:
    with pytest.raises(ValueError, match="pas ALL"):
        open_envelope(build_envelope(_crypto(), SECRET), SECRET, venue=TargetVenue.ALL)


def test_ttl_expiration_boundary_is_exact() -> None:
    envelope = build_envelope(_crypto(timestamp_ms=1_000_000, ttl_ms=1500), SECRET)
    kw: dict[str, Any] = dict(venue=TargetVenue.HYPERLIQUID, max_age_ms=None)
    assert open_envelope(envelope, SECRET, now_ms=1_001_500, **kw).accepted
    assert open_envelope(envelope, SECRET, now_ms=1_001_501, **kw).reject_reason == "expired"


def test_max_age_boundary_is_exact_and_can_be_disabled() -> None:
    envelope = build_envelope(_crypto(timestamp_ms=1_000_000), SECRET)  # sans ttl_ms
    kw: dict[str, Any] = dict(venue=TargetVenue.HYPERLIQUID, max_age_ms=60_000)
    assert open_envelope(envelope, SECRET, now_ms=1_060_000, **kw).accepted
    assert open_envelope(envelope, SECRET, now_ms=1_060_001, **kw).reject_reason == "too_old"
    assert open_envelope(envelope, SECRET, now_ms=9_999_999_999, venue=TargetVenue.HYPERLIQUID, max_age_ms=None).accepted


def test_future_timestamp_beyond_the_skew_is_rejected() -> None:
    envelope = build_envelope(_crypto(timestamp_ms=1_000_000), SECRET)
    kw: dict[str, Any] = dict(venue=TargetVenue.HYPERLIQUID, max_future_skew_ms=5_000)
    assert open_envelope(envelope, SECRET, now_ms=995_000, **kw).accepted  # 5 000 ms d'avance : toléré
    assert open_envelope(envelope, SECRET, now_ms=994_999, **kw).reject_reason == "from_future"


def test_replayed_envelope_is_rejected_the_second_time() -> None:
    guard = ReplayGuard()
    envelope = build_envelope(_crypto(), SECRET)
    assert open_envelope(envelope, SECRET, venue=TargetVenue.HYPERLIQUID, replay_guard=guard).accepted
    assert open_envelope(envelope, SECRET, venue=TargetVenue.HYPERLIQUID, replay_guard=guard).reject_reason == "replay"


def test_a_rejected_message_does_not_burn_its_id_in_the_replay_guard() -> None:
    """Un message refusé pour âge ne doit pas empêcher la version valide d'être traitée ensuite."""
    guard = ReplayGuard()
    envelope = build_envelope(_crypto(timestamp_ms=1_000_000), SECRET)
    late = open_envelope(envelope, SECRET, venue=TargetVenue.HYPERLIQUID, replay_guard=guard, now_ms=2_000_000)
    assert late.reject_reason == "too_old"
    on_time = open_envelope(envelope, SECRET, venue=TargetVenue.HYPERLIQUID, replay_guard=guard, now_ms=1_000_100)
    assert on_time.accepted


def test_replay_guard_is_bounded_and_evicts_the_oldest() -> None:
    guard = ReplayGuard(capacity=2)
    a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    assert guard.check_and_add(a) and guard.check_and_add(b)
    assert guard.check_and_add(b) is False
    assert guard.check_and_add(c)  # évince a
    assert guard.check_and_add(a)  # a est de nouveau inconnu
    with pytest.raises(ValueError):
        ReplayGuard(capacity=0)


# --- Publisher sur de vrais sockets ----------------------------------------------------------


@pytest.fixture
def zmq_ctx() -> Iterator[zmq.Context]:
    """Contexte du côté SUB ; détruit à la fin pour ne laisser aucun socket (cf. fuite connue des fixtures ZMQ)."""
    ctx = zmq.Context()
    try:
        yield ctx
    finally:
        ctx.destroy(linger=0)


def _subscribe(ctx: zmq.Context, *endpoints: str, topic: bytes = DEFAULT_TOPIC) -> zmq.Socket:
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt(zmq.SUBSCRIBE, topic)
    for endpoint in endpoints:
        sub.connect(endpoint)
    return sub


def _receive_after_publishing(pub: SecureSignalPublisher, sub: zmq.Socket, signal: PlatformSignal) -> list[bytes]:
    """Republie jusqu'à réception : un SUB qui vient de se connecter perd les premiers messages (slow joiner)."""
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        pub.publish(signal)
        if sub.poll(100):
            return sub.recv_multipart()
    raise AssertionError("aucun message reçu en 5 s")


def test_publisher_sends_topic_then_a_verifiable_envelope(zmq_ctx: zmq.Context) -> None:
    signal = _crypto()
    with SecureSignalPublisher("tcp://127.0.0.1:*", SECRET) as pub:
        sub = _subscribe(zmq_ctx, pub.endpoint)
        frames = _receive_after_publishing(pub, sub, signal)
    assert len(frames) == 2 and frames[0] == DEFAULT_TOPIC
    assert verify_signature(frames[1], SECRET) is True
    verdict = open_envelope(frames[1], SECRET, venue=TargetVenue.HYPERLIQUID)
    assert verdict.signal == signal


def test_a_rogue_publisher_without_the_key_cannot_inject_a_signal(zmq_ctx: zmq.Context) -> None:
    """Menace réelle : un second émetteur sur le réseau publie sur le même topic avec une autre clé."""
    with SecureSignalPublisher("tcp://127.0.0.1:*", SECRET) as legit, SecureSignalPublisher(
        "tcp://127.0.0.1:*", OTHER_SECRET
    ) as rogue:
        sub = _subscribe(zmq_ctx, legit.endpoint, rogue.endpoint)
        forged = _crypto(direction=SignalDirection.SHORT, position_size_pct=100.0)
        got_rogue = _receive_after_publishing(rogue, sub, forged)
    verdict = open_envelope(got_rogue[1], SECRET, venue=TargetVenue.HYPERLIQUID)
    assert verdict.signal is None and verdict.reject_reason == "bad_signature"


def test_subscriber_on_another_topic_receives_nothing(zmq_ctx: zmq.Context) -> None:
    with SecureSignalPublisher("tcp://127.0.0.1:*", SECRET) as pub:
        sub = _subscribe(zmq_ctx, pub.endpoint, topic=b"AUTRE.TOPIC")
        for _ in range(10):
            pub.publish(_crypto())
            time.sleep(0.02)
        assert sub.poll(200) == 0


def test_publish_does_not_block_without_any_subscriber() -> None:
    with SecureSignalPublisher("tcp://127.0.0.1:*", SECRET) as pub:
        start = time.monotonic()
        results = [pub.publish(_crypto()) for _ in range(200)]
        assert time.monotonic() - start < 2.0
    assert all(results)  # PUB sans abonné : le message est simplement jeté, jamais bloquant


def test_publish_refuses_a_signal_that_bypassed_validation() -> None:
    corrupted = _crypto().model_copy(update={"position_size_pct": 500.0})  # model_copy ne revalide pas
    with SecureSignalPublisher("tcp://127.0.0.1:*", SECRET) as pub, pytest.raises(ValidationError):
        pub.publish(corrupted)


def test_publish_refuses_an_already_expired_signal() -> None:
    stale = _crypto(timestamp_ms=1_000, ttl_ms=10)
    with SecureSignalPublisher("tcp://127.0.0.1:*", SECRET) as pub, pytest.raises(ValueError, match="expiré"):
        pub.publish(stale)


def test_publish_refuses_wrong_types_and_a_closed_publisher() -> None:
    pub = SecureSignalPublisher("tcp://127.0.0.1:*", SECRET)
    with pytest.raises(TypeError):
        pub.publish({"direction": "LONG"})  # type: ignore[arg-type]
    pub.close()
    pub.close()  # idempotent
    with pytest.raises(RuntimeError, match="fermé"):
        pub.publish(_crypto())


def test_empty_topic_is_refused() -> None:
    with pytest.raises(ValueError, match="topic"):
        SecureSignalPublisher("tcp://127.0.0.1:*", SECRET, topic=b"")


def test_from_env_builds_a_working_publisher(monkeypatch: pytest.MonkeyPatch, zmq_ctx: zmq.Context) -> None:
    monkeypatch.setenv(SECRET_ENV_VAR, SECRET)
    with SecureSignalPublisher.from_env("tcp://127.0.0.1:*") as pub:
        sub = _subscribe(zmq_ctx, pub.endpoint)
        frames = _receive_after_publishing(pub, sub, _crypto())
    assert verify_signature(frames[1], SECRET) is True


# --- Aiguillage des 8 plateformes ------------------------------------------------------------


def test_each_crypto_worker_accepts_all_crypto_but_a_targeted_signal_only_its_own() -> None:
    crypto_venues = [TargetVenue.HYPERLIQUID, TargetVenue.BINANCE, TargetVenue.BYBIT, TargetVenue.ENZYME]
    fan_out = build_envelope(_crypto(target_venue=TargetVenue.ALL), SECRET)
    binance_only = build_envelope(_crypto(target_venue=TargetVenue.BINANCE), SECRET)
    for venue in crypto_venues:
        assert open_envelope(fan_out, SECRET, venue=venue).accepted
        assert open_envelope(binance_only, SECRET, venue=venue).accepted is (venue is TargetVenue.BINANCE)
    assert open_envelope(fan_out, SECRET, venue=TargetVenue.DARWINEX_ZERO).reject_reason == "wrong_venue"
    assert open_envelope(fan_out, SECRET, venue=TargetVenue.NUMERAI_SIGNALS).reject_reason == "wrong_venue"
