"""Real, executable tests for core/logger.py's _redact_secrets processor.

No dedicated test file existed for this module before this one. Written
specifically to cover the 2026-09-15 fix: a real 27-hour production
shadow run's own logs were found redacting error_type=type(exc).__name__
for ordinary, benign exception classes (SignatureVerificationError,
TimescaleUnavailableError) purely because their PascalCase names are
24+ unbroken letters -- hiding the real, actionable failure reason for
the run's entire duration. See _SENSITIVE_VALUE_PATTERN's own updated
comment in core/logger.py for the full story and the fix (require at
least one digit or base64 special character in the matched run).

Also re-covers the real secret-shape ground truth tests/red_team_hacker.py
already established (its own fake_secret fixture, reused verbatim here),
to keep both real concerns -- catching genuine secrets, not hiding
benign diagnostics -- verified together in one place.
"""

from __future__ import annotations

from core.logger import _redact_secrets

# Same real fixture tests/red_team_hacker.py's test_api_secret_log_leak already
# established -- realistic-shaped, not a real key.
_REALISTIC_FAKE_SECRET = "sk_live_51HijKLmNoPQRsTuVwXyZ0123456789ABCDEFGabc"


def _redact(**event_dict: object) -> dict[str, object]:
    result = _redact_secrets(None, "info", dict(event_dict))
    return dict(result)


# =============================================================================
# The real 2026-09-15 false positive, and its fix
# =============================================================================


def test_does_not_redact_a_real_benign_exception_class_name() -> None:
    """The exact real value that was wrongly hidden in a real 27h production shadow run's logs."""
    result = _redact(error_type="SignatureVerificationError", consecutive_error_count=4280)
    assert result["error_type"] == "SignatureVerificationError"


def test_does_not_redact_another_real_long_exception_class_name() -> None:
    result = _redact(error_type="TimescaleUnavailableError")
    assert result["error_type"] == "TimescaleUnavailableError"


def test_does_not_redact_ordinary_shorter_exception_names_either() -> None:
    for name in ("RedisUnavailableError", "ConnectionRefusedError", "ValueError", "ValidationError"):
        result = _redact(error_type=name)
        assert result["error_type"] == name


def test_still_redacts_a_realistic_secret_as_an_entire_field_value() -> None:
    """Real, pre-existing substitution behavior (unchanged by this fix): the char class excludes '_'/'-'
    (see that class's own comment), so only the high-entropy run after "sk_live_" -- not the literal
    prefix -- gets substituted. The real secret material itself must never survive, which this checks."""
    result = _redact(api_response_body=_REALISTIC_FAKE_SECRET)
    assert _REALISTIC_FAKE_SECRET not in str(result["api_response_body"])
    assert "***REDACTED***" in str(result["api_response_body"])


def test_still_redacts_a_realistic_secret_embedded_mid_sentence() -> None:
    """The real tests/red_team_hacker.py ground truth this project already established -- unaffected by this fix."""
    message = f"binance {{\"code\":-2015}} key_used={_REALISTIC_FAKE_SECRET}"
    result = _redact(error=message)
    assert _REALISTIC_FAKE_SECRET not in str(result["error"])
    assert "***REDACTED***" in str(result["error"])


def test_still_redacts_a_real_shaped_hex_digest() -> None:
    hex_digest = "a3f5c9e21b7d4680f1e2934c5a6b7d8e9f0a1b2c3d4e5f60718293a4b5c6d7e"  # 64-char SHA-256-shaped
    result = _redact(checksum=hex_digest)
    assert result["checksum"] == "***REDACTED***"


# =============================================================================
# Real, existing behavior this fix must not change
# =============================================================================


def test_still_redacts_by_sensitive_key_name_regardless_of_value_shape() -> None:
    result = _redact(api_key="short", password="alsoShort")
    assert result["api_key"] == "***REDACTED***"
    assert result["password"] == "***REDACTED***"


def test_does_not_redact_a_long_pure_letter_agent_id_with_hyphens() -> None:
    """The real, original reason for excluding '-'/'_' from the charset -- must still hold."""
    result = _redact(agent_id="market-data-binance-btc-usdt-perpetual-futures-instance-one")
    assert "REDACTED" not in str(result["agent_id"])


def test_event_key_itself_is_never_redacted() -> None:
    result = _redact(event="a_very_long_event_name_that_happens_to_be_twenty_four_chars_or_more")
    assert "REDACTED" not in str(result["event"])


def test_a_genuinely_pure_letter_secret_is_a_disclosed_residual_gap() -> None:
    """Documents the real, disclosed trade-off in _SENSITIVE_VALUE_PATTERN's own comment -- not a bug, a known limitation."""
    pure_letter_secret = "ThisLooksLikeAnIdentifierButIsActuallyASecretWithNoDigits"
    result = _redact(some_field=pure_letter_secret)
    assert result["some_field"] == pure_letter_secret  # correctly NOT caught -- same shape as a real class name
