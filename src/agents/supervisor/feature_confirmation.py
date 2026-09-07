"""Confidence adjustment from the centrally-computed :class:`~core.schemas.FeatureVector`.

Closes the Phase 1 gap left open earlier this session: :class:`~src.
data_pipeline.feature_store.FeatureStore` computes and publishes each
instrument's technical-indicator snapshot, but until now nothing actually
consumed it -- :class:`~src.agents.supervisor.agent.XgboostSignalAgent`
and :class:`~src.agents.rl_signal.agent.RlSignalAgent` still decided
purely from :class:`~src.agents.supervisor.ml_engine.SupervisorMLModel`'s
own market/macro/sentiment feature vector.

This module deliberately does **not** feed :class:`~core.schemas.
FeatureVector` into either trained model's own input vector: that would
silently change a dimensionality neither model was fit against, for a
model that may already be running against saved (or future) trained
weights. Instead, :func:`apply_feature_confirmation` treats it as an
**independent second opinion** that adjusts confidence, never the
direction: a model's own call is never overridden here, only made louder
when the technical read agrees, or quieter when it doesn't. A model that
never receives a feature vector (no TimescaleDB history yet for that
instrument, or an instrument the Feature Store doesn't track -- see that
class's own crypto-only default) behaves exactly as before this module
existed.

**Why RSI, not the EMA pair**: ``FeatureVector.ema_fast``/``ema_slow`` are
each normalized *independently* via a rolling Z-score against their own
history (see :mod:`~src.data_pipeline.feature_engine`'s docstring) --
comparing the two normalized values is not the same as comparing the
underlying raw EMAs, so it cannot reliably tell a real crossover apart
from two series that simply drifted differently within their own rolling
windows. ``rsi`` is normalized instead via a fixed linear rescale of its
known ``[0, 100]`` range, which preserves ordering exactly:
``normalized_rsi > 0`` if and only if the raw RSI is above the neutral
midpoint (``50``). That is the one field in the vector safe to read for
plain directional momentum without needing the raw (unpublished) value.
"""

from __future__ import annotations

from typing import Final, Literal

from core.schemas import FeatureVector

# Multiplicative confidence adjustment when RSI's momentum read
# agrees/disagrees with the model's own directional signal. Deliberately
# modest: this is a secondary confirmation, not a second vote -- it should
# nudge conviction, not swing a HOLD into a trade or vice versa (the
# caller's own decision_threshold, applied before this runs, already drew
# that line).
_AGREEMENT_BOOST: Final[float] = 1.10
_DISAGREEMENT_PENALTY: Final[float] = 0.70


def apply_feature_confirmation(
    signal: Literal[-1, 0, 1],
    confidence: float,
    feature_vector: FeatureVector | None,
) -> tuple[Literal[-1, 0, 1], float]:
    """Adjust ``confidence`` using ``feature_vector``'s RSI momentum as a second opinion.

    Args:
        signal: The model's own directional call. Never changed by this
            function -- only ``confidence`` is adjusted.
        confidence: The model's own confidence, in ``[0, 1]``.
        feature_vector: The instrument's latest normalized feature
            snapshot, or ``None`` if the Feature Store has no history for
            it yet (e.g. a forex instrument, which the crypto-only
            default Feature Store doesn't track today -- see that
            module's docstring). ``None`` is a normal, expected input,
            not an error: it means "behave exactly as before this
            confirmation layer existed."

    Returns:
        ``(signal, adjusted_confidence)``. ``adjusted_confidence`` is
        clipped to ``[0, 1]``.
    """
    if feature_vector is None or signal == 0:
        return signal, confidence

    if feature_vector.rsi > 0:
        momentum_direction: Literal[-1, 0, 1] = 1
    elif feature_vector.rsi < 0:
        momentum_direction = -1
    else:
        momentum_direction = 0

    if momentum_direction == 0:
        return signal, confidence
    if momentum_direction == signal:
        adjusted = confidence * _AGREEMENT_BOOST
    else:
        adjusted = confidence * _DISAGREEMENT_PENALTY

    return signal, min(1.0, max(0.0, adjusted))
