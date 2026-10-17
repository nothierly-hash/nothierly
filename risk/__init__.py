"""Cross-broker risk aggregation -- distinct from core/security.py's single-process Kill Switch.

``core/security.py`` answers "should any agent place a new order right
now" (a single global SAFE/PANIC flag). This package answers a
different, broker-spanning question: "across every real venue this
firm is connected to, how much of our real margin capacity is used,
and does anything need to happen about it right now."
"""

from __future__ import annotations
