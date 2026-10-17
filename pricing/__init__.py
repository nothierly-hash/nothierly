"""Pre-trade cost estimation -- distinct from src/execution/reporting_engine.py's *post*-trade TCA.

The billing engine (Chapter 13, Étape 5.7) tells a client what
execution actually cost *after the fact*, from real fills. This
package answers a different question, *before* an order is ever sent:
what would it cost, given the real order book right now.
"""

from __future__ import annotations
