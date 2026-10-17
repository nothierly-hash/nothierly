# Replacement-Cost Valuation Analysis — Nothierly IP

**Purpose:** Build-vs-Buy estimate of the cost to recreate the Nothierly codebase, for seller-side price planning.
**Status:** Analytical estimate. Not a fairness opinion, not a buyer-side document, and not to be presented as one. Compensation, productivity and price inputs are assumptions, marked as such, and should be checked against current market data.
**Method:** Replacement cost plus schedule premium. No revenue multiples, no Sharpe-based value.

## 0. Corrections to the requested inputs

Several inputs in the request do not match the repository. They were replaced with the verified facts below, and the estimate follows those facts.

| Requested input | Repository finding | Treatment |
|---|---|---|
| ">80,000 lines of C++/Python" | 0 C++ files. 100,504 lines of Python (75,727 source, 24,637 tests), git-tracked, excluding archive and generated gRPC code | "Lead C++ HFT Architect" priced as Lead Python systems architect. Scope measured in Python LOC |
| Navier-Stokes "blocks overfitting" | The module is a disclosed illustrative simulation on a simulated market state, not a live liquidity model | No value assigned |
| Vast.ai GPU arbitrage plus Storj | Vast.ai only. Storj was never built (`scripts/audit_matrix.py`) | OpEx line covers GPU rental only |
| "6,618 unauthorized executions blocked; 500 packet losses survived, zero leaks" | 6,618 = consensus-approved signals blocked by the global Kill Switch while it was latched on feed loss, in a mock-mode run with zero recorded fills. 500 = Binance WebSocket reconnects. No memory-leak verdict is recorded for that run | No insurance value assigned (Section 3.4) |
| "Day-1 plug-in to FIX/LMAX" | A FIX server module exists, plus ccxt, Oanda and IBKR adapters. No LMAX connector, and no venue conformance certification evidenced | No day-1 premium. Schedule premium only (Section 3.3) |

## 1. Verified scope

| Component | Lines | Treatment |
|---|---|---|
| Engine: `core`, `src/agents`, `src/execution`, `risk`, `execution`, `api_middleware`, data pipeline, MLOps, RL, pricing, backtesting | ~40,000 | Full weight |
| Test suite | ~24,600 | Full weight |
| Scripts, research, root modules | ~12,400 | 50% weight |
| Dashboard UI, `nothierly_genesis` (R&D, not integrated), retail gateway, other | ~23,000 | Zero weight. A fund would not pay to recreate these |
| **Scope-weighted total** | **~71,000** | |

## 2. Assumptions (all adjustable)

- FX: 1 EUR = 1.10 USD.
- Fully loaded annual cost, tier-1 fund, NY/London: Lead architect $450k (350–650k). Quant researcher $400k (300–600k). DevOps/data engineer $250k (200–330k). Blended $375k per person-year.
- Production-grade output including tests, design and review: 12,000 LOC per person-year (8,000 to 20,000).
- Team-formation, coordination and rework multiplier: 1.4.
- Schedule/execution-risk premium over replacement cost: 15% / 20% / 30% (judgment, not measurement).
- GPU rental differential vs on-demand institutional rates: $2.0–3.0 per GPU-hour, 1,500–20,000 GPU-hours per year (no usage data exists; illustrative).

## 3. Line items

### 3.1 Talent acquisition and R&D CapEx

Two methods:

- **A. Scope-based** (what a diligence team will run after reading the repo): 71,000 LOC ÷ 12,000 = 5.9 person-years × $375k × 1.4.
- **B. Prescribed program** (1 architect, 2 quant researchers, 1 DevOps/data engineer, 36 months): $1.50M per year, $4.5M base.

Method B assumes 12 person-years. The existing scope supports about 6. It is an upper bound, not a replacement of what exists.

### 3.2 Infrastructure OpEx

The savings from GPU rental vs institutional on-demand compute are immaterial at any plausible retraining volume (5,000 GPU-hours per year × $2.5 × 3 years ≈ $37k). Storj savings are $0 because nothing was built. Colocation and compute racks are not part of the asset.

### 3.3 Time-to-market / schedule premium

Months saved are already priced in 3.1 as avoided burn, so adding them again would double count. The only separable element is a premium for schedule certainty, taken as a percentage of 3.1. Live-readiness work remains regardless of buy or build: the last 24-hour run analyses to ATTENTION, it ran in mock mode, and no venue certification is evidenced.

### 3.4 Risk avoidance

Capital preserved in the evidence is $0: the run was mock-mode with zero fills. An insurance value would need assumed capital at risk, event probability and severity, none of which the evidence supports. The engineering effort behind the Kill Switch and chaos harness is already inside 3.1. Assigned value: $0.

## 4. Summary

| Category | Estimated Cost ($M), base | Range ($M) |
|---|---|---|
| 3.1A Replacement cost, scope-based | 3.10 | 1.9 – 4.7 |
| 3.2 OpEx savings (GPU rental only) | 0.04 | 0.01 – 0.18 |
| 3.3 Schedule premium (20% of 3.1A) | 0.62 | 0.3 – 1.4 |
| 3.4 Risk avoidance | 0.00 | 0.00 |
| **Total, scope-based** | **3.8 (€3.4M)** | **2.2 – 6.3 (€2.0M – €5.7M)** |
| 3.1B Replacement cost, prescribed 12 person-year program | 4.50 | 3.5 – 6.5 |
| **Total, program-based** (with 3.2 and 20% premium) | **5.4 (€4.9M)** | **4.0 – 8.6 (€3.6M – €7.8M)** |
| Sensitivity: AI-assisted build (2–3× productivity), scope-based | 1.5 (€1.4M) | 1.3 – 1.9 (€1.2M – €1.7M) |

## 5. Conclusion

The requested conclusion, that €6.0M–€8.0M is a discount to internal build cost, is not supported by these numbers.

- Against the verified scope, €6.0M–€8.0M is 1.8–2.4× the base-case replacement cost.
- Against the full 12 person-year program, it is 1.2–1.6× the base case.
- The range is reached only in the program-based high case (€7.8M). That requires every assumption at its ceiling at once: top-of-market compensation, a team twice the size the existing scope needs, and a 30% premium.

At €6M–€8M the price is a premium to replacement cost. A premium can be justified, but not by this method.

A buyer's diligence team will most likely run the scope-based method. They will also discount for single-author key-person risk, shadow-only status, no audited live P&L, and lower build cost with AI-assisted development (expect questions on authorship and development process). Acqui-hire consideration is also commonly paid as retention compensation over several years rather than an upfront lump sum.

My planning range for a realistic outcome in the current state is roughly €0.5M–€3M total consideration. Confidence is low and I have no transaction data, so engage an M&A advisor before anchoring on any figure.

## 6. What legitimately moves the number

1. A clean repeat run of 24 hours or more with the corrected analyzer, reaching GO.
2. Audited paper or live P&L with realistic slippage and fees.
3. An independent code and security audit.
4. Working, certified connectivity to a real venue (FIX conformance or live IBKR).
5. Competing interest from two or more buyers.

Each of these is evidence a buyer can verify, and none of them depends on claims that fail the first repo inspection.
