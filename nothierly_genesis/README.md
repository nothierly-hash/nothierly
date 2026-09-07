# Nothierly Genesis

**Internal research division — post-deep-learning market microstructure.**
Isolated from the production execution stack (`core/`, `src/agents/`, `api_middleware/`) by design: every module in this package depends on at least one heavy, research-grade library (`torch`, `ripser`, `dowhy`, `snntorch`) that has no business on a latency-sensitive production import graph, and every result reported here is scoped exactly as far as it has actually been tested — no further.

## Read this section before any other

Every pillar below pairs real, working, tested code with a **Validation Status** line. That line is not a disclaimer bolted on for legal cover — it is the actual epistemic status of the result, and it governs what this code may be used for:

| Pillar | What is real and tested | What is *not* validated |
|---|---|---|
| §1 Topological Radar | Persistent homology computation (verified against a known analytic case); Fragility Index formula; real historical L2 data pipeline | The alert threshold — no real historical crash exists in this project's sample data to calibrate against |
| §2 Spiking Tick Network | LIF network architecture; real spike encoding; a **live, measured** latency benchmark | The brief's own latency claim — measured and found **false** on this hardware (§2.3) |
| §3 Causal Impact Engine | The DoWhy pipeline, identification, and refutation testing — validated to recover a *known* synthetic ground-truth effect to within sampling error | Any causal claim drawn from real market data — one uncontrolled trading day supports correlational evidence adjusted for a stated DAG, not a proven causal effect |

None of the three pillars has been trained, calibrated against a real historical event, or run in Shadow Mode. Treat every number below as *architecture-verified*, not *production-validated* — the distinction is the entire point of this document.

---

## §1 Topological Liquidity Radar

**File**: [`topological_crash_detector.py`](topological_crash_detector.py)

### 1.1 Mathematical foundation

A limit order book, observed over a sliding window of $N$ snapshots, is embedded as a point cloud $X \subset \mathbb{R}^3$, where each point is a *state transition*, not a raw level:

$$
x_i = \big(\, r_i,\ \Delta\text{OBI}_i,\ \Delta s_i \,\big), \qquad
r_i = \log\frac{m_{i+1}}{m_i}
$$

with $m_i$ the mid-price, $\text{OBI}_i$ the order-book imbalance, and $s_i$ the spread in basis points at snapshot $i$. Using *deltas* rather than levels is deliberate: raw price is not scale-invariant across regimes, so the cloud's shape would otherwise encode which price regime happened to prevail rather than the market's actual dynamics.

Given $X$, the Vietoris–Rips filtration builds a nested sequence of simplicial complexes $\{VR_\epsilon(X)\}_{\epsilon \ge 0}$, and persistent homology tracks the birth $b$ and death $d$ of each topological feature (connected components, $H_0$; loops, $H_1$) across $\epsilon$. This project computes the **Topological Fragility Index** as the total $H_1$ persistence:

$$
\Phi(X) = \sum_{(b,d) \in \text{Dgm}_1(X)} (d - b)^2
$$

a large, long-lived loop structure in the return/OBI/spread-delta cloud indicates the recent window's dynamics trace a persistent cyclical pattern rather than dissipating noise.

### 1.2 The Morse-theoretic foundation, precisely

Persistent homology is not a heuristic bolted onto topology — it is the computable expression of a classical fact from Morse theory. For a generic (Morse) function $f$ on a manifold, the homology of a sublevel set $f^{-1}(-\infty, a]$ changes only as $a$ crosses a *critical value* of $f$, and the change is completely determined by the index of the corresponding critical point (Edelsbrunner & Harer, 2010, Ch. VI). The Vietoris–Rips filtration parameter $\epsilon$ plays the role of $a$; the births and deaths in a persistence diagram are exactly the critical values at which the complex's homotopy type changes. Persistent homology is Morse theory made computable and, via the stability theorem (Cohen-Steiner, Edelsbrunner & Harer, 2007), stable under noise.

### 1.3 Real precedent, not an invented technique

Gidea & Katz (2018) demonstrated a rising $L^p$-norm of persistence landscapes computed from sliding windows of equity-index log-return vectors in the run-up to the 2007–2008 crash — real, peer-reviewed evidence that topological summary statistics carry early-warning signal in financial time series. This module transplants that methodology to a structurally different but related domain (single-instrument L2 order-book state rather than a cross-asset return vector), and uses total $H_1$ persistence rather than a full landscape norm — a simpler, directly computable relative of the same idea, not a reproduction of Gidea & Katz's own result.

**Library note**: `giotto-tda`, named in the original brief, is unusable on this stack — its last release (2021) hardcodes a Windows binary URL for a Python ABI tag removed in Python 3.8. `ripser` is the real substitute, verified against a synthetic circle point cloud (correctly recovering exactly one dominant $H_1$ feature) before use.

### 1.4 Production hand-off

`assess_fragility()` is pure decision logic — it never calls `core.security.trip_kill_switch()`. `publish_fragility_alert()` is the one integration point: it publishes a breached `FragilityAssessment` onto `TOPOLOGY_FRAGILITY_ALERT_TOPIC` via a real `ZmqPublisher`, the same bus `alert_agent.py` already uses for `EMERGENCY_HALT`. A real, separately-running production subscriber — not built in this package — would be the one to act on it. Genesis detects; production decides.

---

## §2 Neuromorphic Tick Processor

**File**: [`spiking_tick_network.py`](spiking_tick_network.py)

### 2.1 Leaky Integrate-and-Fire dynamics

Each `snntorch.Leaky` neuron implements the discretized LIF membrane equation:

$$
V_{\text{mem}}[t] = \beta \, V_{\text{mem}}[t-1] + I_{\text{in}}[t], \qquad
S[t] = \Theta\big(V_{\text{mem}}[t] - V_{\text{th}}\big)
$$

where $\beta$ is the membrane leak constant, $\Theta$ the Heaviside step function, and a spike resets $V_{\text{mem}}$. Each real order-book event (ADD/CANCEL) is encoded as exactly one nonzero input current at the input neuron matching its price-level bucket and side — a genuinely sparse, event-driven encoding, structurally analogous to real published work on spiking networks over event-camera (DVS) data.

### 2.2 What this architecture is real evidence *for*

Treating an L2 update stream as a sequence of discrete, asynchronous events rather than a densely-sampled time series is a sound, principled modeling choice on its own merits, independent of any latency claim — the market genuinely does not tick at a fixed clock rate.

### 2.3 What this architecture is real evidence *against*

The brief's claim — microsecond-scale inference, faster than an LSTM — was tested, not assumed, and found false on this hardware:

| Model | Parameters | Mean latency (200 trials, CPU) |
|---|---:|---:|
| `SpikingTickNetwork` | 1,539 | **≈76,500 µs** |
| `LSTMBaseline` | 22,211 | **≈1,050 µs** |

The spiking network — with 14× *fewer* parameters — was measured at **≈73× slower**. The real, structural reason: neuromorphic efficiency is a property of event-driven computation implemented **in silicon** (Loihi 2, TrueNorth, Akida), where a neuron receiving no spike performs no work at the hardware level. `snntorch` simulates the same discrete-time dynamics in software, via an explicit Python loop over every timestep — real, measurable overhead a single fused LSTM kernel call does not pay. This is not a implementation defect to fix; it is the correct, expected result of running spiking dynamics on von Neumann hardware, and it is the honest answer to what the brief asked for.

---

## §3 Causal Market Impact Oracle

**File**: [`causal_impact_engine.py`](causal_impact_engine.py)

### 3.1 The causal model, not a correlational one

Pearl's causal hierarchy (Pearl, 2009) distinguishes *association* — $P(Y \mid X)$ — from *intervention* — $P(Y \mid \text{do}(X))$. This module computes the latter. Given the structural causal model implied by the DAG

$$
\text{Volume} \to \text{Spread} \to \text{Adverse\_Selection} \to \text{Price\_Slippage}, \qquad
\text{Volume} \to \text{Price\_Slippage}
$$

the treatment effect of an action $\text{do}(\text{Volume} = v)$ is identified via the backdoor adjustment formula:

$$
P(Y \mid \text{do}(X = x)) = \sum_{z} P(Y \mid X = x, Z = z)\, P(Z = z)
$$

for any $Z$ satisfying the backdoor criterion relative to $(X, Y)$ — computed automatically by DoWhy's graph-based identification algorithm from the DAG above, not assumed by the caller.

### 3.2 The DAG is a real theory, not an invented structure

Each edge encodes a specific, cited result from market microstructure theory:

- $\text{Volume} \to \text{Spread}, \text{Price}$: Kyle (1985) — informed order flow moves price through an adverse-selection channel.
- $\text{Spread} \to \text{Adverse\_Selection}$: Glosten & Milgrom (1985) — spread as compensation for informed-trader risk.
- $\text{Adverse\_Selection} \to \text{Price\_Slippage}$: Almgren & Chriss (2000) — the standard market-impact/execution-cost model.

### 3.3 How the pipeline is actually validated

The true causal effect of an intervention is never directly observable in real, unrandomized market data — there is no ground truth to check an estimate against. `generate_synthetic_market_data()` instead constructs data *from* the DAG's own structural equations with known coefficients; the estimation pipeline is validated by confirming DoWhy recovers that known effect:

$$
\hat{\tau} = 57.93, \qquad \tau_{\text{true}} = \beta_{\text{direct}} + \beta_{V \to S}\,\beta_{S \to A}\,\beta_{A \to P} \times 100 = 58.00
$$

Every real refutation test (`random_common_cause`) is run and checked, not treated as decorative — DoWhy's refutation suite is the actual empirical safeguard Pearl's framework has against a misspecified identification assumption, since no observational method can *prove* a causal graph is correct.

**Library note**: DoWhy is pinned to 0.8 — every later release requires Python < 3.14. Version 0.8 calls a `networkx` API renamed in newer `networkx` releases; a single, guarded compatibility alias (`_patch_networkx_compat()`) is applied, disclosed in the module docstring, and verified end-to-end against the synthetic ground-truth case above before adoption.

---

## Path to production, honestly sequenced

1. **Shadow Mode incubation** — run each pillar against live data for an extended period, logging every signal/decision without acting on it, and compare against realized outcomes. This project already has a real `execution_mode=SHADOW` concept (`api_middleware/gateway.py`) this would extend, not duplicate.
2. **Calibration against real events** — the Fragility Index threshold and the SNN's classification head are architecture, not fitted parameters; both need real labeled historical data (crash events, realized next-tick outcomes) this project's current sample data does not contain.
3. **ZeroMQ integration into the live agent fleet** — §1.4 is the template: Genesis publishes, a real production subscriber decides. The same pattern extends naturally to §2 and §3.

No step above is complete. This document states that plainly because the actual scientific and engineering value of this package is in the real, verifiable primitives it gets right — not in appearing further along than it is.

## References

- Cohen-Steiner, D., Edelsbrunner, H., & Harer, J. (2007). *Stability of Persistence Diagrams*. Discrete & Computational Geometry.
- Edelsbrunner, H., & Harer, J. (2010). *Computational Topology: An Introduction*. American Mathematical Society.
- Gidea, M., & Katz, Y. (2018). *Topological Data Analysis of Financial Time Series: Landscapes of Crashes*. Physica A, 491.
- Glosten, L. R., & Milgrom, P. R. (1985). *Bid, Ask and Transaction Prices in a Specialist Market with Heterogeneously Informed Traders*. Journal of Financial Economics.
- Almgren, R., & Chriss, N. (2000). *Optimal Execution of Portfolio Transactions*. Journal of Risk.
- Kyle, A. S. (1985). *Continuous Auctions and Insider Trading*. Econometrica.
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
