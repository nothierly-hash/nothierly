# Mouhamed Ngom
### Systems Architect & Low-Latency Infrastructure Engineer

> *“Engineering deterministic, high-throughput asynchronous execution engines where microsecond predictability and absolute risk containment meet.”*

# Flagship Project : Nothierly HFT Engine
**Nothierly** is an institutional-grade, asynchronous trading execution engine built from scratch (80,000+ lines of CPython & ZeroMQ) to bridge the gap between quantitative predictive models and lightning-fast market execution.

### Audited Benchmark Metrics (150x Real-Time Stress Test)
*   **Tick-to-Decision Latency (p50) :** `126 µs` (Median) / `897 µs` (p99.9) under heavy load.
*   **Memory Footprint (Zero Leak) :** Strictly flat RAM utilization between **293 MB and 439 MB** across **912,179 consecutive ticks**.
*   **Fail-Closed Risk Gates :** Deterministic safety triggers that automatically lock out stale market signals (e.g., rejecting 9,930 out-of-sync signals during a 29-second pipeline lag).
*   **Fault Tolerance :** Validated via continuous 24-hour shadow runs in degraded conditions, forcing 100% of unauthorized orders to be securely blocked by the Kill Switch.
*   **Quality Assurance :** Backed by a rigorous test suite of **1,447 automated tests** utilizing mock clocks to eliminate non-deterministic regressions.

## 🛠️ Core Technical Stack

*   **Languages :** C, C++, Python (CPython internals, GIL mitigation, low-level memory profiling), SQL.
*   **Low-Latency & Concurrency :** ZeroMQ (PUB/SUB, ROUTER/DEALER), Asynchronous I/O, Event-Driven Architectures.
*   **Reliability & Safety :** Deterministic Stress-Testing, Mock Clocks, Automated CI/CD Pipelines, Fail-Closed Risk Logic.
*   **Infrastructure & Tooling :** Docker, Git, Redis, Linux Kernel Tuning, Binary Protocol Decoders (Dukascopy `.bi5` native).

## 📐 Engineering Philosophy

1. **Determinism over Raw Speed :** A fast system that crashes or leaks memory during market volatility is a liability. Stability and predictability always come first.
2. **Fail-Closed by Design :** When anomaly detection triggers or data pipelines stall, the system must choose capital preservation over execution.
3. **Transparency in Testing :** Performance claims are meaningless without rigorous benchmarking, reproducible test suites, and strict boundary documentation.

## 📫 Connect & Collaborate

*   **GitHub :** [@MouhamedNgom](https://github.com/) 
*   **Focus :** High-Frequency Trading Infrastructures, Quantitative Execution Systems, Asynchronous Market Gateways.
