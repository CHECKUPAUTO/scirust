# SciRust Trading Competitive Roadmap

Status date: 2026-09-06

## Goal

Make `scirust-trader` the strongest auditable open-source quantitative crypto-trading core in the comparison set, measured by **verified capability coverage, deterministic behavior, research validation, and agent usability**.

This roadmap does **not** define success as guaranteed profitability. Every quantitative claim must be demonstrated by reproducible tests or benchmarks. Existing SciRust invariants remain dominant: deterministic reductions, simulation-first execution, evidence-backed recommendations, and proof-sealed decisions.

## External reference set

The comparison baseline is the current public code of:

- Freqtrade / FreqAI: https://github.com/freqtrade/freqtrade
- Hummingbot: https://github.com/hummingbot/hummingbot
- NautilusTrader: https://github.com/nautechsystems/nautilus_trader
- Jesse: https://github.com/jesse-ai/jesse
- OctoBot: https://github.com/Drakkar-Software/OctoBot

These projects are references for capability discovery and validation. SciRust implementations must be original and compatible with their source licenses; do not copy incompatible source code.

## Existing SciRust strengths to preserve

`scirust-trader` already contains:

- deterministic technical indicators and strategy primitives;
- event-driven backtesting and performance/risk metrics;
- train/holdout and walk-forward parameter optimization;
- order-book depth, micro-price, imbalance, VWAP-to-fill and slippage analysis;
- OFI, trade-flow imbalance, VPIN and Kyle lambda;
- Avellaneda-Stoikov and GLFT-style market-making primitives;
- TWAP, VWAP, POV, Iceberg and Almgren-Chriss execution schedules;
- statistical arbitrage with hedge-ratio fitting, spread stationarity, mean-reversion half-life, Hurst and pair scanning;
- market-regime detection using realized volatility, normalized trend, Hurst and empirical Markov transitions;
- portfolio, proof, certification, scanner, agent and MCP-oriented surfaces.

The roadmap extends these foundations rather than replacing them.

---

## Phase 1 — Crypto derivatives intelligence

### Objective

Close the largest market-model gap versus crypto-native platforms: perpetual futures state.

### 1A. Core derivatives state — COMPLETE

Implemented deterministic primitives for:

- spot/perpetual basis;
- mark/index basis;
- funding-rate normalization and signed funding cash-flow accounting;
- open-interest change;
- long/short liquidation imbalance;
- a serializable report usable by strategies, scanners and MCP tools.

### 1B. Historical derivatives features — IN PROGRESS

Implemented first historical slice:

- funding trailing z-score and empirical percentile;
- basis trailing z-score and empirical percentile;
- one-step and configurable-lookback open-interest momentum;
- open-interest acceleration from adjacent percentage changes;
- liquidation total, imbalance, trailing z-score and percentile;
- price/OI regime classification (price up/down x OI up/down).

Remaining historical work:

- liquidation-cluster detection with explicit configurable thresholds;
- divergence features combining price, funding, basis and OI;
- deterministic replay fixtures over longer timestamp-aligned histories.

### 1C. Validation

- unit tests for signs, units, invalid input and edge cases;
- deterministic replay tests;
- fixtures checked against independently calculated examples;
- no trading recommendation inferred from a single derivatives metric.

**Definition of done:** SciRust can represent and analyze the core perpetual-futures state needed by funding, basis and liquidation-aware strategies without exchange-specific code.

---

## Phase 2 — Strategy-family parity and extension

### Objective

Cover strategy families that Hummingbot/OctoBot expose explicitly but SciRust does not yet expose as first-class strategy objects.

Implement, in simulation-first form:

1. configurable DCA / Smart-DCA scheduling;
2. bounded grid strategy with explicit inventory accounting;
3. cross-venue arbitrage opportunity model including fees and executable depth;
4. spot/perpetual basis scanner;
5. funding-carry scanner with explicit funding-direction accounting;
6. cross-exchange market-making model using venue-specific quotes and transfer/latency assumptions;
7. basket/index rebalancing strategy.

Every strategy must expose the assumptions required for profitability calculations. Fees, slippage and liquidity cannot default silently to zero in any scanner that labels an opportunity as economically positive.

**Definition of done:** every strategy family verified in Hummingbot and OctoBot has either a SciRust first-class equivalent or a documented reason why SciRust intentionally omits it.

---

## Phase 3 — Execution realism and venue-neutral runtime contracts

### Objective

Match the execution-system strengths of NautilusTrader and crypto-native bots while preserving the separation between quantitative logic and venue I/O.

Implement:

- venue-neutral market-data adapter traits;
- normalized instrument metadata and exchange trading rules;
- order lifecycle state machine;
- partial fills, cancellations and amendments;
- maker/taker fee schedules per venue/instrument;
- latency and queue-position simulation;
- rate-limit/back-pressure model;
- disconnect/reconnect and deterministic event replay;
- reconciliation of local simulated state against imported exchange event streams.

Live order submission is a separate policy surface and must not be required by the quantitative core.

**Definition of done:** the same strategy and execution plan can be replayed deterministically over normalized event streams from multiple venues.

---

## Phase 4 — Indicator and feature-library coverage

### Objective

Use Jesse as the breadth reference while avoiding low-value duplication.

Process:

1. inventory Jesse's indicator catalogue;
2. map exact equivalents in SciRust;
3. rank missing indicators by independent information content and usage in published/systematic strategies;
4. implement high-value gaps first;
5. add cross-check fixtures against published formulas or independent implementations.

Priority families include:

- directional-movement / ADX family;
- Aroon family;
- adaptive moving averages;
- volatility estimators beyond ATR;
- volume/flow indicators;
- cycle/filter indicators where evidence justifies them.

**Definition of done:** SciRust has documented coverage for the reference catalogue and no important indicator family is missing solely because it was overlooked.

---

## Phase 5 — ML baselines for market data

### Objective

Match FreqAI's practical model breadth without weakening SciRust's deterministic/evidence-first architecture.

Add or integrate reproducible baselines for:

- decision trees and random forests;
- gradient-boosted trees;
- linear/logistic baselines;
- MLP and sequence baselines already supported by SciRust components;
- reinforcement-learning experiments through `scirust-rl-algo` only with explicit environment definitions and holdout evaluation.

Required validation:

- time-ordered train/validation/test splits;
- leakage detection;
- deterministic seeds where stochastic algorithms are used;
- feature provenance;
- baseline comparison against simple strategies;
- no promotion of a model solely from in-sample performance.

**Definition of done:** FreqAI model families have reproducible SciRust-native equivalents or well-defined bridges and are evaluated under stricter time-series validation.

---

## Phase 6 — Research-grade robustness

### Objective

Make SciRust harder to fool with backtests than the reference projects.

Extend the existing train/holdout + walk-forward system with, where mathematically appropriate:

- purged/embargoed cross-validation for overlapping labels;
- bootstrap confidence intervals for key metrics;
- multiple-testing correction for strategy searches;
- Deflated Sharpe Ratio;
- Probability of Backtest Overfitting / combinatorial validation;
- parameter-stability surfaces;
- regime-conditional performance;
- transaction-cost stress tests;
- Monte Carlo trade/order perturbation;
- reproducibility manifests for every research result.

**Definition of done:** every promoted strategy can carry a machine-readable robustness report and provenance proof.

---

## Phase 7 — Agent/MCP quantitative workbench

### Objective

Expose all high-value capabilities to an agent without forcing it to parse implementation details.

Add MCP/tool surfaces for:

- derivatives-state analysis;
- order-book and microstructure analysis;
- strategy construction;
- pair/stat-arb scanning;
- funding/basis/arbitrage scanning;
- execution-cost estimation;
- backtest and walk-forward validation;
- optimizer/robustness reports;
- portfolio/risk state;
- proof/provenance retrieval.

Every returned recommendation-like object must include the evidence, assumptions and data timestamp needed to audit it.

**Definition of done:** an agent can perform the full observe -> hypothesize -> simulate -> validate -> compare -> document loop through stable tool contracts.

---

## Phase 8 — Competitive acceptance matrix

For each release candidate, refresh a source-linked matrix against the reference repositories.

A capability counts as SciRust-complete only when all applicable boxes are true:

- [ ] algorithm/model implemented;
- [ ] unit tests;
- [ ] deterministic replay or reproducibility test;
- [ ] transaction-cost assumptions explicit;
- [ ] backtest integration;
- [ ] robustness validation;
- [ ] agent/MCP surface where useful;
- [ ] documentation with units/sign conventions;
- [ ] no unsupported performance claim.

The target is a verified capability superset of the reference projects' open-source quantitative cores, not a claim of guaranteed higher trading returns.

## Execution order

The implementation sequence is:

`derivatives intelligence -> derivatives history/features -> DCA/grid -> arbitrage/basis/funding scanners -> venue-neutral execution contracts -> indicator gaps -> ML baselines -> robustness extensions -> MCP coverage -> refreshed competitive matrix`

Work should advance in small mergeable PRs. Each PR must preserve the existing architecture and include its own validation evidence.
