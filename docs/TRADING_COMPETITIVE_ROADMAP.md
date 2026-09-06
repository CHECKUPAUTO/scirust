# SciRust Trading Competitive Roadmap

Status date: 2026-09-06

## Goal

Make `scirust-trader` the strongest auditable open-source quantitative crypto-trading core in the comparison set, measured by **verified capability coverage, deterministic behavior, research validation, execution realism, and agent usability**.

This roadmap does **not** define success as guaranteed profitability. Every quantitative claim must be demonstrated by reproducible tests or benchmarks. Existing SciRust invariants remain dominant: deterministic reductions, simulation-first execution, evidence-backed recommendations, and proof-sealed decisions.

## External reference set

Current capability references:

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

## Phase 1 — Crypto derivatives intelligence — COMPLETE

### 1A. Core derivatives state — COMPLETE

Implemented deterministic primitives for:

- spot/perpetual basis;
- mark/index basis;
- funding-rate normalization and signed funding cash-flow accounting;
- open-interest change;
- long/short liquidation imbalance;
- a serializable derivatives report.

### 1B. Historical derivatives context — COMPLETE

Implemented:

- funding trailing z-score and empirical percentile;
- basis trailing z-score and empirical percentile;
- one-step and configurable-lookback open-interest momentum;
- open-interest acceleration from adjacent percentage changes;
- liquidation total, imbalance, trailing z-score and percentile;
- price/OI regime classification (price up/down x OI up/down);
- one-step mark return, funding change, basis change and OI change with explicit units;
- descriptive sign-divergence flags across price/funding/basis/OI;
- configurable liquidation-cluster detection using both z-score and percentile gates;
- deterministic repeated-analysis validation.

No single derivatives metric is promoted automatically into a trading recommendation.

**Definition of done:** satisfied for the venue-neutral quantitative layer. Exchange adapters and live derivatives feeds belong to Phase 3.

---

## Phase 2 — Strategy-family parity and extension — IN PROGRESS

### 2A. Deterministic DCA planning — IMPLEMENTED

Implemented foundation:

- explicit quote allocations and trigger/reference prices;
- maker/post-only and taker activation templates;
- instrument tick/lot/min-notional validation;
- requested vs rounded budget accounting;
- quantity-weighted entry reference;
- directional take-profit and stop-loss references.

Remaining runtime work: trailing stop, time-limit behavior and persisted execution lifecycle belong to the execution-runtime layer rather than the closed-bar `Strategy` trait.

### 2B. Bounded Grid planning — IMPLEMENTED

Implemented foundation:

- deterministic arithmetic levels inside inclusive bounds;
- explicit quote budget allocation;
- tick/lot/min-notional validation;
- duplicate/tight rounded-level rejection;
- post-only entry and reduce-only take-profit templates;
- `max_open_orders` throttle contract;
- explicit replayable level lifecycle state machine.

Remaining runtime work: enforcement of throttling, amendments/cancellation and reconciliation belongs to Phase 3.

### 2C. Cross-venue arbitrage — NEXT

Implement a venue-neutral opportunity model that only labels an opportunity economically positive after all applicable assumptions are explicit:

- buy and sell venues/instruments;
- executable depth for the requested quantity rather than top-of-book only;
- quote-currency conversion when venues use different quotes;
- maker/taker fees;
- slippage/market impact;
- network/gas/transfer costs where applicable;
- available inventory/balance constraints;
- minimum required net profitability;
- stale-quote / timestamp checks;
- leg-risk exposure if fills are not simultaneous.

**Acceptance rule:** gross spread alone can never imply profitability.

### 2D. Spot/perpetual basis scanner

Implement:

- executable spot and perpetual prices for a common size;
- explicit basis and annualized basis conventions;
- funding impact over a declared holding horizon;
- fees/slippage on both legs;
- collateral and leverage assumptions;
- basis convergence scenario rather than guaranteed convergence.

### 2E. Funding-carry scanner

Implement:

- signed funding direction;
- expected funding cash flow for a declared horizon;
- hedge-leg cost;
- basis drift sensitivity;
- fees/slippage and re-hedging costs;
- scenario/stress output rather than a single deterministic profit forecast.

### 2F. Remaining strategy families

- cross-exchange market-making model with venue-specific quotes and latency assumptions;
- basket/index rebalancing strategy with cost-aware turnover constraints.

**Phase 2 definition of done:** every strategy family verified in Hummingbot/OctoBot has a SciRust first-class equivalent or a documented intentional omission, and every economically-positive label exposes its cost/liquidity assumptions.

---

## Phase 3 — Execution realism and venue-neutral runtime contracts

### Objective

Match the execution-system strengths of NautilusTrader and crypto-native bots while preserving the separation between quantitative logic and venue I/O.

Implement:

- venue-neutral market-data adapter traits;
- normalized instrument metadata and exchange trading rules;
- explicit order lifecycle state machine;
- partial fills, cancellations and amendments;
- maker/taker fee schedules per venue/instrument;
- latency and queue-position simulation;
- rate-limit/back-pressure model;
- disconnect/reconnect and deterministic event replay;
- reconciliation of local simulated state against imported exchange event streams;
- leg-risk and recovery semantics for multi-venue strategies.

Live order submission is a separate policy surface and must not be required by the quantitative core.

**Definition of done:** the same strategy and execution plan can be replayed deterministically over normalized event streams from multiple venues.

---

## Phase 4 — Indicator and feature-library coverage

Use Jesse as the breadth reference while avoiding low-value duplication.

Process:

1. inventory Jesse's current indicator catalogue;
2. map exact equivalents in SciRust;
3. rank missing indicators by independent information content and usage in published/systematic strategies;
4. implement high-value gaps first;
5. add cross-check fixtures against published formulas or independent implementations.

Priority families include adaptive moving averages, additional volatility estimators, volume/flow indicators and justified cycle/filter indicators. Existing ADX/DMI support must be counted rather than reimplemented.

---

## Phase 5 — ML baselines for market data

Match FreqAI's practical model breadth without weakening SciRust's deterministic/evidence-first architecture.

Add or integrate reproducible baselines for:

- decision trees and random forests;
- gradient-boosted trees;
- linear/logistic baselines;
- MLP and sequence baselines supported by SciRust components;
- reinforcement-learning experiments through `scirust-rl-algo` only with explicit environment definitions and holdout evaluation.

Required validation:

- time-ordered train/validation/test splits;
- leakage detection;
- deterministic seeds where stochastic algorithms are used;
- feature provenance;
- baseline comparison against simple strategies;
- no promotion solely from in-sample performance.

---

## Phase 6 — Research-grade robustness

Extend the existing train/holdout, walk-forward and seeded Monte-Carlo machinery with, where mathematically appropriate:

- purged/embargoed cross-validation for overlapping labels;
- bootstrap confidence intervals;
- multiple-testing correction for strategy searches;
- Deflated Sharpe Ratio;
- Probability of Backtest Overfitting / combinatorial validation;
- parameter-stability surfaces;
- regime-conditional performance;
- transaction-cost stress tests;
- order/trade perturbation tests;
- reproducibility manifests for every promoted result.

**Definition of done:** every promoted strategy can carry a machine-readable robustness report and provenance proof.

---

## Phase 7 — Agent/MCP quantitative workbench

Expose high-value capabilities through stable agent contracts for:

- derivatives-state/history/context analysis;
- DCA/Grid plan construction;
- order-book and microstructure analysis;
- strategy construction and backtesting;
- pair/stat-arb scanning;
- funding/basis/arbitrage scanning;
- execution-cost estimation;
- walk-forward/optimizer/robustness reports;
- portfolio/risk state;
- proof/provenance retrieval.

Every recommendation-like object must include evidence, assumptions and data timestamps needed to audit it.

---

## Phase 8 — Competitive acceptance matrix

For each release candidate, refresh a source-linked matrix against the reference repositories.

A capability counts as SciRust-complete only when all applicable boxes are true:

- [ ] algorithm/model implemented;
- [ ] unit tests;
- [ ] deterministic replay or reproducibility test;
- [ ] transaction-cost assumptions explicit;
- [ ] liquidity/depth assumptions explicit;
- [ ] backtest/simulation integration;
- [ ] robustness validation;
- [ ] agent/MCP surface where useful;
- [ ] documentation with units/sign conventions;
- [ ] no unsupported performance claim.

The target is a verified capability superset of the reference projects' open-source quantitative cores, not a claim of guaranteed higher trading returns.

## Execution order

`converge Phase 1 + DCA/Grid into master -> cross-venue arbitrage -> basis scanner -> funding-carry scanner -> venue-neutral execution runtime -> indicator gaps -> ML baselines -> robustness extensions -> MCP coverage -> competitive matrix refresh`

Work advances in small mergeable PRs. Every PR must preserve the architecture and carry validation evidence. Stacked PRs are not considered integrated until their content reaches `master` and passes CI there.
