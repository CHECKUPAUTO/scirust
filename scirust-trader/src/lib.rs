//! `scirust-trader` — auditable, agent-drivable crypto-trading toolbox.
//!
//! A pure-Rust, deterministic trading stack that gives an agentic LLM the
//! capabilities of a professional crypto platform — indicators, pattern
//! recognition, order-book microstructure, an order/matching engine, portfolio
//! accounting, performance/risk metrics, strategies, an event-driven
//! backtester, an opportunity scanner, micro-order execution algorithms, market
//! making, and SVG charting — all exposed to any MCP agent via `scirust-mcp`.
//!
//! ```text
//!  data ─► indicators ─► patterns ─┐
//!                                   ├─► strategy ─► backtest ─► metrics ─► scanner ─► proof
//!  orderbook ─► orders ─► portfolio ┘         └─► execution / marketmaking / microstructure
//!                    ├─► DCA planning
//!                    ├─► bounded Grid planning
//!                    ├─► cross-venue arbitrage analysis
//!                    ├─► spot/perp basis scenario analysis
//!                    ├─► funding-carry stress scenarios
//!                    ├─► cross-exchange market-making analysis
//!                    └─► cost-aware basket rebalancing
//!  venue events ─► normalized contracts ─► lifecycle ─► replay / reconciliation
//!                                      ├─► latency / queue / back-pressure models
//!                                      └─► multi-leg imbalance / recovery plans
//!  indicators ─► core catalogue + adaptive / OHLC-volatility / flow gaps
//!  ML data ─► temporal leakage checks ─► linear/tree/forest/boosting/sequence baselines
//!                  └─► existing deterministic MLP / SciRust RL components
//!  derivatives ─► funding / basis / OI / liquidations ─► strategy / scanner
//!             ├─► rolling history / price-OI regimes
//!             └─► divergences / liquidation clusters
//!  model ─► certify ─► agent+LLM ─► proof   (certified, LLM-bounded prediction)
//! ```
//!
//! Design rules
//! ------------
//! 1. **Determinism first** — every numeric step uses pinned reduction order;
//!    same inputs ⇒ same outputs and same proof hashes.
//! 2. **Simulation first** — fills are simulated by a paper matching engine; no
//!    real-money order execution is exposed. Live market data is opt-in behind
//!    the `live` feature.
//! 3. **LLM never decides blind** — the certified-prediction path emits an
//!    IBP-bounded prediction the LLM cannot exceed; the scanner attaches
//!    backtested evidence to every recommendation.
//! 4. **Every decision is sealed** — proofs carry SHA-256 manifests for
//!    third-party replay/audit.

pub mod advanced_indicators;
pub mod agent;
pub mod arbitrage;
pub mod backtest;
pub mod basis;
pub mod basket_rebalance;
pub mod certify;
pub mod chart;
pub mod cli;
pub mod cross_marketmaking;
pub mod dashboard;
pub mod dca;
pub mod derivatives;
pub mod derivatives_context;
pub mod derivatives_history;
pub mod execution;
pub mod execution_realism;
pub mod funding_carry;
pub mod grid;
pub mod indicators;
pub mod market;
pub mod marketmaking;
pub mod metrics;
pub mod microstructure;
pub mod ml_baselines;
pub mod ml_dataset;
pub mod model;
pub mod multileg;
pub mod optimize;
pub mod options;
pub mod order_lifecycle;
pub mod orderbook;
pub mod orders;
pub mod pairs;
pub mod patterns;
pub mod portfolio;
pub mod portfolio_opt;
pub mod proof;
pub mod reconciliation;
pub mod regime;
pub mod risk;
pub mod robustness;
pub mod runtime_replay;
pub mod scanner;
pub mod strategy;
pub mod venue;
pub mod wallet;
