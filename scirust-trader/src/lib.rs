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

pub mod agent;
pub mod backtest;
pub mod certify;
pub mod chart;
pub mod cli;
pub mod dashboard;
pub mod execution;
pub mod indicators;
pub mod market;
pub mod marketmaking;
pub mod metrics;
pub mod microstructure;
pub mod model;
pub mod optimize;
pub mod orderbook;
pub mod orders;
pub mod pairs;
pub mod patterns;
pub mod portfolio;
pub mod portfolio_opt;
pub mod proof;
pub mod regime;
pub mod risk;
pub mod robustness;
pub mod scanner;
pub mod strategy;
pub mod wallet;
