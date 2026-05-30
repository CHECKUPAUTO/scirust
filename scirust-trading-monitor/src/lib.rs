//! `scirust-trading-monitor` — serveur HTTP de monitoring.
//!
//! Expose le bus d'événements et les queries persistence en HTTP léger pour
//! qu'un dashboard web (Leptos, Yew, ou même curl) puisse l'observer en
//! temps réel.
//!
//! ## Endpoints
//!
//! ### SSE (Server-Sent Events) temps réel
//! - `GET /stream/market`     — flux `MarketState`
//! - `GET /stream/news`       — flux `CodifiedEvent`
//! - `GET /stream/decisions`  — flux `Decision`
//! - `GET /stream/bars`       — flux `Bar`
//!
//! ### REST queries
//! - `GET /api/events/recent?limit=50&category=macro`
//! - `GET /api/decisions/recent?limit=50&action=open`
//! - `GET /api/decisions/stats?from=ms&to=ms`
//! - `GET /api/performance?from=ms&to=ms` — outcomes + stats
//! - `GET /api/portfolio` — snapshot du portefeuille virtuel shadow
//! - `GET /api/health` — heartbeat + métriques
//!
//! ## Démarrage
//!
//! ```no_run
//! use scirust_trading_monitor::{MonitorServer, MonitorConfig};
//! # async fn run() {
//! # let bus = scirust_trading_core::EventBus::new();
//! # let api = std::sync::Arc::new(scirust_trading_persistence::QueryApi::open_in_memory().unwrap());
//! let server = MonitorServer::new(MonitorConfig::default(), bus, api, None);
//! server.serve().await.unwrap();
//! # }
//! ```

pub mod routes;
pub mod sse;
pub mod state;

pub use state::{MonitorConfig, MonitorServer, MonitorState};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum MonitorError {
    #[error("bind error: {0}")]
    Bind(String),

    #[error("server error: {0}")]
    Server(String),
}

pub type MonitorResult<T> = Result<T, MonitorError>;
