//! Shared state and server bootstrap.

use crate::{MonitorError, MonitorResult};
use axum::Router;
use scirust_trading_core::EventBus;
use scirust_trading_engine::ShadowEvaluator;
use scirust_trading_persistence::QueryApi;
use std::net::SocketAddr;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct MonitorConfig {
    pub bind_addr: SocketAddr,
    /// Capacité initiale du buffer SSE pour les events backlogged
    pub sse_keep_alive_secs: u64,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            bind_addr: "127.0.0.1:7460".parse().unwrap(),
            sse_keep_alive_secs: 15,
        }
    }
}

/// État partagé entre les handlers axum.
#[derive(Clone)]
pub struct MonitorState {
    pub bus: EventBus,
    pub api: Arc<QueryApi>,
    pub shadow: Option<Arc<ShadowEvaluator>>,
    pub cfg: MonitorConfig,
    pub started_at: chrono::DateTime<chrono::Utc>,
}

pub struct MonitorServer {
    state: MonitorState,
}

impl MonitorServer {
    pub fn new(
        cfg: MonitorConfig,
        bus: EventBus,
        api: Arc<QueryApi>,
        shadow: Option<Arc<ShadowEvaluator>>,
    ) -> Self {
        Self {
            state: MonitorState {
                bus,
                api,
                shadow,
                cfg,
                started_at: chrono::Utc::now(),
            },
        }
    }

    /// Construit le router axum avec toutes les routes.
    pub fn router(&self) -> Router {
        crate::routes::build_router(self.state.clone())
    }

    /// Lance le serveur. Bloque jusqu'à un Ctrl+C ou erreur.
    pub async fn serve(self) -> MonitorResult<()> {
        let addr = self.state.cfg.bind_addr;
        let router = self.router();
        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .map_err(|e| MonitorError::Bind(e.to_string()))?;
        tracing::info!("monitor listening on http://{addr}");
        axum::serve(listener, router.into_make_service())
            .await
            .map_err(|e| MonitorError::Server(e.to_string()))?;
        Ok(())
    }
}
