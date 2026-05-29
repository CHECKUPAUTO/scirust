//! REST routes — queries persistence + health.

use crate::sse;
use crate::state::MonitorState;
use axum::extract::{Query, State};
use axum::http::{header, StatusCode};
use axum::response::{Html, IntoResponse};
use axum::routing::get;
use axum::{Json, Router};
use chrono::{DateTime, Utc};
use scirust_trading_persistence::OutcomeConfig;
use serde::{Deserialize, Serialize};

const DASHBOARD_HTML: &str = include_str!("../static/dashboard.html");

pub fn build_router(state: MonitorState) -> Router {
    Router::new()
        // Dashboard UI
        .route("/", get(dashboard))
        // SSE streams
        .route("/stream/market", get(sse::stream_market))
        .route("/stream/news", get(sse::stream_news))
        .route("/stream/bars", get(sse::stream_bars))
        .route("/stream/decisions", get(sse::stream_decisions))
        // REST
        .route("/api/health", get(health))
        .route("/api/events/recent", get(recent_events))
        .route("/api/decisions/recent", get(recent_decisions))
        .route("/api/decisions/stats", get(decision_stats))
        .route("/api/performance", get(performance_stats))
        .route("/api/portfolio", get(portfolio_snapshot))
        .with_state(state)
}

async fn dashboard() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        Html(DASHBOARD_HTML),
    )
}

// ─── Health ─────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    uptime_seconds: i64,
    started_at: DateTime<Utc>,
    bus_subscribers: BusSubscribers,
}

#[derive(Debug, Serialize)]
struct BusSubscribers {
    market: usize,
    news: usize,
    trades: usize,
    bars: usize,
    orders: usize,
}

async fn health(State(state): State<MonitorState>) -> impl IntoResponse {
    let uptime = (Utc::now() - state.started_at).num_seconds();
    Json(HealthResponse {
        status: "ok",
        uptime_seconds: uptime,
        started_at: state.started_at,
        bus_subscribers: BusSubscribers {
            market: state.bus.market.receiver_count(),
            news: state.bus.news.receiver_count(),
            trades: state.bus.trades.receiver_count(),
            bars: state.bus.bars.receiver_count(),
            orders: state.bus.orders.receiver_count(),
        },
    })
}

// ─── Events ─────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct RecentEventsQuery {
    #[serde(default = "default_limit_50")]
    limit: u32,
    category: Option<String>,
    /// "structural" | "semantic" | "contextual"
    min_enrichment: Option<String>,
}

fn default_limit_50() -> u32 {
    50
}

async fn recent_events(
    State(state): State<MonitorState>,
    Query(q): Query<RecentEventsQuery>,
) -> impl IntoResponse {
    // 🛡️ Sentinel: Enforce max limit to prevent DoS via large resource requests
    let limit = q.limit.min(100);

    let category = q.category.as_deref().and_then(parse_category);
    let min_enrich = q.min_enrichment.as_deref().and_then(parse_enrichment);
    match state
        .api
        .recent_events(limit, category, min_enrich)
        .await
    {
        Ok(events) => Ok(Json(events)),
        Err(e) => {
            // 🛡️ Sentinel: Log internal error and return generic message to prevent info leakage
            tracing::error!("Failed to fetch recent events: {e}");
            Err((StatusCode::INTERNAL_SERVER_ERROR, "Internal Database Error"))
        }
    }
}

fn parse_category(s: &str) -> Option<scirust_trading_core::Category> {
    use scirust_trading_core::Category::*;
    match s.to_lowercase().as_str() {
        "macro" => Some(Macro),
        "regulatory" => Some(Regulatory),
        "exchange_event" => Some(ExchangeEvent),
        "on_chain" => Some(OnChain),
        "narrative" => Some(Narrative),
        "technical" => Some(Technical),
        "liquidation" => Some(Liquidation),
        "funding" => Some(Funding),
        _ => None,
    }
}

fn parse_enrichment(s: &str) -> Option<scirust_trading_core::EnrichmentLevel> {
    use scirust_trading_core::EnrichmentLevel::*;
    match s.to_lowercase().as_str() {
        "raw" => Some(Raw),
        "structural" => Some(Structural),
        "semantic" => Some(Semantic),
        "contextual" => Some(Contextual),
        _ => None,
    }
}

// ─── Decisions ──────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct RecentDecisionsQuery {
    #[serde(default = "default_limit_50")]
    limit: u32,
    action: Option<String>,
}

async fn recent_decisions(
    State(state): State<MonitorState>,
    Query(q): Query<RecentDecisionsQuery>,
) -> impl IntoResponse {
    // 🛡️ Sentinel: Enforce max limit to prevent DoS
    let limit = q.limit.min(100);

    match state.api.recent_decisions(limit, q.action.as_deref()).await {
        Ok(d) => Ok(Json(d)),
        Err(e) => {
            // 🛡️ Sentinel: Prevent information leakage in error response
            tracing::error!("Failed to fetch recent decisions: {e}");
            Err((StatusCode::INTERNAL_SERVER_ERROR, "Internal Database Error"))
        }
    }
}

#[derive(Debug, Deserialize)]
struct RangeQuery {
    /// unix ms
    from: Option<i64>,
    /// unix ms
    to: Option<i64>,
}

impl RangeQuery {
    fn resolve(&self) -> (DateTime<Utc>, DateTime<Utc>) {
        let to = self
            .to
            .and_then(DateTime::<Utc>::from_timestamp_millis)
            .unwrap_or_else(Utc::now);
        let from = self
            .from
            .and_then(DateTime::<Utc>::from_timestamp_millis)
            .unwrap_or(to - chrono::Duration::days(7));
        (from, to)
    }
}

async fn decision_stats(
    State(state): State<MonitorState>,
    Query(q): Query<RangeQuery>,
) -> impl IntoResponse {
    let (from, to) = q.resolve();
    match state.api.decision_stats(from, to).await {
        Ok(s) => Ok(Json(StatsResponse {
            total: s.total,
            no_trade: s.no_trade,
            hold: s.hold,
            open: s.open,
            close: s.close,
            window_from: from,
            window_to: to,
        })),
        Err(e) => {
            tracing::error!("Failed to fetch decision stats: {e}");
            Err((StatusCode::INTERNAL_SERVER_ERROR, "Internal Database Error"))
        }
    }
}

#[derive(Debug, Serialize)]
struct StatsResponse {
    total: u64,
    no_trade: u64,
    hold: u64,
    open: u64,
    close: u64,
    window_from: DateTime<Utc>,
    window_to: DateTime<Utc>,
}

// ─── Performance ────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct PerformanceResponse {
    window_from: DateTime<Utc>,
    window_to: DateTime<Utc>,
    outcomes_count: u64,
    win_rate: f64,
    mean_return_bps: f64,
    median_return_bps: f64,
    std_return_bps: f64,
    stop_loss_hit_rate: f64,
    by_gate: serde_json::Value,
    by_bias: serde_json::Value,
    by_symbol: serde_json::Value,
}

async fn performance_stats(
    State(state): State<MonitorState>,
    Query(q): Query<RangeQuery>,
) -> impl IntoResponse {
    let (from, to) = q.resolve();
    let outcomes = match state
        .api
        .compute_outcomes_in_range(from, to, OutcomeConfig::typical())
        .await
    {
        Ok(o) => o,
        Err(e) => {
            tracing::error!("Failed to compute outcomes: {e}");
            return Err((StatusCode::INTERNAL_SERVER_ERROR, "Internal Database Error".to_string()));
        }
    };
    let stats = state.api.aggregate_stats(&outcomes);
    let by_gate = serde_json::to_value(&serialize_group_map(&stats.by_gate)).unwrap_or_default();
    let by_bias = serde_json::to_value(&serialize_group_map(&stats.by_bias)).unwrap_or_default();
    let by_symbol = serde_json::to_value(&serialize_group_map(&stats.by_symbol)).unwrap_or_default();
    Ok(Json(PerformanceResponse {
        window_from: from,
        window_to: to,
        outcomes_count: stats.n,
        win_rate: stats.overall.win_rate,
        mean_return_bps: stats.overall.mean_return_bps,
        median_return_bps: stats.overall.median_return_bps,
        std_return_bps: stats.overall.std_return_bps,
        stop_loss_hit_rate: stats.overall.stop_loss_hit_rate,
        by_gate,
        by_bias,
        by_symbol,
    }))
}

fn serialize_group_map(
    m: &std::collections::HashMap<String, scirust_trading_persistence::GroupStats>,
) -> std::collections::BTreeMap<String, GroupStatsDto> {
    m.iter()
        .map(|(k, v)| {
            (
                k.clone(),
                GroupStatsDto {
                    n: v.n,
                    win_rate: v.win_rate,
                    mean_return_bps: v.mean_return_bps,
                    median_return_bps: v.median_return_bps,
                    std_return_bps: v.std_return_bps,
                    stop_loss_hit_rate: v.stop_loss_hit_rate,
                },
            )
        })
        .collect()
}

#[derive(Debug, Serialize)]
struct GroupStatsDto {
    n: u64,
    win_rate: f64,
    mean_return_bps: f64,
    median_return_bps: f64,
    std_return_bps: f64,
    stop_loss_hit_rate: f64,
}

// ─── Portfolio ──────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct PortfolioResponse {
    equity_quote: f64,
    realized_pnl: f64,
    total_unrealized_pnl: f64,
    total_exposure: f64,
    position_count: usize,
}

async fn portfolio_snapshot(State(state): State<MonitorState>) -> impl IntoResponse {
    if let Some(shadow) = state.shadow.as_ref() {
        let p = shadow.portfolio_snapshot().await;
        Ok(Json(PortfolioResponse {
            equity_quote: p.equity_quote,
            realized_pnl: p.realized_pnl,
            total_unrealized_pnl: p.total_unrealized_pnl(),
            total_exposure: p.total_exposure(),
            position_count: p.position_count(),
        }))
    } else {
        Err((StatusCode::SERVICE_UNAVAILABLE, "no shadow attached".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    fn make_state() -> MonitorState {
        let bus = scirust_trading_core::EventBus::new();
        let api = std::sync::Arc::new(
            scirust_trading_persistence::QueryApi::open_in_memory().unwrap(),
        );
        MonitorState {
            bus,
            api,
            shadow: None,
            cfg: crate::MonitorConfig::default(),
            started_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn health_endpoint_works() {
        let state = make_state();
        let router = build_router(state);
        let req = Request::builder()
            .uri("/api/health")
            .body(Body::empty())
            .unwrap();
        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body_bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(body["status"], "ok");
        assert!(body["uptime_seconds"].as_i64().unwrap() >= 0);
    }

    #[tokio::test]
    async fn portfolio_endpoint_503_without_shadow() {
        let state = make_state();
        let router = build_router(state);
        let req = Request::builder()
            .uri("/api/portfolio")
            .body(Body::empty())
            .unwrap();
        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn recent_decisions_empty_returns_200() {
        let state = make_state();
        let router = build_router(state);
        let req = Request::builder()
            .uri("/api/decisions/recent?limit=10")
            .body(Body::empty())
            .unwrap();
        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body_bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert!(body.is_array());
        assert_eq!(body.as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn decision_stats_returns_zeros_on_empty() {
        let state = make_state();
        let router = build_router(state);
        let req = Request::builder()
            .uri("/api/decisions/stats")
            .body(Body::empty())
            .unwrap();
        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body_bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(body["total"].as_u64().unwrap(), 0);
        assert_eq!(body["open"].as_u64().unwrap(), 0);
    }

    #[tokio::test]
    async fn parse_category_known_values() {
        assert!(parse_category("macro").is_some());
        assert!(parse_category("MACRO").is_some()); // case-insensitive
        assert!(parse_category("on_chain").is_some());
        assert!(parse_category("bogus").is_none());
    }

    #[tokio::test]
    async fn dashboard_route_serves_html() {
        let state = make_state();
        let router = build_router(state);
        let req = Request::builder().uri("/").body(Body::empty()).unwrap();
        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let headers = resp.headers().clone();
        assert!(headers
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap()
            .contains("text/html"));
        let body_bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .unwrap();
        let html = std::str::from_utf8(&body_bytes).unwrap();
        assert!(html.contains("scirust trading"));
        assert!(html.contains("/stream/news"));
        assert!(html.contains("/api/portfolio"));
    }
}
