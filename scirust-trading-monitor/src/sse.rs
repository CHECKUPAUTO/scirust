//! Transforme les channels broadcast du bus en streams SSE.
//!
//! Chaque connexion HTTP `GET /stream/X` subscribe au canal X et reçoit
//! les events en tant qu'événements SSE. Si le client ralentit, on lui
//! laisse rater des events (les broadcasts sont bornés).

use crate::state::MonitorState;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use futures_util::stream::{Stream, StreamExt};
use std::convert::Infallible;
use std::time::Duration;
use tokio_stream::wrappers::BroadcastStream;

pub async fn stream_market(
    State(state): State<MonitorState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let rx = state.bus.market.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|res| async move {
        match res {
            Ok(market) => serde_json::to_string(&market)
                .ok()
                .map(|json| Ok(Event::default().event("market").data(json))),
            Err(_) => None, // lag → on ignore silencieusement
        }
    });
    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(state.cfg.sse_keep_alive_secs))
            .text("keep-alive"),
    )
}

pub async fn stream_news(
    State(state): State<MonitorState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let rx = state.bus.news.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|res| async move {
        match res {
            Ok(event) => serde_json::to_string(&event)
                .ok()
                .map(|json| Ok(Event::default().event("news").data(json))),
            Err(_) => None,
        }
    });
    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(state.cfg.sse_keep_alive_secs))
            .text("keep-alive"),
    )
}

pub async fn stream_bars(
    State(state): State<MonitorState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let rx = state.bus.bars.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|res| async move {
        match res {
            Ok(bar) => serde_json::to_string(&bar)
                .ok()
                .map(|json| Ok(Event::default().event("bar").data(json))),
            Err(_) => None,
        }
    });
    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(state.cfg.sse_keep_alive_secs))
            .text("keep-alive"),
    )
}

pub async fn stream_decisions(
    State(state): State<MonitorState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // Si pas de shadow attaché → stream vide
    let rx = match state.shadow.as_ref() {
        Some(s) => s.subscribe(),
        None => {
            // On crée un channel vide qui ne reçoit jamais rien
            let (tx, rx) = tokio::sync::broadcast::channel(1);
            drop(tx);
            rx
        }
    };
    let stream = BroadcastStream::new(rx).filter_map(|res| async move {
        match res {
            Ok(decision) => serde_json::to_string(&decision)
                .ok()
                .map(|json| Ok(Event::default().event("decision").data(json))),
            Err(_) => None,
        }
    });
    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(state.cfg.sse_keep_alive_secs))
            .text("keep-alive"),
    )
}
