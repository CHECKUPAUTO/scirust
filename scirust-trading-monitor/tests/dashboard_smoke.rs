//! Smoke test : démarre le monitor, vérifie que / renvoie le dashboard,
//! puis émet un event news et vérifie que le client SSE le reçoit.

use scirust_trading_core::{CodifiedEvent, EventBus, EventTiming, SourceId};
use scirust_trading_monitor::{MonitorConfig, MonitorServer};
use scirust_trading_persistence::QueryApi;
use std::sync::Arc;
use std::time::Duration;

fn random_port() -> u16 {
    use std::net::{SocketAddr, TcpListener};
    let l = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0))).unwrap();
    l.local_addr().unwrap().port()
}

#[tokio::test]
async fn dashboard_loads_and_streams_an_event() {
    let port = random_port();
    let bus = EventBus::new();
    let api = Arc::new(QueryApi::open_in_memory().unwrap());
    let cfg = MonitorConfig {
        bind_addr: format!("127.0.0.1:{port}").parse().unwrap(),
        sse_keep_alive_secs: 15,
    };
    let server_bus = bus.clone();
    tokio::spawn(async move {
        MonitorServer::new(cfg, server_bus, api, None).serve().await.ok();
    });
    tokio::time::sleep(Duration::from_millis(200)).await;

    // 1. Dashboard HTML
    let html = reqwest::get(format!("http://127.0.0.1:{port}/"))
        .await.unwrap().text().await.unwrap();
    assert!(html.contains("scirust trading"));
    assert!(html.contains("EventSource"));

    // 2. Stream news + emit event
    let client = reqwest::Client::new();
    let bus2 = bus.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(150)).await;
        let mut ev = CodifiedEvent::builder(SourceId::new("smoke"), "smoke test news")
            .reliability(0.9)
            .timing(EventTiming::Observed(chrono::Utc::now()))
            .build();
        ev.tags = vec!["smoke".into()];
        let _ = bus2.news.send(ev);
    });

    let resp = client.get(format!("http://127.0.0.1:{port}/stream/news"))
        .send().await.unwrap();
    assert!(resp.status().is_success());

    let body = tokio::time::timeout(Duration::from_secs(3), async {
        use futures_util::StreamExt;
        let mut buf = Vec::with_capacity(2048);
        let mut s = resp.bytes_stream();
        while let Some(chunk) = s.next().await {
            buf.extend_from_slice(&chunk.unwrap());
            if buf.len() > 256 { break; }
        }
        String::from_utf8_lossy(&buf).to_string()
    }).await.unwrap();
    assert!(body.contains("smoke test news") || body.contains("event: news"),
        "got: {body:?}");
}
