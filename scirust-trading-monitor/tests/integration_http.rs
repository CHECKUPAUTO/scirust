//! Integration test : démarre le serveur monitor en vrai, fait des requêtes
//! HTTP, vérifie le SSE.

use scirust_trading_core::{
    Bar, BarKind, CodifiedEvent, Exchange, EventBus, EventTiming, SourceId,
    Symbol,
};
use scirust_trading_monitor::{MonitorConfig, MonitorServer};
use scirust_trading_persistence::QueryApi;
use std::sync::Arc;
use std::time::Duration;

fn random_port() -> u16 {
    // Pick a port and just trust that no one else is using it during the test
    use std::net::{TcpListener, SocketAddr};
    let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0))).unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    port
}

#[tokio::test]
async fn server_responds_to_health_via_real_http() {
    let port = random_port();
    let bus = EventBus::new();
    let api = Arc::new(QueryApi::open_in_memory().unwrap());
    let cfg = MonitorConfig {
        bind_addr: format!("127.0.0.1:{port}").parse().unwrap(),
        sse_keep_alive_secs: 15,
    };
    let server = MonitorServer::new(cfg, bus.clone(), api, None);
    let _h = tokio::spawn(async move {
        let _ = server.serve().await;
    });
    tokio::time::sleep(Duration::from_millis(150)).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://127.0.0.1:{port}/api/health"))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn sse_news_delivers_event() {
    let port = random_port();
    let bus = EventBus::new();
    let api = Arc::new(QueryApi::open_in_memory().unwrap());
    let cfg = MonitorConfig {
        bind_addr: format!("127.0.0.1:{port}").parse().unwrap(),
        sse_keep_alive_secs: 15,
    };
    let server_bus = bus.clone();
    let server = MonitorServer::new(cfg, server_bus, api, None);
    let _h = tokio::spawn(async move {
        let _ = server.serve().await;
    });
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Lance le subscriber HTTP
    let client = reqwest::Client::new();
    let stream_fut = client
        .get(format!("http://127.0.0.1:{port}/stream/news"))
        .send();

    // Émet un event après un petit délai
    let bus_emit = bus.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        let mut ev = CodifiedEvent::builder(SourceId::new("test"), "Hello world")
            .reliability(0.9)
            .timing(EventTiming::Observed(chrono::Utc::now()))
            .build();
        ev.tags = vec!["test".into()];
        let _ = bus_emit.news.send(ev);
    });

    let resp = tokio::time::timeout(Duration::from_secs(3), stream_fut)
        .await
        .unwrap()
        .unwrap();
    assert!(resp.status().is_success());

    // Lire les premiers ~512 bytes du body — devrait contenir notre event
    let body_text = tokio::time::timeout(Duration::from_secs(3), async move {
        let mut buf = Vec::with_capacity(2048);
        use futures_util::StreamExt;
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            buf.extend_from_slice(&chunk.unwrap());
            if buf.len() > 256 {
                break;
            }
        }
        String::from_utf8_lossy(&buf).to_string()
    })
    .await
    .unwrap();
    assert!(
        body_text.contains("Hello world") || body_text.contains("event: news"),
        "expected SSE body to contain our event, got: {body_text:?}"
    );
}

#[tokio::test]
async fn sse_bars_delivers_event() {
    let port = random_port();
    let bus = EventBus::new();
    let api = Arc::new(QueryApi::open_in_memory().unwrap());
    let cfg = MonitorConfig {
        bind_addr: format!("127.0.0.1:{port}").parse().unwrap(),
        sse_keep_alive_secs: 15,
    };
    let server = MonitorServer::new(cfg, bus.clone(), api, None);
    let _h = tokio::spawn(async move {
        let _ = server.serve().await;
    });
    tokio::time::sleep(Duration::from_millis(200)).await;

    let client = reqwest::Client::new();
    let bus_emit = bus.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        let now = chrono::Utc::now();
        let bar = Bar {
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTC", "USDT"),
            kind: BarKind::Tick { ticks_per_bar: 500 },
            start: now,
            end: now,
            open: 50_000.0,
            high: 50_100.0,
            low: 49_950.0,
            close: 50_050.0,
            volume: 1.5,
            trade_count: 500,
            buy_volume: 0.8,
            sell_volume: 0.7,
            vwap: 50_025.0,
        };
        let _ = bus_emit.bars.send(bar);
    });

    let resp = tokio::time::timeout(
        Duration::from_secs(3),
        client.get(format!("http://127.0.0.1:{port}/stream/bars")).send(),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(resp.status().is_success());

    let body_text = tokio::time::timeout(Duration::from_secs(3), async move {
        use futures_util::StreamExt;
        let mut buf = Vec::with_capacity(2048);
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            buf.extend_from_slice(&chunk.unwrap());
            if buf.len() > 256 {
                break;
            }
        }
        String::from_utf8_lossy(&buf).to_string()
    })
    .await
    .unwrap();
    assert!(body_text.contains("event: bar") || body_text.contains("BTC/USDT"));
}
