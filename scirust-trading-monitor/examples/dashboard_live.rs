//! Démo : lance le serveur monitor avec une horloge interne qui pousse
//! des events news synthétiques + market states + décisions sur le bus.
//!
//! Ouvre http://localhost:7460 pour voir le dashboard live.
//!
//! cargo run --example dashboard_live -p scirust-trading-monitor

use chrono::{Duration, Utc};
use scirust_trading_core::{
    Category, CodifiedEvent, EnrichmentLevel, EventBus, EventTiming, Exchange,
    MarketState, Polarity, Side, SourceId, Symbol, Target,
};
use scirust_trading_engine::decision::{
    BiasOutcome, Decision, DecisionAction, GateOutcome, Reasoning,
};
use scirust_trading_monitor::{MonitorConfig, MonitorServer};
use scirust_trading_persistence::QueryApi;
use std::sync::Arc;
use std::time::Duration as StdDuration;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let bus = EventBus::new();
    let api = Arc::new(QueryApi::open_in_memory().unwrap());

    // ─── Synthetic clock ─────────────────────────────────────────────
    let bus_market = bus.clone();
    tokio::spawn(async move {
        let mut tick: u64 = 0;
        let mut mid_btc = 67_000.0;
        let mut mid_eth = 3_500.0;
        loop {
            tokio::time::sleep(StdDuration::from_millis(500)).await;
            tick += 1;
            // Random walk with mean reversion
            mid_btc += (rand_jitter() - 0.5) * 30.0;
            mid_eth += (rand_jitter() - 0.5) * 4.0;
            mid_btc = mid_btc.clamp(60_000.0, 75_000.0);
            mid_eth = mid_eth.clamp(3_200.0, 3_900.0);

            for (sym, mid) in [
                (Symbol::new("BTC", "USDT"), mid_btc),
                (Symbol::new("ETH", "USDT"), mid_eth),
            ] {
                let _ = bus_market.market.send(MarketState {
                    exchange: Exchange::Binance,
                    symbol: sym,
                    timestamp: Utc::now(),
                    mid,
                    microprice: mid + 0.5,
                    spread_bps: 1.0 + rand_jitter() * 3.0,
                    imbalance_5: (rand_jitter() - 0.5) * 0.4,
                    imbalance_20: (rand_jitter() - 0.5) * 0.3,
                    realized_vol_pct: 30.0 + rand_jitter() * 20.0,
                    volume_1min: 5.0 + rand_jitter() * 15.0,
                    flow_imbalance_1min: (rand_jitter() - 0.5) * 0.5,
                    trade_count_1min: 30 + (rand_jitter() * 80.0) as u32,
                });
            }
            let _ = tick;
        }
    });

    // News events every ~3 seconds
    let bus_news = bus.clone();
    tokio::spawn(async move {
        let sources = [
            ("fomc.calendar", 0.98, Category::Macro),
            ("cryptopanic:reuters.com", 0.92, Category::Regulatory),
            ("cryptopanic:theblock.co", 0.78, Category::ExchangeEvent),
            ("whale-alert", 0.90, Category::OnChain),
            ("rss:coindesk", 0.85, Category::Narrative),
            ("rss:x:saylor", 0.55, Category::Narrative),
        ];
        let texts: Vec<(&str, Vec<&str>, f64, f64)> = vec![
            ("SEC approves spot Ethereum ETF from BlackRock", vec!["etf", "sec", "regulatory", "high_impact"], 0.7, 0.85),
            ("FOMC raises rates by 25bp, signals hawkish bias", vec!["fomc", "rate_decision", "hawkish"], -0.5, 0.8),
            ("Whale moves 8500 BTC ($560M) to unknown wallet", vec!["whale", "on_chain", "btc"], -0.3, 0.6),
            ("Binance announces new BTC futures product", vec!["binance", "futures", "listing"], 0.3, 0.5),
            ("CFTC investigates major DEX over compliance", vec!["cftc", "regulatory", "dex"], -0.4, 0.7),
            ("Saylor: 'Bitcoin is the apex property of the human race'", vec!["narrative", "btc"], 0.2, 0.4),
            ("Spread blow-up detected on Bybit BTC perp", vec!["microstructure", "warning"], 0.0, 0.6),
        ];
        let mut idx = 0;
        loop {
            tokio::time::sleep(StdDuration::from_millis(2500 + (rand_jitter() * 3000.0) as u64))
                .await;
            let (text, tags, pol, mag) = &texts[idx % texts.len()];
            let (src, rel, cat) = sources[idx % sources.len()];
            idx += 1;
            let mut ev = CodifiedEvent::builder(SourceId::new(src), *text)
                .category(cat)
                .reliability(rel)
                .timing(EventTiming::Observed(Utc::now()))
                .decay_half_life(3600)
                .expires_in(Duration::hours(2))
                .build();
            ev.tags = tags.iter().map(|s| s.to_string()).collect();
            ev.polarity = Some(Polarity::new(*pol));
            ev.magnitude = Some(*mag);
            ev.semantic_confidence = Some(0.75);
            ev.semantic_summary = Some(text.to_string());
            ev.targets = vec![Target::All];
            ev.enrichment = if idx % 4 == 0 {
                EnrichmentLevel::Contextual
            } else if idx % 2 == 0 {
                EnrichmentLevel::Semantic
            } else {
                EnrichmentLevel::Structural
            };
            ev.explanation = "synthetic demo event".into();
            let _ = bus_news.news.send(ev);
        }
    });

    // Synthetic decisions every ~4 seconds
    let (dec_tx, _dec_rx) = tokio::sync::broadcast::channel(64);
    let dec_tx_clone = dec_tx.clone();
    tokio::spawn(async move {
        let scenarios = [
            ("hold", vec![], vec![], None, "signal trop faible (0.02 < 0.05)"),
            ("open_buy", vec![], vec!["regulatory_neutral"], Some((Side::Buy, 0.005)),
             "signal Buy (score=+0.42, conf=0.75), notional=335.00"),
            ("no_trade", vec!["spread_blowup"], vec![], None,
             "gate `spread_blowup` veto: spread > 8 bps"),
            ("open_sell", vec![], vec!["regulatory_negative"], Some((Side::Sell, 0.003)),
             "signal Sell (score=-0.38, conf=0.72), notional=201.00"),
            ("no_trade", vec!["fomc_window"], vec![], None,
             "gate `fomc_window` veto: FOMC dans 30 min"),
            ("hold", vec![], vec![], None, "signal trop faible (-0.01 < 0.05)"),
        ];
        let mut i = 0;
        loop {
            tokio::time::sleep(StdDuration::from_millis(3500 + (rand_jitter() * 2000.0) as u64))
                .await;
            let (kind, gates, biases, action_data, expl) = &scenarios[i % scenarios.len()];
            i += 1;
            let mut r = Reasoning::empty(expl.to_string());
            r.final_size_multiplier = if !biases.is_empty() { 0.5 } else { 1.0 };
            for g in gates.iter() {
                r.gates_evaluated.push(GateOutcome {
                    name: g.to_string(),
                    triggered: true,
                    description: format!("synthetic {g} fired"),
                });
            }
            for b in biases.iter() {
                r.biases_applied.push(BiasOutcome {
                    name: b.to_string(),
                    applied: true,
                    effect_summary: "size×0.5".into(),
                });
            }
            let action = match (*kind, action_data) {
                ("hold", _) => DecisionAction::Hold,
                ("no_trade", _) => DecisionAction::NoTrade,
                ("open_buy", Some((side, q))) | ("open_sell", Some((side, q))) => {
                    DecisionAction::Open {
                        side: *side,
                        quantity: *q,
                        notional_quote: *q * 67_000.0,
                        limit_price: None,
                        max_hold_seconds: Some(300),
                        stop_loss_bps: Some(30.0),
                    }
                }
                _ => DecisionAction::Hold,
            };
            let d = Decision {
                id: uuid::Uuid::new_v4(),
                timestamp: Utc::now(),
                symbol: Symbol::new("BTC", "USDT"),
                action,
                reasoning: r,
            };
            let _ = dec_tx_clone.send(d);
        }
    });

    // Wire decisions into a fake shadow placeholder by exposing it via state.
    // To keep things simple, we use the existing /stream/decisions path via
    // a real ShadowEvaluator stub. Easier path: bypass and serve decisions
    // by adding a custom router endpoint. For this demo we'll create a
    // minimal shadow that just relays our channel.
    let shadow = make_fake_shadow(dec_tx);

    // Server
    let cfg = MonitorConfig {
        bind_addr: "127.0.0.1:7460".parse().unwrap(),
        sse_keep_alive_secs: 15,
    };
    let server = MonitorServer::new(cfg, bus, api, Some(shadow));

    println!();
    println!("════════════════════════════════════════════════════════════");
    println!("  scirust trading dashboard live");
    println!("  → http://127.0.0.1:7460");
    println!("════════════════════════════════════════════════════════════");
    println!("  Synthetic market data, news events, decisions");
    println!("  Press Ctrl+C to stop");
    println!();
    server.serve().await.unwrap();
}

fn rand_jitter() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    // Simple LCG for portability
    ((now.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) >> 32) & 0xffffff) as f64
        / (0xffffff as f64)
}

fn make_fake_shadow(
    dec_tx: tokio::sync::broadcast::Sender<Decision>,
) -> Arc<scirust_trading_engine::ShadowEvaluator> {
    use scirust_trading_core::{DecisionSchema, Gate, GateCondition, Sizing, SizingMethod};
    use scirust_trading_engine::{Evaluator, ShadowConfig};
    let schema = DecisionSchema {
        gates: vec![Gate {
            name: "spread_blowup".into(),
            description: "spread > 8 bps".into(),
            condition: GateCondition::SpreadAboveBps { bps: 8.0 },
        }],
        biases: vec![],
        sizing: Sizing {
            method: SizingMethod::Fixed { notional: 500.0 },
            max_position_pct: 0.5,
            max_total_exposure_pct: 1.0,
        },
    };
    let shadow = Arc::new(scirust_trading_engine::ShadowEvaluator::new(
        Evaluator::new(schema),
        ShadowConfig::default(),
    ));
    // Override the decisions channel by also broadcasting on shadow's
    // own subscriber. The fake decisions arrive via our channel.
    let shadow_tx = shadow.decisions_tx.clone();
    let mut rx = dec_tx.subscribe();
    tokio::spawn(async move {
        while let Ok(d) = rx.recv().await {
            let _ = shadow_tx.send(d);
        }
    });
    shadow
}
