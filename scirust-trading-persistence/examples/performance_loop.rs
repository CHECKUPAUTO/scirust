//! Démo complète du pipeline : décisions shadow → persistance → outcomes →
//! breakdown par gate/bias. C'est le cycle d'apprentissage complet.
//!
//! cargo run --example performance_loop -p scirust-trading-persistence

use chrono::{Duration, Utc};
use scirust_trading_core::{Exchange, MarketState, Side, Symbol};
use scirust_trading_engine::decision::{
    BiasOutcome, Decision, DecisionAction, GateOutcome, Reasoning,
};
use scirust_trading_persistence::{
    decisions::flush_decisions, writer::flush_market, OutcomeConfig, QueryApi,
};
use uuid::Uuid;

fn make_decision(
    when: chrono::DateTime<Utc>,
    side: Side,
    gates: &[&str],
    biases: &[&str],
) -> Decision {
    let mut r = Reasoning::empty(format!(
        "scenario: {:?} with gates {:?} biases {:?}",
        side, gates, biases
    ));
    for g in gates {
        r.gates_evaluated.push(GateOutcome {
            name: g.to_string(),
            triggered: true,
            description: "".into(),
        });
    }
    for b in biases {
        r.biases_applied.push(BiasOutcome {
            name: b.to_string(),
            applied: true,
            effect_summary: "".into(),
        });
    }
    Decision {
        id: Uuid::new_v4(),
        timestamp: when,
        symbol: Symbol::new("BTC", "USDT"),
        action: DecisionAction::Open {
            side,
            quantity: 0.01,
            notional_quote: 500.0,
            limit_price: None,
            max_hold_seconds: Some(300),
            stop_loss_bps: Some(30.0),
        },
        reasoning: r,
    }
}

fn make_market(when: chrono::DateTime<Utc>, mid: f64) -> MarketState {
    MarketState {
        exchange: Exchange::Binance,
        symbol: Symbol::new("BTC", "USDT"),
        timestamp: when,
        mid,
        microprice: mid,
        spread_bps: 1.0,
        imbalance_5: 0.0,
        imbalance_20: 0.0,
        realized_vol_pct: 30.0,
        volume_1min: 1.0,
        flow_imbalance_1min: 0.0,
        trade_count_1min: 5,
    }
}

#[tokio::main]
async fn main() {
    println!("scirust-trading-persistence performance loop demo");
    println!("══════════════════════════════════════════════════════════════");

    let api = QueryApi::open_in_memory().unwrap();

    // ─── Setup synthétique : 12 décisions historiques avec outcomes connus
    let base = Utc::now() - Duration::hours(2);
    let scenarios: Vec<(&str, Vec<&str>, Vec<&str>, Side, f64)> = vec![
        // (label, gates, biases, side, exit_price_delta_bps)
        ("etf bullish 1",       vec![],                     vec!["regulatory_neutral"], Side::Buy,  60.0),
        ("etf bullish 2",       vec![],                     vec!["regulatory_neutral"], Side::Buy,  45.0),
        ("etf bullish 3",       vec![],                     vec!["regulatory_neutral"], Side::Buy,  72.0),
        ("etf bullish 4",       vec![],                     vec!["regulatory_neutral"], Side::Buy,  -8.0),
        ("vol blowup 1",        vec!["high_volatility"],    vec![],                     Side::Buy,  -45.0),
        ("vol blowup 2",        vec!["high_volatility"],    vec![],                     Side::Buy,  -32.0),
        ("vol blowup 3",        vec!["high_volatility"],    vec![],                     Side::Sell, -28.0),
        ("sec bearish 1",       vec![],                     vec!["regulatory_negative"], Side::Sell, 38.0),
        ("sec bearish 2",       vec![],                     vec!["regulatory_negative"], Side::Sell, 55.0),
        ("sec bearish 3",       vec![],                     vec!["regulatory_negative"], Side::Sell, 12.0),
        ("stop hit 1",          vec![],                     vec![],                     Side::Buy,  -35.0),
        ("stop hit 2",          vec![],                     vec![],                     Side::Buy,  -42.0),
    ];

    {
        let c = api.conn.lock().await;
        for (i, (_, gates, biases, side, exit_delta_bps)) in scenarios.iter().enumerate() {
            // 30 min entre chaque scénario pour éviter les collisions de
            // market_states (primary key sur exchange + symbol + timestamp)
            let ts = base - Duration::hours(6) + Duration::minutes((i as i64) * 30);
            let exit_ts = ts + Duration::minutes(5);
            // Persiste la décision
            let d = make_decision(ts, *side, gates, biases);
            flush_decisions(&c, &[d]).unwrap();
            // Persiste les market states correspondants : entry à 50000,
            // sortie au prix calculé pour donner exit_delta_bps (direction-adjusted)
            let exit_price = match side {
                Side::Buy => 50_000.0 * (1.0 + exit_delta_bps / 10_000.0),
                Side::Sell => 50_000.0 * (1.0 - exit_delta_bps / 10_000.0),
            };
            // Pour les "stop hit", on simule un drawdown qui touche -30 bps
            // en plein milieu avant de revenir au prix de sortie
            let mid_price = if *exit_delta_bps < -30.0 {
                // Stop hit : crée un point intermediaire à -35 bps
                match side {
                    Side::Buy => 50_000.0 * (1.0 - 0.0035),
                    Side::Sell => 50_000.0 * (1.0 + 0.0035),
                }
            } else {
                (50_000.0 + exit_price) / 2.0
            };
            flush_market(
                &c,
                &[
                    make_market(ts, 50_000.0),
                    make_market(ts + Duration::minutes(2), mid_price),
                    make_market(exit_ts, exit_price),
                ],
            )
            .unwrap();
        }
    }

    // ─── Calcule les outcomes pour toute la fenêtre
    println!("\n[1] Calcul des outcomes sur 12 décisions historiques");
    let outcomes = api
        .compute_outcomes_in_range(
            base - Duration::hours(12),
            Utc::now(),
            OutcomeConfig::typical(),
        )
        .await
        .unwrap();
    println!("    → {} outcomes calculés", outcomes.len());

    // Listing détaillé
    println!("\n[2] Détail des outcomes");
    println!(
        "    {:<22} {:>10} {:>12} {:>8} {:>10} {:>10}",
        "scenario_gates_biases", "side", "return_bps", "win?", "max_fav", "max_adv"
    );
    println!("    {}", "─".repeat(80));
    for o in &outcomes {
        let label = format!(
            "{:?}+{:?}",
            o.triggered_gates,
            o.applied_biases
        );
        println!(
            "    {:<22} {:>10?} {:>+12.1} {:>8} {:>+10.1} {:>+10.1}",
            label,
            o.direction,
            o.realized_return_bps,
            if o.is_win() { "✓ win" } else { "× loss" },
            o.max_favorable_bps,
            o.max_adverse_bps
        );
    }

    // ─── Stats globales
    let stats = api.aggregate_stats(&outcomes);
    println!("\n[3] Stats globales");
    println!("    n              : {}", stats.overall.n);
    println!("    win rate       : {:.1}%", stats.overall.win_rate * 100.0);
    println!("    mean return    : {:+.2} bps", stats.overall.mean_return_bps);
    println!("    median         : {:+.2} bps", stats.overall.median_return_bps);
    println!("    std            : {:.2} bps", stats.overall.std_return_bps);
    println!(
        "    stop-hit rate  : {:.1}%",
        stats.overall.stop_loss_hit_rate * 100.0
    );

    // ─── Breakdown par gate
    println!("\n[4] Breakdown par gate déclenché");
    let mut gates: Vec<(&String, &scirust_trading_persistence::GroupStats)> =
        stats.by_gate.iter().collect();
    gates.sort_by(|a, b| {
        b.1.mean_return_bps
            .partial_cmp(&a.1.mean_return_bps)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (gate, gs) in gates {
        println!(
            "    {:<25} n={:<3} win_rate={:>5.1}% mean={:>+7.2} bps median={:>+7.2}",
            gate,
            gs.n,
            gs.win_rate * 100.0,
            gs.mean_return_bps,
            gs.median_return_bps
        );
    }

    // ─── Breakdown par bias
    println!("\n[5] Breakdown par bias appliqué");
    let mut biases: Vec<(&String, &scirust_trading_persistence::GroupStats)> =
        stats.by_bias.iter().collect();
    biases.sort_by(|a, b| {
        b.1.mean_return_bps
            .partial_cmp(&a.1.mean_return_bps)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (bias, gs) in biases {
        println!(
            "    {:<25} n={:<3} win_rate={:>5.1}% mean={:>+7.2} bps median={:>+7.2}",
            bias,
            gs.n,
            gs.win_rate * 100.0,
            gs.mean_return_bps,
            gs.median_return_bps
        );
    }

    println!("\n══════════════════════════════════════════════════════════════");
    println!("Interprétation :");
    println!("  • Le gate 'high_volatility' a un mean négatif → il déclenche");
    println!("    bien dans les moments où le marché bouge contre nous.");
    println!("  • Le bias 'regulatory_negative' a un mean positif → il marque");
    println!("    correctement les opportunités short.");
    println!("  • Le bias 'regulatory_neutral' est mixte (4 décisions, 3 gains");
    println!("    1 perte) → il faut affiner sa condition de déclenchement.");
    println!("\nC'est ce signal qu'on accumule pendant 2-4 semaines pour");
    println!("calibrer le schema empiriquement.");
}
