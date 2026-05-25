//! Évaluation post-hoc des décisions shadow.
//!
//! Pour chaque `Decision::Open` stockée, on regarde les `market_states`
//! enregistrés depuis son timestamp et on calcule :
//!
//!   - `entry_price` = mid au plus proche de `decision.timestamp`
//!   - `exit_price`  = mid au plus proche de `decision.timestamp + max_hold_seconds`
//!   - `stop_loss_hit` = true si à un instant le mid a dépassé le stop
//!   - `realized_return_bps` = direction-ajusté
//!   - `max_favorable_bps` / `max_adverse_bps` = meilleur/pire instant pendant la fenêtre
//!
//! Et on agrège : taux de réussite, moyenne, mediane, écart-type, breakdown
//! par gate déclenché et par bias appliqué. C'est le signal d'apprentissage
//! qui permet de calibrer le `DecisionSchema` empiriquement.

use crate::PersistenceResult;
use chrono::{DateTime, Duration, Utc};
use rusqlite::{Connection, params};
use scirust_trading_core::Side;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

#[derive(Debug, Clone, Default)]
pub struct OutcomeConfig {
    /// Si une décision n'a pas de max_hold_seconds, on évalue à cette fenêtre
    /// par défaut (en secondes).
    pub default_hold_seconds: u32,
    /// Tolérance pour aligner le timestamp d'une décision avec un market_state
    /// (en millisecondes)
    pub price_match_tolerance_ms: i64,
}

impl OutcomeConfig {
    pub fn typical() -> Self {
        Self {
            default_hold_seconds: 600,           // 10 min par défaut
            price_match_tolerance_ms: 5 * 60_000, // ±5 min pour trouver le mid
        }
    }
}

#[derive(Debug, Clone)]
pub struct DecisionOutcome {
    pub decision_id: Uuid,
    pub symbol: String,
    pub direction: Side,
    pub entry_timestamp: DateTime<Utc>,
    pub entry_price: f64,
    pub exit_timestamp: DateTime<Utc>,
    pub exit_price: f64,
    /// Stop-loss configuré sur la décision (en bps depuis l'entrée)
    pub stop_loss_bps: Option<f64>,
    /// True si pendant la fenêtre le mid a dépassé le stop
    pub stop_loss_hit: bool,
    /// True si la fenêtre max_hold est entièrement dépassée
    pub max_hold_reached: bool,
    /// Return ajusté par direction : positif = win, négatif = loss (en bps)
    pub realized_return_bps: f64,
    /// Meilleur point pendant la fenêtre (en bps signés)
    pub max_favorable_bps: f64,
    /// Pire point pendant la fenêtre (en bps signés, donc ≤ 0 typiquement
    /// quand on regarde du côté défavorable)
    pub max_adverse_bps: f64,
    pub holding_duration: Duration,
    /// Liste de gates qui s'étaient déclenchés sur cette décision
    /// (issus de la table decisions, à des fins de breakdown)
    pub triggered_gates: Vec<String>,
    pub applied_biases: Vec<String>,
}

impl DecisionOutcome {
    pub fn is_win(&self) -> bool {
        self.realized_return_bps > 0.0
    }
}

#[derive(Debug, Clone, Default)]
pub struct GroupStats {
    pub n: u64,
    pub win_rate: f64,
    pub mean_return_bps: f64,
    pub median_return_bps: f64,
    pub std_return_bps: f64,
    pub stop_loss_hit_rate: f64,
}

#[derive(Debug, Clone, Default)]
pub struct PerformanceStats {
    pub n: u64,
    pub overall: GroupStats,
    /// Stats restreintes aux décisions où ce gate était déclenché
    pub by_gate: HashMap<String, GroupStats>,
    /// Stats restreintes aux décisions où ce bias était appliqué
    pub by_bias: HashMap<String, GroupStats>,
    /// Stats par symbole
    pub by_symbol: HashMap<String, GroupStats>,
}

impl crate::queries::QueryApi {
    /// Compute outcome for a single Open decision by id.
    /// Returns None if the decision is not found, isn't an Open, or if the
    /// evaluation window hasn't elapsed yet (or there's no market data).
    pub async fn compute_decision_outcome(
        &self,
        decision_id: Uuid,
        cfg: OutcomeConfig,
    ) -> PersistenceResult<Option<DecisionOutcome>> {
        let conn = Arc::clone(&self.conn);
        let id = decision_id.to_string();
        tokio::task::spawn_blocking(move || -> PersistenceResult<Option<DecisionOutcome>> {
            let c = conn.blocking_lock();
            compute_one(&c, &id, &cfg)
        })
        .await?
    }

    /// Batch : tous les Open décisions dans la fenêtre dont l'évaluation
    /// peut être faite (i.e. dont la fenêtre de hold est dans le passé).
    pub async fn compute_outcomes_in_range(
        &self,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
        cfg: OutcomeConfig,
    ) -> PersistenceResult<Vec<DecisionOutcome>> {
        let conn = Arc::clone(&self.conn);
        let from_ms = from.timestamp_millis();
        let to_ms = to.timestamp_millis();
        tokio::task::spawn_blocking(move || -> PersistenceResult<Vec<DecisionOutcome>> {
            let c = conn.blocking_lock();
            let mut stmt = c.prepare(
                "SELECT id FROM decisions
                 WHERE action_kind = 'open' AND timestamp BETWEEN ?1 AND ?2
                 ORDER BY timestamp ASC",
            )?;
            let ids: Vec<String> = stmt
                .query_map(params![from_ms, to_ms], |row| row.get::<_, String>(0))?
                .collect::<Result<Vec<_>, _>>()?;
            let mut outcomes = Vec::with_capacity(ids.len());
            for id in ids {
                if let Some(o) = compute_one(&c, &id, &cfg)? {
                    outcomes.push(o);
                }
            }
            Ok(outcomes)
        })
        .await?
    }

    /// Agrégat statistique d'un set d'outcomes.
    pub fn aggregate_stats(&self, outcomes: &[DecisionOutcome]) -> PerformanceStats {
        aggregate_stats(outcomes)
    }
}

fn compute_one(
    c: &Connection,
    decision_id: &str,
    cfg: &OutcomeConfig,
) -> PersistenceResult<Option<DecisionOutcome>> {
    // 1. Charge la décision
    let row: Option<(i64, String, Option<String>, Option<f64>, Option<i64>, Option<f64>, Option<String>, Option<String>)> = c
        .query_row(
            "SELECT timestamp, symbol, side, quantity, max_hold_seconds, stop_loss_bps,
                    triggered_gates_csv, applied_biases_csv
             FROM decisions WHERE id = ?1 AND action_kind = 'open'",
            params![decision_id],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                    row.get(7)?,
                ))
            },
        )
        .ok();
    let (ts_ms, symbol, side_s, _qty, max_hold_secs, stop_loss_bps, gates_csv, biases_csv) =
        match row {
            Some(r) => r,
            None => return Ok(None),
        };
    let side = match side_s.as_deref() {
        Some("buy") => Side::Buy,
        Some("sell") => Side::Sell,
        _ => return Ok(None),
    };

    let entry_ts = DateTime::<Utc>::from_timestamp_millis(ts_ms).unwrap_or_else(Utc::now);
    let hold_secs = max_hold_secs.unwrap_or(cfg.default_hold_seconds as i64) as u32;
    let exit_ts = entry_ts + Duration::seconds(hold_secs as i64);

    // 2. Si la fenêtre n'est pas encore terminée, on skip (mais on peut
    // quand même fournir un "in-flight outcome" plus tard — pour l'instant
    // on attend que la fenêtre soit complète).
    if Utc::now() < exit_ts {
        return Ok(None);
    }

    // 3. Cherche le mid à l'entrée et à la sortie
    let entry_price = match find_mid_near(c, &symbol, ts_ms, cfg.price_match_tolerance_ms)? {
        Some(p) => p,
        None => return Ok(None),
    };
    let exit_price_ts_ms = exit_ts.timestamp_millis();
    let exit_price =
        match find_mid_near(c, &symbol, exit_price_ts_ms, cfg.price_match_tolerance_ms)? {
            Some(p) => p,
            None => return Ok(None),
        };

    // 4. Parcourt les market_states entre entry et exit pour calculer
    // max_favorable / max_adverse / stop_hit
    let (max_fav_bps, max_adv_bps, stop_hit) = scan_window(
        c,
        &symbol,
        ts_ms,
        exit_price_ts_ms,
        entry_price,
        side,
        stop_loss_bps,
    )?;

    // 5. Return réalisé direction-ajusté
    let raw_return_bps = 10_000.0 * (exit_price / entry_price - 1.0);
    let realized_return_bps = match side {
        Side::Buy => raw_return_bps,
        Side::Sell => -raw_return_bps,
    };

    Ok(Some(DecisionOutcome {
        decision_id: Uuid::parse_str(decision_id).unwrap_or_else(|_| Uuid::nil()),
        symbol: symbol.clone(),
        direction: side,
        entry_timestamp: entry_ts,
        entry_price,
        exit_timestamp: exit_ts,
        exit_price,
        stop_loss_bps,
        stop_loss_hit: stop_hit,
        max_hold_reached: true,
        realized_return_bps,
        max_favorable_bps: max_fav_bps,
        max_adverse_bps: max_adv_bps,
        holding_duration: Duration::seconds(hold_secs as i64),
        triggered_gates: gates_csv
            .filter(|s| !s.is_empty())
            .map(|s| s.split(',').map(|x| x.to_string()).collect())
            .unwrap_or_default(),
        applied_biases: biases_csv
            .filter(|s| !s.is_empty())
            .map(|s| s.split(',').map(|x| x.to_string()).collect())
            .unwrap_or_default(),
    }))
}

fn find_mid_near(
    c: &Connection,
    symbol: &str,
    target_ms: i64,
    tolerance_ms: i64,
) -> PersistenceResult<Option<f64>> {
    let lower = target_ms - tolerance_ms;
    let upper = target_ms + tolerance_ms;
    let mid: Option<f64> = c
        .query_row(
            "SELECT mid FROM market_states
             WHERE symbol = ?1 AND timestamp BETWEEN ?2 AND ?3
             ORDER BY ABS(timestamp - ?4) ASC LIMIT 1",
            params![symbol, lower, upper, target_ms],
            |row| row.get::<_, f64>(0),
        )
        .ok();
    Ok(mid)
}

/// Parcourt tous les market_states dans la fenêtre [from_ms, to_ms] pour
/// calculer (max_favorable_bps, max_adverse_bps, stop_hit).
/// Tous les bps sont ajustés par direction : positif = move dans le sens
/// de la position.
fn scan_window(
    c: &Connection,
    symbol: &str,
    from_ms: i64,
    to_ms: i64,
    entry_price: f64,
    side: Side,
    stop_loss_bps: Option<f64>,
) -> PersistenceResult<(f64, f64, bool)> {
    let mut stmt = c.prepare(
        "SELECT mid FROM market_states
         WHERE symbol = ?1 AND timestamp BETWEEN ?2 AND ?3
         ORDER BY timestamp ASC",
    )?;
    let prices: Vec<f64> = stmt
        .query_map(params![symbol, from_ms, to_ms], |row| {
            row.get::<_, f64>(0)
        })?
        .collect::<Result<Vec<_>, _>>()?;

    if prices.is_empty() {
        return Ok((0.0, 0.0, false));
    }

    let direction_sign = match side {
        Side::Buy => 1.0,
        Side::Sell => -1.0,
    };

    let mut max_fav = f64::NEG_INFINITY;
    let mut max_adv = f64::INFINITY;
    let mut hit = false;
    let stop_threshold_bps = stop_loss_bps.map(|b| -b.abs()); // stop = perte de N bps

    for p in &prices {
        let raw_bps = 10_000.0 * (p / entry_price - 1.0);
        let dir_bps = raw_bps * direction_sign;
        if dir_bps > max_fav {
            max_fav = dir_bps;
        }
        if dir_bps < max_adv {
            max_adv = dir_bps;
        }
        if let Some(threshold) = stop_threshold_bps {
            if dir_bps <= threshold {
                hit = true;
            }
        }
    }

    Ok((max_fav.max(0.0_f64.min(max_fav)), max_adv.min(0.0_f64.max(max_adv)), hit))
}

fn aggregate_stats(outcomes: &[DecisionOutcome]) -> PerformanceStats {
    let mut stats = PerformanceStats {
        n: outcomes.len() as u64,
        overall: compute_group(outcomes),
        ..Default::default()
    };

    // Breakdown par gate
    let mut by_gate: HashMap<String, Vec<&DecisionOutcome>> = HashMap::new();
    for o in outcomes {
        for g in &o.triggered_gates {
            by_gate.entry(g.clone()).or_default().push(o);
        }
    }
    for (gate, group) in by_gate {
        let owned: Vec<DecisionOutcome> = group.into_iter().cloned().collect();
        stats.by_gate.insert(gate, compute_group(&owned));
    }

    // Breakdown par bias
    let mut by_bias: HashMap<String, Vec<&DecisionOutcome>> = HashMap::new();
    for o in outcomes {
        for b in &o.applied_biases {
            by_bias.entry(b.clone()).or_default().push(o);
        }
    }
    for (bias, group) in by_bias {
        let owned: Vec<DecisionOutcome> = group.into_iter().cloned().collect();
        stats.by_bias.insert(bias, compute_group(&owned));
    }

    // Breakdown par symbole
    let mut by_sym: HashMap<String, Vec<&DecisionOutcome>> = HashMap::new();
    for o in outcomes {
        by_sym.entry(o.symbol.clone()).or_default().push(o);
    }
    for (sym, group) in by_sym {
        let owned: Vec<DecisionOutcome> = group.into_iter().cloned().collect();
        stats.by_symbol.insert(sym, compute_group(&owned));
    }

    stats
}

fn compute_group(outcomes: &[DecisionOutcome]) -> GroupStats {
    if outcomes.is_empty() {
        return GroupStats::default();
    }
    let n = outcomes.len() as f64;
    let returns: Vec<f64> = outcomes.iter().map(|o| o.realized_return_bps).collect();
    let mean = returns.iter().sum::<f64>() / n;
    let mut sorted = returns.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };
    let variance = returns.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();
    let wins = outcomes.iter().filter(|o| o.is_win()).count() as f64;
    let stops = outcomes.iter().filter(|o| o.stop_loss_hit).count() as f64;
    GroupStats {
        n: outcomes.len() as u64,
        win_rate: wins / n,
        mean_return_bps: mean,
        median_return_bps: median,
        std_return_bps: std,
        stop_loss_hit_rate: stops / n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decisions::flush_decisions;
    use crate::queries::QueryApi;
    use crate::writer::flush_market;
    use chrono::Duration as CD;
    use scirust_trading_core::{Exchange, MarketState, Side, Symbol};
    use scirust_trading_engine::decision::{Decision, DecisionAction, GateOutcome, Reasoning, BiasOutcome};

    fn mk_decision(when: DateTime<Utc>, side: Side, gates: &[&str], biases: &[&str]) -> Decision {
        let mut r = Reasoning::empty("test");
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

    fn mk_market(when: DateTime<Utc>, mid: f64) -> MarketState {
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

    #[tokio::test]
    async fn winning_long_computes_positive_return() {
        let api = QueryApi::open_in_memory().unwrap();
        // Décision passée il y a 1h
        let entry = Utc::now() - CD::hours(1);
        let decision = mk_decision(entry, Side::Buy, &[], &[]);
        let did = decision.id;
        {
            let c = api.conn.lock().await;
            flush_decisions(&c, &[decision]).unwrap();
            // Market states : 50000 à l'entrée, 50500 à +5min (sortie)
            flush_market(
                &c,
                &[
                    mk_market(entry - CD::seconds(10), 50_000.0),
                    mk_market(entry + CD::seconds(10), 50_000.0),
                    mk_market(entry + CD::minutes(2), 50_300.0),
                    mk_market(entry + CD::minutes(5), 50_500.0),
                ],
            )
            .unwrap();
        }
        let outcome = api
            .compute_decision_outcome(did, OutcomeConfig::typical())
            .await
            .unwrap()
            .expect("should compute");
        assert!(outcome.is_win());
        // 50500 / 50000 - 1 = 0.01 → 100 bps
        assert!((outcome.realized_return_bps - 100.0).abs() < 1.0);
        assert!(outcome.max_favorable_bps >= 100.0);
        assert!(!outcome.stop_loss_hit);
    }

    #[tokio::test]
    async fn losing_long_with_stop_hit() {
        let api = QueryApi::open_in_memory().unwrap();
        let entry = Utc::now() - CD::hours(1);
        let decision = mk_decision(entry, Side::Buy, &[], &[]);
        let did = decision.id;
        {
            let c = api.conn.lock().await;
            flush_decisions(&c, &[decision]).unwrap();
            // Stop à 30 bps → prix qui descend à 49800 (= -40 bps)
            flush_market(
                &c,
                &[
                    mk_market(entry, 50_000.0),
                    mk_market(entry + CD::minutes(1), 49_900.0), // -20 bps
                    mk_market(entry + CD::minutes(2), 49_800.0), // -40 bps → stop hit
                    mk_market(entry + CD::minutes(5), 49_900.0), // exit -20 bps
                ],
            )
            .unwrap();
        }
        let outcome = api
            .compute_decision_outcome(did, OutcomeConfig::typical())
            .await
            .unwrap()
            .expect("should compute");
        assert!(!outcome.is_win());
        assert!(outcome.stop_loss_hit);
        // Final exit_price = 49900, entry = 50000 → -20 bps
        assert!((outcome.realized_return_bps - (-20.0)).abs() < 1.0);
    }

    #[tokio::test]
    async fn aggregate_breakdown_by_gate_and_bias() {
        let api = QueryApi::open_in_memory().unwrap();
        let base = Utc::now() - CD::hours(2);
        // 4 décisions :
        // - 2 avec gate "spread" gainantes (+50 bps)
        // - 2 avec gate "vol" perdantes (-30 bps)
        let scenarios = vec![
            ("spread", true),
            ("spread", true),
            ("vol", false),
            ("vol", false),
        ];
        let mut ids = Vec::new();
        {
            let c = api.conn.lock().await;
            for (i, (gate, winner)) in scenarios.iter().enumerate() {
                let d = mk_decision(
                    base + CD::seconds(i as i64 * 60),
                    Side::Buy,
                    &[gate],
                    &[],
                );
                let did = d.id;
                ids.push(did);
                flush_decisions(&c, &[d]).unwrap();
                let entry_ts = base + CD::seconds(i as i64 * 60);
                let exit_price = if *winner { 50_250.0 } else { 49_850.0 };
                flush_market(
                    &c,
                    &[
                        mk_market(entry_ts, 50_000.0),
                        mk_market(entry_ts + CD::minutes(5), exit_price),
                    ],
                )
                .unwrap();
            }
        }
        let outcomes = api
            .compute_outcomes_in_range(
                base - CD::minutes(1),
                Utc::now() + CD::minutes(1),
                OutcomeConfig::typical(),
            )
            .await
            .unwrap();
        assert_eq!(outcomes.len(), 4);
        let stats = api.aggregate_stats(&outcomes);
        assert_eq!(stats.n, 4);
        // Overall : 2 wins / 4 = 50%
        assert!((stats.overall.win_rate - 0.5).abs() < 1e-9);
        // by_gate spread : 100% wins, +50bps mean
        let spread = stats.by_gate.get("spread").unwrap();
        assert!((spread.win_rate - 1.0).abs() < 1e-9);
        assert!((spread.mean_return_bps - 50.0).abs() < 1.0);
        // by_gate vol : 0% wins, -30bps mean
        let vol = stats.by_gate.get("vol").unwrap();
        assert!((vol.win_rate - 0.0).abs() < 1e-9);
        assert!((vol.mean_return_bps - (-30.0)).abs() < 1.0);
    }

    #[tokio::test]
    async fn outcome_none_if_window_not_elapsed() {
        let api = QueryApi::open_in_memory().unwrap();
        // Décision juste maintenant → fenêtre 5min pas terminée
        let now = Utc::now();
        let decision = mk_decision(now, Side::Buy, &[], &[]);
        let did = decision.id;
        {
            let c = api.conn.lock().await;
            flush_decisions(&c, &[decision]).unwrap();
            flush_market(&c, &[mk_market(now, 50_000.0)]).unwrap();
        }
        let outcome = api
            .compute_decision_outcome(did, OutcomeConfig::typical())
            .await
            .unwrap();
        assert!(outcome.is_none());
    }
}
