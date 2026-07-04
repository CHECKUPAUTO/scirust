//! Statistical arbitrage — trade the *relationship* between two assets, not the
//! direction of either.
//!
//! Directional strategies bet on where a price is going. Pairs trading bets on
//! something far more stable: that two co-moving assets (two L1s, an asset and
//! its perp, a token and a basket) keep a stable *spread*, so when the spread
//! stretches it snaps back. It is market-neutral — long one leg, short the
//! other — so it can profit in a flat or falling tape where directional books
//! stall. This module gives an agent the standard quant toolkit for it:
//!
//! * **Hedge ratio** — OLS of A on B gives the β that makes `A − βB` stationary
//!   (the dollar-neutral leg ratio).
//! * **Cointegration** — the Engle-Granger two-step: regress, then test the
//!   residual spread for stationarity with a Dickey-Fuller-style t-statistic on
//!   the AR(1) mean-reversion coefficient. A stationary spread ⇒ a tradeable
//!   relationship; a unit-root spread ⇒ the two just drift apart.
//! * **Half-life** — from the Ornstein-Uhlenbeck / AR(1) fit, how many bars the
//!   spread takes to revert halfway. Too slow ⇒ the edge decays before it pays.
//! * **Hurst** of the spread (reused from [`crate::regime`]) — an independent
//!   `H < 0.5` confirmation of mean reversion.
//! * **z-score signal** — standardize the current spread; short it when rich,
//!   long it when cheap, flat inside the band.
//! * **Scanner** — test every pair among N assets and rank the tradeable ones.
//!
//! Deterministic: every step is a pure forward-order reduction.

// Nested pair-index loops and windowed regressions read most clearly with
// explicit `i`/`j` index loops here.
#![allow(clippy::needless_range_loop)]

use serde::{Deserialize, Serialize};

use crate::metrics::{mean, stddev};
use crate::regime::hurst_exponent;

/// Simple OLS regression `y = alpha + beta·x`. Returns
/// `(alpha, beta, se_beta, r_squared)`, or `None` if degenerate.
fn ols(x: &[f32], y: &[f32]) -> Option<(f32, f32, f32, f32)> {
    let n = x.len();
    if n < 3 || y.len() != n
    {
        return None;
    }
    let nf = n as f32;
    let mx = x.iter().sum::<f32>() / nf;
    let my = y.iter().sum::<f32>() / nf;
    let mut sxx = 0.0f32;
    let mut sxy = 0.0f32;
    let mut syy = 0.0f32;
    for i in 0..n
    {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    if sxx < 1e-12
    {
        return None;
    }
    let beta = sxy / sxx;
    let alpha = my - beta * mx;
    let mut sse = 0.0f32;
    for i in 0..n
    {
        let e = y[i] - (alpha + beta * x[i]);
        sse += e * e;
    }
    let dof = (nf - 2.0).max(1.0);
    let s2 = sse / dof;
    let se_beta = (s2 / sxx).sqrt();
    let r2 = if syy > 1e-12 { 1.0 - sse / syy } else { 0.0 };
    Some((alpha, beta, se_beta, r2))
}

/// The hedge (cointegrating) regression `A = alpha + beta·B`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HedgeFit {
    /// Intercept.
    pub alpha: f32,
    /// Hedge ratio — units of B shorted per unit of A. The spread is
    /// `A − beta·B − alpha`.
    pub beta: f32,
    /// Regression R².
    pub r_squared: f32,
}

/// Fit the hedge ratio by regressing `a` on `b`.
pub fn hedge_ratio(a: &[f32], b: &[f32]) -> Option<HedgeFit> {
    let (alpha, beta, _se, r2) = ols(b, a)?;
    Some(HedgeFit {
        alpha,
        beta,
        r_squared: r2,
    })
}

/// The spread residual series `A − beta·B − alpha`.
pub fn spread_series(a: &[f32], b: &[f32], fit: &HedgeFit) -> Vec<f32> {
    let n = a.len().min(b.len());
    (0..n)
        .map(|i| a[i] - (fit.beta * b[i] + fit.alpha))
        .collect()
}

/// The AR(1) / Ornstein-Uhlenbeck mean-reversion fit of a spread:
/// `Δs_t = c + lambda·s_{t-1} + ε`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeanReversionFit {
    /// Mean-reversion speed. `lambda < 0` ⇒ the spread pulls back to its mean.
    pub lambda: f32,
    /// Bars to revert halfway, `−ln 2 / lambda`. **`-1` means no reversion**
    /// (`lambda ≥ 0`, a unit-root/divergent spread).
    pub half_life: f32,
    /// Dickey-Fuller-style t-statistic of `lambda`. More negative ⇒ stronger
    /// evidence the spread is stationary (cointegrated).
    pub adf_t: f32,
}

/// Fit the AR(1) mean reversion of `spread`. `None` if too short.
pub fn mean_reversion_fit(spread: &[f32]) -> Option<MeanReversionFit> {
    let n = spread.len();
    if n < 4
    {
        return None;
    }
    let lag = &spread[..n - 1];
    let dy: Vec<f32> = (1..n).map(|i| spread[i] - spread[i - 1]).collect();
    let (_c, lambda, se_lambda, _r2) = ols(lag, &dy)?;
    let adf_t = if se_lambda > 1e-12
    {
        lambda / se_lambda
    }
    else
    {
        0.0
    };
    let half_life = if lambda < 0.0
    {
        -std::f32::consts::LN_2 / lambda
    }
    else
    {
        -1.0
    };
    Some(MeanReversionFit {
        lambda,
        half_life,
        adf_t,
    })
}

/// What to do on the spread right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PairAction {
    /// Spread is cheap (z ≤ −entry): buy A, sell β·B.
    LongSpread,
    /// Spread is rich (z ≥ +entry): sell A, buy β·B.
    ShortSpread,
    /// Inside the band — no position.
    Flat,
}

impl PairAction {
    pub fn label(self) -> &'static str {
        match self
        {
            PairAction::LongSpread => "long spread (buy A, sell B)",
            PairAction::ShortSpread => "short spread (sell A, buy B)",
            PairAction::Flat => "flat (spread within band)",
        }
    }
}

/// Tuning for [`analyze_pair`] and [`scan_pairs`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairConfig {
    /// `|z|` at/above which to open a position.
    pub entry_z: f32,
    /// `|z|` at/below which to close (informational for the exit rule).
    pub exit_z: f32,
    /// ADF t-stat at/below which the spread is called stationary (cointegrated).
    /// A pragmatic threshold, not tied to exact Dickey-Fuller critical values.
    pub adf_threshold: f32,
    /// Longest half-life (bars) still considered tradeable.
    pub max_half_life: f32,
}

impl Default for PairConfig {
    fn default() -> Self {
        Self {
            entry_z: 2.0,
            exit_z: 0.5,
            adf_threshold: -2.5,
            max_half_life: 100.0,
        }
    }
}

fn action_from_z(z: f32, cfg: &PairConfig) -> PairAction {
    if z >= cfg.entry_z
    {
        PairAction::ShortSpread
    }
    else if z <= -cfg.entry_z
    {
        PairAction::LongSpread
    }
    else
    {
        PairAction::Flat
    }
}

/// The full cointegration + signal report for one pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CointegrationReport {
    /// Hedge ratio (units of B per unit of A).
    pub beta: f32,
    /// Hedge-regression intercept.
    pub alpha: f32,
    /// Hedge-regression R².
    pub r_squared: f32,
    /// Bars to revert halfway (`-1` ⇒ no reversion).
    pub half_life: f32,
    /// Dickey-Fuller-style stationarity t-stat of the spread.
    pub adf_t: f32,
    /// Hurst exponent of the spread (`< 0.5` ⇒ mean-reverting).
    pub hurst: f32,
    /// Current spread level.
    pub spread: f32,
    /// Spread mean over the sample (≈ 0 by construction).
    pub spread_mean: f32,
    /// Spread standard deviation.
    pub spread_std: f32,
    /// Current spread z-score — the trade signal.
    pub spread_z: f32,
    /// Whether the spread passes the stationarity gate.
    pub is_cointegrated: bool,
    /// Cointegrated *and* reverting fast enough to trade.
    pub tradeable: bool,
    /// The action implied by the current z-score.
    pub action: PairAction,
    /// Plain-language read for the agent.
    pub verdict: String,
}

/// Analyze a candidate pair from two aligned price series (A, B). Series are
/// tail-aligned to the shorter length. `None` if too short.
pub fn analyze_pair(a: &[f32], b: &[f32], cfg: &PairConfig) -> Option<CointegrationReport> {
    let n = a.len().min(b.len());
    if n < 30
    {
        return None;
    }
    let a = &a[a.len() - n..];
    let b = &b[b.len() - n..];

    let fit = hedge_ratio(a, b)?;
    let spread = spread_series(a, b, &fit);
    let mr = mean_reversion_fit(&spread)?;
    let sm = mean(&spread);
    let ss = stddev(&spread);
    let cur = *spread.last().unwrap();
    let z = if ss > 1e-9 { (cur - sm) / ss } else { 0.0 };
    // Hurst characterizes a process from its *increments* — feed the spread's
    // first differences, not its levels. A mean-reverting spread has
    // anti-persistent changes ⇒ H < 0.5.
    let diffs: Vec<f32> = (1..spread.len())
        .map(|i| spread[i] - spread[i - 1])
        .collect();
    let hurst = hurst_exponent(&diffs);

    let is_cointegrated = mr.lambda < 0.0 && mr.adf_t <= cfg.adf_threshold;
    let tradeable = is_cointegrated && mr.half_life >= 1.0 && mr.half_life <= cfg.max_half_life;
    // The raw z-signal; `tradeable` is a separate quality gate the agent reads
    // alongside it (don't act on a signal from a non-cointegrated pair).
    let action = action_from_z(z, cfg);

    let verdict = if !is_cointegrated
    {
        format!(
            "NOT COINTEGRATED — spread shows no mean reversion (ADF t {:.2} > {:.2}, Hurst \
             {hurst:.2}). The legs drift apart; not a stat-arb pair.",
            mr.adf_t, cfg.adf_threshold
        )
    }
    else if !tradeable
    {
        format!(
            "COINTEGRATED BUT SLOW — reverts with half-life {:.0} bars (limit {:.0}). The edge \
             decays too slowly to trade profitably after costs.",
            mr.half_life, cfg.max_half_life
        )
    }
    else
    {
        format!(
            "TRADEABLE — cointegrated (ADF t {:.2}), half-life {:.0} bars, Hurst {hurst:.2}. \
             Current z {z:+.2}: {}.",
            mr.adf_t,
            mr.half_life,
            action.label()
        )
    };

    Some(CointegrationReport {
        beta: fit.beta,
        alpha: fit.alpha,
        r_squared: fit.r_squared,
        half_life: mr.half_life,
        adf_t: mr.adf_t,
        hurst,
        spread: cur,
        spread_mean: sm,
        spread_std: ss,
        spread_z: z,
        is_cointegrated,
        tradeable,
        action,
        verdict,
    })
}

/// One ranked pair from [`scan_pairs`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairCandidate {
    pub symbol_a: String,
    pub symbol_b: String,
    pub beta: f32,
    pub half_life: f32,
    pub adf_t: f32,
    pub hurst: f32,
    pub current_z: f32,
    pub is_cointegrated: bool,
    pub tradeable: bool,
    pub action: PairAction,
    /// Ranking score (`−adf_t`): more-stationary spreads rank first.
    pub score: f32,
}

/// Test every pair among `symbols` (with matching `series`) for cointegration
/// and return them ranked best-first (most stationary spread first).
pub fn scan_pairs(symbols: &[String], series: &[Vec<f32>], cfg: &PairConfig) -> Vec<PairCandidate> {
    let n = symbols.len().min(series.len());
    let mut out: Vec<PairCandidate> = Vec::new();
    for i in 0..n
    {
        for j in (i + 1)..n
        {
            if let Some(rep) = analyze_pair(&series[i], &series[j], cfg)
            {
                out.push(PairCandidate {
                    symbol_a: symbols[i].clone(),
                    symbol_b: symbols[j].clone(),
                    beta: rep.beta,
                    half_life: rep.half_life,
                    adf_t: rep.adf_t,
                    hurst: rep.hurst,
                    current_z: rep.spread_z,
                    is_cointegrated: rep.is_cointegrated,
                    tradeable: rep.tradeable,
                    action: rep.action,
                    score: -rep.adf_t,
                });
            }
        }
    }
    // Rank by score desc; stable, so ties keep (i, j) enumeration order.
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic LCG in `[-1, 1)` — a reproducible noise source for tests.
    struct Lcg {
        x: u64,
    }
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { x: seed }
        }
        fn next(&mut self) -> f32 {
            self.x = self
                .x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // (x >> 33) is in [0, 2^31); map to [-1, 1).
            let u = (self.x >> 33) as f32 / (1u64 << 31) as f32;
            u * 2.0 - 1.0
        }
    }

    /// A genuine (integrated) random walk.
    fn random_walk(n: usize, seed: u64, start: f32) -> Vec<f32> {
        let mut lcg = Lcg::new(seed);
        let mut p = start;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n
        {
            p += lcg.next();
            out.push(p);
        }
        out
    }

    /// A stationary AR(1) series `s_t = phi·s_{t-1} + noise` (phi in (0,1)).
    fn ar1(n: usize, phi: f32, seed: u64) -> Vec<f32> {
        let mut lcg = Lcg::new(seed);
        let mut s = 0.0f32;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n
        {
            s = phi * s + lcg.next();
            out.push(s);
        }
        out
    }

    #[test]
    fn ols_recovers_a_line() {
        let x: Vec<f32> = (0..40).map(|i| i as f32).collect();
        let y: Vec<f32> = x.iter().map(|&v| 3.0 + 2.0 * v).collect();
        let (a, b, _se, r2) = ols(&x, &y).unwrap();
        assert!((a - 3.0).abs() < 1e-2);
        assert!((b - 2.0).abs() < 1e-3);
        assert!(r2 > 0.999);
    }

    #[test]
    fn action_from_z_thresholds() {
        let cfg = PairConfig::default();
        assert_eq!(action_from_z(3.0, &cfg), PairAction::ShortSpread);
        assert_eq!(action_from_z(-3.0, &cfg), PairAction::LongSpread);
        assert_eq!(action_from_z(0.3, &cfg), PairAction::Flat);
    }

    #[test]
    fn cointegrated_pair_is_tradeable() {
        // B is a random walk; A = B + a fast mean-reverting spread. A and B
        // share B's stochastic trend, so A−βB is stationary ⇒ cointegrated.
        let b = random_walk(300, 12345, 100.0);
        let spread = ar1(300, 0.5, 999);
        let a: Vec<f32> = (0..300).map(|i| b[i] + spread[i]).collect();
        let rep = analyze_pair(&a, &b, &PairConfig::default()).unwrap();
        assert!((rep.beta - 1.0).abs() < 0.15, "beta {}", rep.beta);
        assert!(rep.is_cointegrated, "adf_t {}", rep.adf_t);
        assert!(rep.tradeable, "half_life {}", rep.half_life);
        assert!(rep.half_life > 0.0);
        assert!(rep.hurst < 0.5, "hurst {}", rep.hurst);
    }

    #[test]
    fn independent_walks_are_not_cointegrated() {
        // Two independent random walks share no common trend.
        let a = random_walk(300, 1, 100.0);
        let b = random_walk(300, 2, 100.0);
        let rep = analyze_pair(&a, &b, &PairConfig::default()).unwrap();
        assert!(
            !rep.is_cointegrated,
            "adf_t {} unexpectedly stationary",
            rep.adf_t
        );
        assert!(!rep.tradeable);
    }

    #[test]
    fn rich_spread_signals_short() {
        // Cointegrated pair, then push A up on the last bar so the spread is rich.
        let b = random_walk(300, 7, 100.0);
        let spread = ar1(300, 0.4, 55);
        let mut a: Vec<f32> = (0..300).map(|i| b[i] + spread[i]).collect();
        let last = a.len() - 1;
        a[last] += 20.0; // large positive spread shock
        let rep = analyze_pair(&a, &b, &PairConfig::default()).unwrap();
        assert!(rep.spread_z > 2.0, "z {}", rep.spread_z);
        assert_eq!(rep.action, PairAction::ShortSpread);
    }

    #[test]
    fn too_short_returns_none() {
        let a = vec![1.0f32; 20];
        let b = vec![2.0f32; 20];
        assert!(analyze_pair(&a, &b, &PairConfig::default()).is_none());
    }

    #[test]
    fn scan_ranks_cointegrated_pair_first() {
        // 3 assets: X and Y are cointegrated; Z is an independent walk.
        let x = random_walk(300, 100, 50.0);
        let sp = ar1(300, 0.5, 200);
        let y: Vec<f32> = (0..300).map(|i| x[i] + sp[i]).collect();
        let z = random_walk(300, 300, 80.0);
        let symbols = vec!["X".to_string(), "Y".to_string(), "Z".to_string()];
        let ranked = scan_pairs(&symbols, &[x, y, z], &PairConfig::default());
        assert_eq!(ranked.len(), 3);
        // The X/Y pair should rank first and be tradeable.
        let top = &ranked[0];
        assert!(
            (top.symbol_a == "X" && top.symbol_b == "Y")
                || (top.symbol_a == "Y" && top.symbol_b == "X"),
            "top pair {}/{}",
            top.symbol_a,
            top.symbol_b
        );
        assert!(top.tradeable);
        // Scores are sorted descending.
        for w in ranked.windows(2)
        {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn deterministic() {
        let b = random_walk(300, 42, 100.0);
        let sp = ar1(300, 0.5, 43);
        let a: Vec<f32> = (0..300).map(|i| b[i] + sp[i]).collect();
        let r1 = analyze_pair(&a, &b, &PairConfig::default()).unwrap();
        let r2 = analyze_pair(&a, &b, &PairConfig::default()).unwrap();
        assert_eq!(r1.adf_t, r2.adf_t);
        assert_eq!(r1.beta, r2.beta);
        assert_eq!(r1.spread_z, r2.spread_z);
    }
}
