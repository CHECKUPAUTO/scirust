//! Execution algorithms — slicing a large parent order into fast, small child
//! orders to minimise market impact and slippage.
//!
//! This is the "micro-order" layer a professional execution desk runs on top of
//! the raw order types in [`crate::orders`]. A parent order of `total_qty` is
//! broken into a **schedule** of child orders:
//!
//! * **TWAP** — equal slices, evenly spaced in time.
//! * **VWAP** — slices proportional to a volume profile.
//! * **POV** — participate at a fixed fraction of traded volume.
//! * **Iceberg** — expose only a small display size, replenish until done.
//! * **Almgren–Chriss** — the impact/risk optimal trajectory
//!   `x_j = X·sinh(κ(T−t_j))/sinh(κT)`, which collapses to TWAP at zero risk
//!   aversion and front-loads as risk aversion rises.
//!
//! All schedules are deterministic. A schedule can be dry-run against a price
//! path with [`simulate_execution`] to measure the realised VWAP, slippage vs
//! the arrival price, and fees — the evidence the agent seals into a proof.

use serde::{Deserialize, Serialize};

use crate::market::Candle;
use crate::orders::{simulate_fill, FeeSchedule, Fill, Order, Side, SlippageModel};

/// One child (micro) order in an execution schedule.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct ChildOrder {
    /// 0-based slice index in the schedule.
    pub slice: usize,
    /// Scheduled timestamp (ms) — for a time-based algo.
    pub ts_ms: i64,
    /// Quantity for this child (base units).
    pub qty: f32,
}

/// A complete execution schedule for one parent order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub algo: String,
    pub side: Side,
    pub total_qty: f32,
    pub children: Vec<ChildOrder>,
    /// Model estimate of the temporary-impact cost in bps (0 if not modelled).
    pub expected_impact_bps: f32,
}

impl ExecutionPlan {
    /// Sum of child quantities — should equal `total_qty` up to rounding.
    pub fn scheduled_qty(&self) -> f32 {
        self.children.iter().map(|c| c.qty).sum()
    }

    pub fn num_slices(&self) -> usize {
        self.children.len()
    }
}

/// TWAP: `n` equal child orders spaced `interval_ms` apart from `start_ts`.
pub fn twap(
    side: Side,
    total_qty: f32,
    n: usize,
    start_ts: i64,
    interval_ms: i64,
) -> ExecutionPlan {
    let n = n.max(1);
    let per = total_qty / n as f32;
    let children = (0..n)
        .map(|i| ChildOrder {
            slice: i,
            ts_ms: start_ts + i as i64 * interval_ms,
            qty: per,
        })
        .collect();
    ExecutionPlan {
        algo: format!("twap({n})"),
        side,
        total_qty,
        children,
        expected_impact_bps: 0.0,
    }
}

/// VWAP: child quantities proportional to a `volume_profile` (one weight per
/// slice). Weights need not sum to 1 — they are normalised.
pub fn vwap(
    side: Side,
    total_qty: f32,
    volume_profile: &[f32],
    start_ts: i64,
    interval_ms: i64,
) -> ExecutionPlan {
    let total_w: f32 = volume_profile.iter().map(|v| v.max(0.0)).sum();
    let n = volume_profile.len().max(1);
    let children: Vec<ChildOrder> = if total_w <= 1e-12
    {
        // Degenerate profile -> fall back to equal weighting.
        (0..n)
            .map(|i| ChildOrder {
                slice: i,
                ts_ms: start_ts + i as i64 * interval_ms,
                qty: total_qty / n as f32,
            })
            .collect()
    }
    else
    {
        volume_profile
            .iter()
            .enumerate()
            .map(|(i, &w)| ChildOrder {
                slice: i,
                ts_ms: start_ts + i as i64 * interval_ms,
                qty: total_qty * w.max(0.0) / total_w,
            })
            .collect()
    };
    ExecutionPlan {
        algo: format!("vwap({n})"),
        side,
        total_qty,
        children,
        expected_impact_bps: 0.0,
    }
}

/// POV: participate at `rate` of each slice's expected volume, but never
/// scheduling more than `total_qty` in aggregate (the parent is capped).
pub fn pov(
    side: Side,
    total_qty: f32,
    rate: f32,
    expected_volumes: &[f32],
    start_ts: i64,
    interval_ms: i64,
) -> ExecutionPlan {
    let rate = rate.clamp(0.0, 1.0);
    let mut remaining = total_qty;
    let mut children = Vec::new();
    for (i, &vol) in expected_volumes.iter().enumerate()
    {
        if remaining <= 1e-9
        {
            break;
        }
        let qty = (rate * vol.max(0.0)).min(remaining);
        if qty > 1e-9
        {
            children.push(ChildOrder {
                slice: i,
                ts_ms: start_ts + i as i64 * interval_ms,
                qty,
            });
            remaining -= qty;
        }
    }
    // Any residual (profile too short/thin) goes on the last slice.
    if remaining > 1e-6
    {
        let i = expected_volumes.len().max(1) - 1;
        children.push(ChildOrder {
            slice: i,
            ts_ms: start_ts + i as i64 * interval_ms,
            qty: remaining,
        });
    }
    ExecutionPlan {
        algo: format!("pov({:.2})", rate),
        side,
        total_qty,
        children,
        expected_impact_bps: 0.0,
    }
}

/// Iceberg: expose `display` per slice, replenishing until the full `total_qty`
/// is scheduled.
pub fn iceberg(
    side: Side,
    total_qty: f32,
    display: f32,
    start_ts: i64,
    interval_ms: i64,
) -> ExecutionPlan {
    let display = display.max(1e-9);
    let mut remaining = total_qty;
    let mut children = Vec::new();
    let mut i = 0usize;
    while remaining > 1e-9 && i < 100_000
    {
        let qty = remaining.min(display);
        children.push(ChildOrder {
            slice: i,
            ts_ms: start_ts + i as i64 * interval_ms,
            qty,
        });
        remaining -= qty;
        i += 1;
    }
    ExecutionPlan {
        algo: format!("iceberg(display={:.4})", display),
        side,
        total_qty,
        children,
        expected_impact_bps: 0.0,
    }
}

/// Parameters for the Almgren–Chriss optimal execution schedule.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AlmgrenChriss {
    /// Number of trading intervals.
    pub n: usize,
    /// Total execution horizon (seconds).
    pub horizon_secs: f32,
    /// Per-second price volatility (absolute, price units).
    pub sigma: f32,
    /// Temporary impact coefficient η (price per unit trade rate).
    pub eta: f32,
    /// Permanent impact coefficient γ (price per unit traded).
    pub gamma: f32,
    /// Risk aversion λ (0 → risk-neutral → TWAP).
    pub risk_aversion: f32,
}

impl Default for AlmgrenChriss {
    /// Defaults are scaled so the adjusted temporary impact stays positive
    /// (`η̃ = η − ½γτ > 0`) for the whole schedule — otherwise the model is
    /// ill-posed and degenerates to TWAP.
    fn default() -> Self {
        Self {
            n: 10,
            horizon_secs: 60.0,
            sigma: 0.3,
            eta: 0.1,
            gamma: 1e-4,
            risk_aversion: 1e-4,
        }
    }
}

/// The Almgren–Chriss result: the holdings trajectory, the per-interval trade
/// list, and the expected-cost / variance of the strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcSchedule {
    /// Holdings remaining after each interval, `x_0 = X … x_N = 0` (length N+1).
    pub holdings: Vec<f32>,
    /// Shares traded in each interval, `n_j = x_{j-1} − x_j` (length N).
    pub trades: Vec<f32>,
    /// Decay rate κ of the optimal trajectory (0 = linear/TWAP).
    pub kappa: f32,
    /// Expected implementation-shortfall cost (permanent + temporary).
    pub expected_cost: f32,
    /// Variance of the shortfall.
    pub variance: f32,
}

/// Compute the Almgren–Chriss optimal liquidation trajectory for `total_qty`.
///
/// With `τ = T/N` and adjusted temporary impact `η̃ = η − ½γτ`, the trajectory
/// solves `cosh(κτ) = 1 + ½κ̃²τ²` where `κ̃² = λσ²/η̃`, giving
/// `x_j = X·sinh(κ(T−t_j)) / sinh(κT)`. At `λ→0`, `κ→0` and the schedule is
/// linear (TWAP).
pub fn almgren_chriss(side: Side, total_qty: f32, p: &AlmgrenChriss) -> (ExecutionPlan, AcSchedule) {
    let n = p.n.max(1);
    let x = total_qty;
    let t = p.horizon_secs.max(1e-6);
    let tau = t / n as f32;
    let eta_tilde = p.eta - 0.5 * p.gamma * tau;

    // Solve for κ. If risk aversion or adjusted impact is degenerate, fall back
    // to the linear (TWAP) trajectory.
    let kappa = if p.risk_aversion <= 0.0 || eta_tilde <= 1e-12 || p.sigma <= 0.0
    {
        0.0
    }
    else
    {
        let kappa_tilde_sq = p.risk_aversion * p.sigma * p.sigma / eta_tilde;
        let cosh_arg = 1.0 + 0.5 * kappa_tilde_sq * tau * tau;
        // acosh(x) for x >= 1.
        cosh_arg.max(1.0).acosh() / tau
    };

    // Holdings trajectory.
    let mut holdings = Vec::with_capacity(n + 1);
    if kappa <= 1e-9
    {
        for j in 0..=n
        {
            holdings.push(x * (1.0 - j as f32 / n as f32));
        }
    }
    else
    {
        let sinh_kt = (kappa * t).sinh();
        for j in 0..=n
        {
            let tj = j as f32 * tau;
            let hold = if sinh_kt.abs() > 1e-12
            {
                x * (kappa * (t - tj)).sinh() / sinh_kt
            }
            else
            {
                x * (1.0 - j as f32 / n as f32)
            };
            holdings.push(hold);
        }
    }
    // Pin the endpoints exactly.
    holdings[0] = x;
    holdings[n] = 0.0;

    let trades: Vec<f32> = (1..=n).map(|j| holdings[j - 1] - holdings[j]).collect();

    // Expected cost = permanent (½γX²) + temporary ((η̃/τ)·Σ n_j²).
    let temp_cost: f32 = trades.iter().map(|nj| nj * nj).sum::<f32>() * (eta_tilde.max(0.0) / tau);
    let expected_cost = 0.5 * p.gamma * x * x + temp_cost;
    // Variance = σ²·τ·Σ x_j² over the *held* intervals (j = 1..N).
    let variance: f32 = holdings[1..].iter().map(|xj| xj * xj).sum::<f32>() * (p.sigma * p.sigma * tau);

    let children = trades
        .iter()
        .enumerate()
        .map(|(i, &q)| ChildOrder {
            slice: i,
            ts_ms: (i as f32 * tau * 1000.0) as i64,
            qty: q,
        })
        .collect();

    // Rough impact estimate in bps: temporary cost per share over an assumed
    // unit reference price (caller can rescale). Reported as informational.
    let expected_impact_bps = if x.abs() > 1e-9 { 10_000.0 * temp_cost / x } else { 0.0 };

    let plan = ExecutionPlan {
        algo: format!("almgren_chriss(n={},lambda={:.1e})", n, p.risk_aversion),
        side,
        total_qty,
        children,
        expected_impact_bps,
    };
    let schedule = AcSchedule {
        holdings,
        trades,
        kappa,
        expected_cost,
        variance,
    };
    (plan, schedule)
}

/// A very fast, equal-size micro-order burst: `n` identical child orders with no
/// time spacing (all at `ts_ms`) — the primitive for rapid small-order slicing.
pub fn micro_burst(side: Side, total_qty: f32, n: usize, ts_ms: i64) -> ExecutionPlan {
    twap(side, total_qty, n, ts_ms, 0)
}

/// The realised outcome of running an [`ExecutionPlan`] against a price path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub filled_qty: f32,
    pub avg_price: f32,
    pub arrival_price: f32,
    /// Implementation shortfall vs the arrival price, in bps (≥0 = cost).
    pub slippage_bps: f32,
    pub total_fee: f32,
    pub num_fills: usize,
}

/// Dry-run an execution plan: fill each child as a market micro-order against
/// the matching candle in `candles` (child `i` uses `candles[min(i, last)]`),
/// then measure realised VWAP and slippage vs `arrival_price`.
pub fn simulate_execution(
    plan: &ExecutionPlan,
    candles: &[Candle],
    arrival_price: f32,
    fees: &FeeSchedule,
    slippage: &SlippageModel,
) -> ExecutionResult {
    if candles.is_empty()
    {
        return ExecutionResult {
            filled_qty: 0.0,
            avg_price: 0.0,
            arrival_price,
            slippage_bps: 0.0,
            total_fee: 0.0,
            num_fills: 0,
        };
    }
    let mut fills: Vec<Fill> = Vec::new();
    let mut notional = 0.0f32;
    let mut filled = 0.0f32;
    let mut total_fee = 0.0f32;
    let last = candles.len() - 1;
    for child in &plan.children
    {
        if child.qty <= 1e-12
        {
            continue;
        }
        let candle = &candles[child.slice.min(last)];
        let order = Order::market(child.slice as u64, "EXEC", plan.side, child.qty);
        if let Some(f) = simulate_fill(&order, candle, fees, slippage)
        {
            notional += f.price * f.qty;
            filled += f.qty;
            total_fee += f.fee;
            fills.push(f);
        }
    }
    let avg_price = if filled > 1e-12 { notional / filled } else { arrival_price };
    let slippage_bps = if arrival_price.abs() > 1e-12
    {
        10_000.0 * (avg_price - arrival_price) / arrival_price * plan.side.sign()
    }
    else
    {
        0.0
    };
    ExecutionResult {
        filled_qty: filled,
        avg_price,
        arrival_price,
        slippage_bps,
        total_fee,
        num_fills: fills.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_candles(price: f32, n: usize) -> Vec<Candle> {
        (0..n)
            .map(|i| Candle {
                ts_ms: i as i64,
                open: price,
                high: price * 1.001,
                low: price * 0.999,
                close: price,
                volume: 1000.0,
            })
            .collect()
    }

    #[test]
    fn twap_splits_evenly() {
        let plan = twap(Side::Buy, 100.0, 4, 0, 60_000);
        assert_eq!(plan.num_slices(), 4);
        assert!((plan.scheduled_qty() - 100.0).abs() < 1e-3);
        assert!(plan.children.iter().all(|c| (c.qty - 25.0).abs() < 1e-3));
        assert_eq!(plan.children[1].ts_ms, 60_000);
    }

    #[test]
    fn vwap_weights_by_profile() {
        let profile = [1.0, 2.0, 1.0]; // middle slice gets half
        let plan = vwap(Side::Sell, 100.0, &profile, 0, 1000);
        assert!((plan.scheduled_qty() - 100.0).abs() < 1e-3);
        assert!((plan.children[1].qty - 50.0).abs() < 1e-2);
        assert!((plan.children[0].qty - 25.0).abs() < 1e-2);
    }

    #[test]
    fn pov_caps_at_total() {
        // Rate 0.1 of big volumes but total only 30 -> aggregate scheduled = 30.
        let vols = [1000.0, 1000.0, 1000.0, 1000.0];
        let plan = pov(Side::Buy, 30.0, 0.1, &vols, 0, 1000);
        assert!((plan.scheduled_qty() - 30.0).abs() < 1e-2);
    }

    #[test]
    fn iceberg_chunks_by_display() {
        let plan = iceberg(Side::Buy, 100.0, 30.0, 0, 500);
        // 30+30+30+10
        assert_eq!(plan.num_slices(), 4);
        assert!((plan.scheduled_qty() - 100.0).abs() < 1e-3);
        assert!((plan.children[3].qty - 10.0).abs() < 1e-3);
    }

    #[test]
    fn almgren_chriss_conserves_and_terminates() {
        let (plan, sched) = almgren_chriss(Side::Sell, 1000.0, &AlmgrenChriss::default());
        assert!((plan.scheduled_qty() - 1000.0).abs() < 1.0);
        assert!((sched.holdings[0] - 1000.0).abs() < 1e-3);
        assert!(sched.holdings.last().unwrap().abs() < 1e-3);
        assert!(sched.expected_cost.is_finite());
        assert!(sched.variance >= 0.0);
    }

    #[test]
    fn almgren_chriss_reduces_to_twap_at_zero_risk() {
        let p = AlmgrenChriss {
            risk_aversion: 0.0,
            ..Default::default()
        };
        let (_, sched) = almgren_chriss(Side::Sell, 100.0, &p);
        // Linear trajectory -> equal trades.
        let first = sched.trades[0];
        assert!(sched.trades.iter().all(|t| (t - first).abs() < 1e-2));
        assert!(sched.kappa < 1e-6);
    }

    #[test]
    fn almgren_chriss_front_loads_with_risk_aversion() {
        // Parameters chosen so η̃ = η − ½γτ stays comfortably positive (τ = 6s):
        // η̃ = 0.05 − 0.5·1e-4·6 ≈ 0.0497, giving a strong nonzero κ.
        let p = AlmgrenChriss {
            n: 10,
            horizon_secs: 60.0,
            sigma: 0.5,
            eta: 0.05,
            gamma: 1e-4,
            risk_aversion: 1e-2,
        };
        let (_, sched) = almgren_chriss(Side::Sell, 1000.0, &p);
        // Risk-averse -> trade more early than late (front-loaded liquidation).
        assert!(sched.kappa > 0.0, "kappa should be positive: {}", sched.kappa);
        assert!(
            sched.trades[0] > *sched.trades.last().unwrap(),
            "front-load: first {} vs last {}",
            sched.trades[0],
            sched.trades.last().unwrap()
        );
    }

    #[test]
    fn simulate_execution_measures_slippage() {
        let plan = twap(Side::Buy, 100.0, 5, 0, 1000);
        let candles = flat_candles(100.0, 5);
        let slip = SlippageModel { base_bps: 5.0, impact_bps: 0.0, ref_liquidity: 1.0 };
        let res = simulate_execution(&plan, &candles, 100.0, &FeeSchedule::default(), &slip);
        assert!((res.filled_qty - 100.0).abs() < 1e-2);
        // Buying with 5 bps slippage -> avg price above arrival, positive slippage.
        assert!(res.slippage_bps > 0.0);
        assert!(res.avg_price > 100.0);
        assert!(res.total_fee > 0.0);
    }

    #[test]
    fn micro_burst_is_fast_equal_slices() {
        let plan = micro_burst(Side::Buy, 50.0, 10, 42);
        assert_eq!(plan.num_slices(), 10);
        assert!(plan.children.iter().all(|c| c.ts_ms == 42)); // no time spacing
        assert!((plan.scheduled_qty() - 50.0).abs() < 1e-3);
    }
}
