//! Event-driven backtester — the engine that turns a [`Strategy`] into a
//! performance report, an equity curve, and a trade log.
//!
//! Discipline (the things that separate an honest backtest from a fantasy):
//! * **No look-ahead.** A signal is decided on bar `t`'s close and executed at
//!   bar `t+1`'s open via [`crate::orders::simulate_fill`].
//! * **Costs are real.** Every fill pays maker/taker fees and taker slippage.
//! * **Mark-to-market.** Equity is sampled every bar as `cash + Σ qty·mark`.
//! * **Round-trip accounting.** A trade opens when the position leaves flat and
//!   closes when it returns to flat (or flips), booking net PnL after fees.
//!
//! The result is a [`BacktestReport`] the agent can narrate, chart, and seal
//! into a proof.

use serde::{Deserialize, Serialize};

use crate::agent::Action;
use crate::indicators;
use crate::market::Candle;
use crate::metrics::{periods_per_year, PerformanceReport};
use crate::orders::{simulate_fill, FeeSchedule, Fill, Order, Side, SlippageModel};
use crate::portfolio::Account;
use crate::strategy::Strategy;
use std::collections::BTreeMap;

/// How to size a position from the target signal.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Sizing {
    /// Use `fraction` of current equity as the position notional.
    FixedFraction(f32),
    /// Use a fixed quote-currency notional every time.
    FixedNotional(f32),
    /// Risk `risk_fraction` of equity per trade with the stop set at
    /// `atr_mult · ATR(atr_period)` — the classic volatility-scaled sizing.
    AtrRisk {
        risk_fraction: f32,
        atr_period: usize,
        atr_mult: f32,
    },
}

impl Default for Sizing {
    fn default() -> Self {
        Sizing::FixedFraction(0.5)
    }
}

/// Backtest configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    pub symbol: String,
    pub interval: String,
    pub starting_cash: f32,
    pub fees: FeeSchedule,
    pub slippage: SlippageModel,
    pub sizing: Sizing,
    /// Allow short positions (otherwise Short signals go flat).
    pub allow_short: bool,
    /// Minimum signal strength required to act (0 = act on any non-flat signal).
    pub min_strength: f32,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            symbol: "BTC/USDT".to_string(),
            interval: "1h".to_string(),
            starting_cash: 10_000.0,
            fees: FeeSchedule::default(),
            slippage: SlippageModel::default(),
            sizing: Sizing::default(),
            allow_short: true,
            min_strength: 0.0,
        }
    }
}

/// One completed round-trip trade.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Trade {
    pub symbol: String,
    /// Entry direction.
    pub action: Action,
    pub entry_ts_ms: i64,
    pub exit_ts_ms: i64,
    pub entry_price: f32,
    pub exit_price: f32,
    pub qty: f32,
    pub gross_pnl: f32,
    pub fees: f32,
    pub net_pnl: f32,
    pub return_pct: f32,
    pub bars_held: usize,
}

/// The full backtest result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestReport {
    pub strategy: String,
    pub symbol: String,
    pub interval: String,
    pub starting_cash: f32,
    pub final_equity: f32,
    pub total_return: f32,
    /// Buy-and-hold return over the same window, for comparison.
    pub buy_hold_return: f32,
    pub num_trades: usize,
    pub fees_paid: f32,
    pub equity_curve: Vec<f32>,
    pub trades: Vec<Trade>,
    pub performance: PerformanceReport,
}

/// Tracks the currently-open round trip so it can be booked on close/flip.
struct OpenTrade {
    action: Action,
    entry_ts: i64,
    entry_index: usize,
    entry_price: f32,
    qty: f32,
    fees: f32,
    realized: f32,
}

/// Run `strategy` over `candles` under `cfg`. Candles must be chronological and
/// share `cfg.symbol`.
pub fn run_backtest(strategy: &dyn Strategy, candles: &[Candle], cfg: &BacktestConfig) -> BacktestReport {
    let n = candles.len();
    let mut account = Account::new(cfg.starting_cash);
    // Overwrite the initial curve sample so it stays length-aligned with bars.
    account.equity_curve.clear();

    let mut trades: Vec<Trade> = Vec::new();
    let mut open: Option<OpenTrade> = None;
    let mut order_id = 0u64;
    // Target signed position decided on the previous bar, executed on this bar.
    let mut pending_target: Option<f32> = None;
    let warmup = strategy.warmup().max(2);

    for i in 0..n
    {
        let candle = &candles[i];

        // 1. Execute the target decided on the previous bar at this bar's open.
        if let Some(target_qty) = pending_target.take()
        {
            let current = account.qty(&cfg.symbol);
            let delta = target_qty - current;
            if delta.abs() > 1e-9
            {
                let side = if delta > 0.0 { Side::Buy } else { Side::Sell };
                order_id += 1;
                let order = Order::market(order_id, &cfg.symbol, side, delta.abs());
                if let Some(fill) = simulate_fill(&order, candle, &cfg.fees, &cfg.slippage)
                {
                    apply_and_track(
                        &mut account,
                        &mut open,
                        &mut trades,
                        &cfg.symbol,
                        side,
                        &fill,
                        i,
                    );
                }
            }
        }

        // 2. Mark equity at this bar's close.
        let mut marks = BTreeMap::new();
        marks.insert(cfg.symbol.clone(), candle.close);
        account.mark(&marks);

        // 3. Decide the target for next-bar execution.
        if i + 1 >= warmup && i + 1 < n
        {
            let sig = strategy.evaluate(&candles[..=i]);
            let equity = account.equity(&marks);
            let target = target_qty_for(&sig, equity, candle.close, cfg, &candles[..=i]);
            pending_target = Some(target);
        }
    }

    // Close any residual position at the last close for a clean trade log.
    if account.qty(&cfg.symbol).abs() > 1e-9 && n > 0
    {
        let last = &candles[n - 1];
        let current = account.qty(&cfg.symbol);
        let side = if current > 0.0 { Side::Sell } else { Side::Buy };
        order_id += 1;
        // Fill at the last close (mark), paying fees/slippage.
        let closing_candle = Candle {
            open: last.close,
            ..*last
        };
        let order = Order::market(order_id, &cfg.symbol, side, current.abs());
        if let Some(fill) = simulate_fill(&order, &closing_candle, &cfg.fees, &cfg.slippage)
        {
            apply_and_track(&mut account, &mut open, &mut trades, &cfg.symbol, side, &fill, n - 1);
        }
    }

    let equity_curve = account.equity_curve.clone();
    let final_equity = equity_curve.last().copied().unwrap_or(cfg.starting_cash);
    let total_return = if cfg.starting_cash > 0.0
    {
        final_equity / cfg.starting_cash - 1.0
    }
    else
    {
        0.0
    };
    let buy_hold_return = if n >= 2 && candles[0].close > 0.0
    {
        candles[n - 1].close / candles[0].close - 1.0
    }
    else
    {
        0.0
    };
    let pnls: Vec<f32> = trades.iter().map(|t| t.net_pnl).collect();
    let performance = PerformanceReport::from_curve(
        &equity_curve,
        &pnls,
        periods_per_year(&cfg.interval),
        0.0,
    );

    BacktestReport {
        strategy: strategy.name(),
        symbol: cfg.symbol.clone(),
        interval: cfg.interval.clone(),
        starting_cash: cfg.starting_cash,
        final_equity,
        total_return,
        buy_hold_return,
        num_trades: trades.len(),
        fees_paid: account.fees_paid,
        equity_curve,
        trades,
        performance,
    }
}

/// Translate a signal + sizing into a target signed position quantity.
fn target_qty_for(
    sig: &crate::strategy::Signal,
    equity: f32,
    price: f32,
    cfg: &BacktestConfig,
    hist: &[Candle],
) -> f32 {
    if sig.strength < cfg.min_strength && sig.action != Action::Flat
    {
        // Not confident enough — hold flat.
        return 0.0;
    }
    let dir = match sig.action
    {
        Action::Long => 1.0,
        Action::Short =>
        {
            if cfg.allow_short { -1.0 } else { return 0.0 }
        },
        Action::Flat => return 0.0,
    };
    if price <= 1e-9 || equity <= 0.0
    {
        return 0.0;
    }
    let notional = match cfg.sizing
    {
        Sizing::FixedFraction(f) => equity * f.clamp(0.0, 1.0),
        Sizing::FixedNotional(n) => n.min(equity),
        Sizing::AtrRisk {
            risk_fraction,
            atr_period,
            atr_mult,
        } =>
        {
            let highs: Vec<f32> = hist.iter().map(|c| c.high).collect();
            let lows: Vec<f32> = hist.iter().map(|c| c.low).collect();
            let closes: Vec<f32> = hist.iter().map(|c| c.close).collect();
            let atr_series = indicators::atr(&highs, &lows, &closes, atr_period);
            let atr = atr_series.last().copied().unwrap_or(f32::NAN);
            if atr.is_nan() || atr <= 1e-9
            {
                equity * 0.1 // fall back to a modest fixed fraction
            }
            else
            {
                // qty = riskCapital / stopDistance; notional = qty·price.
                let risk_capital = equity * risk_fraction.clamp(0.0, 1.0);
                let stop_distance = atr_mult * atr;
                let qty = risk_capital / stop_distance;
                (qty * price).min(equity) // cap at 1x equity (no leverage here)
            }
        },
    };
    dir * notional / price
}

/// Apply a fill to the account and update the round-trip trade tracker.
fn apply_and_track(
    account: &mut Account,
    open: &mut Option<OpenTrade>,
    trades: &mut Vec<Trade>,
    symbol: &str,
    side: Side,
    fill: &Fill,
    bar_index: usize,
) {
    let qty_before = account.qty(symbol);
    let realized_before = account.realized_pnl;
    account.apply_fill(symbol, side, fill);
    let qty_after = account.qty(symbol);
    let realized_delta = account.realized_pnl - realized_before;

    match open.take()
    {
        None =>
        {
            // Opening a fresh trade.
            *open = Some(OpenTrade {
                action: if side == Side::Buy { Action::Long } else { Action::Short },
                entry_ts: fill.ts_ms,
                entry_index: bar_index,
                entry_price: fill.price,
                qty: fill.qty,
                fees: fill.fee,
                realized: 0.0,
            });
        },
        Some(mut t) =>
        {
            t.realized += realized_delta;
            let same_dir = (qty_before > 0.0) == (side == Side::Buy);
            if same_dir
            {
                // Added to the position.
                t.fees += fill.fee;
                let denom = t.qty + fill.qty;
                if denom > 1e-12
                {
                    t.entry_price = (t.entry_price * t.qty + fill.price * fill.qty) / denom;
                }
                t.qty = denom;
                *open = Some(t);
            }
            else if qty_after.abs() < 1e-9
            {
                // Closed flat — the whole fill fee belongs to this trade.
                t.fees += fill.fee;
                trades.push(book_trade(&t, symbol, fill, bar_index));
            }
            else if (qty_before > 0.0) != (qty_after > 0.0)
            {
                // Flipped through zero: the fill fee is split by quantity between
                // the closing leg (booked with the old trade) and the opening leg
                // (carried by the new overshoot trade), so neither trade's net
                // PnL absorbs the other's entry/exit fee.
                let close_qty = qty_before.abs();
                let close_fee = if fill.qty > 1e-12
                {
                    fill.fee * close_qty / fill.qty
                }
                else
                {
                    fill.fee
                };
                t.fees += close_fee;
                trades.push(book_trade(&t, symbol, fill, bar_index));
                *open = Some(OpenTrade {
                    action: if side == Side::Buy { Action::Long } else { Action::Short },
                    entry_ts: fill.ts_ms,
                    entry_index: bar_index,
                    entry_price: fill.price,
                    qty: qty_after.abs(),
                    fees: fill.fee - close_fee,
                    realized: 0.0,
                });
            }
            else
            {
                // Partial reduction, still open.
                t.fees += fill.fee;
                t.qty = qty_after.abs();
                *open = Some(t);
            }
        },
    }
}

fn book_trade(t: &OpenTrade, symbol: &str, exit_fill: &Fill, exit_index: usize) -> Trade {
    let gross = t.realized;
    let net = gross - t.fees;
    let notional = (t.entry_price * t.qty).abs().max(1e-9);
    Trade {
        symbol: symbol.to_string(),
        action: t.action,
        entry_ts_ms: t.entry_ts,
        exit_ts_ms: exit_fill.ts_ms,
        entry_price: t.entry_price,
        exit_price: exit_fill.price,
        qty: t.qty,
        gross_pnl: gross,
        fees: t.fees,
        net_pnl: net,
        return_pct: net / notional,
        bars_held: exit_index.saturating_sub(t.entry_index),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::{MaCross, Momentum};

    fn candle(ts: i64, close: f32) -> Candle {
        Candle {
            ts_ms: ts,
            open: close,
            high: close * 1.005,
            low: close * 0.995,
            close,
            volume: 100.0,
        }
    }

    fn uptrend(n: usize) -> Vec<Candle> {
        (0..n).map(|i| candle(i as i64 * 60_000, 100.0 + i as f32)).collect()
    }

    #[test]
    fn backtest_runs_and_reports() {
        let candles = uptrend(120);
        let strat = MaCross::sma(10, 30);
        let report = run_backtest(&strat, &candles, &BacktestConfig::default());
        assert!(report.equity_curve.len() >= 100);
        assert!(report.final_equity > 0.0);
        assert!(report.performance.sharpe.is_finite());
    }

    #[test]
    fn trend_follower_profits_in_clean_uptrend() {
        // A long trend follower on a monotone uptrend should end up profitable
        // and roughly track buy-and-hold (minus costs).
        let candles = uptrend(200);
        let strat = MaCross::sma(5, 20);
        let cfg = BacktestConfig {
            fees: FeeSchedule { maker_bps: 0.0, taker_bps: 0.0 },
            slippage: SlippageModel { base_bps: 0.0, impact_bps: 0.0, ref_liquidity: 1.0 },
            ..Default::default()
        };
        let report = run_backtest(&strat, &candles, &cfg);
        assert!(report.total_return > 0.0, "should profit: {}", report.total_return);
    }

    #[test]
    fn equity_curve_length_matches_bars() {
        let candles = uptrend(60);
        let report = run_backtest(&Momentum::default(), &candles, &BacktestConfig::default());
        assert_eq!(report.equity_curve.len(), candles.len());
    }

    #[test]
    fn no_short_config_stays_flat_on_short_signal() {
        // Downtrend + no-short -> never opens a short; equity stays ~flat.
        let candles: Vec<Candle> = (0..120).map(|i| candle(i as i64, 200.0 - i as f32)).collect();
        let cfg = BacktestConfig {
            allow_short: false,
            fees: FeeSchedule { maker_bps: 0.0, taker_bps: 0.0 },
            slippage: SlippageModel { base_bps: 0.0, impact_bps: 0.0, ref_liquidity: 1.0 },
            ..Default::default()
        };
        let report = run_backtest(&MaCross::sma(5, 20), &candles, &cfg);
        // With no shorting on a pure downtrend, we should not lose (stay in cash).
        assert!(report.total_return >= -0.02, "return {}", report.total_return);
    }

    #[test]
    fn fees_reduce_returns() {
        let candles = uptrend(150);
        let strat = MaCross::sma(5, 15);
        let no_fee = BacktestConfig {
            fees: FeeSchedule { maker_bps: 0.0, taker_bps: 0.0 },
            slippage: SlippageModel { base_bps: 0.0, impact_bps: 0.0, ref_liquidity: 1.0 },
            ..Default::default()
        };
        let with_fee = BacktestConfig {
            fees: FeeSchedule { maker_bps: 10.0, taker_bps: 20.0 },
            ..Default::default()
        };
        let r0 = run_backtest(&strat, &candles, &no_fee);
        let r1 = run_backtest(&strat, &candles, &with_fee);
        assert!(r1.fees_paid > 0.0);
        assert!(r1.total_return <= r0.total_return);
    }
}
