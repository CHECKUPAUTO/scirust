//! MCP tools for `scirust-trader` — the crypto-trading toolbelt an agentic LLM
//! drives by chatting.
//!
//! This is the surface that turns "connect to SciRust and find me trades that
//! respect these conditions, with a profit target of X — and show me the charts
//! and the order proofs" into concrete, deterministic tool calls. Every tool is
//! pure-Rust and simulation-first: the LLM can research, backtest, size, plan
//! execution, and seal proofs without ever touching real money. Live order
//! placement is intentionally **not** exposed here.
//!
//! Tools:
//! * `trader_market_data`         — synth/replay OHLCV candles
//! * `trader_indicators`          — 25+ technical indicators
//! * `trader_patterns`            — candlestick pattern detection
//! * `trader_signal`              — run a named strategy → target position
//! * `trader_backtest`            — event-driven backtest → performance report
//! * `trader_scan_opportunities`  — screen markets×strategies against constraints
//! * `trader_orderbook`           — microstructure + cost-to-trade
//! * `trader_size_position`       — ATR/risk-based position sizing
//! * `trader_execution_plan`      — TWAP/VWAP/POV/Iceberg/Almgren-Chriss slicing
//! * `trader_market_making_quotes`— Avellaneda-Stoikov optimal quotes
//! * `trader_microstructure`      — OFI / VPIN / Kyle's λ / trade-flow
//! * `trader_metrics`             — Sharpe/Sortino/VaR/… from an equity curve
//! * `trader_chart`               — self-contained SVG price/equity charts
//! * `trader_certified_predict`   — IBP-certified ML prediction (LLM-bounded)
//! * `trader_portfolio`           — account state: PnL, equity, exposure, liq price
//! * `trader_rebalance`           — trades to reach target portfolio weights

use crate::registry::McpTool;
use serde_json::{Value, json};
use std::collections::BTreeMap;

use scirust_trader::agent::{Action, StubLlm, TradingAgent};
use scirust_trader::backtest::{BacktestConfig, Sizing, run_backtest};
use scirust_trader::chart::{ChartOptions, Marker, Overlay, candlestick_svg, equity_curve_svg};
use scirust_trader::execution::{
    AlmgrenChriss, almgren_chriss, iceberg, micro_burst, pov, twap, vwap,
};
use scirust_trader::indicators;
use scirust_trader::market::{Candle, MarketFeed, MarketSnapshot, MockExchange};
use scirust_trader::marketmaking::{MmParams, optimal_quotes};
use scirust_trader::metrics::{PerformanceReport, periods_per_year};
use scirust_trader::microstructure::{
    L1Quote, TradePrint, kyle_lambda, order_flow_imbalance, trade_flow_imbalance, vpin,
};
use scirust_trader::model::PricePredictor;
use scirust_trader::orderbook::{Level, OrderBook};
use scirust_trader::orders::{FeeSchedule, Fill, Side, SlippageModel};
use scirust_trader::patterns::detect_patterns;
use scirust_trader::portfolio::{Account, Position, liquidation_price, rebalance_to_weights};
use scirust_trader::scanner::{OpportunityConstraints, ScanRiskConfig, scan};
use scirust_trader::strategy::{STRATEGY_NAMES, strategy_from_spec};

/// All trader tools.
pub fn trader_tools() -> Vec<McpTool> {
    vec![
        market_data_tool(),
        indicators_tool(),
        patterns_tool(),
        signal_tool(),
        backtest_tool(),
        scan_tool(),
        orderbook_tool(),
        size_position_tool(),
        execution_plan_tool(),
        market_making_tool(),
        microstructure_tool(),
        metrics_tool(),
        chart_tool(),
        certified_predict_tool(),
        portfolio_tool(),
        rebalance_tool(),
    ]
}

// ---------------------------------------------------------------------------
// Shared parsing helpers.
// ---------------------------------------------------------------------------

fn f(v: &Value, key: &str, default: f32) -> f32 {
    v.get(key)
        .and_then(|x| x.as_f64())
        .map(|x| x as f32)
        .unwrap_or(default)
}

fn u(v: &Value, key: &str, default: usize) -> usize {
    v.get(key)
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or(default)
}

fn s<'a>(v: &'a Value, key: &str, default: &'a str) -> String {
    v.get(key)
        .and_then(|x| x.as_str())
        .unwrap_or(default)
        .to_string()
}

/// Parse an OHLCV array into candles. Each row may be:
/// `[ts,o,h,l,c,v]` (6), `[o,h,l,c,v]` (5), `[o,h,l,c]` (4), or an object
/// `{ts,open,high,low,close,volume}`.
fn parse_candles(v: &Value) -> Result<Vec<Candle>, String> {
    let arr = v.as_array().ok_or("`ohlcv` must be an array of rows")?;
    let mut out = Vec::with_capacity(arr.len());
    for (i, row) in arr.iter().enumerate()
    {
        let c = if let Some(cols) = row.as_array()
        {
            let n = cols.len();
            let num = |idx: usize| cols.get(idx).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
            match n
            {
                6 => Candle {
                    ts_ms: cols[0].as_i64().unwrap_or(i as i64),
                    open: num(1),
                    high: num(2),
                    low: num(3),
                    close: num(4),
                    volume: num(5),
                },
                5 => Candle {
                    ts_ms: i as i64,
                    open: num(0),
                    high: num(1),
                    low: num(2),
                    close: num(3),
                    volume: num(4),
                },
                4 => Candle {
                    ts_ms: i as i64,
                    open: num(0),
                    high: num(1),
                    low: num(2),
                    close: num(3),
                    volume: 0.0,
                },
                _ => return Err(format!("row {i}: expected 4, 5 or 6 numbers, got {n}")),
            }
        }
        else if row.is_object()
        {
            Candle {
                ts_ms: row.get("ts").and_then(|x| x.as_i64()).unwrap_or(i as i64),
                open: f(row, "open", 0.0),
                high: f(row, "high", 0.0),
                low: f(row, "low", 0.0),
                close: f(row, "close", 0.0),
                volume: f(row, "volume", 0.0),
            }
        }
        else
        {
            return Err(format!("row {i}: must be an array or object"));
        };
        out.push(c);
    }
    if out.is_empty()
    {
        return Err("no candles parsed".to_string());
    }
    Ok(out)
}

fn snapshot_from(symbol: &str, interval: &str, candles: Vec<Candle>) -> MarketSnapshot {
    MarketSnapshot {
        exchange: "provided".to_string(),
        symbol: symbol.to_string(),
        interval: interval.to_string(),
        candles,
    }
}

fn ohlcv_arg(args: &Value) -> Result<Vec<Candle>, String> {
    let v = args.get("ohlcv").ok_or("missing `ohlcv`")?;
    parse_candles(v)
}

fn parse_action(v: &Value, key: &str) -> Option<Action> {
    match v.get(key).and_then(|x| x.as_str())
    {
        Some("long") | Some("LONG") | Some("buy") => Some(Action::Long),
        Some("short") | Some("SHORT") | Some("sell") => Some(Action::Short),
        Some("flat") | Some("FLAT") => Some(Action::Flat),
        _ => None,
    }
}

fn params_map(v: &Value) -> BTreeMap<String, f32> {
    let mut m = BTreeMap::new();
    if let Some(obj) = v.get("params").and_then(|x| x.as_object())
    {
        for (k, val) in obj
        {
            if let Some(x) = val.as_f64()
            {
                m.insert(k.clone(), x as f32);
            }
        }
    }
    m
}

fn to_value<T: serde::Serialize>(t: &T) -> Value {
    serde_json::to_value(t).unwrap_or(Value::Null)
}

// ---------------------------------------------------------------------------
// Tools.
// ---------------------------------------------------------------------------

fn market_data_tool() -> McpTool {
    McpTool {
        name: "trader_market_data".to_string(),
        description: "Generate a deterministic OHLCV candle series for testing/backtesting. \
            source='mock' produces a seeded random-walk (same seed => identical candles, no network). \
            Returns rows [ts_ms, open, high, low, close, volume] you can feed to the other trader tools. \
            (Live exchange data requires building scirust-trader with --features live.)"
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "source": { "type": "string", "enum": ["mock"], "description": "data source (default mock)" },
                "symbol": { "type": "string", "description": "symbol label (default BTC/USDT)" },
                "interval": { "type": "string", "description": "bar interval label (default 1m)" },
                "limit": { "type": "integer", "description": "number of candles (default 200)" },
                "seed": { "type": "integer", "description": "RNG seed for the mock walk (default 42)" },
                "start_price": { "type": "number", "description": "starting price (default 50000)" }
            }
        }),
        handler: Box::new(|args| {
            let limit = u(&args, "limit", 200).min(5000);
            let seed = args.get("seed").and_then(|x| x.as_u64()).unwrap_or(42);
            let start = f(&args, "start_price", 50_000.0);
            let symbol = s(&args, "symbol", "BTC/USDT");
            let interval = s(&args, "interval", "1m");
            let mut feed = MockExchange::new(seed, start);
            let snap = feed.next_snapshot(limit).ok_or("failed to generate candles")?;
            let rows: Vec<Value> = snap
                .candles
                .iter()
                .map(|c| json!([c.ts_ms, c.open, c.high, c.low, c.close, c.volume]))
                .collect();
            Ok(json!({
                "symbol": symbol,
                "interval": interval,
                "count": rows.len(),
                "fingerprint": snap.fingerprint(),
                "ohlcv": rows
            }))
        }),
    }
}

fn indicators_tool() -> McpTool {
    McpTool {
        name: "trader_indicators".to_string(),
        description: "Compute 25+ technical indicators from an OHLCV series and return their \
            latest values: SMA, EMA, RSI, MACD (+signal/hist), ATR, Bollinger bands, Stochastic \
            (%K/%D), ADX/+DI/-DI, OBV, VWAP, Williams %R, CCI, MFI, ROC, Z-score, Supertrend, \
            Donchian and Keltner channels. All use forward-order (deterministic) reductions with \
            Wilder smoothing where standard."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "ohlcv": { "type": "array", "description": "rows [ts,o,h,l,c,v] (ts & v optional)" },
                "params": { "type": "object", "description": "optional periods: rsi, atr, bb, bb_k, adx, stoch_k, stoch_d, ..." }
            },
            "required": ["ohlcv"]
        }),
        handler: Box::new(|args| {
            let candles = ohlcv_arg(&args)?;
            let closes: Vec<f32> = candles.iter().map(|c| c.close).collect();
            let highs: Vec<f32> = candles.iter().map(|c| c.high).collect();
            let lows: Vec<f32> = candles.iter().map(|c| c.low).collect();
            let vols: Vec<f32> = candles.iter().map(|c| c.volume).collect();
            let p = &args;
            let rsi_p = u(p, "rsi", 14);
            let atr_p = u(p, "atr", 14);
            let bb_p = u(p, "bb", 20);
            let bb_k = f(p, "bb_k", 2.0);
            let adx_p = u(p, "adx", 14);
            let stoch_k = u(p, "stoch_k", 14);
            let stoch_d = u(p, "stoch_d", 3);
            let last = |v: &[f32]| v.last().copied().filter(|x| x.is_finite());

            let (k, d) = indicators::stochastic(&highs, &lows, &closes, stoch_k, stoch_d);
            let dmi = indicators::dmi(&highs, &lows, &closes, adx_p);
            let don = indicators::donchian(&highs, &lows, 20);
            let kel = indicators::keltner(&highs, &lows, &closes, 20, 2.0);
            let st = indicators::supertrend(&highs, &lows, &closes, 10, 3.0);

            Ok(json!({
                "n": candles.len(),
                "last_close": closes.last().copied().unwrap_or(0.0),
                "indicators": {
                    "sma20": last(&indicators::sma(&closes, 20)),
                    "ema20": last(&indicators::ema(&closes, 20)),
                    "rsi": last(&indicators::rsi(&closes, rsi_p)),
                    "macd": last(&indicators::macd_line(&closes, 12, 26)),
                    "macd_signal": last(&indicators::macd_signal_line(&closes, 12, 26, 9)),
                    "atr": last(&indicators::atr(&highs, &lows, &closes, atr_p)),
                    "bb_upper": last(&indicators::bollinger_band(&closes, bb_p, bb_k, true)),
                    "bb_mid": last(&indicators::sma(&closes, bb_p)),
                    "bb_lower": last(&indicators::bollinger_band(&closes, bb_p, bb_k, false)),
                    "stoch_k": last(&k),
                    "stoch_d": last(&d),
                    "adx": last(&dmi.adx),
                    "plus_di": last(&dmi.plus_di),
                    "minus_di": last(&dmi.minus_di),
                    "obv": last(&indicators::obv(&closes, &vols)),
                    "vwap": last(&indicators::vwap(&highs, &lows, &closes, &vols, 20)),
                    "williams_r": last(&indicators::williams_r(&highs, &lows, &closes, 14)),
                    "cci": last(&indicators::cci(&highs, &lows, &closes, 20)),
                    "mfi": last(&indicators::mfi(&highs, &lows, &closes, &vols, 14)),
                    "roc": last(&indicators::roc(&closes, 12)),
                    "zscore": last(&indicators::zscore(&closes, 20)),
                    "cmf": last(&indicators::chaikin_money_flow(&highs, &lows, &closes, &vols, 20)),
                    "supertrend": last(&st.line),
                    "supertrend_dir": st.direction.last().copied().unwrap_or(0),
                    "donchian_upper": last(&don.upper),
                    "donchian_lower": last(&don.lower),
                    "keltner_upper": last(&kel.upper),
                    "keltner_lower": last(&kel.lower),
                }
            }))
        }),
    }
}

fn patterns_tool() -> McpTool {
    McpTool {
        name: "trader_patterns".to_string(),
        description: "Detect candlestick patterns (doji, hammer, engulfing, stars, marubozu, \
            three soldiers/crows, piercing/dark-cloud, …) in an OHLC series. Returns each detection \
            with its candle index, bullish/bearish bias, and strength."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "ohlcv": { "type": "array", "description": "rows [ts,o,h,l,c,v]" },
                "only_latest": { "type": "boolean", "description": "return only patterns on the last bar" }
            },
            "required": ["ohlcv"]
        }),
        handler: Box::new(|args| {
            let candles = ohlcv_arg(&args)?;
            let all = detect_patterns(&candles);
            let only_latest = args.get("only_latest").and_then(|x| x.as_bool()).unwrap_or(false);
            let last = candles.len().saturating_sub(1);
            let items: Vec<Value> = all
                .into_iter()
                .filter(|p| !only_latest || p.index == last)
                .map(|p| json!({
                    "pattern": p.kind.label(),
                    "index": p.index,
                    "bullish": p.bullish,
                    "strength": p.strength
                }))
                .collect();
            Ok(json!({ "count": items.len(), "patterns": items }))
        }),
    }
}

fn signal_tool() -> McpTool {
    McpTool {
        name: "trader_signal".to_string(),
        description: format!(
            "Run a named trading strategy over an OHLCV series and return the target position \
             (long/short/flat) implied by the latest closed bar, with a conviction strength and a \
             human reason. Available strategies: {}.",
            STRATEGY_NAMES.join(", ")
        ),
        input_schema: json!({
            "type": "object",
            "properties": {
                "ohlcv": { "type": "array" },
                "strategy": { "type": "string", "description": "strategy name" },
                "params": { "type": "object", "description": "strategy params, e.g. {\"fast\":10,\"slow\":30}" }
            },
            "required": ["ohlcv", "strategy"]
        }),
        handler: Box::new(|args| {
            let candles = ohlcv_arg(&args)?;
            let name = s(&args, "strategy", "sma_cross");
            let strat = strategy_from_spec(&name, &params_map(&args)).ok_or_else(|| {
                format!(
                    "unknown strategy `{name}`; available: {}",
                    STRATEGY_NAMES.join(", ")
                )
            })?;
            let sig = strat.evaluate(&candles);
            Ok(json!({
                "strategy": strat.name(),
                "action": sig.action.label(),
                "strength": sig.strength,
                "reason": sig.reason,
                "warmup": strat.warmup()
            }))
        }),
    }
}

fn build_backtest_cfg(args: &Value, symbol: &str, interval: &str) -> BacktestConfig {
    let sizing = match args.get("sizing").and_then(|x| x.as_str())
    {
        Some("fixed_notional") => Sizing::FixedNotional(f(args, "notional", 1000.0)),
        Some("atr_risk") => Sizing::AtrRisk {
            risk_fraction: f(args, "risk_fraction", 0.01),
            atr_period: u(args, "atr_period", 14),
            atr_mult: f(args, "atr_mult", 2.0),
        },
        _ => Sizing::FixedFraction(f(args, "fraction", 0.5)),
    };
    BacktestConfig {
        symbol: symbol.to_string(),
        interval: interval.to_string(),
        starting_cash: f(args, "capital", 10_000.0),
        fees: FeeSchedule {
            maker_bps: f(args, "maker_bps", 1.0),
            taker_bps: f(args, "taker_bps", 5.0),
        },
        slippage: SlippageModel {
            base_bps: f(args, "slippage_bps", 1.0),
            impact_bps: 0.0,
            ref_liquidity: 1.0,
        },
        sizing,
        allow_short: args
            .get("allow_short")
            .and_then(|x| x.as_bool())
            .unwrap_or(true),
        min_strength: f(args, "min_strength", 0.0),
    }
}

fn backtest_tool() -> McpTool {
    McpTool {
        name: "trader_backtest".to_string(),
        description: "Event-driven backtest of a strategy on an OHLCV series. Decides on each bar's \
            close and executes at the next bar's open (no look-ahead), charging maker/taker fees and \
            slippage. Returns the full performance report (Sharpe, Sortino, Calmar, max drawdown, \
            VaR, win rate, profit factor, …), the trade log, and the equity curve for charting."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "ohlcv": { "type": "array" },
                "strategy": { "type": "string" },
                "params": { "type": "object" },
                "symbol": { "type": "string" },
                "interval": { "type": "string", "description": "for annualising metrics (default 1h)" },
                "capital": { "type": "number", "description": "starting cash (default 10000)" },
                "sizing": { "type": "string", "enum": ["fixed_fraction", "fixed_notional", "atr_risk"] },
                "fraction": { "type": "number" },
                "maker_bps": { "type": "number" },
                "taker_bps": { "type": "number" },
                "slippage_bps": { "type": "number" },
                "allow_short": { "type": "boolean" },
                "include_curve": { "type": "boolean", "description": "include the equity curve array (default true)" }
            },
            "required": ["ohlcv", "strategy"]
        }),
        handler: Box::new(|args| {
            let candles = ohlcv_arg(&args)?;
            let name = s(&args, "strategy", "sma_cross");
            let symbol = s(&args, "symbol", "BTC/USDT");
            let interval = s(&args, "interval", "1h");
            let strat = strategy_from_spec(&name, &params_map(&args))
                .ok_or_else(|| format!("unknown strategy `{name}`"))?;
            let cfg = build_backtest_cfg(&args, &symbol, &interval);
            let report = run_backtest(strat.as_ref(), &candles, &cfg);
            let mut out = json!({
                "strategy": report.strategy,
                "symbol": report.symbol,
                "interval": report.interval,
                "starting_cash": report.starting_cash,
                "final_equity": report.final_equity,
                "total_return": report.total_return,
                "buy_hold_return": report.buy_hold_return,
                "num_trades": report.num_trades,
                "fees_paid": report.fees_paid,
                "performance": to_value(&report.performance),
                "trades": to_value(&report.trades),
            });
            let include_curve = args.get("include_curve").and_then(|x| x.as_bool()).unwrap_or(true);
            if include_curve
            {
                out["equity_curve"] = to_value(&report.equity_curve);
            }
            Ok(out)
        }),
    }
}

fn scan_tool() -> McpTool {
    McpTool {
        name: "trader_scan_opportunities".to_string(),
        description: "THE opportunity finder. Give it a universe of market series and a set of \
            constraints (min backtested return, max drawdown, min Sharpe/win-rate/profit-factor, \
            allowed direction, which strategies), and it backtests every strategy×symbol, reads the \
            live signal on the freshest bar, filters on the constraints, sizes an ATR-based trade \
            plan (entry/stop/take-profit/size), ranks the survivors, and seals each with a SHA-256 \
            proof hash. This is how an agent answers 'find me trades that respect these conditions \
            with a profit target of X' — deterministically and auditably."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "series": {
                    "type": "array",
                    "description": "market series: [{symbol, interval, ohlcv:[[ts,o,h,l,c,v],...]}]",
                    "items": {
                        "type": "object",
                        "properties": {
                            "symbol": { "type": "string" },
                            "interval": { "type": "string" },
                            "ohlcv": { "type": "array" }
                        },
                        "required": ["symbol", "ohlcv"]
                    }
                },
                "constraints": {
                    "type": "object",
                    "description": "min_total_return, min_expectancy, max_drawdown, min_sharpe, min_win_rate, min_profit_factor, min_trades, direction(long|short), min_signal_strength, strategies:[names], max_results"
                },
                "risk": {
                    "type": "object",
                    "description": "capital, risk_per_trade, atr_period, atr_mult, reward_risk"
                }
            },
            "required": ["series"]
        }),
        handler: Box::new(|args| {
            let series_json = args.get("series").and_then(|x| x.as_array()).ok_or("missing `series` array")?;
            let mut snapshots = Vec::new();
            for (i, sj) in series_json.iter().enumerate()
            {
                let symbol = sj.get("symbol").and_then(|x| x.as_str()).ok_or_else(|| format!("series[{i}]: missing symbol"))?;
                let interval = sj.get("interval").and_then(|x| x.as_str()).unwrap_or("1h");
                let ohlcv = sj.get("ohlcv").ok_or_else(|| format!("series[{i}]: missing ohlcv"))?;
                let candles = parse_candles(ohlcv).map_err(|e| format!("series[{i}]: {e}"))?;
                snapshots.push(snapshot_from(symbol, interval, candles));
            }

            let c = args.get("constraints").cloned().unwrap_or(json!({}));
            let mut constraints = OpportunityConstraints::default();
            if let Some(x) = c.get("min_total_return").and_then(|v| v.as_f64()) { constraints.min_total_return = x as f32; }
            if let Some(x) = c.get("min_expectancy").and_then(|v| v.as_f64()) { constraints.min_expectancy = x as f32; }
            if let Some(x) = c.get("max_drawdown").and_then(|v| v.as_f64()) { constraints.max_drawdown = x as f32; }
            if let Some(x) = c.get("min_sharpe").and_then(|v| v.as_f64()) { constraints.min_sharpe = x as f32; }
            constraints.min_win_rate = f(&c, "min_win_rate", constraints.min_win_rate);
            constraints.min_profit_factor = f(&c, "min_profit_factor", constraints.min_profit_factor);
            constraints.min_trades = u(&c, "min_trades", constraints.min_trades);
            constraints.min_signal_strength = f(&c, "min_signal_strength", constraints.min_signal_strength);
            constraints.max_results = u(&c, "max_results", constraints.max_results);
            constraints.direction = parse_action(&c, "direction");
            if let Some(list) = c.get("strategies").and_then(|v| v.as_array())
            {
                constraints.strategies = list.iter().filter_map(|s| s.as_str().map(String::from)).collect();
            }

            let r = args.get("risk").cloned().unwrap_or(json!({}));
            let risk = ScanRiskConfig {
                capital: f(&r, "capital", 10_000.0),
                risk_per_trade: f(&r, "risk_per_trade", 0.01),
                atr_period: u(&r, "atr_period", 14),
                atr_mult: f(&r, "atr_mult", 2.0),
                reward_risk: f(&r, "reward_risk", 2.0),
            };

            let report = scan(&snapshots, &constraints, &risk);
            Ok(to_value(&report))
        }),
    }
}

fn orderbook_tool() -> McpTool {
    McpTool {
        name: "trader_orderbook".to_string(),
        description: "Analyze an order-book snapshot: mid & micro price, spread (bps), depth, \
            order-flow imbalance, and — for a given size — the VWAP a market order would pay, the \
            resulting slippage, and whether the book can absorb it. The read-side an execution or \
            market-making agent needs before sizing."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "symbol": { "type": "string" },
                "bids": { "type": "array", "description": "[[price, qty], ...] (any order)" },
                "asks": { "type": "array", "description": "[[price, qty], ...]" },
                "size": { "type": "number", "description": "optional size to price a market fill for" },
                "depth_levels": { "type": "integer", "description": "levels for depth/imbalance (default 10)" }
            },
            "required": ["bids", "asks"]
        }),
        handler: Box::new(|args| {
            let parse_side = |key: &str| -> Result<Vec<Level>, String> {
                let arr = args
                    .get(key)
                    .and_then(|x| x.as_array())
                    .ok_or(format!("missing `{key}`"))?;
                let mut out = Vec::with_capacity(arr.len());
                for row in arr
                {
                    let cols = row
                        .as_array()
                        .ok_or(format!("`{key}` rows must be [price, qty]"))?;
                    let px = cols.first().and_then(|x| x.as_f64()).ok_or("bad price")? as f32;
                    let qty = cols.get(1).and_then(|x| x.as_f64()).ok_or("bad qty")? as f32;
                    out.push(Level::new(px, qty));
                }
                Ok(out)
            };
            let symbol = s(&args, "symbol", "SYMBOL");
            let book = OrderBook::new(&symbol, 0, parse_side("bids")?, parse_side("asks")?);
            let levels = u(&args, "depth_levels", 10);
            let mut out = to_value(&book.analyze(levels));
            if let Some(size) = args.get("size").and_then(|x| x.as_f64())
            {
                let size = size as f32;
                out["buy_fill"] = to_value(&book.vwap_to_fill(Side::Buy, size));
                out["sell_fill"] = to_value(&book.vwap_to_fill(Side::Sell, size));
            }
            Ok(out)
        }),
    }
}

fn size_position_tool() -> McpTool {
    McpTool {
        name: "trader_size_position".to_string(),
        description: "Risk-based position sizing. Given capital, risk-per-trade, an entry price and \
            either an explicit stop or an ATR (+multiplier), returns the position size that risks \
            exactly the chosen fraction to the stop, plus the take-profit at the target reward:risk. \
            The disciplined way to turn a signal into an order."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "capital": { "type": "number" },
                "risk_per_trade": { "type": "number", "description": "fraction of capital, e.g. 0.01" },
                "entry": { "type": "number" },
                "stop": { "type": "number", "description": "explicit stop price (else use atr)" },
                "atr": { "type": "number", "description": "ATR value (used if stop omitted)" },
                "atr_mult": { "type": "number", "description": "stop distance = atr_mult*atr (default 2)" },
                "reward_risk": { "type": "number", "description": "take-profit reward:risk (default 2)" },
                "direction": { "type": "string", "enum": ["long", "short"] }
            },
            "required": ["capital", "entry"]
        }),
        handler: Box::new(|args| {
            let capital = f(&args, "capital", 10_000.0);
            let risk_frac = f(&args, "risk_per_trade", 0.01).clamp(0.0, 1.0);
            let entry = f(&args, "entry", 0.0);
            let rr = f(&args, "reward_risk", 2.0);
            let long = args.get("direction").and_then(|x| x.as_str()) != Some("short");
            let stop_distance = if let Some(stop) = args.get("stop").and_then(|x| x.as_f64())
            {
                (entry - stop as f32).abs()
            }
            else
            {
                f(&args, "atr", entry * 0.02) * f(&args, "atr_mult", 2.0)
            };
            if stop_distance <= 1e-9 || entry <= 0.0
            {
                return Err("degenerate entry/stop distance".to_string());
            }
            let risk_amount = capital * risk_frac;
            let size = risk_amount / stop_distance;
            let notional = (size * entry).min(capital); // cap at 1x (no leverage)
            let size = notional / entry;
            let (stop, tp) = if long
            {
                (entry - stop_distance, entry + stop_distance * rr)
            }
            else
            {
                (entry + stop_distance, entry - stop_distance * rr)
            };
            Ok(json!({
                "direction": if long { "long" } else { "short" },
                "entry": entry,
                "stop_loss": stop,
                "take_profit": tp,
                "stop_distance": stop_distance,
                "position_size": size,
                "position_notional": notional,
                "risk_amount": risk_amount,
                "reward_risk": rr
            }))
        }),
    }
}

fn execution_plan_tool() -> McpTool {
    McpTool {
        name: "trader_execution_plan".to_string(),
        description: "Slice a large parent order into fast, small child orders to minimise market \
            impact. Algorithms: twap (equal time slices), vwap (proportional to a volume profile), \
            pov (percent-of-volume participation), iceberg (small display, replenished), \
            micro_burst (rapid equal micro-orders), and almgren_chriss (the impact/risk-optimal \
            trajectory that front-loads as risk aversion rises). Returns the child-order schedule."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "side": { "type": "string", "enum": ["buy", "sell"] },
                "total_qty": { "type": "number" },
                "algo": { "type": "string", "enum": ["twap", "vwap", "pov", "iceberg", "micro_burst", "almgren_chriss"] },
                "slices": { "type": "integer", "description": "number of slices (twap/micro_burst)" },
                "interval_ms": { "type": "integer" },
                "volume_profile": { "type": "array", "description": "weights per slice (vwap)" },
                "expected_volumes": { "type": "array", "description": "per-slice volume (pov)" },
                "rate": { "type": "number", "description": "participation rate (pov)" },
                "display": { "type": "number", "description": "display size (iceberg)" },
                "ac": { "type": "object", "description": "almgren_chriss params: n, horizon_secs, sigma, eta, gamma, risk_aversion" }
            },
            "required": ["side", "total_qty", "algo"]
        }),
        handler: Box::new(|args| {
            let side = match args.get("side").and_then(|x| x.as_str())
            {
                Some("sell") => Side::Sell,
                _ => Side::Buy,
            };
            let total = f(&args, "total_qty", 0.0);
            if total <= 0.0
            {
                return Err("total_qty must be > 0".to_string());
            }
            let algo = s(&args, "algo", "twap");
            let n = u(&args, "slices", 10);
            let interval = args
                .get("interval_ms")
                .and_then(|x| x.as_i64())
                .unwrap_or(60_000);
            match algo.as_str()
            {
                "twap" => Ok(to_value(&twap(side, total, n, 0, interval))),
                "micro_burst" => Ok(to_value(&micro_burst(side, total, n, 0))),
                "vwap" =>
                {
                    let profile: Vec<f32> = args
                        .get("volume_profile")
                        .and_then(|x| x.as_array())
                        .map(|a| {
                            a.iter()
                                .filter_map(|v| v.as_f64().map(|x| x as f32))
                                .collect()
                        })
                        .unwrap_or_else(|| vec![1.0; n]);
                    Ok(to_value(&vwap(side, total, &profile, 0, interval)))
                },
                "pov" =>
                {
                    let vols: Vec<f32> = args
                        .get("expected_volumes")
                        .and_then(|x| x.as_array())
                        .map(|a| {
                            a.iter()
                                .filter_map(|v| v.as_f64().map(|x| x as f32))
                                .collect()
                        })
                        .unwrap_or_else(|| vec![total; n]);
                    Ok(to_value(&pov(
                        side,
                        total,
                        f(&args, "rate", 0.1),
                        &vols,
                        0,
                        interval,
                    )))
                },
                "iceberg" => Ok(to_value(&iceberg(
                    side,
                    total,
                    f(&args, "display", total / 10.0).max(1e-6),
                    0,
                    interval,
                ))),
                "almgren_chriss" =>
                {
                    let a = args.get("ac").cloned().unwrap_or(json!({}));
                    let p = AlmgrenChriss {
                        n: u(&a, "n", 10),
                        horizon_secs: f(&a, "horizon_secs", 60.0),
                        sigma: f(&a, "sigma", 0.3),
                        eta: f(&a, "eta", 0.1),
                        gamma: f(&a, "gamma", 1e-4),
                        risk_aversion: f(&a, "risk_aversion", 1e-4),
                    };
                    let (plan, sched) = almgren_chriss(side, total, &p);
                    Ok(json!({ "plan": to_value(&plan), "schedule": to_value(&sched) }))
                },
                other => Err(format!("unknown algo `{other}`")),
            }
        }),
    }
}

fn market_making_tool() -> McpTool {
    McpTool {
        name: "trader_market_making_quotes".to_string(),
        description: "Avellaneda-Stoikov optimal market-making quotes. From the mid price, current \
            signed inventory, time-to-horizon, and (gamma risk aversion, sigma volatility, kappa \
            liquidity) it returns the inventory-skewed reservation price and the optimal bid/ask so \
            the maker earns the spread while mean-reverting inventory to flat."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "mid": { "type": "number" },
                "inventory": { "type": "number", "description": "signed position (long +, short -)" },
                "time_remaining": { "type": "number", "description": "T-t (default 1.0)" },
                "gamma": { "type": "number", "description": "risk aversion (default 0.1)" },
                "sigma": { "type": "number", "description": "volatility (default 2.0)" },
                "kappa": { "type": "number", "description": "order-arrival decay (default 1.5)" }
            },
            "required": ["mid"]
        }),
        handler: Box::new(|args| {
            let mid = f(&args, "mid", 0.0);
            let inv = f(&args, "inventory", 0.0);
            let tr = f(&args, "time_remaining", 1.0);
            let p = MmParams {
                gamma: f(&args, "gamma", 0.1),
                sigma: f(&args, "sigma", 2.0),
                kappa: f(&args, "kappa", 1.5),
            };
            Ok(to_value(&optimal_quotes(mid, inv, tr, &p)))
        }),
    }
}

fn microstructure_tool() -> McpTool {
    McpTool {
        name: "trader_microstructure".to_string(),
        description: "Compute high-frequency microstructure signals: Order-Flow Imbalance (from a \
            series of top-of-book quotes), trade-flow imbalance (from trade prints), VPIN flow \
            toxicity, and Kyle's lambda price impact (from signed volume vs mid changes). The alpha \
            a fast small-order trader reads off the tape."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "quotes": { "type": "array", "description": "L1 quotes: [{bid_px,bid_qty,ask_px,ask_qty}, ...]" },
                "trades": { "type": "array", "description": "prints: [{price,qty,buyer_is_taker}, ...]" },
                "vpin": { "type": "object", "description": "{prices:[...], volumes:[...], bucket_volume, num_buckets}" },
                "kyle": { "type": "object", "description": "{signed_volume:[...], mid_change:[...]}" }
            }
        }),
        handler: Box::new(|args| {
            let mut out = json!({});
            if let Some(q) = args.get("quotes").and_then(|x| x.as_array())
            {
                let quotes: Vec<L1Quote> = q.iter().map(|v| L1Quote {
                    bid_px: f(v, "bid_px", 0.0),
                    bid_qty: f(v, "bid_qty", 0.0),
                    ask_px: f(v, "ask_px", 0.0),
                    ask_qty: f(v, "ask_qty", 0.0),
                }).collect();
                out["order_flow_imbalance"] = json!(order_flow_imbalance(&quotes));
            }
            if let Some(t) = args.get("trades").and_then(|x| x.as_array())
            {
                let trades: Vec<TradePrint> = t.iter().map(|v| TradePrint {
                    price: f(v, "price", 0.0),
                    qty: f(v, "qty", 0.0),
                    buyer_is_taker: v.get("buyer_is_taker").and_then(|x| x.as_bool()).unwrap_or(true),
                }).collect();
                out["trade_flow_imbalance"] = json!(trade_flow_imbalance(&trades));
            }
            if let Some(v) = args.get("vpin")
            {
                let prices: Vec<f32> = v.get("prices").and_then(|x| x.as_array())
                    .map(|a| a.iter().filter_map(|x| x.as_f64().map(|y| y as f32)).collect()).unwrap_or_default();
                let volumes: Vec<f32> = v.get("volumes").and_then(|x| x.as_array())
                    .map(|a| a.iter().filter_map(|x| x.as_f64().map(|y| y as f32)).collect()).unwrap_or_default();
                out["vpin"] = json!(vpin(&prices, &volumes, f(v, "bucket_volume", 1000.0), u(v, "num_buckets", 50)));
            }
            if let Some(v) = args.get("kyle")
            {
                let sv: Vec<f32> = v.get("signed_volume").and_then(|x| x.as_array())
                    .map(|a| a.iter().filter_map(|x| x.as_f64().map(|y| y as f32)).collect()).unwrap_or_default();
                let mc: Vec<f32> = v.get("mid_change").and_then(|x| x.as_array())
                    .map(|a| a.iter().filter_map(|x| x.as_f64().map(|y| y as f32)).collect()).unwrap_or_default();
                out["kyle_lambda"] = json!(kyle_lambda(&sv, &mc));
            }
            Ok(out)
        }),
    }
}

fn metrics_tool() -> McpTool {
    McpTool {
        name: "trader_metrics".to_string(),
        description: "Compute a full performance & risk report (Sharpe, Sortino, Calmar, CAGR, \
            volatility, max drawdown, Ulcer, VaR/CVaR, Kelly, win rate, profit factor, expectancy) \
            from an equity curve (or a return series) plus optional realised trade PnLs. Metrics are \
            annualised for the given bar interval."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "equity": { "type": "array", "description": "equity curve values" },
                "returns": { "type": "array", "description": "per-bar returns (used if equity omitted)" },
                "trade_pnls": { "type": "array", "description": "realised PnL per closed trade" },
                "interval": { "type": "string", "description": "bar interval for annualisation (default 1d)" }
            }
        }),
        handler: Box::new(|args| {
            let interval = s(&args, "interval", "1d");
            let ppy = periods_per_year(&interval);
            let equity: Vec<f32> = if let Some(e) = args.get("equity").and_then(|x| x.as_array())
            {
                e.iter().filter_map(|v| v.as_f64().map(|y| y as f32)).collect()
            }
            else if let Some(r) = args.get("returns").and_then(|x| x.as_array())
            {
                // Rebuild an equity curve from returns starting at 1.0.
                let mut eq = vec![1.0f32];
                for v in r
                {
                    let ret = v.as_f64().unwrap_or(0.0) as f32;
                    let last = *eq.last().unwrap();
                    eq.push(last * (1.0 + ret));
                }
                eq
            }
            else
            {
                return Err("provide `equity` or `returns`".to_string());
            };
            if equity.len() < 2
            {
                return Err("need at least 2 equity points".to_string());
            }
            let pnls: Vec<f32> = args.get("trade_pnls").and_then(|x| x.as_array())
                .map(|a| a.iter().filter_map(|v| v.as_f64().map(|y| y as f32)).collect())
                .unwrap_or_default();
            let report = PerformanceReport::from_curve(&equity, &pnls, ppy, 0.0);
            Ok(to_value(&report))
        }),
    }
}

fn chart_tool() -> McpTool {
    McpTool {
        name: "trader_chart".to_string(),
        description: "Render a self-contained SVG chart the LLM can display directly. type='candles' \
            draws a candlestick chart from OHLCV with optional indicator overlays and entry/exit \
            markers; type='equity' draws an equity curve. No dependencies — the returned string is a \
            complete <svg> document."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "type": { "type": "string", "enum": ["candles", "equity"] },
                "ohlcv": { "type": "array", "description": "for type=candles" },
                "equity": { "type": "array", "description": "for type=equity" },
                "overlays": { "type": "array", "description": "[{name,color,values:[...]}]" },
                "markers": { "type": "array", "description": "[{index,price,bullish,label}]" },
                "title": { "type": "string" },
                "width": { "type": "integer" },
                "height": { "type": "integer" }
            }
        }),
        handler: Box::new(|args| {
            let opts = ChartOptions {
                width: u(&args, "width", 900) as u32,
                height: u(&args, "height", 480) as u32,
                title: s(&args, "title", ""),
            };
            let kind = s(&args, "type", "candles");
            let svg = if kind == "equity"
            {
                let eq: Vec<f32> = args.get("equity").and_then(|x| x.as_array())
                    .map(|a| a.iter().filter_map(|v| v.as_f64().map(|y| y as f32)).collect())
                    .ok_or("type=equity requires `equity`")?;
                equity_curve_svg(&eq, &opts)
            }
            else
            {
                let candles = ohlcv_arg(&args)?;
                let overlays: Vec<Overlay> = args.get("overlays").and_then(|x| x.as_array())
                    .map(|a| a.iter().map(|o| Overlay {
                        name: s(o, "name", "overlay"),
                        color: s(o, "color", "#f6c343"),
                        values: o.get("values").and_then(|v| v.as_array())
                            .map(|vs| vs.iter().map(|v| v.as_f64().map(|y| y as f32).unwrap_or(f32::NAN)).collect())
                            .unwrap_or_default(),
                    }).collect())
                    .unwrap_or_default();
                let markers: Vec<Marker> = args.get("markers").and_then(|x| x.as_array())
                    .map(|a| a.iter().map(|m| Marker {
                        index: u(m, "index", 0),
                        price: f(m, "price", 0.0),
                        bullish: m.get("bullish").and_then(|x| x.as_bool()).unwrap_or(true),
                        label: s(m, "label", ""),
                    }).collect())
                    .unwrap_or_default();
                candlestick_svg(&candles, &overlays, &markers, &opts)
            };
            Ok(json!({ "svg": svg, "format": "svg" }))
        }),
    }
}

fn certified_predict_tool() -> McpTool {
    McpTool {
        name: "trader_certified_predict".to_string(),
        description: "Run SciRust's certified ML predictor on an OHLCV window. A small MLP predicts \
            the next-bar return; Interval Bound Propagation then proves a certified output interval \
            the prediction is guaranteed to lie in, so any LLM narration is hard-bounded and cannot \
            hallucinate a move outside it. Returns the action, the certified bounds, feature \
            attributions, and SHA-256 fingerprints of the input window and model weights."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "ohlcv": { "type": "array" },
                "symbol": { "type": "string" },
                "seed": { "type": "integer", "description": "model seed (default 42, deterministic)" }
            },
            "required": ["ohlcv"]
        }),
        handler: Box::new(|args| {
            let mut candles = ohlcv_arg(&args)?;
            if candles.len() < 30
            {
                return Err("certified_predict needs at least 30 candles".to_string());
            }
            let symbol = s(&args, "symbol", "BTC/USDT");
            let seed = args.get("seed").and_then(|x| x.as_u64()).unwrap_or(42);
            // The model input dim is 13 (lookback 10 + rsi + macd_hist + atr).
            let model = PricePredictor::new(13, &[16, 8], seed);
            let mut agent = TradingAgent::new(model, Box::new(StubLlm));
            agent.lookback = 10;
            // Force the snapshot symbol so the prediction is labelled correctly.
            for c in candles.iter_mut() { let _ = c; }
            let snapshot = snapshot_from(&symbol, "provided", candles);
            let record = agent.process(&snapshot);
            let p = &record.prediction;
            Ok(json!({
                "symbol": p.symbol,
                "action": p.action.label(),
                "raw_prediction": p.raw_prediction,
                "certified_interval": { "lo": p.bounds.output.lo, "hi": p.bounds.output.hi },
                "uncertainty": p.bounds.uncertainty,
                "last_close": p.last_close,
                "snapshot_fingerprint": p.snapshot_fingerprint,
                "weights_fingerprint": p.weights_fingerprint,
                "feature_attribution": to_value(&p.feature_attribution),
                "narration": record.narration,
                "llm_consistent": record.llm_consistent
            }))
        }),
    }
}

/// Build an `Account` from a JSON `{cash, positions:[{symbol,qty,avg_entry,realized_pnl?}]}`.
fn account_from(args: &Value) -> Account {
    let mut acct = Account::new(f(args, "cash", 0.0));
    if let Some(list) = args.get("positions").and_then(|x| x.as_array())
    {
        for p in list
        {
            let symbol = s(p, "symbol", "");
            if symbol.is_empty()
            {
                continue;
            }
            acct.positions.insert(
                symbol.clone(),
                Position {
                    symbol,
                    qty: f(p, "qty", 0.0),
                    avg_entry: f(p, "avg_entry", 0.0),
                    realized_pnl: f(p, "realized_pnl", 0.0),
                },
            );
        }
    }
    acct
}

/// Parse `{marks:{SYM: price}}` into a symbol→price map.
fn marks_from(args: &Value) -> BTreeMap<String, f32> {
    let mut m = BTreeMap::new();
    if let Some(obj) = args.get("marks").and_then(|x| x.as_object())
    {
        for (k, v) in obj
        {
            if let Some(p) = v.as_f64()
            {
                m.insert(k.clone(), p as f32);
            }
        }
    }
    m
}

fn portfolio_tool() -> McpTool {
    McpTool {
        name: "trader_portfolio".to_string(),
        description:
            "Track a paper portfolio's state. Given cash, current positions (signed qty + \
            average entry), and mark prices — plus optional fills to apply first — it returns \
            mark-to-market equity, realized & unrealized PnL, fees, per-position PnL and market \
            value, and gross/net exposure. If `leverage` is given it also reports each position's \
            isolated-margin liquidation price. This is how an agent answers 'what's my PnL / \
            exposure / distance to liquidation?' — deterministically, no real funds."
                .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "cash": { "type": "number", "description": "quote-currency cash balance" },
                "positions": {
                    "type": "array",
                    "description": "[{symbol, qty (signed: +long/-short), avg_entry, realized_pnl?}]"
                },
                "marks": { "type": "object", "description": "{SYMBOL: mark_price}" },
                "fills": {
                    "type": "array",
                    "description": "optional fills to apply first: [{symbol, side(buy|sell), price, qty, fee?}]"
                },
                "leverage": { "type": "number", "description": "optional, for liquidation price" },
                "mmr": { "type": "number", "description": "maintenance-margin rate (default 0.005)" }
            }
        }),
        handler: Box::new(|args| {
            let mut acct = account_from(&args);
            // Apply optional fills.
            if let Some(fills) = args.get("fills").and_then(|x| x.as_array())
            {
                for fj in fills
                {
                    let symbol = s(fj, "symbol", "");
                    if symbol.is_empty()
                    {
                        continue;
                    }
                    let side = match fj.get("side").and_then(|x| x.as_str())
                    {
                        Some("sell") | Some("SELL") => Side::Sell,
                        _ => Side::Buy,
                    };
                    let fill = Fill {
                        price: f(fj, "price", 0.0),
                        qty: f(fj, "qty", 0.0),
                        fee: f(fj, "fee", 0.0),
                        taker: true,
                        ts_ms: 0,
                    };
                    acct.apply_fill(&symbol, side, &fill);
                }
            }
            let marks = marks_from(&args);
            let leverage = args
                .get("leverage")
                .and_then(|x| x.as_f64())
                .map(|x| x as f32);
            let mmr = f(&args, "mmr", 0.005);
            let (gross, net) = acct.exposure(&marks);
            let positions: Vec<Value> = acct
                .positions
                .values()
                .filter(|p| !p.is_flat())
                .map(|p| {
                    let mark = marks.get(&p.symbol).copied().unwrap_or(p.avg_entry);
                    let mut obj = json!({
                        "symbol": p.symbol,
                        "qty": p.qty,
                        "avg_entry": p.avg_entry,
                        "mark": mark,
                        "market_value": p.market_value(mark),
                        "unrealized_pnl": p.unrealized_pnl(mark),
                        "realized_pnl": p.realized_pnl,
                        "side": if p.is_long() { "long" } else { "short" },
                    });
                    if let Some(lev) = leverage
                    {
                        let side = if p.is_long() { Side::Buy } else { Side::Sell };
                        obj["liquidation_price"] =
                            json!(liquidation_price(p.avg_entry, lev, mmr, side));
                    }
                    obj
                })
                .collect();
            Ok(json!({
                "cash": acct.cash,
                "equity": acct.equity(&marks),
                "realized_pnl": acct.realized_pnl,
                "unrealized_pnl": acct.unrealized_pnl(&marks),
                "fees_paid": acct.fees_paid,
                "gross_exposure": gross,
                "net_exposure": net,
                "positions": positions,
            }))
        }),
    }
}

fn rebalance_tool() -> McpTool {
    McpTool {
        name: "trader_rebalance".to_string(),
        description: "Compute the trades that move a portfolio to target weights. Given cash, current \
            positions, mark prices, and target weights (fraction of total equity per symbol; may sum \
            to < 1 to hold cash), it returns the buy/sell trades — with a `band` to suppress trades \
            below a drift threshold. The disciplined way to rebalance a book via chat."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "cash": { "type": "number" },
                "positions": { "type": "array", "description": "[{symbol, qty, avg_entry}]" },
                "marks": { "type": "object", "description": "{SYMBOL: mark_price}" },
                "target_weights": { "type": "object", "description": "{SYMBOL: weight in [0,1]}" },
                "band": { "type": "number", "description": "min weight drift to trade (default 0)" }
            },
            "required": ["marks", "target_weights"]
        }),
        handler: Box::new(|args| {
            let acct = account_from(&args);
            let marks = marks_from(&args);
            let mut targets = BTreeMap::new();
            if let Some(obj) = args.get("target_weights").and_then(|x| x.as_object())
            {
                for (k, v) in obj
                {
                    if let Some(w) = v.as_f64()
                    {
                        targets.insert(k.clone(), w as f32);
                    }
                }
            }
            let band = f(&args, "band", 0.0);
            let trades = rebalance_to_weights(&acct, &targets, &marks, band);
            Ok(json!({
                "equity": acct.equity(&marks),
                "num_trades": trades.len(),
                "trades": to_value(&trades),
            }))
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_ohlcv(n: usize) -> Value {
        let rows: Vec<Value> = (0..n)
            .map(|i| {
                let c = 100.0 + i as f32;
                json!([i, c, c + 1.0, c - 1.0, c, 100.0])
            })
            .collect();
        json!(rows)
    }

    fn tool(name: &str) -> McpTool {
        trader_tools()
            .into_iter()
            .find(|t| t.name == name)
            .expect("tool exists")
    }

    #[test]
    fn all_tools_have_unique_names() {
        let tools = trader_tools();
        let mut names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        names.sort();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len(), "duplicate trader tool name");
        assert!(before >= 16);
    }

    #[test]
    fn portfolio_reports_pnl_and_exposure() {
        let t = tool("trader_portfolio");
        let out = (t.handler)(json!({
            "cash": 5000.0,
            "positions": [{ "symbol": "BTC/USDT", "qty": 1.0, "avg_entry": 50000.0 }],
            "marks": { "BTC/USDT": 55000.0 },
            "leverage": 10.0
        }))
        .unwrap();
        // equity = 5000 cash + 1*55000 = 60000; unrealized = 5000.
        assert!((out["equity"].as_f64().unwrap() - 60000.0).abs() < 1.0);
        assert!((out["unrealized_pnl"].as_f64().unwrap() - 5000.0).abs() < 1.0);
        let pos = &out["positions"][0];
        assert_eq!(pos["side"], json!("long"));
        // 10x long -> liquidation below entry.
        assert!(pos["liquidation_price"].as_f64().unwrap() < 50000.0);
    }

    #[test]
    fn portfolio_applies_fills() {
        let t = tool("trader_portfolio");
        let out = (t.handler)(json!({
            "cash": 10000.0,
            "marks": { "ETH/USDT": 3000.0 },
            "fills": [{ "symbol": "ETH/USDT", "side": "buy", "price": 3000.0, "qty": 2.0, "fee": 6.0 }]
        }))
        .unwrap();
        // cash = 10000 - 6000 - 6 fee = 3994; equity = 3994 + 2*3000 = 9994.
        assert!((out["equity"].as_f64().unwrap() - 9994.0).abs() < 1.0);
        assert!((out["fees_paid"].as_f64().unwrap() - 6.0).abs() < 1e-3);
    }

    #[test]
    fn rebalance_produces_trades_to_targets() {
        let t = tool("trader_rebalance");
        let out = (t.handler)(json!({
            "cash": 10000.0,
            "positions": [],
            "marks": { "BTC/USDT": 100.0 },
            "target_weights": { "BTC/USDT": 0.5 }
        }))
        .unwrap();
        // 50% of 10000 = 5000 / 100 = 50 units to buy.
        assert_eq!(out["num_trades"], json!(1));
        assert!((out["trades"][0]["qty"].as_f64().unwrap() - 50.0).abs() < 1e-2);
        assert_eq!(out["trades"][0]["side"], json!("Buy"));
    }

    #[test]
    fn market_data_is_deterministic() {
        let t = tool("trader_market_data");
        let a = (t.handler)(json!({ "seed": 7, "limit": 20 })).unwrap();
        let b = (t.handler)(json!({ "seed": 7, "limit": 20 })).unwrap();
        assert_eq!(a["fingerprint"], b["fingerprint"]);
        assert_eq!(a["count"], json!(20));
    }

    #[test]
    fn indicators_compute() {
        let t = tool("trader_indicators");
        let out = (t.handler)(json!({ "ohlcv": mock_ohlcv(60) })).unwrap();
        assert!(out["indicators"]["rsi"].is_number());
        assert!(out["indicators"]["adx"].is_number() || out["indicators"]["adx"].is_null());
    }

    #[test]
    fn signal_runs_named_strategy() {
        let t = tool("trader_signal");
        let out = (t.handler)(json!({ "ohlcv": mock_ohlcv(80), "strategy": "sma_cross" })).unwrap();
        assert_eq!(out["action"], json!("LONG")); // clean uptrend
    }

    #[test]
    fn signal_rejects_unknown_strategy() {
        let t = tool("trader_signal");
        assert!((t.handler)(json!({ "ohlcv": mock_ohlcv(80), "strategy": "nope" })).is_err());
    }

    #[test]
    fn backtest_returns_performance() {
        let t = tool("trader_backtest");
        let out =
            (t.handler)(json!({ "ohlcv": mock_ohlcv(150), "strategy": "sma_cross" })).unwrap();
        assert!(out["performance"]["sharpe"].is_number());
        assert!(out["final_equity"].is_number());
    }

    #[test]
    fn scan_finds_and_seals() {
        let t = tool("trader_scan_opportunities");
        let out = (t.handler)(json!({
            "series": [{ "symbol": "BTC/USDT", "interval": "1h", "ohlcv": mock_ohlcv(200) }]
        }))
        .unwrap();
        assert!(out["manifest_hash"].is_string());
        assert!(out["opportunities"].is_array());
    }

    #[test]
    fn orderbook_analysis() {
        let t = tool("trader_orderbook");
        let out = (t.handler)(json!({
            "bids": [[99.0, 5.0], [98.0, 10.0]],
            "asks": [[101.0, 4.0], [102.0, 8.0]],
            "size": 6.0
        }))
        .unwrap();
        assert!(out["mid"].as_f64().unwrap() > 99.0);
        assert!(out["buy_fill"]["vwap"].is_number());
    }

    #[test]
    fn size_position_computes_stop_and_tp() {
        let t = tool("trader_size_position");
        let out = (t.handler)(json!({
            "capital": 10000.0, "risk_per_trade": 0.01, "entry": 100.0, "atr": 2.0
        }))
        .unwrap();
        assert!(out["position_size"].as_f64().unwrap() > 0.0);
        assert!(out["stop_loss"].as_f64().unwrap() < 100.0);
        assert!(out["take_profit"].as_f64().unwrap() > 100.0);
    }

    #[test]
    fn execution_plans() {
        let t = tool("trader_execution_plan");
        let twap =
            (t.handler)(json!({ "side": "buy", "total_qty": 100.0, "algo": "twap", "slices": 4 }))
                .unwrap();
        assert_eq!(twap["children"].as_array().unwrap().len(), 4);
        let ac =
            (t.handler)(json!({ "side": "sell", "total_qty": 1000.0, "algo": "almgren_chriss" }))
                .unwrap();
        assert!(ac["schedule"]["kappa"].is_number());
    }

    #[test]
    fn market_making_quotes() {
        let t = tool("trader_market_making_quotes");
        let out = (t.handler)(json!({ "mid": 100.0, "inventory": 5.0 })).unwrap();
        // Long inventory -> reservation below mid.
        assert!(out["reservation_price"].as_f64().unwrap() < 100.0);
        assert!(out["bid"].as_f64().unwrap() < out["ask"].as_f64().unwrap());
    }

    #[test]
    fn microstructure_signals() {
        let t = tool("trader_microstructure");
        let out = (t.handler)(json!({
            "trades": [{"price":100.0,"qty":3.0,"buyer_is_taker":true},{"price":100.0,"qty":1.0,"buyer_is_taker":false}]
        })).unwrap();
        assert!((out["trade_flow_imbalance"].as_f64().unwrap() - 0.5).abs() < 1e-3);
    }

    #[test]
    fn metrics_from_equity() {
        let t = tool("trader_metrics");
        let eq: Vec<f32> = (0..50).map(|i| 10_000.0 + i as f32 * 20.0).collect();
        let out = (t.handler)(json!({ "equity": eq, "interval": "1d" })).unwrap();
        assert!(out["sharpe"].is_number());
        assert!(out["max_drawdown"].as_f64().unwrap() >= 0.0);
    }

    #[test]
    fn chart_renders_svg() {
        let t = tool("trader_chart");
        let out = (t.handler)(json!({ "type": "candles", "ohlcv": mock_ohlcv(30), "title": "T" }))
            .unwrap();
        assert!(out["svg"].as_str().unwrap().starts_with("<svg"));
    }

    #[test]
    fn certified_predict_bounds() {
        let t = tool("trader_certified_predict");
        let out = (t.handler)(json!({ "ohlcv": mock_ohlcv(50) })).unwrap();
        let lo = out["certified_interval"]["lo"].as_f64().unwrap();
        let hi = out["certified_interval"]["hi"].as_f64().unwrap();
        assert!(lo <= hi);
        let raw = out["raw_prediction"].as_f64().unwrap();
        assert!(
            raw >= lo - 1e-4 && raw <= hi + 1e-4,
            "raw {raw} must be within [{lo},{hi}]"
        );
    }
}
