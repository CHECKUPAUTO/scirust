use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketState {
    pub exchange: Exchange,
    pub symbol: Symbol,
    pub timestamp: DateTime<Utc>,
    pub mid: f64,
    pub microprice: f64,
    pub spread_bps: f64,
    pub imbalance_5: f64,
    pub imbalance_20: f64,
    pub realized_vol_pct: f64,
    pub volume_1min: f64,
    pub flow_imbalance_1min: f64,
    pub trade_count_1min: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Exchange { Binance, Coinbase, Kraken }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Symbol {
    pub base: String,
    pub quote: String,
}

impl Symbol {
    pub fn new(base: impl Into<String>, quote: impl Into<String>) -> Self {
        Self { base: base.into(), quote: quote.into() }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Side { Buy, Sell }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketReaction {
    pub horizon_seconds: u32,
    pub mean_move_bps: f64,
    pub win_rate: f64,
}
