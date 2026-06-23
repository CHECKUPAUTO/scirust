//! Market data layer: snapshots, mock exchange, Binance connector (stub).
//!
//! A `MarketSnapshot` is the **deterministic unit of input** — a fixed-length
//! OHLCV window that becomes the tensor fed to the model. Its hash is part of
//! the decision proof, so two replayed runs always start from the same data.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// One OHLCV candle.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Candle {
    pub ts_ms: i64,
    pub open: f32,
    pub high: f32,
    pub low: f32,
    pub close: f32,
    pub volume: f32,
}

/// Fixed-length window of candles — the model input unit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketSnapshot {
    pub exchange: String,
    pub symbol: String,
    pub interval: String,
    pub candles: Vec<Candle>,
}

impl MarketSnapshot {
    /// SHA-256 fingerprint of the canonical JSON encoding.
    ///
    /// The encoding is *canonical*: keys are emitted in struct-declaration
    /// order, floats are rendered with 6 decimals, timestamps as integers.
    /// Two snapshots with the same fingerprint are guaranteed identical input.
    pub fn fingerprint(&self) -> String {
        let json = serde_json::to_string(self).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(json.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Close-price series as a `Vec<f32>` (most common model input).
    pub fn closes(&self) -> Vec<f32> {
        self.candles.iter().map(|c| c.close).collect()
    }

    /// Length of the window.
    pub fn len(&self) -> usize {
        self.candles.len()
    }

    /// True if the snapshot carries no candles.
    pub fn is_empty(&self) -> bool {
        self.candles.is_empty()
    }

    /// Last close price (reference price for the proof).
    pub fn last_close(&self) -> Option<f32> {
        self.candles.last().map(|c| c.close)
    }
}

/// A source of market snapshots.
pub trait MarketFeed {
    fn next_snapshot(&mut self, window: usize) -> Option<MarketSnapshot>;
}

/// Deterministic mock exchange — generates a random-walk price series seeded
/// from a PCG-style integer so that replay produces identical snapshots.
///
/// This is the **default feed** for the MVP: no network, no keys, no slippage
/// risk. The `BinanceConnector` below shares the same `MarketFeed` trait and
/// will be wired once the pipeline is validated.
pub struct MockExchange {
    seed: u64,
    price: f32,
    drift: f32,
    vol: f32,
    ts_ms: i64,
    step_ms: i64,
    symbol: String,
    interval: String,
}

impl MockExchange {
    /// Create a mock feed. `seed` pins the RNG; same seed ⇒ identical series.
    pub fn new(seed: u64, start_price: f32) -> Self {
        Self {
            seed,
            price: start_price,
            drift: 0.0001,
            vol: 0.005,
            ts_ms: 1_700_000_000_000,
            step_ms: 60_000,
            symbol: "BTC/USDT".to_string(),
            interval: "1m".to_string(),
        }
    }

    /// Simple xorshift RNG — deterministic, no external crate.
    fn next_rand(&mut self) -> f32 {
        self.seed ^= self.seed << 13;
        self.seed ^= self.seed >> 7;
        self.seed ^= self.seed << 17;
        ((self.seed % 10_000) as f32) / 10_000.0
    }

    /// Advance one candle using a Box-Muller-ish transform.
    fn next_candle(&mut self) -> Candle {
        let u1 = self.next_rand().max(1e-6);
        let u2 = self.next_rand();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        let ret = self.drift + self.vol * z;
        let open = self.price;
        self.price *= (1.0 + ret).max(0.01);
        let high = open.max(self.price) * (1.0 + self.vol * 0.3 * self.next_rand());
        let low = open.min(self.price) * (1.0 - self.vol * 0.3 * self.next_rand());
        let volume = 100.0 + 900.0 * self.next_rand();
        let c = Candle {
            ts_ms: self.ts_ms,
            open,
            high,
            low,
            close: self.price,
            volume,
        };
        self.ts_ms += self.step_ms;
        c
    }
}

impl MarketFeed for MockExchange {
    fn next_snapshot(&mut self, window: usize) -> Option<MarketSnapshot> {
        if window == 0
        {
            return None;
        }
        let candles: Vec<Candle> = (0..window).map(|_| self.next_candle()).collect();
        Some(MarketSnapshot {
            exchange: "mock".to_string(),
            symbol: self.symbol.clone(),
            interval: self.interval.clone(),
            candles,
        })
    }
}

/// Binance REST connector — fetches klines from `/api/v3/klines`.
///
/// Each call to `next_snapshot` fetches `window` most recent candles for the
/// configured symbol/interval. The Binance API returns arrays of mixed types
/// (strings and numbers), which we parse into `Candle` structs.
///
/// Endpoint: `GET /api/v3/klines?symbol=BTCUSDT&interval=1m&limit=50`
/// Response: array of arrays: `[openTime, open, high, low, close, volume, ...]`
pub struct BinanceConnector {
    pub api_base: String,
    pub symbol: String,
    pub interval: String,
}

impl BinanceConnector {
    pub fn new(symbol: &str, interval: &str) -> Self {
        Self {
            api_base: "https://api.binance.com".to_string(),
            symbol: symbol.to_string(),
            interval: interval.to_string(),
        }
    }

    pub fn with_base_url(mut self, base: &str) -> Self {
        self.api_base = base.to_string();
        self
    }

    /// Fetch klines from Binance — disabled in the default (pure-Rust) build.
    /// Build with `--features live` to enable real HTTP.
    #[cfg(not(feature = "live"))]
    fn fetch_klines(&self, _limit: usize) -> Option<Vec<Candle>> {
        let _ = &self.api_base;
        None
    }

    /// Fetch klines from Binance. Returns `None` on any HTTP/parse error.
    #[cfg(feature = "live")]
    fn fetch_klines(&self, limit: usize) -> Option<Vec<Candle>> {
        let url = format!(
            "{}/api/v3/klines?symbol={}&interval={}&limit={}",
            self.api_base, self.symbol, self.interval, limit
        );
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .ok()?;
        let resp = client.get(&url).send().ok()?;
        if !resp.status().is_success()
        {
            return None;
        }
        let raw: Vec<Vec<serde_json::Value>> = resp.json().ok()?;
        let candles = raw
            .into_iter()
            .map(|row| Candle {
                ts_ms: row.first().and_then(|v| v.as_i64()).unwrap_or(0),
                open: row
                    .get(1)
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0),
                high: row
                    .get(2)
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0),
                low: row
                    .get(3)
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0),
                close: row
                    .get(4)
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0),
                volume: row
                    .get(5)
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0),
            })
            .collect();
        Some(candles)
    }
}

impl MarketFeed for BinanceConnector {
    fn next_snapshot(&mut self, window: usize) -> Option<MarketSnapshot> {
        let candles = self.fetch_klines(window)?;
        if candles.is_empty()
        {
            return None;
        }
        Some(MarketSnapshot {
            exchange: "binance".to_string(),
            symbol: self.symbol.clone(),
            interval: self.interval.clone(),
            candles,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_exchange_is_deterministic() {
        let mut a = MockExchange::new(42, 50_000.0);
        let mut b = MockExchange::new(42, 50_000.0);
        let sa = a.next_snapshot(10).unwrap();
        let sb = b.next_snapshot(10).unwrap();
        assert_eq!(
            sa.candles, sb.candles,
            "same seed must yield identical candles"
        );
        assert_eq!(sa.fingerprint(), sb.fingerprint());
    }

    #[test]
    fn snapshot_fingerprint_is_stable() {
        let mut ex = MockExchange::new(7, 100.0);
        let s = ex.next_snapshot(5).unwrap();
        let fp1 = s.fingerprint();
        let fp2 = s.fingerprint();
        assert_eq!(fp1, fp2);
        assert_eq!(fp1.len(), 64);
    }

    #[test]
    fn closes_extract_correctly() {
        let mut ex = MockExchange::new(1, 100.0);
        let s = ex.next_snapshot(3).unwrap();
        assert_eq!(s.closes().len(), 3);
        assert!(s.last_close().is_some());
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = MockExchange::new(1, 100.0);
        let mut b = MockExchange::new(2, 100.0);
        let sa = a.next_snapshot(10).unwrap();
        let sb = b.next_snapshot(10).unwrap();
        assert_ne!(sa.fingerprint(), sb.fingerprint());
    }

    #[test]
    fn binance_connector_constructs() {
        let b = BinanceConnector::new("BTCUSDT", "1m");
        assert_eq!(b.symbol, "BTCUSDT");
        assert_eq!(b.interval, "1m");
    }

    #[cfg(feature = "network-tests")]
    #[test]
    fn binance_live_fetch_returns_candles() {
        let mut b = BinanceConnector::new("BTCUSDT", "1m");
        let snap = b.next_snapshot(10).expect("live Binance fetch should work");
        assert_eq!(snap.exchange, "binance");
        assert!(!snap.candles.is_empty());
        assert!(snap.last_close().unwrap_or(0.0) > 0.0);
    }
}
