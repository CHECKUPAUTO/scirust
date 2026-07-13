//! Market data layer: snapshots, deterministic mock exchange, and an optional
//! live Binance connector.
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
    /// SHA-256 fingerprint of a canonical binary encoding.
    ///
    /// Strings are length-prefixed, integers use little-endian bytes, and every
    /// float is represented by its exact IEEE-754 bit pattern. This encoding is
    /// total (including for NaN and infinities), deterministic, and never falls
    /// back to hashing an empty serialization.
    pub fn fingerprint(&self) -> String {
        let mut hasher = Sha256::new();
        hash_string(&mut hasher, &self.exchange);
        hash_string(&mut hasher, &self.symbol);
        hash_string(&mut hasher, &self.interval);
        hasher.update((self.candles.len() as u64).to_le_bytes());
        for candle in &self.candles
        {
            hasher.update(candle.ts_ms.to_le_bytes());
            hasher.update(candle.open.to_bits().to_le_bytes());
            hasher.update(candle.high.to_bits().to_le_bytes());
            hasher.update(candle.low.to_bits().to_le_bytes());
            hasher.update(candle.close.to_bits().to_le_bytes());
            hasher.update(candle.volume.to_bits().to_le_bytes());
        }
        format!("{:x}", hasher.finalize())
    }

    /// Fingerprint a snapshot only if every market value is finite.
    pub fn try_fingerprint(&self) -> Result<String, &'static str> {
        if self.candles.iter().any(|c| {
            !c.open.is_finite()
                || !c.high.is_finite()
                || !c.low.is_finite()
                || !c.close.is_finite()
                || !c.volume.is_finite()
        })
        {
            return Err("market snapshot contains a non-finite OHLCV value");
        }
        Ok(self.fingerprint())
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

fn hash_string(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
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
            .iter()
            .map(|row| parse_kline(row))
            .collect::<Option<Vec<_>>>()?;
        Some(candles)
    }
}

#[cfg(any(feature = "live", test))]
fn parse_kline(row: &[serde_json::Value]) -> Option<Candle> {
    fn finite_f32(value: &serde_json::Value) -> Option<f32> {
        let parsed = if let Some(text) = value.as_str()
        {
            text.parse::<f32>().ok()?
        }
        else
        {
            value.as_f64()? as f32
        };
        parsed.is_finite().then_some(parsed)
    }

    Some(Candle {
        ts_ms: row.first()?.as_i64()?,
        open: finite_f32(row.get(1)?)?,
        high: finite_f32(row.get(2)?)?,
        low: finite_f32(row.get(3)?)?,
        close: finite_f32(row.get(4)?)?,
        volume: finite_f32(row.get(5)?)?,
    })
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

    #[test]
    fn non_finite_snapshots_never_collapse_to_one_fingerprint() {
        let mut a = MockExchange::new(1, 100.0).next_snapshot(1).unwrap();
        let mut b = a.clone();
        a.candles[0].close = f32::NAN;
        b.candles[0].close = f32::INFINITY;
        assert_ne!(a.fingerprint(), b.fingerprint());
        assert!(a.try_fingerprint().is_err());
        assert!(b.try_fingerprint().is_err());
    }

    #[test]
    fn malformed_binance_rows_are_rejected_instead_of_zero_filled() {
        let valid = serde_json::json!([
            1_700_000_000_000_i64,
            "100.0",
            "101.0",
            "99.0",
            "100.5",
            "42.0"
        ]);
        let valid_row = valid.as_array().unwrap();
        assert_eq!(parse_kline(valid_row).unwrap().close, 100.5);

        let missing_volume =
            serde_json::json!([1_700_000_000_000_i64, "100", "101", "99", "100.5"]);
        assert!(parse_kline(missing_volume.as_array().unwrap()).is_none());

        let malformed_price = serde_json::json!([
            1_700_000_000_000_i64,
            "not-a-price",
            "101",
            "99",
            "100.5",
            "42"
        ]);
        assert!(parse_kline(malformed_price.as_array().unwrap()).is_none());

        let non_finite =
            serde_json::json!([1_700_000_000_000_i64, "NaN", "101", "99", "100.5", "42"]);
        assert!(parse_kline(non_finite.as_array().unwrap()).is_none());
    }
}
