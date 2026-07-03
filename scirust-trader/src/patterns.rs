//! Candlestick pattern recognition — the price-action layer.
//!
//! Every detector is a pure function over a slice of [`Candle`]s. A pattern is
//! reported with the index of its **last** candle, a bullish/bearish bias, and a
//! `strength` in `[0, 1]` so a strategy or the agent can rank signals. Detection
//! uses only closed bars (no look-ahead) and guards against zero-range candles.

use serde::{Deserialize, Serialize};

use crate::market::Candle;

/// The catalogue of recognised patterns.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PatternKind {
    Doji,
    Hammer,
    InvertedHammer,
    ShootingStar,
    HangingMan,
    BullishMarubozu,
    BearishMarubozu,
    BullishEngulfing,
    BearishEngulfing,
    PiercingLine,
    DarkCloudCover,
    MorningStar,
    EveningStar,
    ThreeWhiteSoldiers,
    ThreeBlackCrows,
}

impl PatternKind {
    pub fn label(&self) -> &'static str {
        match self
        {
            PatternKind::Doji => "doji",
            PatternKind::Hammer => "hammer",
            PatternKind::InvertedHammer => "inverted_hammer",
            PatternKind::ShootingStar => "shooting_star",
            PatternKind::HangingMan => "hanging_man",
            PatternKind::BullishMarubozu => "bullish_marubozu",
            PatternKind::BearishMarubozu => "bearish_marubozu",
            PatternKind::BullishEngulfing => "bullish_engulfing",
            PatternKind::BearishEngulfing => "bearish_engulfing",
            PatternKind::PiercingLine => "piercing_line",
            PatternKind::DarkCloudCover => "dark_cloud_cover",
            PatternKind::MorningStar => "morning_star",
            PatternKind::EveningStar => "evening_star",
            PatternKind::ThreeWhiteSoldiers => "three_white_soldiers",
            PatternKind::ThreeBlackCrows => "three_black_crows",
        }
    }
}

/// A detected pattern instance.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Pattern {
    pub kind: PatternKind,
    /// Index of the pattern's final candle in the input slice.
    pub index: usize,
    pub bullish: bool,
    /// Confidence/prominence in `[0, 1]`.
    pub strength: f32,
}

/// Geometry of one candle, all in price units.
#[derive(Debug, Clone, Copy)]
struct Geom {
    body: f32,
    range: f32,
    upper: f32,
    lower: f32,
    bull: bool,
    open: f32,
    close: f32,
}

fn geom(c: &Candle) -> Geom {
    let body = (c.close - c.open).abs();
    let range = (c.high - c.low).max(1e-9);
    let upper = c.high - c.open.max(c.close);
    let lower = c.open.min(c.close) - c.low;
    Geom {
        body,
        range,
        upper: upper.max(0.0),
        lower: lower.max(0.0),
        bull: c.close >= c.open,
        open: c.open,
        close: c.close,
    }
}

/// Detect all patterns ending at or before the last candle. Results are sorted
/// by candle index ascending. Trend context (for hammer vs hanging-man, etc.)
/// uses the mean close of the preceding `trend_lookback` candles.
pub fn detect_patterns(candles: &[Candle]) -> Vec<Pattern> {
    detect_patterns_with(candles, 5)
}

/// As [`detect_patterns`] with an explicit trend look-back.
pub fn detect_patterns_with(candles: &[Candle], trend_lookback: usize) -> Vec<Pattern> {
    let n = candles.len();
    let mut out = Vec::new();
    for i in 0..n
    {
        let g = geom(&candles[i]);
        let downtrend = in_downtrend(candles, i, trend_lookback);
        let uptrend = in_uptrend(candles, i, trend_lookback);

        // --- single-candle patterns ---
        // Doji: body is a tiny fraction of the range.
        if g.body <= 0.1 * g.range
        {
            out.push(Pattern {
                kind: PatternKind::Doji,
                index: i,
                bullish: false,
                strength: (1.0 - g.body / g.range).clamp(0.0, 1.0),
            });
        }
        // Marubozu: body fills almost the whole range (tiny shadows).
        else if g.body >= 0.9 * g.range
        {
            out.push(Pattern {
                kind: if g.bull { PatternKind::BullishMarubozu } else { PatternKind::BearishMarubozu },
                index: i,
                bullish: g.bull,
                strength: (g.body / g.range).clamp(0.0, 1.0),
            });
        }

        // Hammer / hanging man: small body near the top, long lower shadow,
        // a stubby upper shadow (no larger than the body).
        if g.body >= 0.03 * g.range
            && g.lower >= 2.0 * g.body
            && g.upper <= g.body
            && g.body <= 0.4 * g.range
        {
            let strength = (g.lower / (2.0 * g.body)).min(2.0) / 2.0;
            if downtrend
            {
                out.push(Pattern {
                    kind: PatternKind::Hammer,
                    index: i,
                    bullish: true,
                    strength,
                });
            }
            else if uptrend
            {
                out.push(Pattern {
                    kind: PatternKind::HangingMan,
                    index: i,
                    bullish: false,
                    strength,
                });
            }
        }
        // Inverted hammer / shooting star: small body near the bottom, long upper.
        if g.body >= 0.03 * g.range
            && g.upper >= 2.0 * g.body
            && g.lower <= g.body
            && g.body <= 0.4 * g.range
        {
            let strength = (g.upper / (2.0 * g.body)).min(2.0) / 2.0;
            if uptrend
            {
                out.push(Pattern {
                    kind: PatternKind::ShootingStar,
                    index: i,
                    bullish: false,
                    strength,
                });
            }
            else if downtrend
            {
                out.push(Pattern {
                    kind: PatternKind::InvertedHammer,
                    index: i,
                    bullish: true,
                    strength,
                });
            }
        }

        // --- two-candle patterns ---
        if i >= 1
        {
            let p = geom(&candles[i - 1]);
            // Bullish engulfing: prev bearish, curr bullish, curr body covers prev.
            if !p.bull
                && g.bull
                && g.close >= p.open
                && g.open <= p.close
                && g.body > p.body
            {
                out.push(Pattern {
                    kind: PatternKind::BullishEngulfing,
                    index: i,
                    bullish: true,
                    strength: (g.body / (p.body + 1e-9)).min(3.0) / 3.0,
                });
            }
            // Bearish engulfing: prev bullish, curr bearish, curr body covers prev.
            if p.bull
                && !g.bull
                && g.open >= p.close
                && g.close <= p.open
                && g.body > p.body
            {
                out.push(Pattern {
                    kind: PatternKind::BearishEngulfing,
                    index: i,
                    bullish: false,
                    strength: (g.body / (p.body + 1e-9)).min(3.0) / 3.0,
                });
            }
            // Piercing line: prev bearish, curr bullish opening below prev low,
            // closing above the midpoint of the prev body.
            let prev_mid = (p.open + p.close) / 2.0;
            if !p.bull
                && g.bull
                && g.open < p.close
                && g.close > prev_mid
                && g.close < p.open
            {
                out.push(Pattern {
                    kind: PatternKind::PiercingLine,
                    index: i,
                    bullish: true,
                    strength: 0.6,
                });
            }
            // Dark cloud cover: mirror of piercing line.
            if p.bull
                && !g.bull
                && g.open > p.close
                && g.close < prev_mid
                && g.close > p.open
            {
                out.push(Pattern {
                    kind: PatternKind::DarkCloudCover,
                    index: i,
                    bullish: false,
                    strength: 0.6,
                });
            }
        }

        // --- three-candle patterns ---
        if i >= 2
        {
            let a = geom(&candles[i - 2]);
            let b = geom(&candles[i - 1]);
            let mid_a = (a.open + a.close) / 2.0;
            // Morning star: big bearish, small-bodied star, big bullish closing
            // back above the midpoint of the first candle.
            if !a.bull
                && b.body <= 0.5 * a.body
                && g.bull
                && g.close > mid_a
                && a.body > 1e-9
            {
                out.push(Pattern {
                    kind: PatternKind::MorningStar,
                    index: i,
                    bullish: true,
                    strength: 0.75,
                });
            }
            // Evening star: mirror.
            if a.bull
                && b.body <= 0.5 * a.body
                && !g.bull
                && g.close < mid_a
                && a.body > 1e-9
            {
                out.push(Pattern {
                    kind: PatternKind::EveningStar,
                    index: i,
                    bullish: false,
                    strength: 0.75,
                });
            }
            // Three white soldiers: three rising bullish candles.
            if a.bull
                && b.bull
                && g.bull
                && b.close > a.close
                && g.close > b.close
                && b.open > a.open
                && g.open > b.open
            {
                out.push(Pattern {
                    kind: PatternKind::ThreeWhiteSoldiers,
                    index: i,
                    bullish: true,
                    strength: 0.85,
                });
            }
            // Three black crows: three falling bearish candles.
            if !a.bull
                && !b.bull
                && !g.bull
                && b.close < a.close
                && g.close < b.close
                && b.open < a.open
                && g.open < b.open
            {
                out.push(Pattern {
                    kind: PatternKind::ThreeBlackCrows,
                    index: i,
                    bullish: false,
                    strength: 0.85,
                });
            }
        }
    }
    out
}

/// Patterns whose final candle is exactly the last candle of the slice — the
/// "what just printed?" query an agent asks on the freshest bar.
pub fn latest_patterns(candles: &[Candle]) -> Vec<Pattern> {
    if candles.is_empty()
    {
        return Vec::new();
    }
    let last = candles.len() - 1;
    detect_patterns(candles)
        .into_iter()
        .filter(|p| p.index == last)
        .collect()
}

fn mean_close_before(candles: &[Candle], i: usize, lookback: usize) -> Option<f32> {
    if i == 0
    {
        return None;
    }
    let start = i.saturating_sub(lookback);
    let slice = &candles[start..i];
    if slice.is_empty()
    {
        return None;
    }
    Some(slice.iter().map(|c| c.close).sum::<f32>() / slice.len() as f32)
}

fn in_downtrend(candles: &[Candle], i: usize, lookback: usize) -> bool {
    match mean_close_before(candles, i, lookback)
    {
        Some(m) => candles[i].close < m,
        None => false,
    }
}

fn in_uptrend(candles: &[Candle], i: usize, lookback: usize) -> bool {
    match mean_close_before(candles, i, lookback)
    {
        Some(m) => candles[i].close > m,
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(o: f32, h: f32, l: f32, cl: f32) -> Candle {
        Candle {
            ts_ms: 0,
            open: o,
            high: h,
            low: l,
            close: cl,
            volume: 1.0,
        }
    }

    #[test]
    fn doji_detected() {
        let candles = vec![c(100.0, 105.0, 95.0, 100.2)];
        let pats = detect_patterns(&candles);
        assert!(pats.iter().any(|p| p.kind == PatternKind::Doji));
    }

    #[test]
    fn marubozu_detected() {
        let candles = vec![c(100.0, 110.05, 99.98, 110.0)];
        let pats = detect_patterns(&candles);
        assert!(pats.iter().any(|p| p.kind == PatternKind::BullishMarubozu));
    }

    #[test]
    fn hammer_after_downtrend() {
        // Preceding downtrend, then a hammer (small body top, long lower shadow).
        let mut candles: Vec<Candle> = (0..5).map(|i| c(110.0 - i as f32 * 2.0, 111.0 - i as f32 * 2.0, 108.0 - i as f32 * 2.0, 109.0 - i as f32 * 2.0)).collect();
        // Hammer: open 100, close 100.5, low 95 (long lower), high 100.8 (small upper).
        candles.push(c(100.0, 100.8, 95.0, 100.5));
        let pats = detect_patterns(&candles);
        assert!(pats.iter().any(|p| p.kind == PatternKind::Hammer && p.bullish));
    }

    #[test]
    fn bullish_engulfing() {
        let candles = vec![
            c(100.0, 100.5, 98.0, 98.5), // bearish
            c(98.0, 102.0, 97.5, 101.5), // bullish, engulfs
        ];
        let pats = detect_patterns(&candles);
        assert!(pats.iter().any(|p| p.kind == PatternKind::BullishEngulfing && p.bullish));
    }

    #[test]
    fn bearish_engulfing() {
        let candles = vec![
            c(100.0, 102.0, 99.5, 101.5), // bullish
            c(102.0, 102.5, 99.0, 99.5),  // bearish engulfs
        ];
        let pats = detect_patterns(&candles);
        assert!(pats.iter().any(|p| p.kind == PatternKind::BearishEngulfing && !p.bullish));
    }

    #[test]
    fn morning_star() {
        let candles = vec![
            c(110.0, 110.5, 104.0, 104.5), // big bearish
            c(104.0, 104.6, 103.4, 104.1), // small star
            c(104.5, 109.0, 104.2, 108.5), // big bullish above midpoint of first (107.25)
        ];
        let pats = detect_patterns(&candles);
        assert!(pats.iter().any(|p| p.kind == PatternKind::MorningStar && p.bullish));
    }

    #[test]
    fn three_white_soldiers() {
        let candles = vec![
            c(100.0, 103.0, 99.5, 102.5),
            c(102.0, 105.0, 101.5, 104.5),
            c(104.0, 107.0, 103.5, 106.5),
        ];
        let pats = detect_patterns(&candles);
        assert!(pats.iter().any(|p| p.kind == PatternKind::ThreeWhiteSoldiers));
    }

    #[test]
    fn latest_only_returns_last_index() {
        let candles = vec![
            c(100.0, 100.5, 98.0, 98.5),
            c(98.0, 102.0, 97.5, 101.5),
        ];
        let latest = latest_patterns(&candles);
        assert!(latest.iter().all(|p| p.index == 1));
    }

    #[test]
    fn empty_input_is_safe() {
        assert!(detect_patterns(&[]).is_empty());
        assert!(latest_patterns(&[]).is_empty());
    }
}
