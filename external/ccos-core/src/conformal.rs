//! # Conformal calibration — a distribution-free false-alarm bound
//!
//! The injection signal today flags on a hard-coded probability cut (`>= 0.5`), a knob with no
//! guarantee about how often it wrongly flags benign text. **Split-conformal prediction** replaces
//! that guess with a threshold calibrated on a corpus of known-benign scores: pick a target
//! miscoverage `alpha`, and any future *exchangeable* benign input is flagged with probability at
//! most `alpha` — a **distribution-free, finite-sample** guarantee (no assumption on the score
//! distribution). It is the same guarantee SciRust's OT guards give Modbus/DNP3 traffic, on the
//! code-ingest wire.
//!
//! Pure and deterministic — a sort plus an integer index (`total_cmp`, no RNG) — so a calibrated
//! guard is bit-reproducible and `replay == live` holds. Distilled from `scirust-core`'s
//! `conformal_quantile` (oracle only; nothing linked). A stateless cosine ranker cannot even express
//! such a bound on its own false-alarm rate.

/// The finite-sample split-conformal quantile of `scores` at miscoverage `alpha`: the
/// `⌈(n+1)(1−alpha)⌉`-th smallest score. A future exchangeable score **above** it is flagged, and by
/// the exchangeability argument a genuinely in-distribution point is flagged with probability at most
/// `alpha`.
///
/// Returns `+∞` when the sample is too small for the level (`⌈(n+1)(1−alpha)⌉ > n`) — **fail-open**:
/// flag nothing rather than fabricate a bound. Deterministic: a total-order sort (`total_cmp`) and an
/// integer index, no RNG.
pub fn conformal_quantile(scores: &[f64], alpha: f64) -> f64 {
    let n = scores.len();
    if n == 0 {
        return f64::INFINITY;
    }
    let alpha = alpha.clamp(0.0, 1.0);
    // 1-based rank of the order statistic; the finite-sample ceil correction is what makes the
    // guarantee hold for a *future* (n+1)-th point, not just the calibration set.
    let rank = (((n + 1) as f64) * (1.0 - alpha)).ceil() as usize;
    if rank == 0 {
        return f64::NEG_INFINITY; // alpha ≥ 1 ⇒ flag everything
    }
    if rank > n {
        return f64::INFINITY; // too few samples for this level ⇒ fail-open
    }
    let mut sorted: Vec<f64> = scores.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    sorted[rank - 1]
}

/// A calibrated accept/flag decision with a stated false-alarm bound. Built by
/// [`ConformalGuard::calibrate`] over a corpus of **known-benign** nonconformity scores (e.g. the
/// injection probability of trusted source files); [`ConformalGuard::flags`] then decides any future
/// score with the distribution-free guarantee that a benign input is flagged with probability at most
/// [`alarm_bound`](Self::alarm_bound).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConformalGuard {
    threshold: f64,
    alpha: f64,
}

impl ConformalGuard {
    /// Calibrate a guard at target miscoverage `alpha` over `calibration` benign scores. The
    /// threshold is the [`conformal_quantile`]; with too few samples it is `+∞` (fail-open).
    pub fn calibrate(calibration: &[f64], alpha: f64) -> Self {
        Self {
            threshold: conformal_quantile(calibration, alpha),
            alpha: alpha.clamp(0.0, 1.0),
        }
    }

    /// The calibrated flag threshold (a score `>=` this is flagged).
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// The guaranteed upper bound on the probability a genuinely-benign input is flagged.
    pub fn alarm_bound(&self) -> f64 {
        self.alpha
    }

    /// Whether `score` crosses the calibrated threshold. Fail-open when uncalibrated (`+∞` ⇒ never
    /// flags).
    pub fn flags(&self, score: f64) -> bool {
        score >= self.threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantile_matches_the_order_statistic() {
        // n=5, alpha=0.2 ⇒ rank = ⌈6·0.8⌉ = ⌈4.8⌉ = 5 ⇒ the 5th smallest = 0.5.
        let s = [0.5, 0.1, 0.4, 0.2, 0.3];
        assert!((conformal_quantile(&s, 0.2) - 0.5).abs() < 1e-12);
        // alpha=0.5 ⇒ rank = ⌈6·0.5⌉ = 3 ⇒ the 3rd smallest = 0.3.
        assert!((conformal_quantile(&s, 0.5) - 0.3).abs() < 1e-12);
    }

    #[test]
    fn small_sample_fails_open() {
        // n=3, alpha=0.05 ⇒ rank = ⌈4·0.95⌉ = 4 > 3 ⇒ +∞ (flag nothing, never fabricate a bound).
        assert_eq!(conformal_quantile(&[0.1, 0.2, 0.3], 0.05), f64::INFINITY);
        assert_eq!(conformal_quantile(&[], 0.1), f64::INFINITY);
        assert!(!ConformalGuard::calibrate(&[0.1, 0.2, 0.3], 0.05).flags(0.99));
    }

    #[test]
    fn tighter_alpha_raises_the_threshold() {
        let cal: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let lax = conformal_quantile(&cal, 0.20);
        let strict = conformal_quantile(&cal, 0.02);
        assert!(
            strict >= lax,
            "a smaller false-alarm budget demands a higher bar"
        );
    }

    #[test]
    fn guard_separates_benign_from_anomalous_and_is_deterministic() {
        // A benign band around ~0.2; calibrate at alpha=0.1.
        let cal: Vec<f64> = (0..50).map(|i| 0.10 + (i as f64) * 0.004).collect(); // 0.10..0.30
        let g = ConformalGuard::calibrate(&cal, 0.1);
        assert!(
            g.flags(0.95),
            "an obvious injection well above the band is flagged"
        );
        assert!(
            !g.flags(0.05),
            "a clearly-benign score below the band is accepted"
        );
        assert!((g.alarm_bound() - 0.1).abs() < 1e-12);
        assert_eq!(
            g,
            ConformalGuard::calibrate(&cal, 0.1),
            "calibration is deterministic"
        );
    }
}
