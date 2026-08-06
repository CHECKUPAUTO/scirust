//! # Retrodiction — a deterministic Kalman smoother over a belief trajectory
//!
//! Time-travel in CCOS today *replays* the past exactly as it happened. **Retrodiction** is the
//! structural inverse: given everything learned *after* step `t`, what was the minimum-variance
//! estimate of a claim's belief *at* `t`? A claim that was noisy and uncertain early but became
//! firmly established later is reconstructed with lower variance at those early steps — the future
//! folded back into the past. A stateless similarity retriever has no time axis and no belief state
//! to smooth, so it cannot even phrase the question; a deterministic, replayable belief trajectory
//! can answer it exactly.
//!
//! This is a **scalar local-level** (random-walk) Rauch–Tung–Striebel fixed-interval smoother: a
//! forward Kalman filter followed by a backward pass that injects every future measurement into each
//! past estimate. The scalar form needs no matrix inversion, so the whole thing is a handful of
//! fixed-order `f64` operations with no RNG and no wall-clock — bit-reproducible, so `replay == live`
//! holds. Distilled from `scirust-estimation`'s `RtsSmoother` (used only as the correctness oracle;
//! nothing is linked — the "distill, don't link" discipline of `src/lsa.rs`).
//!
//! Model: latent `x_t = x_{t-1} + w_t` (`w ~ N(0, q)`), measured `z_t = x_t + v_t` (`v ~ N(0, r)`).
//! `q` is how fast belief is allowed to drift, `r` how noisy each raw `QBelief` reading is; both are
//! caller parameters (never hard-coded), matching the `qbelief_decayed(half_life)` contract.

/// Rauch–Tung–Striebel fixed-interval smoother for a scalar local-level state. Folds **future**
/// measurements back into every **past** step, returning the minimum-variance retrodicted trajectory
/// the noisy `measurements` sample (same length as the input).
///
/// `q` is the process (random-walk) variance — larger `q` trusts the measurements more and tracks
/// faster; `r` is the measurement variance — larger `r` smooths harder. `q` is clamped to `≥ 0` and
/// `r` to a tiny positive floor so the Kalman gain is always well defined. Deterministic scalar
/// arithmetic (no matrix inversion, no RNG), so the result is a pure, bit-reproducible function of
/// the inputs. An empty input yields an empty output; a single sample is returned unchanged.
pub fn rts_smooth(measurements: &[f64], q: f64, r: f64) -> Vec<f64> {
    let n = measurements.len();
    if n == 0 {
        return Vec::new();
    }
    let q = q.max(0.0);
    let r = r.max(1e-12); // a positive measurement variance keeps the gain finite

    // Forward Kalman filter (scalar local-level): store the filtered mean/variance at each step.
    let mut x_filt = vec![0.0f64; n];
    let mut p_filt = vec![0.0f64; n];
    x_filt[0] = measurements[0];
    p_filt[0] = r; // the first reading carries one measurement's worth of uncertainty
    for t in 1..n {
        let x_pred = x_filt[t - 1]; // random walk ⇒ predicted mean is the previous estimate
        let p_pred = p_filt[t - 1] + q;
        let gain = p_pred / (p_pred + r);
        x_filt[t] = x_pred + gain * (measurements[t] - x_pred);
        p_filt[t] = (1.0 - gain) * p_pred;
    }

    // Backward RTS pass: inject each future smoothed estimate into the past.
    let mut x_smooth = x_filt.clone();
    for t in (0..n.saturating_sub(1)).rev() {
        let p_pred = p_filt[t] + q; // P_pred[t+1]
        let x_pred = x_filt[t]; // x_pred[t+1] (random walk)
        let c = if p_pred > 0.0 {
            p_filt[t] / p_pred
        } else {
            0.0
        };
        x_smooth[t] = x_filt[t] + c * (x_smooth[t + 1] - x_pred);
    }
    x_smooth
}

#[cfg(test)]
mod tests {
    use super::*;

    fn variance(xs: &[f64]) -> f64 {
        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n
    }

    #[test]
    fn empty_and_singleton_are_identity() {
        assert!(rts_smooth(&[], 0.01, 0.1).is_empty());
        assert_eq!(rts_smooth(&[0.5], 0.01, 0.1), vec![0.5]);
    }

    #[test]
    fn folds_future_evidence_into_the_past() {
        // A hard step 0,0,0 → 1,1,1. Retrodiction rounds the corner: the pre-jump steps are pulled
        // UP by the future 1s (a thing forward-only filtering, and any stateless retriever, cannot do).
        let s = rts_smooth(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 0.01, 0.1);
        assert_eq!(s.len(), 6);
        assert!(
            s[2] > 0.1,
            "the last pre-jump step is lifted by future evidence: {s:?}"
        );
        assert!(
            s[0] > 0.0,
            "even the first step feels the higher future: {s:?}"
        );
        assert!(s[0] < s[5], "the smoothed trajectory follows the step up");
        assert!(
            s.windows(2).all(|w| w[1] >= w[0] - 1e-9),
            "the smoothed step is monotone non-decreasing: {s:?}"
        );
    }

    #[test]
    fn reduces_variance_of_a_noisy_constant() {
        // Alternating noise around a constant 0.5. The smoother must dampen the jitter.
        let raw = [0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3];
        let s = rts_smooth(&raw, 0.001, 0.1);
        assert!(
            variance(&s) < variance(&raw),
            "smoothed variance {} should be below raw {}",
            variance(&s),
            variance(&raw)
        );
    }

    #[test]
    fn is_bit_deterministic() {
        let raw = [0.0, 0.2, -0.1, 0.4, 0.35, 0.5, 0.48];
        let a = rts_smooth(&raw, 0.02, 0.15);
        let b = rts_smooth(&raw, 0.02, 0.15);
        let a_bits: Vec<u64> = a.iter().map(|x| x.to_bits()).collect();
        let b_bits: Vec<u64> = b.iter().map(|x| x.to_bits()).collect();
        assert_eq!(
            a_bits, b_bits,
            "retrodiction is bit-reproducible (replay == live)"
        );
    }
}
