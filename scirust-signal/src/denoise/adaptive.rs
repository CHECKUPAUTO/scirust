//! Adaptive / model-based denoisers — the family for non-stationary noise.
//!
//! Unlike the fixed filters in the other families, these methods carry an internal
//! model that *tracks* the signal as it evolves:
//!
//! * [`kalman_smooth`] — a local-level (random-walk) **Kalman filter** followed by
//!   a **Rauch-Tung-Striebel smoother**: the optimal linear estimator when a slowly
//!   varying level is observed through additive white noise, and the natural tool
//!   once the noise statistics drift over time.
//! * [`kalman_smooth_auto`] — the same smoother with its process/measurement
//!   variances tuned automatically by maximizing the **whiteness of the
//!   innovations**: a correctly specified Kalman filter produces white innovations,
//!   so the whiteness of the one-step prediction errors is a reference-free score
//!   of model fit — the same falsification idea as
//!   [`super::detect::separate`]'s residual test.
//! * [`lms_line_enhancer`] / [`rls_line_enhancer`] — **adaptive line enhancers**:
//!   a self-tuning predictor whose reference input is the signal *delayed by Δ*.
//!   Periodic/narrowband content stays correlated across the delay and is
//!   predicted (the enhanced output); broadband noise decorrelates and lands in
//!   the prediction error. No noise reference or prior tuning to a frequency is
//!   needed — the filter finds the line itself, and follows it if it drifts.

use super::estimate_noise_std_helper;

/// Local-level Kalman filter + Rauch-Tung-Striebel smoother.
///
/// State model: `x_k = x_{k-1} + w_k` (`w ~ N(0, process_var)`), observation
/// `y_k = x_k + v_k` (`v ~ N(0, meas_var)`). The forward pass filters causally;
/// the backward RTS pass re-estimates every state from the *whole* record, so the
/// result is smooth and phase-free. `process_var` sets agility: small values give
/// heavy smoothing, large values track fast changes.
pub fn kalman_smooth(signal: &[f64], process_var: f64, meas_var: f64) -> Vec<f64> {
    let n = signal.len();
    if n == 0 || process_var <= 0.0 || meas_var <= 0.0
    {
        return signal.to_vec();
    }
    let (xs, _) = kalman_forward_rts(signal, process_var, meas_var);
    xs
}

/// The auto-tuned Kalman smoother's result: the smoothed signal plus the variances
/// that were selected and the innovation-whiteness score that selected them.
#[derive(Debug, Clone)]
pub struct KalmanFit {
    /// The RTS-smoothed signal.
    pub output: Vec<f64>,
    /// Selected process (state random-walk) variance.
    pub process_var: f64,
    /// Selected measurement-noise variance.
    pub meas_var: f64,
    /// Whiteness of the innovations in [0, 1]: fraction of autocorrelation lags of
    /// the one-step prediction errors inside the white-noise confidence band. Close
    /// to 1 means the local-level model explains the data well.
    pub innovation_whiteness: f64,
}

/// Kalman smoother with automatic variance selection.
///
/// `meas_var` is fixed to the robust MAD noise estimate. `process_var` is chosen
/// from a logarithmic grid by a **parsimony rule on innovation whiteness**: a
/// correctly specified filter produces white one-step prediction errors, but on a
/// misspecified (non-random-walk) signal the whiteness keeps creeping up as `q`
/// grows and the filter degenerates into tracking everything — so instead of the
/// argmax, we take the *smallest* `q` whose whiteness comes within a tolerance of
/// the best. That is the smoothest model the whiteness diagnostic does not
/// reject, which is where the actual SNR gain lives.
pub fn kalman_smooth_auto(signal: &[f64]) -> KalmanFit {
    let n = signal.len();
    let sigma = estimate_noise_std_helper(signal);
    if n < 8 || sigma <= 0.0
    {
        return KalmanFit {
            output: signal.to_vec(),
            process_var: 0.0,
            meas_var: sigma * sigma,
            innovation_whiteness: 1.0,
        };
    }
    let r = sigma * sigma;
    // log grid over q/r from 1e-4 (very smooth) to 1e2 (fast tracking).
    let mut grid: Vec<(f64, f64)> = Vec::new();
    let mut ratio = 1.0e-4;
    while ratio <= 1.0e2 + 1.0e-9
    {
        let (_, innov) = kalman_forward_rts(signal, r * ratio, r);
        grid.push((ratio, whiteness_score(&innov)));
        ratio *= 10.0_f64.sqrt();
    }
    let best_white = grid.iter().map(|&(_, w)| w).fold(0.0_f64, f64::max);
    let tolerance = 0.1;
    let (best_ratio, picked_white) = grid
        .iter()
        .copied()
        .find(|&(_, w)| w >= best_white - tolerance)
        .unwrap_or((1.0e-4, best_white));
    let q = r * best_ratio;
    let (output, _) = kalman_forward_rts(signal, q, r);
    KalmanFit {
        output,
        process_var: q,
        meas_var: r,
        innovation_whiteness: picked_white,
    }
}

/// Forward Kalman pass + RTS backward pass for the local-level model.
/// Returns `(smoothed states, innovations)`.
fn kalman_forward_rts(signal: &[f64], q: f64, r: f64) -> (Vec<f64>, Vec<f64>) {
    let n = signal.len();
    let mut xf = vec![0.0; n]; // filtered state
    let mut pf = vec![0.0; n]; // filtered variance
    let mut pp = vec![0.0; n]; // predicted variance
    let mut innov = vec![0.0; n];

    // Semi-diffuse start: trust the first sample, with generous uncertainty.
    let mut x_pred = signal[0];
    let mut p_pred = r + q;
    for i in 0..n
    {
        if i > 0
        {
            x_pred = xf[i - 1];
            p_pred = pf[i - 1] + q;
        }
        pp[i] = p_pred;
        innov[i] = signal[i] - x_pred;
        let k = p_pred / (p_pred + r);
        xf[i] = x_pred + k * innov[i];
        pf[i] = (1.0 - k) * p_pred;
    }

    // RTS: for the local level, the predicted state at i+1 equals xf[i].
    let mut xs = xf.clone();
    for i in (0..n.saturating_sub(1)).rev()
    {
        let c = pf[i] / pp[i + 1];
        xs[i] = xf[i] + c * (xs[i + 1] - xf[i]);
    }
    (xs, innov)
}

/// Fraction of autocorrelation lags of `x` inside the `±1.96/√N` white-noise band.
fn whiteness_score(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 8
    {
        return 1.0;
    }
    let max_lag = (n / 4).clamp(1, 40);
    let mean = x.iter().sum::<f64>() / n as f64;
    let c0: f64 = x.iter().map(|&v| (v - mean) * (v - mean)).sum();
    if c0 <= 0.0
    {
        return 1.0;
    }
    let band = 1.96 / (n as f64).sqrt();
    let mut within = 0usize;
    for lag in 1..=max_lag
    {
        let mut s = 0.0;
        for i in 0..(n - lag)
        {
            s += (x[i] - mean) * (x[i + lag] - mean);
        }
        if (s / c0).abs() < band
        {
            within += 1;
        }
    }
    within as f64 / max_lag as f64
}

/// Local **linear-trend** Kalman filter + RTS smoother (2-D state: level + slope).
///
/// State model: `level_k = level_{k-1} + slope_{k-1} + w_l`,
/// `slope_k = slope_{k-1} + w_b`, observation `y_k = level_k + v_k`. Where the
/// local-*level* smoother ([`kalman_smooth`]) must trade lag against noise on a
/// trending signal, the trend model tracks ramps *unbiasedly*: a clean ramp is
/// reproduced almost exactly, and drifting sensor baselines are followed without
/// the systematic under-shoot of the level-only model. `process_var_slope` sets
/// how fast the slope itself may wander (small ⇒ near-linear trend).
pub fn kalman_trend_smooth(
    signal: &[f64],
    process_var_level: f64,
    process_var_slope: f64,
    meas_var: f64,
) -> Vec<f64> {
    let n = signal.len();
    if n < 3 || process_var_level < 0.0 || process_var_slope <= 0.0 || meas_var <= 0.0
    {
        return signal.to_vec();
    }
    let (ql, qb, r) = (process_var_level, process_var_slope, meas_var);

    // Forward pass. State z = (level, slope); F = [[1,1],[0,1]]; H = [1,0].
    // Filtered state/covariance and predicted covariance per step (symmetric 2×2
    // stored as (p11, p12, p22)).
    let mut lf = vec![0.0; n];
    let mut bf = vec![0.0; n];
    let mut pf = vec![(0.0, 0.0, 0.0); n];
    let mut pp = vec![(0.0, 0.0, 0.0); n];

    // Semi-diffuse start: level from the first sample, slope unknown but bounded.
    let mut l_pred = signal[0];
    let mut b_pred = 0.0;
    let mut p_pred = (r + ql, 0.0, r + qb);
    for i in 0..n
    {
        if i > 0
        {
            let (p11, p12, p22) = pf[i - 1];
            l_pred = lf[i - 1] + bf[i - 1];
            b_pred = bf[i - 1];
            p_pred = (p11 + 2.0 * p12 + p22 + ql, p12 + p22, p22 + qb);
        }
        pp[i] = p_pred;
        let (pp11, pp12, pp22) = p_pred;
        let s = pp11 + r;
        let k1 = pp11 / s;
        let k2 = pp12 / s;
        let e = signal[i] - l_pred;
        lf[i] = l_pred + k1 * e;
        bf[i] = b_pred + k2 * e;
        pf[i] = ((1.0 - k1) * pp11, (1.0 - k1) * pp12, pp22 - k2 * pp12);
    }

    // RTS backward pass: G = P_f·Fᵀ·(P_pred[k+1])⁻¹, z_s = z_f + G(z_{s,k+1} − z_{p,k+1}).
    let mut ls = lf.clone();
    let mut bs = bf.clone();
    for i in (0..n - 1).rev()
    {
        let (p11, p12, p22) = pf[i];
        let (pp11, pp12, pp22) = pp[i + 1];
        let det = pp11 * pp22 - pp12 * pp12;
        if det.abs() < 1.0e-300
        {
            continue;
        }
        // P_f·Fᵀ with Fᵀ = [[1,0],[1,1]] → [[p11+p12, p12],[p12+p22, p22]].
        let a11 = p11 + p12;
        let a12 = p12;
        let a21 = p12 + p22;
        let a22 = p22;
        // (P_pred)⁻¹ = 1/det · [[pp22, −pp12],[−pp12, pp11]].
        let g11 = (a11 * pp22 - a12 * pp12) / det;
        let g12 = (-a11 * pp12 + a12 * pp11) / det;
        let g21 = (a21 * pp22 - a22 * pp12) / det;
        let g22 = (-a21 * pp12 + a22 * pp11) / det;
        let dl = ls[i + 1] - (lf[i] + bf[i]);
        let db = bs[i + 1] - bf[i];
        ls[i] = lf[i] + g11 * dl + g12 * db;
        bs[i] = bf[i] + g21 * dl + g22 * db;
    }
    ls
}

/// Adaptive line enhancer with a normalized-LMS predictor.
///
/// The filter predicts `y[i]` from the *delayed* samples
/// `y[i-delay], …, y[i-delay-taps+1]`. Narrowband (periodic) content survives the
/// delay correlated and is captured by the prediction — the returned enhanced
/// signal; broadband noise decorrelates over `delay` samples and is rejected.
/// `mu` in (0, 2) is the normalized step size (0.05–0.5 typical) — NLMS is
/// mean-square stable only below 2, so out-of-range values pass the input through
/// unchanged, like the other denoisers' parameter guards. The filter needs a
/// convergence run-in of a few hundred samples.
pub fn lms_line_enhancer(signal: &[f64], taps: usize, delay: usize, mu: f64) -> Vec<f64> {
    let n = signal.len();
    if n == 0 || taps == 0 || delay == 0 || mu <= 0.0 || mu >= 2.0
    {
        return signal.to_vec();
    }
    let mut w = vec![0.0; taps];
    let mut out = vec![0.0; n];
    for i in 0..n
    {
        let mut y = 0.0;
        let mut norm = 1.0e-12;
        for (j, &wj) in w.iter().enumerate()
        {
            let idx = i as isize - delay as isize - j as isize;
            let v = if idx >= 0 { signal[idx as usize] } else { 0.0 };
            y += wj * v;
            norm += v * v;
        }
        let e = signal[i] - y;
        let g = mu * e / norm;
        for (j, wj) in w.iter_mut().enumerate()
        {
            let idx = i as isize - delay as isize - j as isize;
            let v = if idx >= 0 { signal[idx as usize] } else { 0.0 };
            *wj += g * v;
        }
        out[i] = y;
    }
    out
}

/// Adaptive line enhancer with a recursive-least-squares predictor.
///
/// Same delayed-prediction structure as [`lms_line_enhancer`], but the weights are
/// the exact least-squares solution over an exponentially-forgotten window —
/// convergence in roughly `2·taps` samples instead of hundreds, at `O(taps²)` per
/// sample. `forgetting` in (0, 1] (0.98–0.999 typical); smaller values track
/// faster drift.
pub fn rls_line_enhancer(signal: &[f64], taps: usize, delay: usize, forgetting: f64) -> Vec<f64> {
    let n = signal.len();
    if n == 0 || taps == 0 || delay == 0 || forgetting <= 0.0 || forgetting > 1.0
    {
        return signal.to_vec();
    }
    let mut w = vec![0.0; taps];
    // Inverse correlation matrix, initialized large (weak prior on the weights).
    let mut p = vec![vec![0.0; taps]; taps];
    for (j, row) in p.iter_mut().enumerate()
    {
        row[j] = 100.0;
    }
    let mut out = vec![0.0; n];
    let mut x = vec![0.0; taps];
    let mut px = vec![0.0; taps];
    for i in 0..n
    {
        for (j, xj) in x.iter_mut().enumerate()
        {
            let idx = i as isize - delay as isize - j as isize;
            *xj = if idx >= 0 { signal[idx as usize] } else { 0.0 };
        }
        // A-priori prediction (the enhanced output).
        let y: f64 = w.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
        out[i] = y;
        let e = signal[i] - y;
        // Gain k = P·x / (λ + xᵀ·P·x).
        for (j, pxj) in px.iter_mut().enumerate()
        {
            *pxj = p[j].iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
        }
        let denom = forgetting + x.iter().zip(px.iter()).map(|(&a, &b)| a * b).sum::<f64>();
        // Weight update and P ← (P − k·xᵀP)/λ.
        for (j, wj) in w.iter_mut().enumerate()
        {
            *wj += px[j] / denom * e;
        }
        for j in 0..taps
        {
            let kj = px[j] / denom;
            for l in 0..taps
            {
                p[j][l] = (p[j][l] - kj * px[l]) / forgetting;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{Lcg, snr_db};
    use super::*;
    use core::f64::consts::PI;

    fn noisy_sine(n: usize, freq_cycles: f64, noise: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let mut rng = Lcg::new(seed);
        let clean: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * freq_cycles * i as f64 / n as f64).sin())
            .collect();
        let obs: Vec<f64> = clean.iter().map(|&c| c + noise * rng.gauss()).collect();
        (clean, obs)
    }

    #[test]
    fn kalman_smooth_reduces_noise() {
        let (clean, obs) = noisy_sine(512, 3.0, 0.4, 41);
        let out = kalman_smooth(&obs, 0.4 * 0.4 * 1.0e-2, 0.4 * 0.4);
        assert_eq!(out.len(), obs.len());
        assert!(snr_db(&clean, &out) > snr_db(&clean, &obs) + 3.0);
    }

    #[test]
    fn kalman_auto_improves_and_reports_white_innovations() {
        let (clean, obs) = noisy_sine(1024, 4.0, 0.35, 43);
        let fit = kalman_smooth_auto(&obs);
        assert!(snr_db(&clean, &fit.output) > snr_db(&clean, &obs) + 2.0);
        assert!(
            fit.innovation_whiteness > 0.7,
            "whiteness {}",
            fit.innovation_whiteness
        );
        assert!(fit.process_var > 0.0 && fit.meas_var > 0.0);
    }

    #[test]
    fn kalman_tracks_a_level_shift() {
        // A step: the smoother must follow it (bias decays), not smear it away.
        let mut rng = Lcg::new(47);
        let clean: Vec<f64> = (0..512).map(|i| if i < 256 { 0.0 } else { 4.0 }).collect();
        let obs: Vec<f64> = clean.iter().map(|&c| c + 0.3 * rng.gauss()).collect();
        let fit = kalman_smooth_auto(&obs);
        // Well after the step the estimate must sit near the new level.
        let tail = &fit.output[300..500];
        let tail_mean = tail.iter().sum::<f64>() / tail.len() as f64;
        assert!((tail_mean - 4.0).abs() < 0.3, "tail mean {tail_mean}");
    }

    #[test]
    fn rts_backward_pass_removes_phase_lag() {
        // The causal forward filter alone lags the signal (cross-correlation with
        // the clean reference peaks at a positive lag ≈ 1/k samples); the RTS
        // backward pass re-centers the estimate so the peak sits at lag 0. This is
        // the discriminating test for the backward pass: degrading the smoother to
        // the forward filter moves the peak to lag ≈ 8 and fails the assert.
        let (clean, obs) = noisy_sine(1024, 4.0, 0.35, 71);
        let r = 0.35 * 0.35;
        let out = kalman_smooth(&obs, r * 1.0e-2, r);
        let mut best_lag = 0usize;
        let mut best_corr = f64::NEG_INFINITY;
        for lag in 0..=16usize
        {
            let mut c = 0.0;
            for i in lag..clean.len()
            {
                c += clean[i - lag] * out[i];
            }
            if c > best_corr
            {
                best_corr = c;
                best_lag = lag;
            }
        }
        assert_eq!(
            best_lag, 0,
            "smoothed output lags the signal by {best_lag} samples"
        );
    }

    #[test]
    fn lms_rejects_divergent_step_size() {
        // NLMS is mean-square stable only for mu < 2: out-of-range values must
        // pass the input through instead of returning divergent garbage.
        let (_, obs) = noisy_sine(256, 10.0, 0.3, 73);
        assert_eq!(lms_line_enhancer(&obs, 16, 1, 2.0), obs.to_vec());
        assert_eq!(lms_line_enhancer(&obs, 16, 1, 2.5), obs.to_vec());
        // In-range values stay finite.
        let out = lms_line_enhancer(&obs, 16, 1, 1.9);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn kalman_degenerate_inputs_pass_through() {
        assert!(kalman_smooth(&[], 1.0, 1.0).is_empty());
        let x = [1.0, 2.0];
        assert_eq!(kalman_smooth(&x, 0.0, 1.0), x.to_vec());
        assert_eq!(kalman_smooth_auto(&x).output, x.to_vec());
    }

    #[test]
    fn trend_smoother_reproduces_clean_ramp_where_level_model_fails() {
        // On a noiseless ramp the linear-trend model is exact — the smoother must
        // reproduce it almost perfectly, while the local-level model (with the
        // same variances) systematically distorts it. This is the discriminating
        // test for the 2-D state.
        let ramp: Vec<f64> = (0..256).map(|i| 0.05 * i as f64).collect();
        let r = 1.0e-2;
        let trend = kalman_trend_smooth(&ramp, 1.0e-8, 1.0e-8, r);
        let level = kalman_smooth(&ramp, 1.0e-8, r);
        let max_err = |est: &[f64]| {
            est.iter()
                .zip(ramp.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max)
        };
        let e_trend = max_err(&trend);
        let e_level = max_err(&level);
        assert!(e_trend < 1.0e-3, "trend max error {e_trend}");
        assert!(
            e_trend < e_level / 100.0,
            "trend {e_trend} should be ≪ level {e_level}"
        );
    }

    #[test]
    fn trend_smoother_denoises_a_trending_signal() {
        let n = 512;
        let mut rng = Lcg::new(97);
        let clean: Vec<f64> = (0..n)
            .map(|i| 0.02 * i as f64 + (2.0 * PI * 2.0 * i as f64 / n as f64).sin())
            .collect();
        let obs: Vec<f64> = clean.iter().map(|&c| c + 0.4 * rng.gauss()).collect();
        let r = 0.4 * 0.4;
        let out = kalman_trend_smooth(&obs, 1.0e-4 * r, 1.0e-4 * r, r);
        assert!(snr_db(&clean, &out) > snr_db(&clean, &obs) + 3.0);
    }

    #[test]
    fn trend_smoother_degenerate_inputs_pass_through() {
        assert!(kalman_trend_smooth(&[], 1.0, 1.0, 1.0).is_empty());
        let x = [1.0, 2.0];
        assert_eq!(kalman_trend_smooth(&x, 1.0, 1.0, 1.0), x.to_vec());
        let y = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(kalman_trend_smooth(&y, 1.0, 0.0, 1.0), y.to_vec());
    }

    #[test]
    fn lms_ale_extracts_tone_from_white_noise() {
        let n = 4096;
        let (clean, obs) = noisy_sine(n, 200.0, 0.5, 53);
        let out = lms_line_enhancer(&obs, 16, 1, 0.2);
        // Judge after the convergence run-in.
        let half = n / 2;
        let s_out = snr_db(&clean[half..], &out[half..]);
        let s_obs = snr_db(&clean[half..], &obs[half..]);
        assert!(s_out > s_obs + 3.0, "{s_out} vs {s_obs}");
    }

    #[test]
    fn rls_ale_converges_on_short_record() {
        let n = 512;
        let (clean, obs) = noisy_sine(n, 25.0, 0.4, 59);
        let out = rls_line_enhancer(&obs, 12, 1, 0.995);
        let q = n / 4;
        let s_out = snr_db(&clean[q..], &out[q..]);
        let s_obs = snr_db(&clean[q..], &obs[q..]);
        assert!(s_out > s_obs + 2.0, "{s_out} vs {s_obs}");
    }

    #[test]
    fn ale_degenerate_inputs_pass_through() {
        let x = [1.0, 2.0, 3.0];
        assert_eq!(lms_line_enhancer(&x, 0, 1, 0.2), x.to_vec());
        assert_eq!(rls_line_enhancer(&x, 4, 0, 0.99), x.to_vec());
    }
}
