//! Spatio-temporal MIMO adaptive filter + innovation-whiteness λ tuning.
//!
//! ## [`MimoFirRls`] — the real "multi-channel adaptive filter"
//!
//! [`crate::rls::RlsFilter`] learns an *instantaneous* map `u → d`. Physical
//! multi-channel problems (echo/crosstalk cancellation, vibration-path
//! identification, MIMO equalization) are **convolutive**: each output is a sum
//! of FIR-filtered inputs, `d_o[k] = Σ_c Σ_t h[o][c][t]·u_c[k−t]`. This wrapper
//! makes that structure first-class: it maintains one delay line per input
//! channel, stacks them into the regressor, drives an [`crate::rls::RlsFilter`]
//! core over it, and exposes the identified FIR kernel per (output, input)
//! channel pair. The temporal dimension the instantaneous filter lacks is
//! exactly what this adds.
//!
//! ## [`tune_lambda`] — forgetting factor chosen by falsification
//!
//! The forgetting factor λ is the one knob users guess. This helper chooses it
//! the same way `scirust-signal`'s auto-Kalman chooses its process variance:
//! a correctly tuned filter produces **white innovations**, so each candidate λ
//! is scored by the fraction of innovation autocorrelation lags inside the
//! `±1.96/√N` white-noise band, and the **largest λ within tolerance of the
//! best score** wins — the most averaging (most stable) setting the whiteness
//! diagnostic does not reject, mirroring the parsimony rule of
//! `denoise::adaptive::kalman_smooth_auto`.

use crate::rls::RlsFilter;
use serde::{Deserialize, Serialize};

/// Multi-channel FIR-structured RLS: `n_in` input channels × `taps` delays,
/// `n_out` output channels, identified online.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MimoFirRls {
    n_in: usize,
    n_out: usize,
    taps: usize,
    /// Stacked regressor: input channel `c`, delay `t` at index `c·taps + t`.
    regressor: Vec<f64>,
    core: RlsFilter,
}

impl MimoFirRls {
    /// `n_in` input channels, `n_out` output channels, `taps` FIR taps per
    /// channel pair; `lambda`/`delta` as in [`RlsFilter::new`].
    pub fn new(n_in: usize, n_out: usize, taps: usize, lambda: f64, delta: f64) -> Self {
        assert!(taps >= 1, "need at least one tap");
        Self {
            n_in,
            n_out,
            taps,
            regressor: vec![0.0; n_in * taps],
            core: RlsFilter::new(n_in * taps, n_out, lambda, delta),
        }
    }

    /// Push one time step: current sample of every input channel (`inputs`,
    /// length `n_in`) and every target channel (`targets`, length `n_out`).
    /// Shifts the delay lines, then adapts. Returns the a-priori errors.
    #[allow(clippy::needless_range_loop)]
    pub fn update(&mut self, inputs: &[f64], targets: &[f64]) -> &[f64] {
        assert_eq!(inputs.len(), self.n_in);
        assert_eq!(targets.len(), self.n_out);
        // Shift each channel's delay line one step and insert the new sample.
        for c in 0..self.n_in
        {
            let base = c * self.taps;
            for t in (1..self.taps).rev()
            {
                self.regressor[base + t] = self.regressor[base + t - 1];
            }
            self.regressor[base] = inputs[c];
        }
        self.core.update(&self.regressor, targets)
    }

    /// Identified FIR kernel from input channel `in_ch` to output channel
    /// `out_ch` (length `taps`, tap 0 = instantaneous).
    pub fn kernel(&self, out_ch: usize, in_ch: usize) -> &[f64] {
        assert!(out_ch < self.n_out && in_ch < self.n_in);
        let row = out_ch * self.n_in * self.taps + in_ch * self.taps;
        &self.core.weights()[row..row + self.taps]
    }

    /// Number of FIR taps per channel pair.
    pub fn taps(&self) -> usize {
        self.taps
    }

    /// Access the underlying flat-regressor RLS core.
    pub fn core(&self) -> &RlsFilter {
        &self.core
    }
}

/// Result of [`tune_lambda`].
#[derive(Debug, Clone)]
pub struct LambdaFit {
    /// Selected forgetting factor.
    pub lambda: f64,
    /// Innovation whiteness of the selected λ, in [0, 1].
    pub whiteness: f64,
}

/// Choose the forgetting factor on a calibration record by **innovation
/// whiteness** (see the module docs). `records` is a sequence of
/// `(input, target)` pairs for a scalar-output filter of dimension `n_in`;
/// `candidates` is scanned and the largest λ whose whiteness comes within
/// `0.1` of the best is returned. Burn-in (`2·n_in` samples, min 10) is
/// excluded from the whiteness statistic so convergence transients don't
/// count against a candidate.
pub fn tune_lambda(
    records: &[(Vec<f64>, f64)],
    n_in: usize,
    delta: f64,
    candidates: &[f64],
) -> LambdaFit {
    assert!(!candidates.is_empty(), "need at least one candidate λ");
    let burn_in = (2 * n_in).max(10).min(records.len() / 2);
    let mut scored: Vec<(f64, f64)> = Vec::with_capacity(candidates.len());
    for &lambda in candidates
    {
        let mut filter = crate::rls::VectorRls::new(n_in, lambda, delta);
        let mut innovations = Vec::with_capacity(records.len().saturating_sub(burn_in));
        for (k, (u, d)) in records.iter().enumerate()
        {
            let e = filter.update(u, *d);
            if k >= burn_in
            {
                innovations.push(e);
            }
        }
        scored.push((lambda, whiteness_score(&innovations)));
    }
    let best = scored.iter().map(|&(_, w)| w).fold(0.0_f64, f64::max);
    let tolerance = 0.1;
    // Largest λ (most averaging) the whiteness diagnostic does not reject.
    let (lambda, whiteness) = scored
        .iter()
        .copied()
        .filter(|&(_, w)| w >= best - tolerance)
        .fold(
            (candidates[0], -1.0),
            |acc, x| if x.0 > acc.0 { x } else { acc },
        );
    LambdaFit { lambda, whiteness }
}

/// Fraction of autocorrelation lags inside the `±1.96/√N` white-noise band —
/// the same falsification statistic as `scirust-signal`'s residual and
/// innovation whiteness tests (duplicated here because `scirust-signal`
/// depends on this crate, not the other way around).
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

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn identifies_a_known_2x2_fir_coupling() {
        // Ground-truth convolutive MIMO system: 2 inputs → 2 outputs, 3 taps.
        let h: [[[f64; 3]; 2]; 2] = [
            [[0.9, -0.3, 0.1], [0.2, 0.1, 0.0]],
            [[-0.5, 0.4, 0.0], [1.0, 0.0, -0.2]],
        ];
        let (n_in, n_out, taps) = (2, 2, 3);
        let mut filter = MimoFirRls::new(n_in, n_out, taps, 0.999, 100.0);
        let mut rng = Lcg(31);
        let mut hist = [[0.0; 3]; 2]; // per-input delay line for the oracle
        for _ in 0..5000
        {
            let u = [rng.next(), rng.next()];
            for c in 0..n_in
            {
                hist[c][2] = hist[c][1];
                hist[c][1] = hist[c][0];
                hist[c][0] = u[c];
            }
            let mut d = [0.0; 2];
            for o in 0..n_out
            {
                for c in 0..n_in
                {
                    for t in 0..taps
                    {
                        d[o] += h[o][c][t] * hist[c][t];
                    }
                }
            }
            filter.update(&u, &d);
        }
        for o in 0..n_out
        {
            for c in 0..n_in
            {
                let k = filter.kernel(o, c);
                for t in 0..taps
                {
                    assert!(
                        (k[t] - h[o][c][t]).abs() < 1.0e-3,
                        "kernel[{o}][{c}][{t}] = {} vs {}",
                        k[t],
                        h[o][c][t]
                    );
                }
            }
        }
    }

    #[test]
    fn tune_lambda_prefers_one_for_static_system() {
        // Static system: nothing to forget, the parsimony rule must keep the
        // largest candidate.
        let mut rng = Lcg(41);
        let records: Vec<(Vec<f64>, f64)> = (0..800)
            .map(|_| {
                let u = vec![rng.next(), rng.next()];
                let d = 2.0 * u[0] - u[1] + 0.05 * rng.next();
                (u, d)
            })
            .collect();
        let fit = tune_lambda(&records, 2, 100.0, &[0.90, 0.95, 0.99, 1.0]);
        assert_eq!(fit.lambda, 1.0, "static system should keep λ = 1");
        assert!(fit.whiteness > 0.8, "whiteness {}", fit.whiteness);
    }

    #[test]
    fn tune_lambda_rejects_one_for_drifting_system() {
        // Fast-drifting system: λ = 1 averages the drift into colored
        // innovations; the diagnostic must push λ below 1.
        let mut rng = Lcg(43);
        let records: Vec<(Vec<f64>, f64)> = (0..1500)
            .map(|k| {
                let u = vec![rng.next(), rng.next()];
                let w0 = 1.0 + 3.0 * (k as f64 / 150.0).sin();
                let d = w0 * u[0] - u[1] + 0.02 * rng.next();
                (u, d)
            })
            .collect();
        let fit = tune_lambda(&records, 2, 100.0, &[0.90, 0.95, 0.99, 1.0]);
        assert!(fit.lambda < 1.0, "drifting system kept λ = 1");
    }
}
