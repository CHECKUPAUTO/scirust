//! Composite **wavelet–RLS–RTS** denoising pipeline.
//!
//! Ported from the standalone implementation contributed in PR #278 (developed
//! on a Jetson, in parallel with the `denoise` framework) and re-based onto the
//! framework's periodized multi-wavelet DWT — so the pipeline gains Db4/Db6/Db8
//! and the robust finest-band noise estimate for free.
//!
//! The pipeline implements the composite estimation equation
//!
//! ```text
//! x̂_{|N} = M_RTS · [(I − Δ_RLS(x, λ)) · Wᵀ · 𝒯_τ(W · s)]
//! ```
//!
//! in three stages:
//!
//! 1. **Wavelet shrinkage** `Wᵀ·𝒯_τ(W·s)` — the transform-domain denoiser.
//!    Soft thresholding removes noise but *biases amplitudes toward zero*
//!    (every surviving coefficient is shrunk by `τ`).
//! 2. **RLS bias correction** `(I − Δ_RLS)` — a causal recursive-least-squares
//!    filter ([`scirust_estimation::RlsFilter`]) learns, sample by sample, the
//!    scalar gain mapping the shrunken reconstruction back to the observation.
//!    Because the shrinkage bias is multiplicative and the residual noise is
//!    zero-mean, the learned gain converges to the bias inverse — restoring
//!    amplitude without re-fitting the noise.
//! 3. **RTS smoothing** `M_RTS` — a fixed-interval Rauch-Tung-Striebel pass
//!    ([`scirust_estimation::RtsSmoother`]) over a caller-chosen linear-Gaussian
//!    model re-smooths what the gain correction re-amplified.
//!
//! Every block is deterministic `f64` — a run is bit-reproducible.

use scirust_estimation::linalg::Mat;
use scirust_estimation::{RlsFilter, RtsSmoother};

use super::mad;
use super::transform::{ThresholdMode, Wavelet, apply_threshold, dwt_forward, dwt_inverse};

/// Parameters controlling the full wavelet–RLS–RTS pipeline.
#[derive(Debug, Clone)]
pub struct WaveletRlsRtsParams {
    /// Wavelet basis for the shrinkage stage.
    pub wavelet: Wavelet,
    /// Number of decomposition levels (`0` = automatic).
    pub wavelet_levels: usize,
    /// Soft threshold `τ`; `None` selects the universal threshold
    /// `σ√(2 ln N)` with `σ` estimated robustly from the finest detail band.
    pub tau: Option<f64>,
    /// RLS forgetting factor in `(0, 1]`.
    pub rls_lambda: f64,
    /// RLS initial covariance scale `P(0) = δ·I`.
    pub rls_delta: f64,
}

impl Default for WaveletRlsRtsParams {
    fn default() -> Self {
        Self {
            wavelet: Wavelet::Haar,
            wavelet_levels: 3,
            tau: None,
            rls_lambda: 0.98,
            rls_delta: 100.0,
        }
    }
}

/// Run the full wavelet–RLS–RTS pipeline on a 1-D signal.
///
/// `x0`, `f`, `q`, `h`, `r`, `p0` parameterize the RTS smoother's linear-Gaussian
/// state model (state transition, process noise, observation, measurement noise,
/// initial covariance). For the common scalar case use
/// [`wavelet_rls_rts_smooth_1d`].
///
/// Returns `(denoised, delta_norm)` where `delta_norm = ‖diag(1 − w_k)‖_F`
/// summarizes how much the RLS stage corrected the wavelet reconstruction
/// (`0` ⇒ the shrinkage was already unbiased).
#[allow(clippy::too_many_arguments)]
pub fn wavelet_rls_rts_smooth(
    signal: &[f64],
    params: &WaveletRlsRtsParams,
    x0: &[f64],
    f: &Mat,
    q: &Mat,
    h: &Mat,
    r: &Mat,
    p0: &Mat,
) -> (Vec<f64>, f64) {
    let n = signal.len();
    let taps = params.wavelet.lowpass();
    if n < taps.len().max(2)
    {
        return (signal.to_vec(), 0.0);
    }

    // 1. Wavelet shrinkage: Wᵀ·𝒯_τ(W·s) on the framework's periodized DWT
    //    (arbitrary length via reflection padding, any Wavelet basis).
    //
    //    Faithful to the original pipeline, 𝒯_τ is applied to EVERY coefficient
    //    — approximation band included. That is more aggressive than the
    //    details-only convention of `wavelet_denoise_with` and introduces a
    //    systematic amplitude bias… which is exactly the bias stage 2's RLS gain
    //    is there to learn and undo. The two stages are designed as a pair.
    let Some((mut approx, mut details, n_pad)) = dwt_forward(signal, params.wavelet_levels, &taps)
    else
    {
        return (signal.to_vec(), 0.0);
    };
    let tau = params.tau.unwrap_or_else(|| {
        // σ from the *finest* detail band — details[0] by construction, which
        // fixes the fixed-offset window of the original implementation.
        let sigma = mad(&details[0]) / 0.6745;
        sigma * (2.0 * (n_pad as f64).ln()).sqrt()
    });
    for band in details.iter_mut()
    {
        for d in band.iter_mut()
        {
            *d = apply_threshold(*d, tau, ThresholdMode::Soft);
        }
    }
    for a in approx.iter_mut()
    {
        *a = apply_threshold(*a, tau, ThresholdMode::Soft);
    }
    let reconstructed = dwt_inverse(approx, &details, &taps, n);

    // 2. RLS bias correction. The scalar filter learns, causally, the gain w_k
    //    mapping reconstruction → observation; the running correction matrix is
    //    Δ = diag(1 − w_k), applied as corrected_k = w_k · recon_k. Its
    //    Frobenius norm is accumulated directly — no n×n matrix is built.
    let mut rls = RlsFilter::new(1, 1, params.rls_lambda, params.rls_delta);
    let mut corrected = vec![0.0; n];
    let mut delta_sq = 0.0;
    for (i, (rec, obs)) in reconstructed.iter().zip(signal.iter()).enumerate()
    {
        rls.update(&[*rec], &[*obs]);
        let w_k = rls.weights()[0];
        corrected[i] = rec * w_k;
        delta_sq += (1.0 - w_k) * (1.0 - w_k);
    }
    let delta_norm = delta_sq.sqrt();

    // 3. RTS smoothing over the caller's state model.
    let measurements: Vec<Vec<f64>> = corrected.iter().map(|&v| vec![v]).collect();
    let smoothed = RtsSmoother::smooth(x0, p0, f, q, h, r, &measurements);
    let denoised: Vec<f64> = smoothed.iter().map(|state| state[0]).collect();

    (denoised, delta_norm)
}

/// [`wavelet_rls_rts_smooth`] for the scalar local-level case:
/// `F = [1]`, `Q = [process_noise]`, `H = [1]`, `R = [measurement_noise]`.
pub fn wavelet_rls_rts_smooth_1d(
    signal: &[f64],
    params: &WaveletRlsRtsParams,
    process_noise: f64,
    measurement_noise: f64,
) -> (Vec<f64>, f64) {
    if signal.is_empty()
    {
        return (Vec::new(), 0.0);
    }
    let x0 = vec![signal[0]];
    let p0 = Mat::new(1, 1, vec![1.0]);
    let f = Mat::new(1, 1, vec![1.0]);
    let q = Mat::new(1, 1, vec![process_noise]);
    let h = Mat::new(1, 1, vec![1.0]);
    let r = Mat::new(1, 1, vec![measurement_noise]);
    wavelet_rls_rts_smooth(signal, params, &x0, &f, &q, &h, &r, &p0)
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{Lcg, snr_db};
    use super::super::wavelet_denoise_with;
    use super::*;

    fn piecewise(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                if i < n / 4
                {
                    1.0
                }
                else if i < n / 2
                {
                    3.0
                }
                else if i < 3 * n / 4
                {
                    -1.0
                }
                else
                {
                    0.5
                }
            })
            .collect()
    }

    #[test]
    fn pipeline_beats_raw_signal() {
        let n = 256;
        let mut rng = Lcg::new(113);
        let clean = piecewise(n);
        let noisy: Vec<f64> = clean.iter().map(|&c| c + 0.3 * rng.gauss()).collect();
        let params = WaveletRlsRtsParams::default();
        let (out, delta_norm) = wavelet_rls_rts_smooth_1d(&noisy, &params, 0.05, 0.09);
        assert_eq!(out.len(), n);
        assert!(snr_db(&clean, &out) > snr_db(&clean, &noisy));
        assert!(delta_norm >= 0.0);
    }

    #[test]
    fn rls_stage_corrects_shrinkage_bias() {
        // Discriminating test for the RLS block: with an aggressive soft
        // threshold the wavelet stage shrinks amplitudes hard; the pipeline
        // (gain correction + light RTS) must recover part of that bias and
        // beat wavelet-only denoising with the same threshold.
        let n = 512;
        let mut rng = Lcg::new(127);
        let clean = piecewise(n);
        let noisy: Vec<f64> = clean.iter().map(|&c| c + 0.25 * rng.gauss()).collect();

        let params = WaveletRlsRtsParams {
            wavelet: Wavelet::Haar,
            wavelet_levels: 3,
            tau: Some(0.8),
            rls_lambda: 0.995,
            rls_delta: 100.0,
        };
        // Light RTS (q ≫ r ⇒ near pass-through) so the comparison isolates the
        // RLS bias-correction stage.
        let (pipeline_out, delta_norm) = wavelet_rls_rts_smooth_1d(&noisy, &params, 1.0, 0.01);

        // The shrinkage stage alone (identical all-band soft threshold), i.e.
        // the pipeline with stages 2 and 3 removed — the mutant this test kills.
        let taps = Wavelet::Haar.lowpass();
        let (mut approx, mut details, _) = dwt_forward(&noisy, 3, &taps).unwrap();
        for band in details.iter_mut()
        {
            for d in band.iter_mut()
            {
                *d = apply_threshold(*d, 0.8, ThresholdMode::Soft);
            }
        }
        for a in approx.iter_mut()
        {
            *a = apply_threshold(*a, 0.8, ThresholdMode::Soft);
        }
        let wavelet_only = dwt_inverse(approx, &details, &taps, n);

        let s_pipeline = snr_db(&clean, &pipeline_out);
        let s_wavelet = snr_db(&clean, &wavelet_only);
        assert!(
            s_pipeline > s_wavelet,
            "pipeline {s_pipeline} dB should beat wavelet-only {s_wavelet} dB (Δ = {delta_norm})"
        );
        assert!(delta_norm > 0.0, "RLS must have corrected something");
    }

    #[test]
    fn pipeline_universal_threshold_and_db_bases_work() {
        let n = 300; // non power of two: exercises the padding path
        let mut rng = Lcg::new(131);
        let clean: Vec<f64> = (0..n)
            .map(|i| (2.0 * core::f64::consts::PI * 3.0 * i as f64 / n as f64).sin())
            .collect();
        let noisy: Vec<f64> = clean.iter().map(|&c| c + 0.3 * rng.gauss()).collect();
        for wavelet in [Wavelet::Haar, Wavelet::Db4, Wavelet::Db8]
        {
            let params = WaveletRlsRtsParams {
                wavelet,
                wavelet_levels: 0,
                tau: None,
                rls_lambda: 0.98,
                rls_delta: 100.0,
            };
            let (out, _) = wavelet_rls_rts_smooth_1d(&noisy, &params, 0.01, 0.09);
            assert_eq!(out.len(), n);
            assert!(
                snr_db(&clean, &out) > snr_db(&clean, &noisy),
                "{wavelet:?} pipeline must beat raw"
            );
        }
    }

    #[test]
    fn pipeline_is_consistent_with_framework_wavelet_denoiser() {
        // Sanity: on a smooth signal the pipeline should be in the same quality
        // class as the framework's plain wavelet denoiser (it adds bias
        // correction + smoothing, not a regression).
        let n = 512;
        let mut rng = Lcg::new(137);
        let clean: Vec<f64> = (0..n)
            .map(|i| (2.0 * core::f64::consts::PI * 4.0 * i as f64 / n as f64).sin())
            .collect();
        let noisy: Vec<f64> = clean.iter().map(|&c| c + 0.35 * rng.gauss()).collect();
        let params = WaveletRlsRtsParams {
            wavelet: Wavelet::Db4,
            wavelet_levels: 0,
            tau: None,
            rls_lambda: 0.98,
            rls_delta: 100.0,
        };
        let (out, _) = wavelet_rls_rts_smooth_1d(&noisy, &params, 0.02, 0.35 * 0.35);
        let plain = wavelet_denoise_with(&noisy, 0, ThresholdMode::Soft, Wavelet::Db4);
        let s_pipe = snr_db(&clean, &out);
        let s_plain = snr_db(&clean, &plain);
        assert!(
            s_pipe > s_plain - 3.0,
            "pipeline {s_pipe} dB unreasonably below plain wavelet {s_plain} dB"
        );
    }

    #[test]
    fn pipeline_degenerate_inputs() {
        let params = WaveletRlsRtsParams::default();
        let (out, d) = wavelet_rls_rts_smooth_1d(&[], &params, 0.1, 0.1);
        assert!(out.is_empty());
        assert_eq!(d, 0.0);
        let (out, _) = wavelet_rls_rts_smooth_1d(&[1.0], &params, 0.1, 0.1);
        assert_eq!(out, vec![1.0]);
    }
}
