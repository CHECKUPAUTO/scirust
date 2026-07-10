//! Wavelet–RLS–RTS denoising pipeline.
//!
//! Implements the composite estimation equation:
//!
//! ```text
//! x̂_{|N} = M_RTS · [(I - Δ_RLS(x, λ)) · W^T · 𝒯_τ(W · s)]
//! ```
//!
//! This combines:
//! 1. **Wavelet decomposition** (`W`) — projects the noisy signal into the
//!    wavelet domain using the multi-level Haar DWT.
//! 2. **Soft thresholding** (`𝒯_τ`) — shrinks wavelet coefficients to remove
//!    noise while preserving sharp features.
//! 3. **Wavelet reconstruction** (`W^T`) — inverts the wavelet transform.
//! 4. **RLS correction** (`I - Δ_RLS`) — an RLS filter learns the optimal
//!    adaptation between the wavelet reconstruction and the raw observation,
//!    correcting systematic shrinkage bias.
//! 5. **RTS smoothing** (`M_RTS`) — a Rauch–Tung–Striebel fixed-interval
//!    smoother that uses a forward–backward pass to give the minimum-variance
//!    estimate given a linear-Gaussian model.
//!
//! Every block is deterministic `f64` — a run is bit-reproducible.

use scirust_estimation::linalg::Mat;
use scirust_estimation::{RlsFilter, RtsSmoother};

use crate::threshold::soft_threshold;
use crate::wavelet::{haar_dwt_multilevel, haar_idwt_multilevel};

/// Parameters controlling the full wavelet–RLS–RTS pipeline.
#[derive(Debug, Clone)]
pub struct WaveletRlsRtsParams {
    /// Number of wavelet decomposition levels.
    pub wavelet_levels: usize,
    /// Threshold for soft-thresholding wavelet coefficients.
    /// Use `None` for universal threshold (Donoho–Johnstone).
    pub tau: Option<f64>,
    /// RLS forgetting factor in `(0, 1]`.
    pub rls_lambda: f64,
    /// RLS initial covariance scale `P(0) = δ·I`.
    pub rls_delta: f64,
}

impl Default for WaveletRlsRtsParams {
    fn default() -> Self {
        Self {
            wavelet_levels: 3,
            tau: None,
            rls_lambda: 0.98,
            rls_delta: 100.0,
        }
    }
}

/// Run the full wavelet–RLS–RTS denoising pipeline on a 1-D signal.
///
/// The RLS block learns the mapping from the wavelet-denoised reconstruction
/// to the original noisy signal, producing a correction matrix `Δ = I - W_rls`
/// that compensates for shrinkage bias introduced by the soft threshold.  This
/// is then smoothed by RTS for the final estimate.
///
/// # Arguments
/// * `signal` — the noisy input signal.  Length must be a power of two and
///   divisible by `2^levels`.
/// * `params` — pipeline parameterisation.
/// * `f` — state transition matrix `(n×n)` for the RTS smoother model.
/// * `q` — process noise covariance `(n×n)`.
/// * `h` — observation matrix `(m×n)`.
/// * `r` — measurement noise covariance `(m×m)`.
/// * `p0` — initial state covariance.
///
/// # Returns
/// A tuple `(denoised_signal, delta_norm)` where the first element is
/// the smooth estimate and the second is the Frobenius norm of the RLS
/// correction `Δ`, giving a measure of how much the adaptive block adjusted
/// the wavelet reconstruction.
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
    assert!(
        n.is_power_of_two(),
        "signal length must be a power of two, got {n}"
    );

    // ----------------------------------------------------------------
    // 1. Wavelet decomposition: W·s
    // ----------------------------------------------------------------
    let mut coeffs = signal.to_vec();
    haar_dwt_multilevel(&mut coeffs, params.wavelet_levels);

    // ----------------------------------------------------------------
    // 2. Soft thresholding: 𝒯_τ(W·s)
    // ----------------------------------------------------------------
    let tau = params.tau.unwrap_or_else(|| {
        let n_fine = n / (1 << (params.wavelet_levels - 1));
        let detail_start = n - n_fine;
        let mut abs_vals: Vec<f64> = coeffs[detail_start..]
            .iter()
            .map(|x| x.abs())
            .collect();
        if abs_vals.is_empty()
        {
            return 0.0;
        }
        abs_vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let median_abs = if abs_vals.len() % 2 == 0
        {
            (abs_vals[abs_vals.len() / 2 - 1] + abs_vals[abs_vals.len() / 2]) / 2.0
        }
        else
        {
            abs_vals[abs_vals.len() / 2]
        };
        let sigma = median_abs / 0.6745;
        sigma * (2.0 * (n as f64).ln()).sqrt()
    });
    soft_threshold(&mut coeffs, tau);

    // ----------------------------------------------------------------
    // 3. Wavelet reconstruction: W^T · 𝒯_τ(W·s)
    // ----------------------------------------------------------------
    haar_idwt_multilevel(&mut coeffs, params.wavelet_levels);
    let reconstructed = coeffs;

    // ----------------------------------------------------------------
    // 4. RLS correction: (I - Δ_RLS(x, λ)) · reconstructed
    //
    //    The RLS filter learns the mapping from reconstructed → noisy
    //    signal *causally*, one sample at a time.  At each time step
    //    the **current** weight w_k gives the instantaneous correction:
    //
    //      Δ_k = 1 - w_k
    //      corrected_k = w_k · recon_k
    //
    //    The full correction matrix Δ = diag(1 - w_0, ..., 1 - w_{n-1})
    //    is assembled so that (I - Δ) · reconstructed = diag(w_k) · recon,
    //    preserving the trajectory-aware, causal nature of the operator.
    //
    //    Δ_norm is the Frobenius norm of this diagonal matrix, giving a
    //    scalar summary of how much the RLS adapted across the signal.
    // ----------------------------------------------------------------
    let mut rls = RlsFilter::new(1, 1, params.rls_lambda, params.rls_delta);
    let mut corrected = vec![0.0; n];
    let mut delta_mat = Mat::zeros(n, n);
    let mut w_traj = Vec::with_capacity(n);
    for (i, (r, s)) in reconstructed.iter().zip(signal.iter()).enumerate()
    {
        rls.update(&[*r], &[*s]);
        let w_k = rls.weights()[0];
        w_traj.push(w_k);
        corrected[i] = r * w_k;
        delta_mat.set(i, i, 1.0 - w_k);
    }
    let delta_norm: f64 = delta_mat.data.iter().map(|&x| x * x).sum::<f64>().sqrt();

    // ----------------------------------------------------------------
    // 5. RTS smoothing: M_RTS · (corrected)
    // ----------------------------------------------------------------
    let measurements: Vec<Vec<f64>> = corrected.iter().map(|&v| vec![v]).collect();
    let smoothed = RtsSmoother::smooth(x0, p0, f, q, h, r, &measurements);
    let denoised: Vec<f64> = smoothed.iter().map(|state| state[0]).collect();

    (denoised, delta_norm)
}

/// Convenience function for the common 1-D scalar case where the state is
/// the signal value itself (position-only model):
///
/// `F = [1]`, `Q = [q]`, `H = [1]`, `R = [r]`.
pub fn wavelet_rls_rts_smooth_1d(
    signal: &[f64],
    wavelet_levels: usize,
    tau: Option<f64>,
    rls_lambda: f64,
    process_noise: f64,
    measurement_noise: f64,
) -> (Vec<f64>, f64) {
    let params = WaveletRlsRtsParams {
        wavelet_levels,
        tau,
        rls_lambda,
        rls_delta: 100.0,
    };
    let x0 = vec![signal[0]];
    let p0 = Mat::new(1, 1, vec![1.0]);
    let f = Mat::new(1, 1, vec![1.0]);
    let q = Mat::new(1, 1, vec![process_noise]);
    let h = Mat::new(1, 1, vec![1.0]);
    let r = Mat::new(1, 1, vec![measurement_noise]);

    wavelet_rls_rts_smooth(signal, &params, &x0, &f, &q, &h, &r, &p0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_denoises_piecewise_constant() {
        let n = 128;
        let mut rng = Rng::new(42);
        let mut clean = vec![1.0; n];
        for i in 32..64
        {
            clean[i] = 3.0;
        }
        for i in 96..n
        {
            clean[i] = -1.0;
        }
        let noisy: Vec<f64> = clean.iter().map(|x| x + rng.normal(0.3)).collect();

        let (denoised, delta_norm) = wavelet_rls_rts_smooth_1d(
            &noisy,
            2,
            Some(0.4),
            0.99,
            0.01,
            0.09,
        );

        let noisy_mse: f64 = noisy.iter()
            .zip(&clean)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>() / n as f64;
        let denoised_mse: f64 = denoised.iter()
            .zip(&clean)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>() / n as f64;
        assert!(
            denoised_mse < noisy_mse,
            "denoised MSE {denoised_mse} should be < noisy MSE {noisy_mse}; Δ = {delta_norm}"
        );
        assert!(delta_norm >= 0.0, "delta must be non-negative");
    }

    #[test]
    fn wavelet_only_improves_mse() {
        let n = 128;
        let mut rng = Rng::new(7);
        let clean: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let noisy: Vec<f64> = clean.iter().map(|x| x + rng.normal(0.25)).collect();

        let noisy_mse: f64 = noisy.iter()
            .zip(&clean)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>() / n as f64;

        // Wavelet-only denoising
        let mut coeffs = noisy.clone();
        haar_dwt_multilevel(&mut coeffs, 3);
        let tau = 0.35;
        soft_threshold(&mut coeffs, tau);
        haar_idwt_multilevel(&mut coeffs, 3);
        let wavelet_mse: f64 = coeffs.iter()
            .zip(&clean)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>() / n as f64;

        assert!(
            wavelet_mse < noisy_mse,
            "wavelet MSE {wavelet_mse} should be < noisy MSE {noisy_mse}"
        );
    }

    struct Rng {
        s: u64,
    }
    impl Rng {
        fn new(seed: u64) -> Self {
            Self { s: seed }
        }
        fn u01(&mut self) -> f64 {
            self.s = self.s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
        }
        fn normal(&mut self, sd: f64) -> f64 {
            let (u1, u2) = (self.u01(), self.u01());
            sd * (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
        }
    }
}
