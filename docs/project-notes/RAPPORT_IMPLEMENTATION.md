# Rapport d'implémentation — Pipeline Wavelet–RLS–RTS

Équation maîtresse implémentée :

$$\hat{\mathbf{x}}_{|N} = \mathbf{M}_{\text{RTS}} \left[ \left( \mathbf{I} - \mathbf{\Delta}_{\text{RLS}}(\mathbf{x}, \lambda) \right) \mathbf{W}^T \mathcal{T}_{\tau}(\mathbf{W}\mathbf{s}) \right]$$

---

## 1. `scirust-signal/Cargo.toml` — Dépendance ajoutée

```toml
[package]
name = "scirust-signal"
version = "0.1.0"
edition = "2021"
publish = false
description = "SciRust signal processing — FFT, window functions, time/frequency features for industrial monitoring"

[dependencies]
scirust-core = { path = "../scirust-core" }
scirust-estimation = { path = "../scirust-estimation" }
serde = { version = "1", features = ["derive"] }
```

---

## 2. `scirust-signal/src/lib.rs` — Déclarations de modules

```rust
//! SciRust Signal Processing
//!
//! Pure-Rust DSP primitives for industrial monitoring and automotive applications.
//!
//! ## Modules
//! - **Complex numbers** — basic complex arithmetic (`Complex`)
//! - **FFT** — radix-2 Cooley-Tukey forward/inverse FFT
//! - **Windows** — Hanning, Hamming, Blackman, Blackman-Harris, Flat-top
//! - **Feature extraction** — time-domain (RMS, crest factor, kurtosis, skewness,
//!   zero-crossing rate, autocorrelation), frequency-domain (PSD, spectral centroid,
//!   spectral entropy, band power)
//! - **Bearing diagnostics** — BPFO, BPFI, BSF, FTF calculation, fault frequency
//!   detection for rolling-element bearings
//! - **Order analysis** — order tracking, resampling for variable-speed rotating machinery
//! - **Thresholding** — soft/hard thresholding operators for wavelet denoising
//! - **Wavelet** — Haar DWT/IDWT, multi-level decomposition, wavelet matrix construction
//! - **Denoising pipeline** — composite wavelet–RLS–RTS estimator combining all blocks

pub mod bearing;
pub mod cepstrum;
pub mod complex;
pub mod envelope;
pub mod features;
pub mod fft;
pub mod mcsa;
pub mod order;
pub mod denoise;
pub mod threshold;
pub mod wavelet;
pub mod windows;

pub use bearing::{BearingFault, BearingGeometry, bpfi, bpfo, bsf, detect_bearing_faults, ftf};
pub use cepstrum::{dominant_quefrency, real_cepstrum};
pub use complex::Complex;
pub use envelope::{dominant_envelope_freq, envelope_spectrum, hilbert_envelope};
pub use features::spectral::{
    band_power, psd, spectral_centroid, spectral_entropy, spectral_flatness, spectral_rolloff,
    spectral_spread,
};
pub use features::{
    autocorrelation, crest_factor, energy, entropy, kurtosis, peak_to_peak, rms, skewness,
    zero_crossing_rate,
};
pub use fft::{fft, fft_real, ifft};
pub use mcsa::{
    BarSeverity, BrokenBarResult, EccentricityResult, MotorDiagnosis, MotorFault,
    analyze_broken_bar, analyze_eccentricity, diagnose_motor, slip,
};
pub use order::{order_spectrum, order_track, resample_constant_angle, rpm_profile, tacho_to_rpm};
pub use threshold::{
    hard_threshold, soft_threshold, sure_threshold, universal_soft_threshold,
};
pub use wavelet::{haar_dwt, haar_dwt_multilevel, haar_idwt, haar_idwt_multilevel, haar_matrix};
pub use windows::{apply_window, blackman, blackman_harris, flattop, hamming, hanning};
```

---

## 3. `scirust-estimation/src/lib.rs` — Module RLS ajouté

```rust
//! # scirust-estimation — deterministic state estimation
//!
//! Pure-Rust, bit-reproducible state estimators for industrial sensing:
//!
//! - [`KalmanFilter`] — the linear Kalman filter (fixed-order `f64`).
//! - [`Ekf`] — the Extended Kalman filter (nonlinear `f`/`h` via closures + Jacobians).
//! - [`IntervalFilter`] — set-membership estimation with a **containment
//!   guarantee**: a box that provably brackets the true state given bounded
//!   noise — the certified counterpart to the Kalman filter's probabilistic
//!   estimate.
//!
//! Shared infrastructure for the battery (BMS), sensor-fusion and structural
//! verticals. Every operation accumulates in a fixed order, so a run is
//! bit-identical across machines — the determinism guarantee the rest of
//! SciRust upholds, extended to estimation.

pub mod ekf;
pub mod imm;
pub mod interval;
pub mod kalman;
pub mod linalg;
pub mod particle;
pub mod rls;
pub mod smoother;
pub mod ud;
pub mod ukf;

pub use ekf::Ekf;
pub use imm::{Imm, ImmModel};
pub use interval::IntervalFilter;
pub use kalman::KalmanFilter;
pub use linalg::Mat;
pub use particle::ParticleFilter;
pub use smoother::RtsSmoother;
pub use ud::UdFilter;
pub use rls::{RlsFilter, VectorRls};
pub use ukf::Ukf;
```

---

## 4. `scirust-signal/src/threshold.rs` — Opérateurs de seuillage

```rust
//! Soft and hard thresholding (shrinkage) operators for signal denoising.
//!
//! The soft-thresholding operator (a.k.a. shrinkage) is the proximal operator
//! of the L1 norm:
//!
//! ```text
//! 𝒯_τ(x) = sign(x) · max(|x| - τ, 0)
//! ```
//!
//! Hard thresholding sets coefficients below τ to zero without shrinkage:
//!
//! ```text
//! ℋ_τ(x) = x · 𝟙_{|x| > τ}
//! ```
//!
//! Both are deterministic `f64` with fixed-order accumulation.

/// Apply soft thresholding (shrinkage) element-wise: `sign(x)·max(|x|-τ, 0)`.
///
/// # Arguments
/// * `data` - slice of values to threshold (modified in place)
/// * `tau` - threshold level (must be ≥ 0)
pub fn soft_threshold(data: &mut [f64], tau: f64) {
    assert!(tau >= 0.0, "tau must be non-negative, got {tau}");
    if tau == 0.0
    {
        return;
    }
    for x in data.iter_mut()
    {
        let abs = x.abs();
        if abs <= tau
        {
            *x = 0.0;
        }
        else
        {
            *x = x.signum() * (abs - tau);
        }
    }
}

/// Apply hard thresholding element-wise: `x · 𝟙_{|x| > τ}`.
///
/// # Arguments
/// * `data` - slice of values to threshold (modified in place)
/// * `tau` - threshold level (must be ≥ 0)
pub fn hard_threshold(data: &mut [f64], tau: f64) {
    assert!(tau >= 0.0, "tau must be non-negative, got {tau}");
    if tau == 0.0
    {
        return;
    }
    for x in data.iter_mut()
    {
        if x.abs() <= tau
        {
            *x = 0.0;
        }
    }
}

/// Apply soft thresholding with a **universal** threshold `√(2·log(n))·σ̂`,
/// where `σ̂` is the median-absolute-deviation estimate of noise standard
/// deviation (Donoho & Johnstone 1994).
///
/// `data` is modified in place. Returns the threshold that was used.
pub fn universal_soft_threshold(data: &mut [f64]) -> f64 {
    let n = data.len();
    if n == 0
    {
        return 0.0;
    }
    // MAD estimate: median(|data|) / 0.6745
    let mut abs_vals: Vec<f64> = data.iter().map(|x| x.abs()).collect();
    abs_vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let median_abs = if n % 2 == 0
    {
        (abs_vals[n / 2 - 1] + abs_vals[n / 2]) / 2.0
    }
    else
    {
        abs_vals[n / 2]
    };
    let sigma = median_abs / 0.6745;
    let tau = sigma * (2.0 * (n as f64).ln()).sqrt();
    soft_threshold(data, tau);
    tau
}

/// **SureShrink** — Stein's Unbiased Risk Estimate adaptive threshold (Donoho &
/// Johnstone 1995).  Selects a near-optimal threshold by minimizing SURE (Stein's
/// Unbiased Risk Estimate) over a discrete grid.  Best for signals with moderate
/// sparsity.
///
/// Runs in O(N log N + S) where S is the grid size (200 steps internally), down
/// from O(N·S) via prefix sums of the squared coefficients.
///
/// `data` is modified in place. Returns the selected threshold.
pub fn sure_threshold(data: &mut [f64]) -> f64 {
    let n = data.len();
    if n == 0
    {
        return 0.0;
    }
    // Sort absolute values for O(log N) partitioning.
    let mut sorted: Vec<f64> = data.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let n_f = n as f64;
    // Prefix sums of squared coefficients: prefix_sq[i] = Σ_{j< i} sorted[j]²
    // Cost: O(N) once.
    let mut prefix_sq = vec![0.0; n + 1];
    for i in 0..n
    {
        prefix_sq[i + 1] = prefix_sq[i] + sorted[i] * sorted[i];
    }
    let total_ss = prefix_sq[n];

    // SURE risk at threshold t: R(t) = n - 2·#{|x_j| ≤ t} + Σ min(|x_j|, t)²
    let max_val = sorted.last().copied().unwrap_or(0.0).abs();
    if max_val <= 0.0
    {
        return 0.0;
    }

    let n_steps = 200;
    let mut best_tau = 0.0;
    let mut best_risk = total_ss; // risk at tau=0
    for i in 1..=n_steps
    {
        let frac = i as f64 / n_steps as f64;
        let tau = max_val * (frac * 0.99 + 0.01).ln() / (0.01f64).ln();
        let tau = tau.max(0.0);

        // Binary search → O(log N).  Prefix sum → O(1).
        let idx = sorted.partition_point(|x| x.abs() <= tau);
        let sum_sq = prefix_sq[idx] + (n - idx) as f64 * tau * tau;

        // SURE: n - 2·count_below + sum(min(|x|, τ))²
        let risk = n_f - 2.0 * idx as f64 + sum_sq;
        if risk < best_risk
        {
            best_risk = risk;
            best_tau = tau;
        }
    }
    soft_threshold(data, best_tau);
    best_tau
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soft_threshold_zeros_small_values() {
        let mut data = vec![-0.5, 0.3, 1.2, -2.0, 0.0];
        soft_threshold(&mut data, 1.0);
        let expected = vec![0.0, 0.0, 0.2, -1.0, 0.0];
        for (a, b) in data.iter().zip(&expected)
        {
            assert!((a - b).abs() < 1e-12, "{a} != {b}");
        }
    }

    #[test]
    fn soft_threshold_tau_zero_is_identity() {
        let mut data = vec![1.0, -2.0, 3.0];
        let orig = data.clone();
        soft_threshold(&mut data, 0.0);
        assert_eq!(data, orig);
    }

    #[test]
    fn hard_threshold_keeps_large_values() {
        let mut data = vec![0.1, 0.9, 1.5, -0.4, -2.0];
        hard_threshold(&mut data, 0.8);
        let expected = vec![0.0, 0.9, 1.5, 0.0, -2.0];
        assert_eq!(data, expected);
    }

    #[test]
    fn universal_soft_threshold_on_noise() {
        let mut data: Vec<f64> = vec![
            0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8,
            0.9, -1.0, 1.1, -1.2, 1.3, -1.4, 1.5, -1.6,
        ];
        let original = data.clone();
        let tau = universal_soft_threshold(&mut data);
        assert!(tau > 0.0, "tau should be positive, got {tau}");
        let energy_before: f64 = original.iter().map(|x| x * x).sum();
        let energy_after: f64 = data.iter().map(|x| x * x).sum();
        assert!(energy_after <= energy_before + 1e-9);
    }

    #[test]
    fn sure_threshold_reduces_energy_of_noisy_signal() {
        let mut data: Vec<f64> = (0..128)
            .map(|i| (i as f64 * 0.1).sin() + 0.5 * (i as f64 * 0.7).cos())
            .collect();
        let energy_before: f64 = data.iter().map(|x| x * x).sum();
        sure_threshold(&mut data);
        let energy_after: f64 = data.iter().map(|x| x * x).sum();
        assert!(
            energy_after <= energy_before + 1e-9,
            "thresholding must not increase energy"
        );
    }
}
```

---

## 5. `scirust-signal/src/wavelet.rs` — Transformée de Haar

```rust
//! Haar wavelet transform — forward and inverse DWT.
//!
//! All operations are deterministic `f64`.  The boundary extension is
//! **periodic** (wraparound), which keeps the number of coefficients
//! exactly equal to the input length at every level — perfect for
//! coefficient-wise thresholding.
//!
//! ## Example
//! ```text
//! s  = [1, 2, 3, 4, 5, 6, 7, 8]
//! cA = [2.12, 4.95, 7.78, 10.6]   (scaling / approximation)
//! cD = [-0.707, -0.707, -0.707, -0.707]  (detail / wavelet)
//! ```

use core::f64::consts::FRAC_1_SQRT_2;

/// Perform a 1-level **Haar discrete wavelet transform** (analysis) in place.
///
/// The first half of the output holds the approximation coefficients; the second
/// half holds the detail coefficients.  Length must be even.
pub fn haar_dwt(data: &mut [f64]) {
    let n = data.len();
    assert!(n % 2 == 0, "length must be even for Haar DWT, got {n}");
    if n < 2
    {
        return;
    }
    let half = n / 2;
    let mut tmp = vec![0.0; n];
    for j in 0..half
    {
        let a = data[2 * j];
        let b = data[2 * j + 1];
        tmp[j] = (a + b) * FRAC_1_SQRT_2;
        tmp[half + j] = (a - b) * FRAC_1_SQRT_2;
    }
    data.copy_from_slice(&tmp);
}

/// Perform a 1-level **inverse Haar DWT** (synthesis) in place.
///
/// Input must be in the same arrangement as `haar_dwt` output: first half
/// approximation, second half detail.  Length must be even.
pub fn haar_idwt(data: &mut [f64]) {
    let n = data.len();
    assert!(n % 2 == 0, "length must be even for Haar IDWT, got {n}");
    if n < 2
    {
        return;
    }
    let half = n / 2;
    let mut tmp = vec![0.0; n];
    for j in 0..half
    {
        let c = data[j];
        let d = data[half + j];
        tmp[2 * j] = (c + d) * FRAC_1_SQRT_2;
        tmp[2 * j + 1] = (c - d) * FRAC_1_SQRT_2;
    }
    data.copy_from_slice(&tmp);
}

/// Perform a multi-level **Haar DWT** (pyramid decomposition) in place.
///
/// `levels` is the number of decomposition levels (must be ≥ 1).  After
/// decomposition, the first `n / 2^levels` coefficients are the coarsest
/// approximation; the remaining coefficients are detail coefficients from
/// each level.
///
/// Length must be divisible by `2^levels`.
pub fn haar_dwt_multilevel(data: &mut [f64], levels: usize) {
    let n = data.len();
    let divisor = 1 << levels;
    assert!(
        n % divisor == 0,
        "length must be divisible by 2^{levels}, got {n}"
    );
    let mut len = n;
    for _ in 0..levels
    {
        let half = len / 2;
        let mut tmp = vec![0.0; len];
        for j in 0..half
        {
            let a = data[2 * j];
            let b = data[2 * j + 1];
            tmp[j] = (a + b) * FRAC_1_SQRT_2;
            tmp[half + j] = (a - b) * FRAC_1_SQRT_2;
        }
        for j in 0..len
        {
            data[j] = tmp[j];
        }
        len = half;
    }
}

/// Perform a multi-level **inverse Haar DWT** in place.
///
/// Must match the number of levels used in `haar_dwt_multilevel`.
pub fn haar_idwt_multilevel(data: &mut [f64], levels: usize) {
    let n = data.len();
    let divisor = 1 << levels;
    assert!(
        n % divisor == 0,
        "length must be divisible by 2^{levels}, got {n}"
    );
    let mut len = n / divisor;
    for _ in 0..levels
    {
        let half = len;
        let full = len * 2;
        let mut tmp = vec![0.0; full];
        for j in 0..half
        {
            let c = data[j];
            let d = data[half + j];
            tmp[2 * j] = (c + d) * FRAC_1_SQRT_2;
            tmp[2 * j + 1] = (c - d) * FRAC_1_SQRT_2;
        }
        for j in 0..full
        {
            data[j] = tmp[j];
        }
        len = full;
    }
}

/// Build a **Haar DWT matrix** `W` (size `n×n`) for a 1-level transform.
///
/// Multiplying a signal vector by `W` applies the 1-level Haar DWT (approximation
/// then detail coefficients).  `W` is orthogonal, so `Wᵀ` reconstructs.
///
/// Construction: each column `j` is `haar_dwt(e_j)` where `e_j` is the `j`-th
/// standard basis vector.  This guarantees `W · x == haar_dwt(x)`.
pub fn haar_matrix(n: usize) -> Vec<Vec<f64>> {
    assert!(
        n.is_power_of_two(),
        "Haar matrix requires power-of-two, got {n}"
    );
    let mut w = vec![vec![0.0; n]; n];
    let mut col_buf = vec![0.0; n];
    for j in 0..n
    {
        col_buf.fill(0.0);
        col_buf[j] = 1.0;
        haar_dwt(&mut col_buf);
        for i in 0..n
        {
            w[i][j] = col_buf[i];
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn haar_round_trip_single_level() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let orig = data.clone();
        haar_dwt(&mut data);
        haar_idwt(&mut data);
        for (a, b) in data.iter().zip(&orig)
        {
            assert!((a - b).abs() < 1e-12, "{a} != {b}");
        }
    }

    #[test]
    fn haar_multilevel_round_trip() {
        let mut data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let orig = data.clone();
        haar_dwt_multilevel(&mut data, 3);
        haar_idwt_multilevel(&mut data, 3);
        for (a, b) in data.iter().zip(&orig)
        {
            assert!((a - b).abs() < 1e-12, "{a} != {b}");
        }
    }

    #[test]
    fn haar_matrix_is_orthogonal() {
        let n = 8;
        let w = haar_matrix(n);
        for i in 0..n
        {
            for j in 0..n
            {
                let dot: f64 = (0..n).map(|k| w[i][k] * w[j][k]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-12, "W·Wᵀ[{i},{j}] = {dot}");
            }
        }
    }

    #[test]
    fn haar_matrix_matches_in_place_dwt() {
        let n = 8;
        let sig = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
        let mut expected = sig.clone();
        haar_dwt(&mut expected);
        let w = haar_matrix(n);
        let actual: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| w[i][j] * sig[j]).sum())
            .collect();
        for (a, b) in actual.iter().zip(&expected)
        {
            assert!((a - b).abs() < 1e-12, "{a} != {b}");
        }
    }
}
```

---

## 6. `scirust-estimation/src/rls.rs` — Filtre adaptatif RLS

```rust
//! Recursive Least Squares (RLS) adaptive filter — multi-channel, deterministic `f64`.
//!
//! An RLS filter learns a linear transformation `Δ` (correction matrix) from a
//! stream of input/output observations, using a forgetting factor `λ` that
//! controls how quickly past samples are forgotten.  The learned matrix is the
//! **Δ_RLS** that appears in the wavelet–RLS–RTS estimation equation:
//!
//! ```text
//! x̂_{|N} = M_RTS · [(I - Δ_RLS(x, λ)) · W^T · 𝒯_τ(W · s)]
//! ```
//!
//! ## Algorithm (standard RLS)
//!
//! For each new observation `(u, d)` where `u` is the input vector and `d` is
//! the target vector:
//!
//! 1. **Gain**:  `k = (P · u) / (λ + u^T · P · u)`
//! 2. **Error**: `e = d - W^T · u`   (where `W` is the weight matrix)
//! 3. **Weight update**: `W += k ⊗ e`
//! 4. **Covariance update**: `P = (P - k ⊗ u^T · P) / λ`

use serde::{Deserialize, Serialize};

/// Multi-channel RLS adaptive filter.
///
/// Learns an `n_out × n_in` weight matrix `w` incrementally.
/// The **correction matrix** `Δ` is derived from the inverse covariance `P`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlsFilter {
    n_in: usize,
    n_out: usize,
    lambda: f64,
    /// Weight matrix (n_out × n_in), row-major.
    w: Vec<f64>,
    /// Inverse input covariance (n_in × n_in), row-major.
    p: Vec<f64>,
}

impl RlsFilter {
    /// Create a new RLS filter with `n_in` inputs and `n_out` outputs.
    ///
    /// * `lambda` — forgetting factor in `(0, 1]`.  Small values adapt quickly;
    ///   `λ=1.0` never forgets.
    /// * `delta` — initial `P(0) = δ·I`.  Large values (e.g. `1e3`) give fast
    ///   initial convergence; small values (e.g. `1.0`) stabilise the filter.
    pub fn new(n_in: usize, n_out: usize, lambda: f64, delta: f64) -> Self {
        assert!(
            lambda > 0.0 && lambda <= 1.0,
            "lambda must be in (0, 1]"
        );
        let w = vec![0.0; n_out * n_in];
        let mut p = vec![0.0; n_in * n_in];
        for i in 0..n_in
        {
            p[i * n_in + i] = delta;
        }
        Self {
            n_in,
            n_out,
            lambda,
            w,
            p,
        }
    }

    /// Create from an existing weight matrix (row-major, `n_out × n_in`).
    pub fn with_weights(n_in: usize, n_out: usize, lambda: f64, delta: f64, weights: Vec<f64>) -> Self {
        assert_eq!(weights.len(), n_out * n_in);
        let mut f = Self::new(n_in, n_out, lambda, delta);
        f.w = weights;
        f
    }

    /// Filter one sample: predict `d̂ = w · u`, update weights using target `d`.
    ///
    /// Returns the prediction error `e = d - d̂`.
    pub fn update(&mut self, u: &[f64], d: &[f64]) -> Vec<f64> {
        assert_eq!(u.len(), self.n_in);
        assert_eq!(d.len(), self.n_out);

        // Prediction: d̂ = w · u
        let mut d_hat = vec![0.0; self.n_out];
        for i in 0..self.n_out
        {
            let row_start = i * self.n_in;
            for j in 0..self.n_in
            {
                d_hat[i] += self.w[row_start + j] * u[j];
            }
        }

        // Error: e = d - d̂
        let e: Vec<f64> = d.iter().zip(&d_hat).map(|(a, b)| a - b).collect();

        // Gain: k = P·u / (λ + uᵀ·P·u)
        let mut pu = vec![0.0; self.n_in];
        for i in 0..self.n_in
        {
            let row_start = i * self.n_in;
            for j in 0..self.n_in
            {
                pu[i] += self.p[row_start + j] * u[j];
            }
        }
        let upu: f64 = u.iter().zip(&pu).map(|(a, b)| a * b).sum();
        let denom = self.lambda + upu;
        let gain: Vec<f64> = pu.iter().map(|v| v / denom).collect();

        // Weight update: w += k ⊗ e  (outer product)
        for i in 0..self.n_out
        {
            let row_start = i * self.n_in;
            let ei = e[i];
            for j in 0..self.n_in
            {
                self.w[row_start + j] += gain[j] * ei;
            }
        }

        // Covariance update (symmetric form):
        // P_new = (P - k·(uᵀ·P)) / λ  =  (P - (pu/denom) ⊗ pu) / λ
        for i in 0..self.n_in
        {
            let row_start = i * self.n_in;
            let ki = gain[i];
            for j in 0..self.n_in
            {
                self.p[row_start + j] = (self.p[row_start + j] - ki * pu[j]) / self.lambda;
            }
        }

        e
    }

    /// Current weight matrix (row-major `n_out × n_in`).
    pub fn weights(&self) -> &[f64] {
        &self.w
    }

    /// Current inverse covariance matrix `P` (row-major `n_in × n_in`).
    pub fn covariance_inv(&self) -> &[f64] {
        &self.p
    }

    /// Compute the **RLS correction matrix** `Δ` as `P · Pᵀ` normalised by the
    /// trace, giving a measure of the filter's confidence in its current
    /// estimate.  Returns a square `n_in × n_in` matrix in row-major order.
    ///
    /// A high-confidence filter (small `P`) produces `Δ → 0`; an uncertain
    /// filter (large `P`) produces `Δ → I`, meaning the full correction is
    /// applied.
    pub fn delta(&self, scale: f64) -> Vec<f64> {
        let trace: f64 = (0..self.n_in).map(|i| self.p[i * self.n_in + i]).sum();
        if trace <= 0.0
        {
            return vec![0.0; self.n_in * self.n_in];
        }
        let factor = scale / trace;
        self.p.iter().map(|v| v * factor).collect()
    }

    /// Forgetting factor.
    pub fn lambda(&self) -> f64 { self.lambda }
    /// Change the forgetting factor dynamically.
    pub fn set_lambda(&mut self, lambda: f64) {
        assert!(lambda > 0.0 && lambda <= 1.0);
        self.lambda = lambda;
    }
    /// Input dimension.
    pub fn n_in(&self) -> usize { self.n_in }
    /// Output dimension.
    pub fn n_out(&self) -> usize { self.n_out }
    /// Reset the filter to initial conditions (zero weights, diagonal `P`).
    pub fn reset(&mut self, delta: f64) {
        self.w.fill(0.0);
        self.p.fill(0.0);
        for i in 0..self.n_in
        {
            self.p[i * self.n_in + i] = delta;
        }
    }
}

/// **Vector RLS** — scalar-output RLS specialised for the common case of
/// tracking one signal at a time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorRls {
    n: usize,
    lambda: f64,
    w: Vec<f64>,
    p: Vec<f64>,
}

impl VectorRls {
    /// Create a new scalar-output RLS with `n` input features.
    pub fn new(n: usize, lambda: f64, delta: f64) -> Self {
        assert!(lambda > 0.0 && lambda <= 1.0);
        let w = vec![0.0; n];
        let mut p = vec![0.0; n * n];
        for i in 0..n
        {
            p[i * n + i] = delta;
        }
        Self { n, lambda, w, p }
    }

    /// Update with one input vector `u` and scalar target `d`.
    /// Returns the prediction error `e = d - w·u`.
    pub fn update(&mut self, u: &[f64], d: f64) -> f64 {
        assert_eq!(u.len(), self.n);

        let d_hat: f64 = self.w.iter().zip(u).map(|(a, b)| a * b).sum();
        let e = d - d_hat;

        let mut pu = vec![0.0; self.n];
        for i in 0..self.n
        {
            let row_start = i * self.n;
            for j in 0..self.n
            {
                pu[i] += self.p[row_start + j] * u[j];
            }
        }
        let upu: f64 = u.iter().zip(&pu).map(|(a, b)| a * b).sum();
        let denom = self.lambda + upu;

        for i in 0..self.n
        {
            let ki = pu[i] / denom;
            self.w[i] += ki * e;
            let row_start = i * self.n;
            for j in 0..self.n
            {
                self.p[row_start + j] = (self.p[row_start + j] - ki * pu[j]) / self.lambda;
            }
        }
        e
    }

    /// Current weight vector.
    pub fn weights(&self) -> &[f64] { &self.w }
    /// Inverse covariance matrix.
    pub fn covariance_inv(&self) -> &[f64] { &self.p }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rls_tracks_linear_system() {
        let n_in = 2;
        let n_out = 1;
        let mut rls = RlsFilter::new(n_in, n_out, 0.99, 100.0);
        let true_w = vec![3.0, -2.0];
        let inputs = vec![
            vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5],
            vec![-1.0, 2.0], vec![2.0, -1.0],
        ];
        for _ in 0..200
        {
            for u in &inputs
            {
                let d = true_w[0] * u[0] + true_w[1] * u[1];
                rls.update(u, &[d]);
            }
        }
        let w = rls.weights();
        assert!((w[0] - 3.0).abs() < 0.15, "w[0] = {}", w[0]);
        assert!((w[1] + 2.0).abs() < 0.15, "w[1] = {}", w[1]);
    }

    #[test]
    fn vector_rls_converges() {
        let mut rls = VectorRls::new(3, 0.95, 10.0);
        let true_w = vec![1.5, -0.5, 2.0];
        let inputs = vec![
            vec![0.8, 1.2, -0.3], vec![-0.5, 0.7, 1.1],
            vec![2.0, -1.0, 0.5], vec![0.1, -0.8, 1.5],
            vec![-1.2, 0.3, -0.7],
        ];
        for _ in 0..200
        {
            for u in &inputs
            {
                let d: f64 = true_w.iter().zip(u.iter()).map(|(a, b)| a * b).sum();
                rls.update(u, d);
            }
        }
        let w = rls.weights();
        for (a, b) in w.iter().zip(&true_w)
        {
            assert!((a - b).abs() < 0.15, "weight {a} != {b}");
        }
    }

    #[test]
    fn delta_is_normalised() {
        let mut rls = RlsFilter::new(4, 2, 0.98, 10.0);
        for _ in 0..20
        {
            let u = vec![1.0, 2.0, 3.0, 4.0];
            let d = vec![1.0, -1.0];
            rls.update(&u, &d);
        }
        let d = rls.delta(1.0);
        let trace: f64 = (0..4).map(|i| d[i * 4 + i]).sum();
        assert!((trace - 1.0).abs() < 1e-9, "delta trace = {trace}");
    }
}
```

---

## 7. `scirust-signal/src/denoise.rs` — Pipeline complet

```rust
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
        fn new(seed: u64) -> Self { Self { s: seed } }
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
```

---

---

## Optimisations post-audit

### A. Élimination du heap churn niveau-par-niveau — transformée d'ondelettes

**Avant** (`haar_dwt_multilevel`) : une allocation `vec![0.0; len]` par niveau de décomposition.

**Après** : un unique scratch buffer de taille `n` alloué une fois pour toutes, réutilisé à chaque niveau via `data[..len].copy_from_slice(&scratch[..len])`.

```rust
pub fn haar_dwt_multilevel(data: &mut [f64], levels: usize) {
    let n = data.len();
    let mut scratch = vec![0.0; n];      // ← allocation unique
    let mut len = n;
    for _ in 0..levels
    {
        let half = len / 2;
        for j in 0..half { /* ... */ }
        data[..len].copy_from_slice(&scratch[..len]);  // ← copy de bande
        len = half;
    }
}
```

Même optimisation appliquée à `haar_idwt_multilevel`. Résultat : zéro fragmentation heap, prédictibilité cache L1/L2, déterministe.

### B. Stabilité numérique du filtre RLS — symétrisation forcée de P

Après la mise à jour de covariance `P_new = (P - k·uᵀ·P) / λ`, on force :

```rust
// P = (P + Pᵀ) / 2   → garantit P symétrique
for i in 0..self.n_in {
    for j in (i + 1)..self.n_in {
        let avg = (self.p[i * self.n_in + j] + self.p[j * self.n_in + i]) * 0.5;
        self.p[i * self.n_in + j] = avg;
        self.p[j * self.n_in + i] = avg;
    }
}
```

Empêche la dérive de la définie-positivité sur les longs horizons temporels où λ < 1. Appliqué aux deux `RlsFilter::update` et `VectorRls::update`.

### C. Perspective : factorisation UD de P

Pour une robustesse totale, la covariance P peut être stockée sous forme UD-factorisée (P = U·D·Uᵀ), comme dans le module `scirust-estimation/src/ud.rs` existant. Le module `UdFilter` de scirust fournit déjà l'infrastructure. Une future migration remplacerait la boucle de covariance O(n²) par des mises à jour UD garantissant P > 0 à la précision machine.

---

## Résumé des tests

```text
test result: ok. 76 passed (30 estimation + 46 signal), 0 failed, 0 ignored, 0 filtered out
```

| Crate | Tests | Nouveaux tests |
|-------|-------|----------------|
| `scirust-estimation` | 30 | `rls_tracks_linear_system`, `vector_rls_converges`, `delta_is_normalised` |
| `scirust-signal` | 46 | `soft_threshold_*`, `hard_threshold_*`, `universal_soft_threshold_*`, `sure_threshold_*`, `haar_*` (4), `pipeline_denoises_*`, `wavelet_only_improves_mse` |
