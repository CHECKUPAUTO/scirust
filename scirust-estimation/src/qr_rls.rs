//! Square-root RLS — the numerically hardened form of [`crate::rls::VectorRls`].
//!
//! Standard RLS propagates the inverse covariance `P` directly. Under a
//! forgetting factor `λ < 1`, finite precision and poorly exciting inputs, `P`
//! can drift away from symmetry and positive definiteness (the classic RLS
//! "explosion"); the plain implementation counters this with a forced
//! re-symmetrization, which helps but proves nothing.
//!
//! This filter never stores `P` at all. It propagates a **square-root factor**
//! `S` with `P = S·Sᵀ`, updated in place by Potter's rank-1 formula (the same
//! family of methods as the crate's [`crate::ud::UdFilter`] square-root Kalman
//! filter):
//!
//! ```text
//! v = Sᵀ·u,   β = λ + ‖v‖²,   k = S·v / β
//! S ← (S − α·(S·v)·vᵀ) / √λ,  with  α = 1 / (β + √(λ·β))
//! ```
//!
//! Whatever rounding errors accumulate inside `S`, the implied covariance
//! `S·Sᵀ` satisfies `xᵀ(S·Sᵀ)x = ‖Sᵀx‖² ≥ 0` for every `x` — positive
//! semi-definiteness holds **by construction**, not by trust. This is the
//! honest answer to RLS divergence: not a guarantee that the *estimate* is
//! always good (no algorithm offers that under insufficient excitation), but a
//! structural guarantee that the covariance can never turn indefinite.
//!
//! Cost is the same order as standard RLS, `O(n²)` per sample, zero heap
//! allocation in the hot loop.

use serde::{Deserialize, Serialize};

/// Scalar-output square-root RLS with `n` input features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QrRls {
    n: usize,
    lambda: f64,
    /// Weight vector.
    w: Vec<f64>,
    /// Square-root covariance factor `S` (row-major `n × n`), `P = S·Sᵀ`.
    s: Vec<f64>,
    /// Scratch `v = Sᵀ·u`.
    #[serde(skip, default)]
    scratch_v: Vec<f64>,
    /// Scratch `f = S·v` (also holds the unscaled gain).
    #[serde(skip, default)]
    scratch_f: Vec<f64>,
}

impl QrRls {
    /// Create the filter: `lambda ∈ (0, 1]`, initial `S(0) = √delta·I`
    /// (so `P(0) = delta·I`, matching [`crate::rls::VectorRls::new`]).
    pub fn new(n: usize, lambda: f64, delta: f64) -> Self {
        assert!(lambda > 0.0 && lambda <= 1.0, "lambda must be in (0, 1]");
        assert!(delta > 0.0, "delta must be positive");
        let mut s = vec![0.0; n * n];
        let sqrt_delta = delta.sqrt();
        for i in 0..n
        {
            s[i * n + i] = sqrt_delta;
        }
        Self {
            n,
            lambda,
            w: vec![0.0; n],
            s,
            scratch_v: vec![0.0; n],
            scratch_f: vec![0.0; n],
        }
    }

    /// Update with one input vector `u` and scalar target `d`.
    /// Returns the a-priori prediction error `e = d − w·u`.
    #[allow(clippy::needless_range_loop)]
    pub fn update(&mut self, u: &[f64], d: f64) -> f64 {
        assert_eq!(u.len(), self.n);
        let n = self.n;
        if self.scratch_v.len() != n
        {
            self.scratch_v.resize(n, 0.0);
        }
        if self.scratch_f.len() != n
        {
            self.scratch_f.resize(n, 0.0);
        }

        let e: f64 = d - self.w.iter().zip(u).map(|(a, b)| a * b).sum::<f64>();

        // v = Sᵀ·u
        for j in 0..n
        {
            let mut acc = 0.0;
            for i in 0..n
            {
                acc += self.s[i * n + j] * u[i];
            }
            self.scratch_v[j] = acc;
        }
        // β = λ + ‖v‖² ;  f = S·v (= P·u)
        let beta: f64 = self.lambda + self.scratch_v.iter().map(|&x| x * x).sum::<f64>();
        for i in 0..n
        {
            let row = i * n;
            let mut acc = 0.0;
            for j in 0..n
            {
                acc += self.s[row + j] * self.scratch_v[j];
            }
            self.scratch_f[i] = acc;
        }

        // Weight update: w += (f/β)·e.
        for i in 0..n
        {
            self.w[i] += self.scratch_f[i] / beta * e;
        }

        // Potter factor update: S ← (S − α·f·vᵀ)/√λ, α = 1/(β + √(λβ)).
        let alpha = 1.0 / (beta + (self.lambda * beta).sqrt());
        let inv_sqrt_lambda = 1.0 / self.lambda.sqrt();
        for i in 0..n
        {
            let row = i * n;
            let fi = alpha * self.scratch_f[i];
            for j in 0..n
            {
                self.s[row + j] = (self.s[row + j] - fi * self.scratch_v[j]) * inv_sqrt_lambda;
            }
        }

        e
    }

    /// Current weight vector.
    pub fn weights(&self) -> &[f64] {
        &self.w
    }

    /// Reconstruct the inverse covariance `P = S·Sᵀ` (allocates; diagnostic use).
    pub fn covariance_inv(&self) -> Vec<f64> {
        let n = self.n;
        let mut p = vec![0.0; n * n];
        for i in 0..n
        {
            for j in 0..n
            {
                let mut acc = 0.0;
                for k in 0..n
                {
                    acc += self.s[i * n + k] * self.s[j * n + k];
                }
                p[i * n + j] = acc;
            }
        }
        p
    }

    /// Forgetting factor.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

/// Multi-output (MIMO) square-root RLS — the hardened twin of
/// [`crate::rls::RlsFilter`].
///
/// The covariance factor `S` depends only on the *inputs*, so a single Potter
/// recursion is shared across all `n_out` outputs; each output row of the
/// weight matrix is corrected with the same gain. Per-sample cost is therefore
/// `O(n_in² + n_out·n_in)` — the same order as the standard MIMO RLS, with the
/// square-root PSD-by-construction guarantee of [`QrRls`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QrRlsMimo {
    n_in: usize,
    n_out: usize,
    lambda: f64,
    /// Weight matrix (n_out × n_in), row-major.
    w: Vec<f64>,
    /// Square-root covariance factor `S` (n_in × n_in), `P = S·Sᵀ`.
    s: Vec<f64>,
    #[serde(skip, default)]
    scratch_v: Vec<f64>,
    #[serde(skip, default)]
    scratch_f: Vec<f64>,
    #[serde(skip, default)]
    scratch_e: Vec<f64>,
}

impl QrRlsMimo {
    /// Create the filter: `lambda ∈ (0, 1]`, `S(0) = √delta·I` (`P(0) = delta·I`).
    pub fn new(n_in: usize, n_out: usize, lambda: f64, delta: f64) -> Self {
        assert!(lambda > 0.0 && lambda <= 1.0, "lambda must be in (0, 1]");
        assert!(delta > 0.0, "delta must be positive");
        let mut s = vec![0.0; n_in * n_in];
        let sqrt_delta = delta.sqrt();
        for i in 0..n_in
        {
            s[i * n_in + i] = sqrt_delta;
        }
        Self {
            n_in,
            n_out,
            lambda,
            w: vec![0.0; n_out * n_in],
            s,
            scratch_v: vec![0.0; n_in],
            scratch_f: vec![0.0; n_in],
            scratch_e: vec![0.0; n_out],
        }
    }

    /// Update with one input vector `u` (n_in) and target vector `d` (n_out).
    /// Returns the a-priori errors (view into internal storage). Zero heap
    /// allocation per call.
    #[allow(clippy::needless_range_loop)]
    pub fn update(&mut self, u: &[f64], d: &[f64]) -> &[f64] {
        assert_eq!(u.len(), self.n_in);
        assert_eq!(d.len(), self.n_out);
        let n = self.n_in;
        if self.scratch_v.len() != n
        {
            self.scratch_v.resize(n, 0.0);
        }
        if self.scratch_f.len() != n
        {
            self.scratch_f.resize(n, 0.0);
        }
        if self.scratch_e.len() != self.n_out
        {
            self.scratch_e.resize(self.n_out, 0.0);
        }

        // A-priori errors e = d − W·u.
        for i in 0..self.n_out
        {
            let row = i * n;
            let mut d_hat = 0.0;
            for j in 0..n
            {
                d_hat += self.w[row + j] * u[j];
            }
            self.scratch_e[i] = d[i] - d_hat;
        }

        // Shared input-side recursion: v = Sᵀ·u, β = λ + ‖v‖², f = S·v (= P·u).
        for j in 0..n
        {
            let mut acc = 0.0;
            for i in 0..n
            {
                acc += self.s[i * n + j] * u[i];
            }
            self.scratch_v[j] = acc;
        }
        let beta: f64 = self.lambda + self.scratch_v.iter().map(|&x| x * x).sum::<f64>();
        for i in 0..n
        {
            let row = i * n;
            let mut acc = 0.0;
            for j in 0..n
            {
                acc += self.s[row + j] * self.scratch_v[j];
            }
            self.scratch_f[i] = acc;
        }

        // Weight update, every output row with the same gain f/β.
        for i in 0..self.n_out
        {
            let row = i * n;
            let ei = self.scratch_e[i];
            for j in 0..n
            {
                self.w[row + j] += self.scratch_f[j] / beta * ei;
            }
        }

        // Potter factor update (identical to the scalar QrRls).
        let alpha = 1.0 / (beta + (self.lambda * beta).sqrt());
        let inv_sqrt_lambda = 1.0 / self.lambda.sqrt();
        for i in 0..n
        {
            let row = i * n;
            let fi = alpha * self.scratch_f[i];
            for j in 0..n
            {
                self.s[row + j] = (self.s[row + j] - fi * self.scratch_v[j]) * inv_sqrt_lambda;
            }
        }

        &self.scratch_e
    }

    /// Current weight matrix (row-major `n_out × n_in`).
    pub fn weights(&self) -> &[f64] {
        &self.w
    }

    /// Reconstruct `P = S·Sᵀ` (allocates; diagnostic use).
    pub fn covariance_inv(&self) -> Vec<f64> {
        let n = self.n_in;
        let mut p = vec![0.0; n * n];
        for i in 0..n
        {
            for j in 0..n
            {
                let mut acc = 0.0;
                for k in 0..n
                {
                    acc += self.s[i * n + k] * self.s[j * n + k];
                }
                p[i * n + j] = acc;
            }
        }
        p
    }

    /// Forgetting factor.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rls::VectorRls;

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
    fn qr_rls_matches_standard_rls_on_well_conditioned_data() {
        // On benign data the square-root form is algebraically the same filter:
        // weights must agree to fine tolerance.
        let n = 4;
        let mut std_rls = VectorRls::new(n, 0.98, 100.0);
        let mut qr = QrRls::new(n, 0.98, 100.0);
        let true_w = [1.5, -0.5, 2.0, 0.7];
        let mut rng = Lcg(7);
        for _ in 0..2000
        {
            let u: Vec<f64> = (0..n).map(|_| rng.next()).collect();
            let d: f64 = true_w.iter().zip(&u).map(|(a, b)| a * b).sum();
            std_rls.update(&u, d);
            qr.update(&u, d);
        }
        for (a, b) in std_rls.weights().iter().zip(qr.weights())
        {
            assert!((a - b).abs() < 1.0e-6, "weights diverged: {a} vs {b}");
        }
        for (w, t) in qr.weights().iter().zip(&true_w)
        {
            assert!((w - t).abs() < 1.0e-6, "did not converge: {w} vs {t}");
        }
    }

    #[test]
    fn implied_covariance_is_psd_by_construction_under_stress() {
        // Aggressive forgetting + nearly collinear inputs over a long horizon:
        // the regime where plain RLS covariances degrade. The Potter factor
        // guarantees xᵀ(S·Sᵀ)x = ‖Sᵀx‖² ≥ 0 for all x, so every diagonal entry
        // and every 2×2 principal minor of the reconstructed P must be
        // non-negative, and everything must stay finite.
        let n = 3;
        let mut qr = QrRls::new(n, 0.9, 1000.0);
        let mut rng = Lcg(99);
        for _ in 0..100_000
        {
            let x = rng.next();
            // Nearly collinear: second/third features almost copies of the first.
            let u = [x, x + 1.0e-9 * rng.next(), 2.0 * x + 1.0e-9 * rng.next()];
            let d = 0.5 * x;
            let e = qr.update(&u, d);
            assert!(e.is_finite(), "error diverged");
        }
        let p = qr.covariance_inv();
        for i in 0..n
        {
            assert!(p[i * n + i].is_finite(), "P[{i}][{i}] not finite");
            assert!(p[i * n + i] >= 0.0, "P[{i}][{i}] = {} < 0", p[i * n + i]);
            for j in 0..n
            {
                assert!(p[i * n + j].is_finite());
                // 2×2 principal minor non-negative (Cauchy-Schwarz on S rows).
                let minor = p[i * n + i] * p[j * n + j] - p[i * n + j] * p[j * n + i];
                assert!(minor >= -1.0e-12, "minor ({i},{j}) = {minor}");
            }
        }
        // The weight along the excited direction stays sane.
        assert!(qr.weights().iter().all(|w| w.is_finite() && w.abs() < 10.0));
    }

    #[test]
    fn qr_rls_tracks_a_drifting_system() {
        // λ < 1 must let the filter follow a slowly changing true weight.
        let n = 2;
        let mut qr = QrRls::new(n, 0.95, 100.0);
        let mut rng = Lcg(3);
        let mut true_w = [1.0, -1.0];
        let mut last_err = 0.0;
        for t in 0..5000
        {
            true_w[0] = 1.0 + 0.0005 * t as f64;
            let u = [rng.next(), rng.next()];
            let d: f64 = true_w.iter().zip(&u).map(|(a, b)| a * b).sum();
            last_err = qr.update(&u, d);
        }
        assert!((qr.weights()[0] - true_w[0]).abs() < 0.05, "lagging drift");
        assert!(last_err.abs() < 0.1, "steady-state error {last_err}");
    }

    #[test]
    fn mimo_first_row_is_bit_identical_to_scalar_qr_rls() {
        // Same S recursion, same op order per weight row ⇒ output row 0 of the
        // MIMO filter must reproduce the scalar filter to the last bit.
        let n = 4;
        let mut scalar = QrRls::new(n, 0.97, 50.0);
        let mut mimo = QrRlsMimo::new(n, 2, 0.97, 50.0);
        let mut rng = Lcg(11);
        for _ in 0..500
        {
            let u: Vec<f64> = (0..n).map(|_| rng.next()).collect();
            let d0: f64 = 1.5 * u[0] - 0.5 * u[2] + 0.01 * rng.next();
            let d1: f64 = -2.0 * u[1] + u[3] + 0.01 * rng.next();
            let e_s = scalar.update(&u, d0);
            let e_m0 = mimo.update(&u, &[d0, d1])[0];
            assert_eq!(e_s.to_bits(), e_m0.to_bits(), "errors diverged");
        }
        for j in 0..n
        {
            assert_eq!(
                scalar.weights()[j].to_bits(),
                mimo.weights()[j].to_bits(),
                "weight {j} diverged"
            );
        }
    }

    #[test]
    fn mimo_matches_standard_mimo_rls() {
        use crate::rls::RlsFilter;
        let (n_in, n_out) = (3, 2);
        let mut std_rls = RlsFilter::new(n_in, n_out, 0.99, 100.0);
        let mut qr = QrRlsMimo::new(n_in, n_out, 0.99, 100.0);
        let true_w = [[2.0, -1.0, 0.5], [0.3, 1.2, -0.7]];
        let mut rng = Lcg(23);
        for _ in 0..3000
        {
            let u: Vec<f64> = (0..n_in).map(|_| rng.next()).collect();
            let d: Vec<f64> = true_w
                .iter()
                .map(|row| row.iter().zip(&u).map(|(a, b)| a * b).sum())
                .collect();
            std_rls.update(&u, &d);
            qr.update(&u, &d);
        }
        for (a, b) in std_rls.weights().iter().zip(qr.weights())
        {
            assert!((a - b).abs() < 1.0e-6, "weights diverged: {a} vs {b}");
        }
        for (i, row) in true_w.iter().enumerate()
        {
            for (j, t) in row.iter().enumerate()
            {
                let w = qr.weights()[i * n_in + j];
                assert!((w - t).abs() < 1.0e-6, "w[{i}][{j}] = {w} vs {t}");
            }
        }
    }
}
