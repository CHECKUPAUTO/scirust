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
//!

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
    /// Scratch: gain numerator `P·u` (n_in). Persistent so [`Self::update`]
    /// performs **zero heap allocations** per sample.
    #[serde(skip, default)]
    scratch_pu: Vec<f64>,
    /// Scratch: a-priori error `e = d − ŷ` (n_out), also the return storage.
    #[serde(skip, default)]
    scratch_e: Vec<f64>,
}

impl RlsFilter {
    /// Create a new RLS filter with `n_in` inputs and `n_out` outputs.
    ///
    /// * `lambda` — forgetting factor in `(0, 1]`.  Small values adapt quickly;
    ///   `λ=1.0` never forgets.
    /// * `delta` — initial `P(0) = δ·I`.  Large values (e.g. `1e3`) give fast
    ///   initial convergence; small values (e.g. `1.0`) stabilise the filter.
    pub fn new(n_in: usize, n_out: usize, lambda: f64, delta: f64) -> Self {
        assert!(lambda > 0.0 && lambda <= 1.0, "lambda must be in (0, 1]");
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
            scratch_pu: vec![0.0; n_in],
            scratch_e: vec![0.0; n_out],
        }
    }

    /// Create from an existing weight matrix (row-major, `n_out × n_in`).
    pub fn with_weights(
        n_in: usize,
        n_out: usize,
        lambda: f64,
        delta: f64,
        weights: Vec<f64>,
    ) -> Self {
        assert_eq!(weights.len(), n_out * n_in);
        let mut f = Self::new(n_in, n_out, lambda, delta);
        f.w = weights;
        f
    }

    /// Filter one sample: predict `d̂ = w · u`, update weights using target `d`.
    ///
    /// Returns the a-priori prediction error `e = d - d̂` (a view into internal
    /// storage, valid until the next call). The hot loop performs **no heap
    /// allocation**: all intermediates live in persistent scratch buffers, and
    /// the Kalman gain `k = pu/denom` is folded into the updates on the fly.
    #[allow(clippy::needless_range_loop)]
    pub fn update(&mut self, u: &[f64], d: &[f64]) -> &[f64] {
        assert_eq!(u.len(), self.n_in);
        assert_eq!(d.len(), self.n_out);
        // One-time resize after deserialization (scratch is #[serde(skip)]).
        if self.scratch_pu.len() != self.n_in
        {
            self.scratch_pu.resize(self.n_in, 0.0);
        }
        if self.scratch_e.len() != self.n_out
        {
            self.scratch_e.resize(self.n_out, 0.0);
        }

        // Error: e = d - w·u  (prediction folded in).
        for i in 0..self.n_out
        {
            let row_start = i * self.n_in;
            let mut d_hat = 0.0;
            for j in 0..self.n_in
            {
                d_hat += self.w[row_start + j] * u[j];
            }
            self.scratch_e[i] = d[i] - d_hat;
        }

        // Gain numerator: pu = P·u ; denominator: λ + uᵀ·P·u.
        for i in 0..self.n_in
        {
            let row_start = i * self.n_in;
            let mut acc = 0.0;
            for j in 0..self.n_in
            {
                acc += self.p[row_start + j] * u[j];
            }
            self.scratch_pu[i] = acc;
        }
        let upu: f64 = u.iter().zip(&self.scratch_pu).map(|(a, b)| a * b).sum();
        let denom = self.lambda + upu;

        // Weight update: w += k ⊗ e with k[j] = pu[j]/denom.
        for i in 0..self.n_out
        {
            let row_start = i * self.n_in;
            let ei = self.scratch_e[i];
            for j in 0..self.n_in
            {
                self.w[row_start + j] += self.scratch_pu[j] / denom * ei;
            }
        }

        // Covariance update (symmetric form):
        // P_new = (P - k·(uᵀ·P)) / λ  =  (P - (pu/denom) ⊗ pu) / λ
        for i in 0..self.n_in
        {
            let row_start = i * self.n_in;
            let ki = self.scratch_pu[i] / denom;
            for j in 0..self.n_in
            {
                self.p[row_start + j] =
                    (self.p[row_start + j] - ki * self.scratch_pu[j]) / self.lambda;
            }
        }
        // Enforce exact symmetry: P = (P + Pᵀ) / 2
        // Prevents positive-definiteness drift over long horizons (λ < 1).
        for i in 0..self.n_in
        {
            for j in (i + 1)..self.n_in
            {
                let avg = (self.p[i * self.n_in + j] + self.p[j * self.n_in + i]) * 0.5;
                self.p[i * self.n_in + j] = avg;
                self.p[j * self.n_in + i] = avg;
            }
        }

        &self.scratch_e
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
        // Δ = scale · P / tr(P)   — normalised inverse covariance.
        let trace: f64 = (0..self.n_in).map(|i| self.p[i * self.n_in + i]).sum();
        if trace <= 0.0
        {
            return vec![0.0; self.n_in * self.n_in];
        }
        let factor = scale / trace;
        self.p.iter().map(|v| v * factor).collect()
    }

    /// Forgetting factor.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Change the forgetting factor dynamically.
    pub fn set_lambda(&mut self, lambda: f64) {
        assert!(lambda > 0.0 && lambda <= 1.0);
        self.lambda = lambda;
    }

    /// Input dimension.
    pub fn n_in(&self) -> usize {
        self.n_in
    }

    /// Output dimension.
    pub fn n_out(&self) -> usize {
        self.n_out
    }

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
    /// Scratch buffer `P·u` — persistent so `update` never allocates.
    #[serde(skip, default)]
    scratch_pu: Vec<f64>,
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
        Self {
            n,
            lambda,
            w,
            p,
            scratch_pu: vec![0.0; n],
        }
    }

    /// Update with one input vector `u` and scalar target `d`.
    /// Returns the prediction error `e = d - w·u`. Zero heap allocation per call.
    #[allow(clippy::needless_range_loop)]
    pub fn update(&mut self, u: &[f64], d: f64) -> f64 {
        assert_eq!(u.len(), self.n);
        if self.scratch_pu.len() != self.n
        {
            self.scratch_pu.resize(self.n, 0.0);
        }

        let d_hat: f64 = self.w.iter().zip(u).map(|(a, b)| a * b).sum();
        let e = d - d_hat;

        // Gain numerator: pu = P·u
        for i in 0..self.n
        {
            let row_start = i * self.n;
            let mut acc = 0.0;
            for j in 0..self.n
            {
                acc += self.p[row_start + j] * u[j];
            }
            self.scratch_pu[i] = acc;
        }
        let upu: f64 = u.iter().zip(&self.scratch_pu).map(|(a, b)| a * b).sum();
        let denom = self.lambda + upu;

        // Weight and covariance updates (symmetric form)
        for i in 0..self.n
        {
            let ki = self.scratch_pu[i] / denom;
            self.w[i] += ki * e;
            let row_start = i * self.n;
            for j in 0..self.n
            {
                self.p[row_start + j] =
                    (self.p[row_start + j] - ki * self.scratch_pu[j]) / self.lambda;
            }
        }
        // Enforce exact symmetry: P = (P + Pᵀ) / 2
        for i in 0..self.n
        {
            for j in (i + 1)..self.n
            {
                let avg = (self.p[i * self.n + j] + self.p[j * self.n + i]) * 0.5;
                self.p[i * self.n + j] = avg;
                self.p[j * self.n + i] = avg;
            }
        }
        e
    }

    /// Current weight vector.
    pub fn weights(&self) -> &[f64] {
        &self.w
    }

    /// Inverse covariance matrix.
    pub fn covariance_inv(&self) -> &[f64] {
        &self.p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rls_tracks_linear_system() {
        let n_in = 2;
        let n_out = 1;
        let mut rls = RlsFilter::new(n_in, n_out, 0.99, 100.0);
        // True system: y = 3·x₁ - 2·x₂
        let true_w = [3.0, -2.0];
        // Vary inputs so the covariance stays full-rank
        let inputs = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.5, 0.5],
            vec![-1.0, 2.0],
            vec![2.0, -1.0],
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
        // Vary inputs for full-rank covariance
        let inputs = vec![
            vec![0.8, 1.2, -0.3],
            vec![-0.5, 0.7, 1.1],
            vec![2.0, -1.0, 0.5],
            vec![0.1, -0.8, 1.5],
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
        // Feed some data so P evolves
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

    #[test]
    fn rls_is_a_static_state_kalman_filter_cross_oracle() {
        // At λ = 1 the RLS recursion IS a Kalman filter whose state is the
        // weight vector: F = I, Q = 0, H_k = u_kᵀ, R = 1, P(0) = δ·I. The
        // crate's own KalmanFilter (generic matrix path with an explicit
        // innovation-covariance inversion) is therefore an independent oracle:
        // both implementations must agree along the whole trajectory. Since H
        // changes per sample, a fresh KalmanFilter is rebuilt each step from
        // the carried state/covariance — pure reuse, no reimplementation.
        use crate::kalman::KalmanFilter;
        use crate::linalg::Mat;

        let n = 3;
        let delta = 100.0;
        let mut rls = VectorRls::new(n, 1.0, delta);

        let mut x = vec![0.0; n];
        let mut p = Mat::identity(n);
        for i in 0..n
        {
            p.set(i, i, delta);
        }
        let identity = Mat::identity(n);
        let zero_q = Mat::zeros(n, n);
        let r = Mat::new(1, 1, vec![1.0]);

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
        let mut rng = Lcg(57);
        let true_w = [0.8, -1.3, 2.1];
        for _ in 0..300
        {
            let u: Vec<f64> = (0..n).map(|_| rng.next()).collect();
            let d: f64 = true_w.iter().zip(&u).map(|(a, b)| a * b).sum::<f64>() + 0.1 * rng.next();

            rls.update(&u, d);

            let h = Mat::new(1, n, u.clone());
            let mut kf = KalmanFilter::new(
                x.clone(),
                p.clone(),
                identity.clone(),
                zero_q.clone(),
                h,
                r.clone(),
            );
            assert!(kf.update(&[d]), "Kalman update failed");
            x = kf.state().to_vec();
            p = kf.covariance().clone();

            for (a, b) in rls.weights().iter().zip(&x)
            {
                assert!(
                    (a - b).abs() < 1.0e-8 * (1.0 + a.abs()),
                    "RLS {a} vs Kalman {b}"
                );
            }
        }
        // Both converged to the truth.
        for (w, t) in rls.weights().iter().zip(&true_w)
        {
            assert!((w - t).abs() < 0.05, "{w} vs {t}");
        }
    }
}
