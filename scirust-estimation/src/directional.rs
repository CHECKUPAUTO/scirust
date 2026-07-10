//! Directional-forgetting RLS (Kulhavý / Cao-Schwartz) — the principled
//! anti-windup.
//!
//! ## The failure mode this fixes
//!
//! Exponential forgetting discounts the information matrix `R = P⁻¹` in **every
//! direction**: `R ← λ·R + u·uᵀ`. Under poor excitation — inputs confined to a
//! subspace, the normal industrial regime (steady setpoint, idle machine) — the
//! unexcited directions receive no fresh information yet keep being discounted,
//! so their information decays like `λᵏ` and the covariance **blows up** like
//! `λ⁻ᵏ` ("covariance windup"). The next stray input then produces a huge gain
//! and a violent, wrong weight jump. The stress test on [`crate::qr_rls::QrRls`]
//! shows the square-root form keeps `P` *positive*; it cannot keep it *bounded*
//! — windup is a model problem, not a rounding problem.
//!
//! ## The directional fix
//!
//! Forget **only in the excited direction** (Kulhavý's restricted forgetting,
//! in the rank-1 form of Cao & Schwartz): split `R` into its component along
//! the incoming regressor (in the `R`-metric) and the rest, discount only the
//! former:
//!
//! ```text
//! g = R·u,  r = uᵀ·g            (information carried by the direction of u)
//! R ← R − ((1−λ)/r)·g·gᵀ        (forget: only that rank-1 slice)   [if r > 0]
//! R ← R + u·uᵀ                  (learn from the new sample)
//! ```
//!
//! Directions orthogonal to the excitation keep their information **unchanged**
//! — provably: if `u` never excites a direction, its diagonal of `R` never
//! moves, so `P` stays bounded there by construction. The filter is maintained
//! in *both* forms (`P` for the gain via Sherman-Morrison, `R` for the
//! directional split), each update `O(n²)`; [`DirectionalRls::recondition`]
//! resynchronizes `P = R⁻¹` exactly to wash out long-horizon drift between the
//! two representations (see the crate's cross-oracle test against
//! `scirust-solvers`' Cholesky).

use crate::linalg::Mat;
use serde::{Deserialize, Serialize};

/// Scalar-output RLS with directional forgetting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectionalRls {
    n: usize,
    lambda: f64,
    /// Weight vector.
    w: Vec<f64>,
    /// Covariance `P` (row-major n×n).
    p: Vec<f64>,
    /// Information matrix `R = P⁻¹` (row-major n×n), the directional ledger.
    r: Vec<f64>,
    /// Excitation threshold: below `r ≤ eps` no forgetting is applied (there is
    /// nothing to forget in an unexcited direction).
    eps: f64,
    #[serde(skip, default)]
    scratch_g: Vec<f64>,
    #[serde(skip, default)]
    scratch_pu: Vec<f64>,
}

impl DirectionalRls {
    /// `lambda ∈ (0, 1]` (directional forgetting factor), `P(0) = delta·I`
    /// (so `R(0) = I/delta`).
    pub fn new(n: usize, lambda: f64, delta: f64) -> Self {
        assert!(lambda > 0.0 && lambda <= 1.0, "lambda must be in (0, 1]");
        assert!(delta > 0.0, "delta must be positive");
        let mut p = vec![0.0; n * n];
        let mut r = vec![0.0; n * n];
        for i in 0..n
        {
            p[i * n + i] = delta;
            r[i * n + i] = 1.0 / delta;
        }
        Self {
            n,
            lambda,
            w: vec![0.0; n],
            p,
            r,
            eps: 1.0e-12,
            scratch_g: vec![0.0; n],
            scratch_pu: vec![0.0; n],
        }
    }

    /// Update with input `u` and scalar target `d`; returns the a-priori error.
    /// Zero heap allocation per call.
    #[allow(clippy::needless_range_loop)]
    pub fn update(&mut self, u: &[f64], d: f64) -> f64 {
        assert_eq!(u.len(), self.n);
        let n = self.n;
        if self.scratch_g.len() != n
        {
            self.scratch_g.resize(n, 0.0);
        }
        if self.scratch_pu.len() != n
        {
            self.scratch_pu.resize(n, 0.0);
        }

        let e: f64 = d - self.w.iter().zip(u).map(|(a, b)| a * b).sum::<f64>();

        // Directional split: g = R·u, r_info = uᵀ·g.
        for i in 0..n
        {
            let row = i * n;
            let mut acc = 0.0;
            for j in 0..n
            {
                acc += self.r[row + j] * u[j];
            }
            self.scratch_g[i] = acc;
        }
        let r_info: f64 = u.iter().zip(&self.scratch_g).map(|(a, b)| a * b).sum();

        if r_info > self.eps && self.lambda < 1.0
        {
            // Forget ONLY the rank-1 slice of R along the excitation:
            // R ← R − ρ·g·gᵀ with ρ = (1−λ)/r_info; the matched Sherman-Morrison
            // update of P is P ← P + (ρ/λ)·u·uᵀ (using P·(R·u) = u and
            // 1 − ρ·r_info = λ).
            let rho = (1.0 - self.lambda) / r_info;
            let rho_over_lambda = rho / self.lambda;
            for i in 0..n
            {
                let row = i * n;
                let gi = self.scratch_g[i];
                let ui = u[i];
                for j in 0..n
                {
                    self.r[row + j] -= rho * gi * self.scratch_g[j];
                    self.p[row + j] += rho_over_lambda * ui * u[j];
                }
            }
        }

        // Information add (measurement): R ← R + u·uᵀ; P via Sherman-Morrison;
        // gain K = P_new·u = P'·u / (1 + uᵀ·P'·u).
        for i in 0..n
        {
            let row = i * n;
            let mut acc = 0.0;
            for j in 0..n
            {
                acc += self.p[row + j] * u[j];
            }
            self.scratch_pu[i] = acc;
        }
        let s = 1.0
            + u.iter()
                .zip(&self.scratch_pu)
                .map(|(a, b)| a * b)
                .sum::<f64>();
        for i in 0..n
        {
            self.w[i] += self.scratch_pu[i] / s * e;
            let row = i * n;
            let pui = self.scratch_pu[i];
            let ui = u[i];
            for j in 0..n
            {
                self.p[row + j] -= pui * self.scratch_pu[j] / s;
                self.r[row + j] += ui * u[j];
            }
        }
        // Keep both representations exactly symmetric.
        for i in 0..n
        {
            for j in (i + 1)..n
            {
                let ap = (self.p[i * n + j] + self.p[j * n + i]) * 0.5;
                self.p[i * n + j] = ap;
                self.p[j * n + i] = ap;
                let ar = (self.r[i * n + j] + self.r[j * n + i]) * 0.5;
                self.r[i * n + j] = ar;
                self.r[j * n + i] = ar;
            }
        }

        e
    }

    /// Resynchronize `P = R⁻¹` exactly — the long-horizon hygiene step. The two
    /// representations are updated by matched rank-1 formulas each sample, but
    /// floating-point drift slowly decouples them; call this every `10³–10⁶`
    /// samples (cost `O(n³)`, uses the crate's own [`Mat::inverse`]). Returns
    /// `false` (leaving `P` untouched) if `R` is numerically singular.
    pub fn recondition(&mut self) -> bool {
        let n = self.n;
        let r_mat = Mat::new(n, n, self.r.clone());
        let Some(p_mat) = r_mat.inverse()
        else
        {
            return false;
        };
        for i in 0..n
        {
            for j in 0..n
            {
                self.p[i * n + j] = p_mat.get(i, j);
            }
        }
        true
    }

    /// Consistency diagnostic: `max |P·R − I|` over all entries. Grows slowly
    /// with the horizon; [`Self::recondition`] resets it to rounding level.
    pub fn consistency_error(&self) -> f64 {
        let n = self.n;
        let mut worst = 0.0_f64;
        for i in 0..n
        {
            for j in 0..n
            {
                let mut acc = 0.0;
                for k in 0..n
                {
                    acc += self.p[i * n + k] * self.r[k * n + j];
                }
                let target = if i == j { 1.0 } else { 0.0 };
                worst = worst.max((acc - target).abs());
            }
        }
        worst
    }

    /// Current weight vector.
    pub fn weights(&self) -> &[f64] {
        &self.w
    }

    /// Current covariance `P` (row-major n×n).
    pub fn covariance(&self) -> &[f64] {
        &self.p
    }

    /// Current information matrix `R` (row-major n×n).
    pub fn information(&self) -> &[f64] {
        &self.r
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
    fn windup_immunity_where_standard_rls_explodes() {
        // THE discriminating test. Excite only direction 0 for a long stretch
        // with aggressive forgetting: standard RLS discounts the unexcited
        // direction every step, so its P[1][1] grows like λ⁻ᵏ — astronomically.
        // Directional forgetting leaves the unexcited direction untouched:
        // P[1][1] stays at its initial value.
        let n = 2;
        let (lambda, delta) = (0.9, 100.0);
        let mut standard = VectorRls::new(n, lambda, delta);
        let mut directional = DirectionalRls::new(n, lambda, delta);
        let mut rng = Lcg(3);
        for _ in 0..2000
        {
            let x = 1.0 + 0.1 * rng.next();
            let u = [x, 0.0];
            let d = 2.0 * x;
            standard.update(&u, d);
            directional.update(&u, d);
        }
        let p_std_11 = standard.covariance_inv()[n + 1];
        let p_dir_11 = directional.covariance()[n + 1];
        // Standard RLS: δ/λ^2000 ≈ 10²⁹³ — windup. (If this ever stops holding,
        // the standard filter gained its own guard and this test should change.)
        assert!(
            p_std_11 > 1.0e50,
            "standard RLS unexpectedly bounded: {p_std_11}"
        );
        // Directional: bounded at the initial value (nothing learned, nothing
        // forgotten in that direction).
        assert!(
            p_dir_11 <= delta * 1.0001,
            "directional P11 = {p_dir_11}, expected ≤ {delta}"
        );
        // And the excited direction converged.
        assert!((directional.weights()[0] - 2.0).abs() < 1.0e-6);
    }

    #[test]
    fn recovers_gracefully_when_excitation_returns() {
        // After a long starvation phase, direction 1 wakes up: the directional
        // filter must adapt with a sane gain (its P11 never wound up).
        let n = 2;
        let mut dir = DirectionalRls::new(n, 0.95, 100.0);
        let mut rng = Lcg(7);
        for _ in 0..1000
        {
            let x = 1.0 + 0.1 * rng.next();
            dir.update(&[x, 0.0], 2.0 * x);
        }
        for _ in 0..500
        {
            let u = [rng.next(), rng.next()];
            let d = 2.0 * u[0] - 3.0 * u[1];
            let e = dir.update(&u, d);
            assert!(e.is_finite() && e.abs() < 100.0, "gain blew up: e = {e}");
        }
        assert!((dir.weights()[0] - 2.0).abs() < 0.05);
        assert!((dir.weights()[1] + 3.0).abs() < 0.05);
    }

    #[test]
    fn lambda_one_matches_growing_window_rls() {
        // With λ = 1 there is no forgetting at all: directional and standard
        // coincide (same growing-window least squares).
        let n = 3;
        let mut std_rls = VectorRls::new(n, 1.0, 50.0);
        let mut dir = DirectionalRls::new(n, 1.0, 50.0);
        let mut rng = Lcg(11);
        for _ in 0..800
        {
            let u: Vec<f64> = (0..n).map(|_| rng.next()).collect();
            let d = 1.5 * u[0] - 0.5 * u[1] + 2.0 * u[2] + 0.01 * rng.next();
            std_rls.update(&u, d);
            dir.update(&u, d);
        }
        for (a, b) in std_rls.weights().iter().zip(dir.weights())
        {
            assert!((a - b).abs() < 1.0e-8, "{a} vs {b}");
        }
    }

    #[test]
    fn tracks_a_drifting_system_in_the_excited_direction() {
        let n = 2;
        let mut dir = DirectionalRls::new(n, 0.95, 100.0);
        let mut rng = Lcg(13);
        let mut w0 = 1.0;
        for _ in 0..3000
        {
            w0 += 0.001;
            let u = [rng.next(), rng.next()];
            let d = w0 * u[0] - u[1];
            dir.update(&u, d);
        }
        assert!((dir.weights()[0] - w0).abs() < 0.05, "lagging the drift");
    }

    #[test]
    fn recondition_restores_p_r_consistency() {
        // Long horizon: the matched rank-1 updates of P and R drift apart in
        // floating point; recondition() must collapse the consistency error
        // back to rounding level.
        let n = 4;
        let mut dir = DirectionalRls::new(n, 0.98, 100.0);
        let mut rng = Lcg(17);
        for _ in 0..200_000
        {
            let u: Vec<f64> = (0..n).map(|_| rng.next()).collect();
            let d = u[0] - 2.0 * u[1] + 0.5 * u[3] + 0.01 * rng.next();
            dir.update(&u, d);
        }
        let before = dir.consistency_error();
        assert!(dir.recondition(), "R became singular");
        let after = dir.consistency_error();
        assert!(
            after <= before,
            "recondition made things worse: {before} → {after}"
        );
        assert!(after < 1.0e-9, "consistency after recondition: {after}");
    }
}
