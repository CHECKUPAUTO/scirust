//! Probabilistic data association (PDAF) for tracking in clutter.
//!
//! [`super::mtt`] associates each track to at most one measurement by a hard
//! nearest-neighbour choice inside a gate. In clutter that choice is brittle: a
//! single false return closer than the true one hijacks the track. The
//! **Probabilistic Data Association Filter** replaces the hard decision with a
//! soft one — every measurement in the validation gate is kept, weighted by its
//! association probability `βᵢ` (with `β₀` the probability that *none* of them is
//! the target, i.e. the target was missed and all returns are clutter), and the
//! track is updated with the probability-weighted combined innovation. The
//! covariance carries an extra "spread of the innovations" term that inflates it
//! to honestly reflect the association ambiguity.
//!
//! This is the single-target PDAF over a constant-velocity Cartesian state with
//! position measurements, built on the shared dense-matrix helpers from
//! [`super::imm2d`]; dependency-free.

// Dense small-matrix linear algebra; indexed sweeps are the algorithm.
#![allow(clippy::needless_range_loop)]

use super::imm2d::{chol_solve, cholesky, mat_add, mat_mul, mat_t, mat_vec};
use std::f64::consts::PI;

const POS_X: usize = 0;
const POS_Y: usize = 2;

/// A single-target **Probabilistic Data Association Filter** over the Cartesian
/// constant-velocity state `[x, vₓ, y, v_y]` with `(x, y)` position
/// measurements, tracking through clutter.
#[derive(Debug, Clone, PartialEq)]
pub struct PdaFilter {
    x: Vec<f64>,
    p: Vec<Vec<f64>>,
    f: Vec<Vec<f64>>,
    q: Vec<Vec<f64>>,
    h: Vec<Vec<f64>>,
    r: Vec<Vec<f64>>,
    p_detect: f64,
    gate: f64,
    clutter_density: f64,
}

impl PdaFilter {
    /// A PDAF at frame interval `dt`, process-noise intensity `q`, and
    /// measurement variance `meas_var` (per axis), initialised at `(x0, y0)`
    /// with zero velocity and isotropic covariance `var0`. `p_detect` is the
    /// probability of detecting the target on a scan, `gate` the χ² (2 d.o.f.)
    /// validation-gate threshold, and `clutter_density` the spatial density of
    /// false returns (per unit area).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dt: f64,
        q: f64,
        meas_var: f64,
        x0: f64,
        y0: f64,
        var0: f64,
        p_detect: f64,
        gate: f64,
        clutter_density: f64,
    ) -> Self {
        let f = vec![
            vec![1.0, dt, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, dt],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let (qa, qb, qd) = (q * dt * dt * dt / 3.0, q * dt * dt / 2.0, q * dt);
        let qm = vec![
            vec![qa, qb, 0.0, 0.0],
            vec![qb, qd, 0.0, 0.0],
            vec![0.0, 0.0, qa, qb],
            vec![0.0, 0.0, qb, qd],
        ];
        let h = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 0.0, 1.0, 0.0]];
        let r = vec![vec![meas_var, 0.0], vec![0.0, meas_var]];
        let mut p0 = vec![vec![0.0; 4]; 4];
        for i in 0..4
        {
            p0[i][i] = var0;
        }
        Self {
            x: vec![x0, 0.0, y0, 0.0],
            p: p0,
            f,
            q: qm,
            h,
            r,
            p_detect,
            gate,
            clutter_density,
        }
    }

    /// Time update: `x ← F·x`, `P ← F·P·Fᵀ + Q`.
    pub fn predict(&mut self) {
        self.x = mat_vec(&self.f, &self.x);
        let fp = mat_mul(&self.f, &self.p);
        self.p = mat_add(&mat_mul(&fp, &mat_t(&self.f)), &self.q);
    }

    /// PDAF measurement update over the scan's `measurements` (`(x, y)` pairs,
    /// target and clutter mixed). Returns `β₀`, the probability that the target
    /// was not detected (no measurement associated). The state is left at the
    /// prediction when `β₀ = 1` (no validated measurement).
    pub fn update(&mut self, measurements: &[(f64, f64)]) -> f64 {
        let zhat = mat_vec(&self.h, &self.x);
        let ht = mat_t(&self.h);
        let mut s = mat_mul(&mat_mul(&self.h, &self.p), &ht);
        for i in 0..2
        {
            for j in 0..2
            {
                s[i][j] += self.r[i][j];
            }
        }
        let l = match cholesky(&s)
        {
            Some(l) => l,
            None => return 1.0,
        };
        // Kalman gain K = P·Hᵀ·S⁻¹.
        let pht = mat_mul(&self.p, &ht);
        let mut k = vec![vec![0.0; 2]; 4];
        for i in 0..4
        {
            k[i] = chol_solve(&l, &pht[i]);
        }
        // Gate the measurements; keep innovations and their likelihoods.
        let mut nus: Vec<[f64; 2]> = Vec::new();
        let mut es: Vec<f64> = Vec::new();
        for &(zx, zy) in measurements
        {
            let nu = [zx - zhat[0], zy - zhat[1]];
            let sinv = chol_solve(&l, &nu);
            let d2 = nu[0] * sinv[0] + nu[1] * sinv[1];
            if d2 <= self.gate
            {
                nus.push(nu);
                es.push((-0.5 * d2).exp());
            }
        }
        // Parametric PDA: b = λ·|2πS|^{1/2}·(1 − P_D·P_G)/P_D, with the χ²(2)
        // gate mass P_G = 1 − e^{−gate/2} and |2πS|^{1/2} = 2π·√det S.
        let det_s = l[0][0] * l[0][0] * l[1][1] * l[1][1];
        let p_g = 1.0 - (-0.5 * self.gate).exp();
        let b = self.clutter_density * 2.0 * PI * det_s.sqrt() * (1.0 - self.p_detect * p_g)
            / self.p_detect;
        let sum_e: f64 = es.iter().sum();
        let denom = b + sum_e;
        if denom <= 0.0
        {
            return 1.0; // nothing to associate — coast on the prediction.
        }
        let beta0 = b / denom;
        // Combined innovation ν̄ = Σ βᵢ νᵢ.
        let mut nu_bar = [0.0, 0.0];
        for (nu, &e) in nus.iter().zip(&es)
        {
            let beta = e / denom;
            nu_bar[0] += beta * nu[0];
            nu_bar[1] += beta * nu[1];
        }
        for i in 0..4
        {
            self.x[i] += k[i][0] * nu_bar[0] + k[i][1] * nu_bar[1];
        }
        // P_c = (I − K·H)·P (the standard updated covariance).
        let kh = mat_mul(&k, &self.h);
        let mut imkh = kh;
        for i in 0..4
        {
            for j in 0..4
            {
                let id = if i == j { 1.0 } else { 0.0 };
                imkh[i][j] = id - imkh[i][j];
            }
        }
        let p_c = mat_mul(&imkh, &self.p);
        // Spread of innovations: Σ βᵢ νᵢνᵢᵀ − ν̄ν̄ᵀ, mapped through K.
        let mut spread = vec![vec![0.0; 2]; 2];
        for (nu, &e) in nus.iter().zip(&es)
        {
            let beta = e / denom;
            for a in 0..2
            {
                for c in 0..2
                {
                    spread[a][c] += beta * nu[a] * nu[c];
                }
            }
        }
        for a in 0..2
        {
            for c in 0..2
            {
                spread[a][c] -= nu_bar[a] * nu_bar[c];
            }
        }
        let p_tilde = mat_mul(&mat_mul(&k, &spread), &mat_t(&k));
        // P = β₀·P_pred + (1−β₀)·P_c + P̃, then symmetrise.
        let mut newp = vec![vec![0.0; 4]; 4];
        for i in 0..4
        {
            for j in 0..4
            {
                newp[i][j] = beta0 * self.p[i][j] + (1.0 - beta0) * p_c[i][j] + p_tilde[i][j];
            }
        }
        for i in 0..4
        {
            for j in 0..4
            {
                self.p[i][j] = 0.5 * (newp[i][j] + newp[j][i]);
            }
        }
        beta0
    }

    /// Predict then update in one frame; returns `β₀`.
    pub fn step(&mut self, measurements: &[(f64, f64)]) -> f64 {
        self.predict();
        self.update(measurements)
    }

    /// The current filtered Cartesian position `(x, y)`.
    pub fn position(&self) -> (f64, f64) {
        (self.x[POS_X], self.x[POS_Y])
    }

    /// The current filtered Cartesian velocity `(vₓ, v_y)`.
    pub fn velocity(&self) -> (f64, f64) {
        (self.x[1], self.x[3])
    }

    /// The current state vector `[x, vₓ, y, v_y]`.
    pub fn state(&self) -> &[f64] {
        &self.x
    }

    /// The current position-estimate variance `Pₓₓ + P_yy`.
    pub fn position_variance(&self) -> f64 {
        self.p[POS_X][POS_X] + self.p[POS_Y][POS_Y]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic LCG for reproducible clutter and measurement noise.
    struct Lcg(u64);
    impl Lcg {
        fn unit(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        fn signed(&mut self) -> f64 {
            2.0 * self.unit() - 1.0
        }
    }

    #[test]
    fn clutter_free_single_measurement_tracks_the_truth() {
        // With no clutter (λ = 0) and one measurement per scan, PDAF reduces to a
        // standard Kalman filter and follows a constant-velocity target.
        let (vx, vy) = (1.5_f64, -0.8_f64);
        let mut f = PdaFilter::new(1.0, 1e-4, 0.25, 0.0, 0.0, 10.0, 0.99, 16.0, 0.0);
        for k in 1..=60
        {
            let (tx, ty) = (vx * k as f64, vy * k as f64);
            f.step(&[(tx, ty)]);
        }
        let (px, py) = f.position();
        let (tx, ty) = (vx * 60.0, vy * 60.0);
        assert!(
            (px - tx).abs() < 0.3 && (py - ty).abs() < 0.3,
            "pos ({px},{py})"
        );
    }

    #[test]
    fn tracks_a_target_through_dense_clutter() {
        // Each scan carries the true (noisy) measurement plus several uniform
        // clutter returns near the prediction. PDAF must stay locked to truth.
        let mut rng = Lcg(0x00C1_A5ED);
        let (vx, vy) = (1.0_f64, 0.5_f64);
        let mut f = PdaFilter::new(1.0, 1e-3, 0.25, 0.0, 0.0, 10.0, 0.9, 16.0, 1e-4);
        for k in 1..=50
        {
            let (tx, ty) = (vx * k as f64, vy * k as f64);
            let mut scan = vec![(tx + 0.3 * rng.signed(), ty + 0.3 * rng.signed())];
            for _ in 0..5
            {
                scan.push((tx + 8.0 * rng.signed(), ty + 8.0 * rng.signed()));
            }
            f.step(&scan);
        }
        let (px, py) = f.position();
        let (tx, ty) = (vx * 50.0, vy * 50.0);
        assert!(
            (px - tx).abs() < 2.0 && (py - ty).abs() < 2.0,
            "pos ({px},{py}) vs ({tx},{ty})"
        );
    }

    #[test]
    fn a_missed_scan_coasts_and_grows_the_covariance() {
        let mut f = PdaFilter::new(1.0, 0.5, 0.25, 0.0, 0.0, 5.0, 0.9, 16.0, 1e-4);
        for k in 1..=20
        {
            f.step(&[(2.0 * k as f64, 0.0)]);
        }
        let before_pos = f.position();
        let before_var = f.position_variance();
        // An empty scan: β₀ = 1, the state coasts on its prediction and the
        // covariance grows.
        let beta0 = f.step(&[]);
        assert!((beta0 - 1.0).abs() < 1e-12, "β₀ = {beta0}");
        let after_pos = f.position();
        // Position advanced by one velocity step (coast), not corrected.
        assert!(after_pos.0 > before_pos.0);
        assert!(f.position_variance() > before_var, "covariance should grow");
    }

    #[test]
    fn beta0_falls_when_a_measurement_sits_on_the_prediction() {
        // A single measurement exactly at the prediction ⇒ strong association,
        // small β₀; only far/no measurements ⇒ β₀ near 1.
        let mut f = PdaFilter::new(1.0, 1e-3, 0.25, 0.0, 0.0, 5.0, 0.9, 16.0, 1e-4);
        f.predict();
        let (px, py) = f.position();
        let beta0_on = {
            let mut g = f.clone();
            g.update(&[(px, py)])
        };
        let beta0_empty = {
            let mut g = f.clone();
            g.update(&[])
        };
        assert!(beta0_on < 0.5, "on-prediction β₀ = {beta0_on}");
        assert!(
            (beta0_empty - 1.0).abs() < 1e-12,
            "empty β₀ = {beta0_empty}"
        );
    }
}
