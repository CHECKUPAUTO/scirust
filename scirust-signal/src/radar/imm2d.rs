//! Manoeuvring-target tracking in the plane — a coordinated-turn IMM.
//!
//! The 1-D IMM ([`super::kalman`]) switches between a quiet and an agile
//! *constant-velocity* model, which follows a manoeuvre by inflating its process
//! noise — a proxy for a turn, not a turn. A real turning target (an aircraft or
//! missile pulling a constant-rate turn) is tracked far better by a model that
//! *knows* about turns: the **coordinated-turn** (CT) model, whose transition
//! rotates the velocity vector at a constant angular rate ω while integrating
//! position along the arc.
//!
//! This module provides:
//!
//! - [`KalmanLinear`] — a general n-state, m-measurement **linear Kalman
//!   filter** with dense matrices, a Cholesky-based measurement update, and the
//!   Gaussian innovation likelihood. Reusable well beyond tracking.
//! - [`cv_model_2d`] / [`ct_model_2d`] — the two planar motion models over the
//!   Cartesian state `[x, vₓ, y, v_y]`: nearly-constant-velocity, and
//!   coordinated turn at a fixed rate ω.
//! - [`Imm2D`] — an Interacting Multiple Model estimator over a bank of these
//!   (typically CV plus one or two CT models at ±ω). During straight flight the
//!   CV model wins; the instant the target turns, the matching CT model's
//!   likelihood takes over, so the track follows the arc instead of overshooting
//!   it. Dependency-free.

// This module is dense small-matrix linear algebra; indexed sweeps are the
// algorithm, not an oversight.
#![allow(clippy::needless_range_loop)]

use std::f64::consts::PI;

/// The Cartesian tracking state layout `[x, vₓ, y, v_y]` produced by the model
/// constructors: position is indices `0`/`2`, velocity `1`/`3`.
const POS_X: usize = 0;
const VEL_X: usize = 1;
const POS_Y: usize = 2;
const VEL_Y: usize = 3;

/// A general linear Kalman filter with an `n`-vector state, `m`-vector
/// measurement, and dense transition/process/observation/noise matrices.
///
/// `predict` applies `x ← F·x`, `P ← F·P·Fᵀ + Q`; `update` applies the standard
/// measurement correction with observation `H` and noise `R`, returning the
/// Gaussian innovation likelihood.
#[derive(Debug, Clone, PartialEq)]
pub struct KalmanLinear {
    x: Vec<f64>,
    p: Vec<Vec<f64>>,
    f: Vec<Vec<f64>>,
    q: Vec<Vec<f64>>,
    h: Vec<Vec<f64>>,
    r: Vec<Vec<f64>>,
}

impl KalmanLinear {
    /// A filter with initial state `x0`, covariance `p0`, transition `f`,
    /// process noise `q`, observation `h`, and measurement noise `r`.
    pub fn new(
        x0: Vec<f64>,
        p0: Vec<Vec<f64>>,
        f: Vec<Vec<f64>>,
        q: Vec<Vec<f64>>,
        h: Vec<Vec<f64>>,
        r: Vec<Vec<f64>>,
    ) -> Self {
        Self {
            x: x0,
            p: p0,
            f,
            q,
            h,
            r,
        }
    }

    /// Time update: `x ← F·x`, `P ← F·P·Fᵀ + Q`.
    pub fn predict(&mut self) {
        self.x = mat_vec(&self.f, &self.x);
        let fp = mat_mul(&self.f, &self.p);
        let fpft = mat_mul(&fp, &mat_t(&self.f));
        self.p = mat_add(&fpft, &self.q);
    }

    /// Measurement update with `z`; returns the Gaussian innovation likelihood
    /// `𝒩(residual; 0, S)`. If the innovation covariance is not positive
    /// definite the state is left unchanged and `0.0` is returned.
    pub fn update(&mut self, z: &[f64]) -> f64 {
        let n = self.x.len();
        let m = z.len();
        let hx = mat_vec(&self.h, &self.x);
        let y: Vec<f64> = (0..m).map(|i| z[i] - hx[i]).collect();
        let ht = mat_t(&self.h);
        let s = mat_add(&mat_mul(&mat_mul(&self.h, &self.p), &ht), &self.r);
        let l = match cholesky(&s)
        {
            Some(l) => l,
            None => return 0.0,
        };
        // Kalman gain K = P·Hᵀ·S⁻¹, solved row-wise: S·K[i]ᵀ = (P·Hᵀ)[i].
        let pht = mat_mul(&self.p, &ht);
        let mut k = vec![vec![0.0; m]; n];
        for i in 0..n
        {
            k[i] = chol_solve(&l, &pht[i]);
        }
        for i in 0..n
        {
            let mut acc = 0.0;
            for j in 0..m
            {
                acc += k[i][j] * y[j];
            }
            self.x[i] += acc;
        }
        // P ← (I − K·H)·P, then symmetrise for numerical hygiene.
        let kh = mat_mul(&k, &self.h);
        let mut imkh = kh;
        for i in 0..n
        {
            for j in 0..n
            {
                let ident = if i == j { 1.0 } else { 0.0 };
                imkh[i][j] = ident - imkh[i][j];
            }
        }
        let np = mat_mul(&imkh, &self.p);
        for i in 0..n
        {
            for j in 0..n
            {
                self.p[i][j] = 0.5 * (np[i][j] + np[j][i]);
            }
        }
        // Likelihood from the same Cholesky: exp(−½ yᵀS⁻¹y) / √((2π)^m·det S).
        let sinv_y = chol_solve(&l, &y);
        let quad: f64 = (0..m).map(|i| y[i] * sinv_y[i]).sum();
        let mut det = 1.0;
        for i in 0..m
        {
            det *= l[i][i] * l[i][i];
        }
        (-0.5 * quad).exp() / ((2.0 * PI).powi(m as i32) * det).sqrt()
    }

    /// Predict then update in one frame; returns the innovation likelihood.
    pub fn step(&mut self, z: &[f64]) -> f64 {
        self.predict();
        self.update(z)
    }

    /// The current state vector.
    pub fn state(&self) -> &[f64] {
        &self.x
    }

    /// The current state covariance.
    pub fn covariance(&self) -> &[Vec<f64>] {
        &self.p
    }
}

/// Continuous-white-noise-acceleration process noise for one axis over `dt`
/// with intensity `q`: `[[dt³/3, dt²/2], [dt²/2, dt]]·q`.
fn axis_q(dt: f64, q: f64) -> (f64, f64, f64) {
    (q * dt * dt * dt / 3.0, q * dt * dt / 2.0, q * dt)
}

/// Assemble the block-diagonal planar `[x, vₓ, y, v_y]` process-noise matrix
/// from the per-axis white-noise-acceleration terms.
fn planar_q(dt: f64, q: f64) -> Vec<Vec<f64>> {
    let (a, b, d) = axis_q(dt, q);
    vec![
        vec![a, b, 0.0, 0.0],
        vec![b, d, 0.0, 0.0],
        vec![0.0, 0.0, a, b],
        vec![0.0, 0.0, b, d],
    ]
}

/// The planar position-only observation and its noise: `H` picks `x`, `y`;
/// `R = r·I₂`.
fn planar_h_r(r: f64) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let h = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 0.0, 1.0, 0.0]];
    let rr = vec![vec![r, 0.0], vec![0.0, r]];
    (h, rr)
}

/// A **nearly-constant-velocity** planar model over `[x, vₓ, y, v_y]` at frame
/// interval `dt`, process-noise intensity `q`, and measurement variance `r`,
/// initialised at `(x0, y0)` with zero velocity and isotropic covariance `var0`.
pub fn cv_model_2d(dt: f64, q: f64, r: f64, x0: f64, y0: f64, var0: f64) -> KalmanLinear {
    let f = vec![
        vec![1.0, dt, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, dt],
        vec![0.0, 0.0, 0.0, 1.0],
    ];
    let (h, rr) = planar_h_r(r);
    let x = vec![x0, 0.0, y0, 0.0];
    let p0 = diag(&[var0; 4]);
    KalmanLinear::new(x, p0, f, planar_q(dt, q), h, rr)
}

/// A **coordinated-turn** planar model at constant angular rate `omega` (rad per
/// frame) over `[x, vₓ, y, v_y]`. The transition rotates the velocity vector at
/// `omega` and integrates position along the arc; as `omega → 0` it degenerates
/// to [`cv_model_2d`]. Other arguments match `cv_model_2d`.
pub fn ct_model_2d(
    dt: f64,
    omega: f64,
    q: f64,
    r: f64,
    x0: f64,
    y0: f64,
    var0: f64,
) -> KalmanLinear {
    let wt = omega * dt;
    let (s, c) = (wt.sin(), wt.cos());
    // sinωt/ω and (1−cosωt)/ω, with the ω→0 limits dt and 0.
    let (a, b) = if omega.abs() < 1e-9
    {
        (dt, 0.0)
    }
    else
    {
        (s / omega, (1.0 - c) / omega)
    };
    let f = vec![
        vec![1.0, a, 0.0, -b],
        vec![0.0, c, 0.0, -s],
        vec![0.0, b, 1.0, a],
        vec![0.0, s, 0.0, c],
    ];
    let (h, rr) = planar_h_r(r);
    let x = vec![x0, 0.0, y0, 0.0];
    let p0 = diag(&[var0; 4]);
    KalmanLinear::new(x, p0, f, planar_q(dt, q), h, rr)
}

/// An **Interacting Multiple Model** estimator over a bank of planar
/// [`KalmanLinear`] models sharing the `[x, vₓ, y, v_y]` state — typically one
/// [`cv_model_2d`] and one or two [`ct_model_2d`] at ±ω.
///
/// Each [`step`](Self::step) mixes the models' states by the transition-weighted
/// mode probabilities, filters every model on the measurement, updates the mode
/// probabilities from the likelihoods, and reports the probability-weighted
/// combined estimate.
#[derive(Debug, Clone)]
pub struct Imm2D {
    models: Vec<KalmanLinear>,
    mu: Vec<f64>,
    trans: Vec<Vec<f64>>,
}

impl Imm2D {
    /// An IMM over `models` with Markov mode-transition matrix `trans`
    /// (`trans[i][j]` = probability of switching from model `i` to `j`) and
    /// initial mode probabilities `mu0`; `trans` rows and `mu0` are normalised.
    pub fn new(models: Vec<KalmanLinear>, trans: Vec<Vec<f64>>, mu0: Vec<f64>) -> Self {
        let n = models.len();
        let mut mu = mu0;
        mu.resize(n, 0.0);
        normalise(&mut mu);
        let mut trans = trans;
        trans.resize(n, vec![0.0; n]);
        for row in &mut trans
        {
            row.resize(n, 0.0);
            normalise(row);
        }
        Self { models, mu, trans }
    }

    /// Advance one frame with planar position measurement `z = [x, y]`.
    pub fn step(&mut self, z: &[f64]) {
        let n = self.models.len();
        if n == 0
        {
            return;
        }
        let dim = self.models[0].x.len();
        let cbar: Vec<f64> = (0..n)
            .map(|j| (0..n).map(|i| self.trans[i][j] * self.mu[i]).sum())
            .collect();
        let xs: Vec<Vec<f64>> = self.models.iter().map(|m| m.x.clone()).collect();
        let ps: Vec<Vec<Vec<f64>>> = self.models.iter().map(|m| m.p.clone()).collect();
        for j in 0..n
        {
            let cj = cbar[j].max(1e-300);
            let mut xmix = vec![0.0; dim];
            for i in 0..n
            {
                let w = self.trans[i][j] * self.mu[i] / cj;
                for d in 0..dim
                {
                    xmix[d] += w * xs[i][d];
                }
            }
            let mut pmix = vec![vec![0.0; dim]; dim];
            for i in 0..n
            {
                let w = self.trans[i][j] * self.mu[i] / cj;
                let dx: Vec<f64> = (0..dim).map(|d| xs[i][d] - xmix[d]).collect();
                for a in 0..dim
                {
                    for b in 0..dim
                    {
                        pmix[a][b] += w * (ps[i][a][b] + dx[a] * dx[b]);
                    }
                }
            }
            self.models[j].x = xmix;
            self.models[j].p = pmix;
        }
        let like: Vec<f64> = self.models.iter_mut().map(|m| m.step(z)).collect();
        let mut newmu: Vec<f64> = (0..n).map(|j| cbar[j] * like[j]).collect();
        let norm: f64 = newmu.iter().sum();
        if norm <= 1e-300
        {
            newmu = cbar;
        }
        normalise(&mut newmu);
        self.mu = newmu;
    }

    /// The combined position estimate `(x, y) = Σ_j μ_j·(x_j, y_j)`.
    pub fn position(&self) -> (f64, f64) {
        let mut xy = (0.0, 0.0);
        for (m, &w) in self.models.iter().zip(&self.mu)
        {
            xy.0 += w * m.x[POS_X];
            xy.1 += w * m.x[POS_Y];
        }
        xy
    }

    /// The combined velocity estimate `(vₓ, v_y) = Σ_j μ_j·(vₓ_j, v_y_j)`.
    pub fn velocity(&self) -> (f64, f64) {
        let mut vxy = (0.0, 0.0);
        for (m, &w) in self.models.iter().zip(&self.mu)
        {
            vxy.0 += w * m.x[VEL_X];
            vxy.1 += w * m.x[VEL_Y];
        }
        vxy
    }

    /// The current mode probabilities, one per model, summing to one.
    pub fn mode_probabilities(&self) -> &[f64] {
        &self.mu
    }
}

// ---- small dense-matrix helpers (f64, row-major); shared within `radar` ----

/// The `n×n` diagonal matrix with the given diagonal.
fn diag(d: &[f64]) -> Vec<Vec<f64>> {
    let n = d.len();
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n
    {
        m[i][i] = d[i];
    }
    m
}

/// Matrix product `a·b`.
pub(super) fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let (n, k, m) = (a.len(), b.len(), b[0].len());
    let mut c = vec![vec![0.0; m]; n];
    for i in 0..n
    {
        for l in 0..k
        {
            let ail = a[i][l];
            if ail != 0.0
            {
                for j in 0..m
                {
                    c[i][j] += ail * b[l][j];
                }
            }
        }
    }
    c
}

/// Matrix–vector product `a·v`.
pub(super) fn mat_vec(a: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(v).map(|(x, y)| x * y).sum())
        .collect()
}

/// Elementwise sum `a + b`.
pub(super) fn mat_add(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b)
        .map(|(ra, rb)| ra.iter().zip(rb).map(|(x, y)| x + y).collect())
        .collect()
}

/// Transpose.
pub(super) fn mat_t(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let (rows, cols) = (a.len(), a[0].len());
    let mut t = vec![vec![0.0; rows]; cols];
    for i in 0..rows
    {
        for j in 0..cols
        {
            t[j][i] = a[i][j];
        }
    }
    t
}

/// Lower-triangular Cholesky factor `L` with `A = L·Lᵀ`; `None` if `A` is not
/// positive definite.
pub(super) fn cholesky(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n
    {
        for j in 0..=i
        {
            let mut sum = a[i][j];
            for k in 0..j
            {
                sum -= l[i][k] * l[j][k];
            }
            if i == j
            {
                if sum <= 0.0
                {
                    return None;
                }
                l[i][j] = sum.sqrt();
            }
            else
            {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    Some(l)
}

/// Solve `A·x = b` given the Cholesky factor `L` of `A` (`A = L·Lᵀ`).
pub(super) fn chol_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut y = vec![0.0; n];
    for i in 0..n
    {
        let mut s = b[i];
        for k in 0..i
        {
            s -= l[i][k] * y[k];
        }
        y[i] = s / l[i][i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev()
    {
        let mut s = y[i];
        for k in (i + 1)..n
        {
            s -= l[k][i] * x[k];
        }
        x[i] = s / l[i][i];
    }
    x
}

/// Scale a vector to sum to one; a degenerate all-zero vector becomes uniform.
fn normalise(v: &mut [f64]) {
    let s: f64 = v.iter().sum();
    if s > 0.0
    {
        for x in v.iter_mut()
        {
            *x /= s;
        }
    }
    else if !v.is_empty()
    {
        let u = 1.0 / v.len() as f64;
        for x in v.iter_mut()
        {
            *x = u;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Advance a true state by a transition matrix (noise-free ground truth).
    fn advance(f: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
        mat_vec(f, x)
    }

    #[test]
    fn ct_reduces_to_cv_as_omega_vanishes() {
        // A coordinated turn at a negligible rate predicts like constant velocity.
        let dt = 1.0;
        let mut cv = cv_model_2d(dt, 0.0, 1.0, 2.0, -3.0, 1.0);
        let mut ct = ct_model_2d(dt, 1e-10, 0.0, 1.0, 2.0, -3.0, 1.0);
        cv.x = vec![2.0, 1.5, -3.0, 0.5];
        ct.x = vec![2.0, 1.5, -3.0, 0.5];
        cv.predict();
        ct.predict();
        for i in 0..4
        {
            assert!((cv.state()[i] - ct.state()[i]).abs() < 1e-6, "state {i}");
        }
    }

    #[test]
    fn linear_kalman_recovers_constant_velocity_in_2d() {
        // Noise-free straight-line motion: the filtered velocity converges.
        let dt = 0.5;
        let (vx, vy) = (2.0, -1.0);
        let mut f = cv_model_2d(dt, 1e-6, 1.0, 0.0, 0.0, 10.0);
        for k in 1..=200
        {
            let t = k as f64 * dt;
            f.step(&[vx * t, vy * t]);
        }
        assert!(
            (f.state()[VEL_X] - vx).abs() < 1e-2,
            "vx {}",
            f.state()[VEL_X]
        );
        assert!(
            (f.state()[VEL_Y] - vy).abs() < 1e-2,
            "vy {}",
            f.state()[VEL_Y]
        );
    }

    #[test]
    fn ct_filter_tracks_a_circular_trajectory_far_better_than_cv() {
        // Ground truth: a coordinated turn at rate ω0 (exact circular motion).
        let (dt, omega) = (1.0_f64, 0.10_f64);
        let truth_model = ct_model_2d(dt, omega, 0.0, 1.0, 0.0, 0.0, 1.0);
        let f_truth = truth_model.f.clone();
        let mut xt = vec![0.0, 1.0, 0.0, 0.0]; // moving +x, will curve
        let mut ct = ct_model_2d(dt, omega, 1e-4, 0.25, 0.0, 0.0, 5.0);
        let mut cv = cv_model_2d(dt, 1e-4, 0.25, 0.0, 0.0, 5.0);
        let (mut ct_err, mut cv_err) = (0.0, 0.0);
        for k in 1..=60
        {
            xt = advance(&f_truth, &xt);
            let z = [xt[POS_X], xt[POS_Y]];
            ct.step(&z);
            cv.step(&z);
            if k > 30
            {
                ct_err += (ct.state()[POS_X] - z[0]).hypot(ct.state()[POS_Y] - z[1]);
                cv_err += (cv.state()[POS_X] - z[0]).hypot(cv.state()[POS_Y] - z[1]);
            }
        }
        assert!(
            ct_err < 0.5 * cv_err,
            "CT ({ct_err}) should clearly beat CV ({cv_err}) on a turn"
        );
    }

    #[test]
    fn imm2d_picks_the_turn_model_and_beats_cv_on_a_manoeuvre() {
        // Straight run, then a coordinated turn. The IMM's CT mode should take
        // over and the track should beat a lone CV filter over the turn.
        let (dt, omega) = (1.0_f64, 0.12_f64);
        let cv_truth = cv_model_2d(dt, 0.0, 1.0, 0.0, 0.0, 1.0).f.clone();
        let ct_truth = ct_model_2d(dt, omega, 0.0, 1.0, 0.0, 0.0, 1.0).f.clone();
        let k_turn = 20usize;
        let mut xt = vec![0.0, 1.0, 0.0, 0.0];
        let mut imm = Imm2D::new(
            vec![
                cv_model_2d(dt, 1e-3, 0.25, 0.0, 0.0, 5.0),
                ct_model_2d(dt, omega, 1e-3, 0.25, 0.0, 0.0, 5.0),
            ],
            vec![vec![0.95, 0.05], vec![0.05, 0.95]],
            vec![0.5, 0.5],
        );
        let mut lone = cv_model_2d(dt, 1e-3, 0.25, 0.0, 0.0, 5.0);
        let mut ct_prob_before = 0.0;
        let (mut imm_err, mut lone_err) = (0.0, 0.0);
        for k in 1..=45
        {
            xt = if k <= k_turn
            {
                advance(&cv_truth, &xt)
            }
            else
            {
                advance(&ct_truth, &xt)
            };
            let z = [xt[POS_X], xt[POS_Y]];
            imm.step(&z);
            lone.step(&z);
            if k == k_turn
            {
                ct_prob_before = imm.mode_probabilities()[1];
            }
            if k > k_turn && k <= k_turn + 15
            {
                let (ix, iy) = imm.position();
                imm_err += (ix - z[0]).hypot(iy - z[1]);
                lone_err += (lone.state()[POS_X] - z[0]).hypot(lone.state()[POS_Y] - z[1]);
            }
        }
        let ct_prob_after = imm.mode_probabilities()[1];
        assert!(
            ct_prob_after > ct_prob_before,
            "CT mode probability should rise on the turn: {ct_prob_before} -> {ct_prob_after}"
        );
        assert!(
            imm_err < lone_err,
            "IMM ({imm_err}) should beat the lone CV filter ({lone_err}) on the turn"
        );
    }

    #[test]
    fn imm2d_mode_probabilities_are_a_valid_distribution() {
        let mut imm = Imm2D::new(
            vec![
                cv_model_2d(1.0, 1e-3, 1.0, 0.0, 0.0, 10.0),
                ct_model_2d(1.0, 0.1, 1e-3, 1.0, 0.0, 0.0, 10.0),
            ],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![0.5, 0.5],
        );
        for k in 1..=25
        {
            imm.step(&[k as f64, 0.5 * k as f64]);
        }
        let mu = imm.mode_probabilities();
        assert_eq!(mu.len(), 2);
        assert!((mu.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(mu.iter().all(|&p| (0.0..=1.0).contains(&p)));
    }

    #[test]
    fn cholesky_solves_a_known_system() {
        // A = [[4,2],[2,3]]; solve A x = [2, 1] ⇒ x = [0.25, 0.166…].
        let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
        let l = cholesky(&a).unwrap();
        let x = chol_solve(&l, &[2.0, 1.0]);
        // Verify by re-multiplying A·x ≈ b.
        let b = mat_vec(&a, &x);
        assert!((b[0] - 2.0).abs() < 1e-12 && (b[1] - 1.0).abs() < 1e-12);
        // A non-positive-definite matrix is rejected.
        assert!(cholesky(&[vec![1.0, 2.0], vec![2.0, 1.0]]).is_none());
    }

    #[test]
    fn imm2d_empty_bank_is_inert() {
        let mut imm = Imm2D::new(Vec::new(), Vec::new(), Vec::new());
        imm.step(&[1.0, 2.0]);
        assert_eq!(imm.position(), (0.0, 0.0));
        assert!(imm.mode_probabilities().is_empty());
    }
}
