//! Extended Kalman tracking from polar (range/bearing) radar measurements.
//!
//! The trackers in [`super::imm2d`] assume the measurement is already the
//! Cartesian position `(x, y)`. A real radar reports **range and bearing** — a
//! *polar* measurement, a nonlinear function of the Cartesian state — so a plain
//! linear Kalman filter does not apply. The **extended Kalman filter** (EKF)
//! bridges this: it keeps the motion model linear (a Cartesian
//! constant-velocity state, so the *prediction* is exact) but linearises the
//! nonlinear measurement `h(x) = (√(x²+y²), atan2(y, x))` about the current
//! estimate via its Jacobian for the correction. The result is a filter that
//! tracks a target in clean Cartesian coordinates directly from the raw polar
//! returns, with the bearing innovation wrapped so a target crossing the ±π
//! azimuth boundary is handled without a discontinuity.
//!
//! Built on the shared dense-matrix helpers from [`super::imm2d`];
//! dependency-free.

// Dense small-matrix linear algebra; indexed sweeps are the algorithm.
#![allow(clippy::needless_range_loop)]

use super::imm2d::{chol_solve, cholesky, mat_add, mat_mul, mat_t, mat_vec};
use std::f64::consts::PI;

const POS_X: usize = 0;
const POS_Y: usize = 2;

/// Wrap an angle to `(−π, π]` so a bearing innovation never jumps by `2π`.
fn wrap_pi(a: f64) -> f64 {
    let two_pi = 2.0 * PI;
    let w = (a + PI).rem_euclid(two_pi) - PI;
    // rem_euclid maps to [−π, π); nudge the −π endpoint to +π for symmetry.
    if w <= -PI { w + two_pi } else { w }
}

/// An extended Kalman filter tracking a target in the Cartesian state
/// `[x, vₓ, y, v_y]` from polar **range/bearing** measurements.
///
/// The prediction is a linear constant-velocity step (`x ← F·x`,
/// `P ← F·P·Fᵀ + Q`); the [`update`](Self::update) linearises the polar
/// observation `h(x) = (range, bearing)` about the current state.
#[derive(Debug, Clone, PartialEq)]
pub struct RadarEkf {
    x: Vec<f64>,
    p: Vec<Vec<f64>>,
    f: Vec<Vec<f64>>,
    q: Vec<Vec<f64>>,
}

impl RadarEkf {
    /// A tracker at frame interval `dt` and process-noise intensity `q`,
    /// initialised at Cartesian position `(x0, y0)` with zero velocity and
    /// isotropic covariance `var0`.
    pub fn new(dt: f64, q: f64, x0: f64, y0: f64, var0: f64) -> Self {
        let f = vec![
            vec![1.0, dt, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, dt],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        // Continuous-white-noise-acceleration process noise, per axis.
        let (qa, qb, qd) = (q * dt * dt * dt / 3.0, q * dt * dt / 2.0, q * dt);
        let qm = vec![
            vec![qa, qb, 0.0, 0.0],
            vec![qb, qd, 0.0, 0.0],
            vec![0.0, 0.0, qa, qb],
            vec![0.0, 0.0, qb, qd],
        ];
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
        }
    }

    /// The predicted polar measurement `h(x) = (range, bearing)` for the current
    /// state.
    pub fn predicted_measurement(&self) -> (f64, f64) {
        let (px, py) = (self.x[POS_X], self.x[POS_Y]);
        (px.hypot(py), py.atan2(px))
    }

    /// Time update: linear constant-velocity prediction of state and covariance.
    pub fn predict(&mut self) {
        self.x = mat_vec(&self.f, &self.x);
        let fp = mat_mul(&self.f, &self.p);
        self.p = mat_add(&mat_mul(&fp, &mat_t(&self.f)), &self.q);
    }

    /// Measurement update with a polar return `(range, bearing)` and the
    /// respective measurement variances. Linearises `h` about the current state
    /// via its Jacobian; the bearing residual is wrapped to `(−π, π]`. Returns
    /// the Gaussian innovation likelihood. If the target is at (or numerically
    /// at) the origin the bearing is undefined, so the state is left unchanged
    /// and `0.0` is returned.
    pub fn update(&mut self, range: f64, bearing: f64, range_var: f64, bearing_var: f64) -> f64 {
        let (px, py) = (self.x[POS_X], self.x[POS_Y]);
        let r2 = px * px + py * py;
        let r = r2.sqrt();
        if r < 1e-12
        {
            return 0.0;
        }
        // Jacobian H = ∂h/∂x (2×4), columns [x, vₓ, y, v_y].
        let h = vec![
            vec![px / r, 0.0, py / r, 0.0],
            vec![-py / r2, 0.0, px / r2, 0.0],
        ];
        // Innovation, bearing wrapped.
        let y = [range - r, wrap_pi(bearing - py.atan2(px))];
        let ht = mat_t(&h);
        // S = H·P·Hᵀ + R.
        let mut s = mat_mul(&mat_mul(&h, &self.p), &ht);
        s[0][0] += range_var;
        s[1][1] += bearing_var;
        let l = match cholesky(&s)
        {
            Some(l) => l,
            None => return 0.0,
        };
        // Gain K = P·Hᵀ·S⁻¹, solved row-wise.
        let pht = mat_mul(&self.p, &ht);
        let mut k = vec![vec![0.0; 2]; 4];
        for i in 0..4
        {
            k[i] = chol_solve(&l, &pht[i]);
        }
        for i in 0..4
        {
            self.x[i] += k[i][0] * y[0] + k[i][1] * y[1];
        }
        // P ← (I − K·H)·P, symmetrised.
        let kh = mat_mul(&k, &h);
        let mut imkh = kh;
        for i in 0..4
        {
            for j in 0..4
            {
                let ident = if i == j { 1.0 } else { 0.0 };
                imkh[i][j] = ident - imkh[i][j];
            }
        }
        let np = mat_mul(&imkh, &self.p);
        for i in 0..4
        {
            for j in 0..4
            {
                self.p[i][j] = 0.5 * (np[i][j] + np[j][i]);
            }
        }
        let sinv_y = chol_solve(&l, &y);
        let quad = y[0] * sinv_y[0] + y[1] * sinv_y[1];
        let det = l[0][0] * l[0][0] * l[1][1] * l[1][1];
        (-0.5 * quad).exp() / ((2.0 * PI).powi(2) * det).sqrt()
    }

    /// Predict then update with a polar return in one frame.
    pub fn step(&mut self, range: f64, bearing: f64, range_var: f64, bearing_var: f64) -> f64 {
        self.predict();
        self.update(range, bearing, range_var, bearing_var)
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

    /// The current position-estimate variance `Pₓₓ + P_yy` (a scalar spread).
    pub fn position_variance(&self) -> f64 {
        self.p[POS_X][POS_X] + self.p[POS_Y][POS_Y]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The true polar measurement of a Cartesian point.
    fn polar(x: f64, y: f64) -> (f64, f64) {
        (x.hypot(y), y.atan2(x))
    }

    #[test]
    fn wrap_pi_maps_into_the_principal_interval() {
        assert!((wrap_pi(0.3) - 0.3).abs() < 1e-12);
        assert!((wrap_pi(PI + 0.1) - (-PI + 0.1)).abs() < 1e-12);
        assert!((wrap_pi(-PI - 0.1) - (PI - 0.1)).abs() < 1e-12);
        // Small differences straddling ±π stay small after wrapping.
        let d = wrap_pi((PI - 0.05) - (-(PI - 0.05)));
        assert!(d.abs() < 0.11, "straddle diff {d}");
    }

    #[test]
    fn ekf_recovers_a_cartesian_track_from_polar_measurements() {
        // Straight-line target; feed it exact range/bearing and check the filter
        // converges to the Cartesian truth.
        let (dt, x0, y0, vx, vy) = (1.0_f64, 40.0_f64, 5.0_f64, -0.6_f64, 1.2_f64);
        let mut ekf = RadarEkf::new(dt, 1e-3, 38.0, 6.0, 25.0);
        let (r_var, b_var) = (1e-4, 1e-8);
        for k in 1..=80
        {
            let (tx, ty) = (x0 + vx * k as f64 * dt, y0 + vy * k as f64 * dt);
            let (r, b) = polar(tx, ty);
            ekf.step(r, b, r_var, b_var);
        }
        let (ex, ey) = ekf.position();
        let (tx, ty) = (x0 + vx * 80.0 * dt, y0 + vy * 80.0 * dt);
        assert!(
            (ex - tx).abs() < 0.2 && (ey - ty).abs() < 0.2,
            "pos ({ex},{ey}) vs ({tx},{ty})"
        );
        let (evx, evy) = ekf.velocity();
        assert!(
            (evx - vx).abs() < 0.1 && (evy - vy).abs() < 0.1,
            "vel ({evx},{evy})"
        );
    }

    #[test]
    fn ekf_tracks_across_the_bearing_wrap() {
        // Target moving along the −x axis (bearing near ±π), crossing the wrap.
        // With correct innovation wrapping the track stays glued to the truth.
        let dt = 1.0_f64;
        let (x0, y0, vx, vy) = (-30.0_f64, 3.0_f64, -0.4_f64, -0.5_f64);
        let mut ekf = RadarEkf::new(dt, 1e-3, -28.0, 2.0, 20.0);
        let mut max_err: f64 = 0.0;
        for k in 1..=60
        {
            let (tx, ty) = (x0 + vx * k as f64 * dt, y0 + vy * k as f64 * dt);
            let (r, b) = polar(tx, ty);
            ekf.step(r, b, 1e-4, 1e-8);
            if k > 30
            {
                let (ex, ey) = ekf.position();
                max_err = max_err.max((ex - tx).hypot(ey - ty));
            }
        }
        assert!(max_err < 0.3, "bearing-wrap tracking error {max_err}");
    }

    #[test]
    fn ekf_update_shrinks_position_variance() {
        let mut ekf = RadarEkf::new(1.0, 0.5, 20.0, 20.0, 100.0);
        ekf.predict();
        let before = ekf.position_variance();
        let (r, b) = polar(20.0, 20.0);
        ekf.update(r, b, 0.1, 1e-4);
        assert!(
            ekf.position_variance() < before,
            "update should reduce variance"
        );
    }

    #[test]
    fn ekf_predicted_measurement_matches_state() {
        let ekf = RadarEkf::new(1.0, 0.1, 3.0, 4.0, 1.0);
        let (r, b) = ekf.predicted_measurement();
        assert!((r - 5.0).abs() < 1e-12, "range {r}");
        assert!((b - (4.0_f64).atan2(3.0)).abs() < 1e-12, "bearing {b}");
    }

    #[test]
    fn ekf_update_at_the_origin_is_inert() {
        let mut ekf = RadarEkf::new(1.0, 0.1, 0.0, 0.0, 1.0);
        let before = ekf.state().to_vec();
        let like = ekf.update(1.0, 0.0, 0.1, 0.1);
        assert_eq!(like, 0.0);
        assert_eq!(ekf.state(), &before[..]);
    }
}
