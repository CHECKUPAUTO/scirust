//! Loosely-coupled GNSS/INS fusion (total-state Kalman filter).
//!
//! The IMU is fast but drifts; GNSS is slow but absolute. Loosely-coupled
//! fusion runs a constant-velocity Kalman filter whose **prediction** is driven
//! by IMU acceleration (a control input) and whose **correction** is the GNSS
//! position fix. Between fixes — under a bridge, in an urban canyon — the filter
//! dead-reckons on the IMU alone and its covariance grows; when GNSS returns,
//! the fix pulls the estimate back and shrinks the covariance.
//!
//! State is `[pₓ, p_y, vₓ, v_y]` in a local tangent frame. Process noise comes
//! from an acceleration-uncertainty `σ_a`; GNSS noise from a position `σ_gnss`.

use scirust_estimation::Mat;

/// A loosely-coupled GNSS/INS fusion filter.
#[derive(Debug, Clone)]
pub struct GnssInsFusion {
    /// State `[pₓ, p_y, vₓ, v_y]`.
    pub x: Vec<f64>,
    p: Mat,
    sigma_a: f64,
    sigma_gnss: f64,
}

impl GnssInsFusion {
    /// New filter from initial position/velocity, acceleration process-noise
    /// std `sigma_a` (m/s²), GNSS position std `sigma_gnss` (m), and initial
    /// per-state standard deviations `p0_std` (`[pₓ, p_y, vₓ, v_y]`).
    pub fn new(
        pos: [f64; 2],
        vel: [f64; 2],
        sigma_a: f64,
        sigma_gnss: f64,
        p0_std: [f64; 4],
    ) -> Self {
        let x = vec![pos[0], pos[1], vel[0], vel[1]];
        let p = Mat::diag(&p0_std.map(|s| s * s));
        Self {
            x,
            p,
            sigma_a,
            sigma_gnss,
        }
    }

    /// IMU-driven prediction over `dt` seconds with nav-frame acceleration
    /// `accel` `[aₓ, a_y]`.
    pub fn predict(&mut self, accel: [f64; 2], dt: f64) {
        // Constant-velocity transition with acceleration as a control input.
        #[rustfmt::skip]
        let f = Mat::new(4, 4, vec![
            1.0, 0.0, dt,  0.0,
            0.0, 1.0, 0.0, dt,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]);
        let half = 0.5 * dt * dt;
        #[rustfmt::skip]
        let b = Mat::new(4, 2, vec![
            half, 0.0,
            0.0,  half,
            dt,   0.0,
            0.0,  dt,
        ]);
        let fx = f.matvec(&self.x);
        let bu = b.matvec(&accel);
        self.x = fx.iter().zip(&bu).map(|(a, c)| a + c).collect();

        // Q from the acceleration random walk: per axis
        // [[dt⁴/4, dt³/2],[dt³/2, dt²]]·σ_a².
        let sa2 = self.sigma_a * self.sigma_a;
        let q_pp = sa2 * dt.powi(4) / 4.0;
        let q_pv = sa2 * dt.powi(3) / 2.0;
        let q_vv = sa2 * dt * dt;
        let mut q = Mat::zeros(4, 4);
        // x axis: p=0, v=2.
        q.set(0, 0, q_pp);
        q.set(0, 2, q_pv);
        q.set(2, 0, q_pv);
        q.set(2, 2, q_vv);
        // y axis: p=1, v=3.
        q.set(1, 1, q_pp);
        q.set(1, 3, q_pv);
        q.set(3, 1, q_pv);
        q.set(3, 3, q_vv);

        self.p = f.matmul(&self.p).matmul(&f.t()).add(&q);
    }

    /// Correct with a GNSS position fix `pos` `[pₓ, p_y]`. Returns `false` if the
    /// innovation covariance is singular (correction skipped).
    pub fn update_gnss(&mut self, pos: [f64; 2]) -> bool {
        #[rustfmt::skip]
        let h = Mat::new(2, 4, vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
        ]);
        let r = Mat::diag(&[self.sigma_gnss * self.sigma_gnss; 2]);
        let y = [pos[0] - self.x[0], pos[1] - self.x[1]];
        let ht = h.t();
        let s = h.matmul(&self.p).matmul(&ht).add(&r);
        let s_inv = match s.inverse()
        {
            Some(m) => m,
            None => return false,
        };
        let k = self.p.matmul(&ht).matmul(&s_inv); // 4×2
        let ky = k.matvec(&y);
        for (xi, d) in self.x.iter_mut().zip(&ky)
        {
            *xi += d;
        }
        // P = (I − K H) P (I − K H)ᵀ + K R Kᵀ — the Joseph form. Algebraically
        // equivalent to the shorter (I − K H) P, but the short form is only
        // guaranteed symmetric/PSD for an exactly optimal K; the Joseph form
        // stays symmetric and PSD by construction regardless of rounding
        // (Bucy & Joseph 1968; Grewal & Andrews, Kalman Filtering, §6.3.4) —
        // the standard choice for a navigation filter meant to run unattended
        // for long stretches (dead-reckoning under a bridge, an urban canyon).
        let kh = k.matmul(&h);
        let i_kh = Mat::identity(4).sub(&kh);
        let i_kh_t = i_kh.t();
        let krk_t = k.matmul(&r).matmul(&k.t());
        self.p = i_kh.matmul(&self.p).matmul(&i_kh_t).add(&krk_t);
        true
    }

    /// Current position estimate `[pₓ, p_y]`.
    pub fn position(&self) -> [f64; 2] {
        [self.x[0], self.x[1]]
    }

    /// Current velocity estimate `[vₓ, v_y]`.
    pub fn velocity(&self) -> [f64; 2] {
        [self.x[2], self.x[3]]
    }

    /// Position uncertainty as the RMS of the two position standard deviations.
    pub fn position_uncertainty(&self) -> f64 {
        ((self.p.get(0, 0) + self.p.get(1, 1)) / 2.0).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ins::Ins2d;

    // Deterministic zero-mean noise in [-0.5, 0.5).
    struct Rng {
        s: u64,
    }
    impl Rng {
        fn new(seed: u64) -> Self {
            Self { s: seed }
        }
        fn unit(&mut self) -> f64 {
            self.s = self.s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 11) as f64 + 0.5) / ((1u64 << 53) as f64) - 0.5
        }
    }

    #[test]
    fn fusion_beats_ins_only_dead_reckoning() {
        // Truth: a gentle S-curve. The IMU has a constant acceleration bias, so
        // INS-only drifts; GNSS fixes (every 10 steps) should keep fusion tight.
        let dt = 0.1;
        let n = 600;
        let accel_bias = [0.02, -0.015];
        let mut rng = Rng::new(0xA1D);

        let mut fusion = GnssInsFusion::new([0.0, 0.0], [1.0, 0.0], 0.3, 1.5, [2.0, 2.0, 1.0, 1.0]);
        let mut ins = Ins2d::new([0.0, 0.0], [1.0, 0.0]);

        // Truth state integrated with the *true* acceleration.
        let (mut tp, mut tv) = ([0.0, 0.0], [1.0, 0.0]);
        let mut fusion_sq = 0.0;
        let mut ins_sq = 0.0;
        for k in 0..n
        {
            // True manoeuvring acceleration.
            let t = k as f64 * dt;
            let a_true = [0.2 * (0.3 * t).cos(), 0.15 * (0.2 * t).sin()];
            // The IMU reads truth + a fixed bias (what both dead-reckoners see).
            let a_imu = [a_true[0] + accel_bias[0], a_true[1] + accel_bias[1]];

            // Advance truth.
            for i in 0..2
            {
                tp[i] += tv[i] * dt + 0.5 * a_true[i] * dt * dt;
                tv[i] += a_true[i] * dt;
            }
            // Both estimators propagate on the biased IMU.
            fusion.predict(a_imu, dt);
            ins.propagate(a_imu, dt);
            // GNSS fix every 10 steps (1 Hz), noisy.
            if k % 10 == 0
            {
                let fix = [tp[0] + 1.5 * rng.unit(), tp[1] + 1.5 * rng.unit()];
                fusion.update_gnss(fix);
            }
            // Accumulate squared position error after a short settling window.
            if k > 100
            {
                let fp = fusion.position();
                fusion_sq += (fp[0] - tp[0]).powi(2) + (fp[1] - tp[1]).powi(2);
                ins_sq += (ins.pos[0] - tp[0]).powi(2) + (ins.pos[1] - tp[1]).powi(2);
            }
        }
        let fusion_rmse = (fusion_sq / (n - 100) as f64).sqrt();
        let ins_rmse = (ins_sq / (n - 100) as f64).sqrt();
        assert!(
            fusion_rmse < 0.25 * ins_rmse,
            "fusion {fusion_rmse} should be far tighter than INS {ins_rmse}"
        );
    }

    #[test]
    fn uncertainty_grows_during_outage_and_shrinks_on_fix() {
        let dt = 0.1;
        let mut fusion = GnssInsFusion::new([0.0, 0.0], [1.0, 0.0], 0.3, 1.0, [1.0, 1.0, 0.5, 0.5]);
        // Settle with a few fixes.
        for _ in 0..20
        {
            fusion.predict([0.0, 0.0], dt);
            fusion.update_gnss([0.0, 0.0]);
        }
        let settled = fusion.position_uncertainty();
        // GNSS outage: predict only.
        for _ in 0..50
        {
            fusion.predict([0.0, 0.0], dt);
        }
        let outage = fusion.position_uncertainty();
        assert!(outage > settled, "uncertainty must grow in outage");
        // A single fix shrinks it again.
        fusion.update_gnss([0.0, 0.0]);
        let recovered = fusion.position_uncertainty();
        assert!(recovered < outage, "a fix must shrink uncertainty");
    }

    #[test]
    fn predict_propagates_state_and_grows_covariance_by_the_exact_q() {
        // Isolate the prediction. Pick σ_a = 2, dt = 1 so the discrete white-
        // noise-acceleration block is clean integers per axis:
        //   Q_axis = σ_a²·[[dt⁴/4, dt³/2],[dt³/2, dt²]] = 4·[[¼,½],[½,1]]
        //          = [[1, 2],[2, 4]].
        // Start from P = 0 so after one step P holds exactly Q. The mean moves by
        // the constant-velocity transition plus ½·a·dt² control: from
        // x=[0,0,1,−2] with a=[3,−1] over dt=1 ⇒ p=[0+1+1.5, 0−2−0.5]=[2.5,−2.5],
        // v=[1+3, −2−1]=[4,−3].
        let mut f = GnssInsFusion::new([0.0, 0.0], [1.0, -2.0], 2.0, 1.0, [0.0, 0.0, 0.0, 0.0]);
        f.predict([3.0, -1.0], 1.0);
        let p = f.position();
        let v = f.velocity();
        assert!(
            (p[0] - 2.5).abs() < 1e-12 && (p[1] + 2.5).abs() < 1e-12,
            "pos {p:?}"
        );
        assert!(
            (v[0] - 4.0).abs() < 1e-12 && (v[1] + 3.0).abs() < 1e-12,
            "vel {v:?}"
        );
        // Covariance now equals Q exactly: position variance 1, velocity 4, RMS
        // position std = √((1+1)/2) = 1.
        assert!(
            (f.position_uncertainty() - 1.0).abs() < 1e-12,
            "unc {}",
            f.position_uncertainty()
        );
        assert!((f.p.get(0, 0) - 1.0).abs() < 1e-12, "Pxx {}", f.p.get(0, 0));
        assert!(
            (f.p.get(2, 2) - 4.0).abs() < 1e-12,
            "Pvxvx {}",
            f.p.get(2, 2)
        );
        assert!(
            (f.p.get(0, 2) - 2.0).abs() < 1e-12,
            "Pxvx {}",
            f.p.get(0, 2)
        );
        assert!(
            (f.p.get(2, 0) - 2.0).abs() < 1e-12,
            "symmetry {}",
            f.p.get(2, 0)
        );
    }

    #[test]
    fn noise_free_constant_velocity_fixes_track_the_truth_exactly() {
        // A true constant-velocity body (no acceleration) with perfect, noise-free
        // GNSS fixes. A correct loosely-coupled filter must converge to and then
        // sit *on* the truth — both position and the (only indirectly observed)
        // velocity. After 40 steps of dt=0.5 the body is at (40,20) moving (2,1).
        let dt = 0.5;
        let mut f = GnssInsFusion::new([0.0, 0.0], [2.0, 1.0], 0.1, 0.5, [2.0, 2.0, 2.0, 2.0]);
        let tv = [2.0, 1.0];
        let mut tp = [0.0, 0.0];
        for _ in 0..40
        {
            f.predict([0.0, 0.0], dt);
            tp[0] += tv[0] * dt;
            tp[1] += tv[1] * dt;
            f.update_gnss(tp);
        }
        let p = f.position();
        let v = f.velocity();
        assert!(
            (p[0] - tp[0]).abs() < 1e-6 && (p[1] - tp[1]).abs() < 1e-6,
            "pos {p:?} vs {tp:?}"
        );
        assert!(
            (v[0] - tv[0]).abs() < 1e-6 && (v[1] - tv[1]).abs() < 1e-6,
            "vel {v:?} vs {tv:?}"
        );
    }

    #[test]
    fn a_near_perfect_fix_snaps_position_and_corrects_velocity_via_cross_covariance() {
        // One GNSS update with a tiny σ_gnss. The position should jump essentially
        // onto the fix. Crucially, the *velocity* must also move, pulled through
        // the position–velocity cross-covariance: with K = P·Hᵀ(H·P·Hᵀ)⁻¹ and a
        // near-zero R, the velocity gain is P_pv/P_pp, so
        //   Δv = (P_pv/P_pp)·innovation.
        // Construct a known cross term with one noise-free predict. Starting from
        // the diagonal P₀ = diag(4,4,4,4) (std 2 on every state) with σ_a = 0
        // (so Q = 0), the prediction P ← F·P₀·Fᵀ with F coupling p ← v·dt gives,
        // per axis:  P_pp = 4 + 4·dt², P_pv = 4·dt, P_vv = 4.
        // At dt = 0.5 that is P_pp = 5, P_pv = 2, P_vv = 4.
        let dt = 0.5;
        let mut f = GnssInsFusion::new([0.0, 0.0], [0.0, 0.0], 0.0, 1e-4, [2.0, 2.0, 2.0, 2.0]);
        f.predict([0.0, 0.0], dt);
        // Confirm the constructed covariance matches the hand value before the fix.
        assert!((f.p.get(0, 0) - 5.0).abs() < 1e-12, "Ppp {}", f.p.get(0, 0));
        assert!((f.p.get(0, 2) - 2.0).abs() < 1e-12, "Ppv {}", f.p.get(0, 2));
        let fix = [3.0, -5.0];
        assert!(f.update_gnss(fix));
        let p = f.position();
        let v = f.velocity();
        // Position snaps to the fix (R → 0).
        assert!(
            (p[0] - 3.0).abs() < 1e-3 && (p[1] + 5.0).abs() < 1e-3,
            "pos {p:?}"
        );
        // Velocity correction = (P_pv/P_pp)·innovation = (2/5)·fix.
        assert!((v[0] - 0.4 * 3.0).abs() < 1e-3, "vx {}", v[0]);
        assert!((v[1] - 0.4 * -5.0).abs() < 1e-3, "vy {}", v[1]);
    }

    #[test]
    fn velocity_is_recovered_from_position_fixes_alone() {
        // GNSS measures position only (H has no velocity row), yet a constant-
        // velocity body's speed is observable through the sequence of fixes. Start
        // with a deliberately wrong velocity (0,0) and a large velocity prior;
        // noise-free position fixes of a body moving (2,−1) must pull the velocity
        // estimate to the truth.
        let dt = 0.5;
        let mut f = GnssInsFusion::new(
            [0.0, 0.0],
            [0.0, 0.0],
            0.05,
            0.2,
            [1.0, 1.0, 10.0_f64.sqrt(), 10.0_f64.sqrt()],
        );
        let tv = [2.0, -1.0];
        let mut tp = [0.0, 0.0];
        for _ in 0..60
        {
            f.predict([0.0, 0.0], dt);
            tp[0] += tv[0] * dt;
            tp[1] += tv[1] * dt;
            f.update_gnss(tp);
        }
        let v = f.velocity();
        assert!((v[0] - tv[0]).abs() < 1e-3, "vx {} should approach 2", v[0]);
        assert!(
            (v[1] - tv[1]).abs() < 1e-3,
            "vy {} should approach -1",
            v[1]
        );
    }

    #[test]
    fn joseph_form_keeps_covariance_symmetric_and_psd_under_confident_fixes() {
        // Invariant test for a P1 audit finding: the Kalman covariance
        // update now uses the Joseph form (I − K H) P (I − K H)ᵀ + K R Kᵀ,
        // which is symmetric and PSD by construction regardless of
        // finite-precision rounding — unlike the algebraically equivalent
        // but only conditionally-safe short form (I − K H) P this filter
        // used before (Bucy & Joseph 1968; Grewal & Andrews, Kalman
        // Filtering, §6.3.4). Exercised under repeated, very confident GNSS
        // fixes (tiny sigma_gnss), the stiffest regime for this filter.
        let mut fusion =
            GnssInsFusion::new([0.0, 0.0], [1.0, 0.0], 0.3, 1e-4, [2.0, 2.0, 1.0, 1.0]);
        for k in 0..500
        {
            fusion.predict([0.01, -0.01], 0.1);
            let ok = fusion.update_gnss([k as f64 * 0.1, -(k as f64) * 0.05]);
            assert!(ok, "update_gnss unexpectedly rejected at step {k}");
            let p = &fusion.p;
            for i in 0..4
            {
                assert!(
                    p.get(i, i) >= -1e-9,
                    "negative variance at step {k}, index {i}: {}",
                    p.get(i, i)
                );
            }
            for i in 0..4
            {
                for j in (i + 1)..4
                {
                    let asym = (p.get(i, j) - p.get(j, i)).abs();
                    assert!(
                        asym < 1e-6,
                        "asymmetric covariance at step {k}, ({i},{j}): {asym:e}"
                    );
                }
            }
        }
    }

    #[test]
    fn singular_innovation_covariance_is_rejected_without_mutating_state() {
        // If GNSS noise is zero *and* the predicted position variance is zero, the
        // innovation covariance S = H P Hᵀ + R is exactly zero and not invertible.
        // update_gnss must report false and leave the state untouched (no NaNs
        // from a division by a singular S).
        let mut f = GnssInsFusion::new([1.0, 2.0], [3.0, 4.0], 0.0, 0.0, [0.0, 0.0, 1.0, 1.0]);
        let before = f.x.clone();
        let ok = f.update_gnss([99.0, -99.0]);
        assert!(!ok, "singular S must be rejected");
        assert_eq!(
            f.x, before,
            "state must be unchanged when the update is skipped"
        );
    }
}
