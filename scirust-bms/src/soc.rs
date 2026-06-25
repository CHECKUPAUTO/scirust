//! State-of-Charge estimation with an Extended Kalman Filter.
//!
//! A 1-RC equivalent-circuit cell model — state `[SoC, V₁]` (charge + one
//! polarization voltage) — driven by the measured current, with the terminal
//! voltage `V = OCV(SoC) − V₁ − I·R₀` as the (nonlinear) measurement. The EKF
//! fuses Coulomb counting (which drifts) with the voltage curve (which anchors),
//! recovering SoC even from a wrong initial guess. Built on the deterministic
//! [`scirust_estimation::Ekf`], so a run is bit-reproducible.

use scirust_estimation::{Ekf, Mat};
use serde::{Deserialize, Serialize};

/// 1-RC equivalent-circuit cell parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellParams {
    /// Usable capacity in ampere-seconds (`Ah · 3600`).
    pub q_cap: f64,
    /// Ohmic resistance `R₀` (Ω).
    pub r0: f64,
    /// Polarization resistance `R₁` (Ω).
    pub r1: f64,
    /// Polarization capacitance `C₁` (F).
    pub c1: f64,
    /// Open-circuit-voltage polynomial `OCV(s) = a0 + a1·s + a2·s²`.
    pub ocv: [f64; 3],
}

impl CellParams {
    /// Open-circuit voltage at state of charge `s`.
    pub fn ocv(&self, s: f64) -> f64 {
        self.ocv[0] + self.ocv[1] * s + self.ocv[2] * s * s
    }

    /// `dOCV/ds`.
    pub fn docv(&self, s: f64) -> f64 {
        self.ocv[1] + 2.0 * self.ocv[2] * s
    }
}

/// SoC/SoH-oriented EKF over a 1-RC cell model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatteryEkf {
    ekf: Ekf,
    params: CellParams,
}

impl BatteryEkf {
    /// Initialise at an (possibly wrong) initial SoC guess `soc0`.
    pub fn new(params: CellParams, soc0: f64) -> Self {
        let p0 = Mat::diag(&[0.05, 1e-3]); // SoC quite uncertain, V1 near 0
        let q = Mat::diag(&[1e-7, 1e-6]);
        let r = Mat::diag(&[1e-4]); // voltage measurement variance
        let ekf = Ekf::new(vec![soc0, 0.0], p0, q, r);
        Self { ekf, params }
    }

    /// One predict/update with measured `current` (A, +discharge), step `dt`
    /// (s) and measured terminal voltage `v_meas` (V).
    pub fn step(&mut self, current: f64, dt: f64, v_meas: f64) {
        let CellParams {
            q_cap,
            r0,
            r1,
            c1,
            ocv,
        } = self.params.clone();
        let [a0, a1, a2] = ocv;
        let i = current;
        let alpha = (-dt / (r1 * c1)).exp();

        let f = move |x: &[f64]| vec![x[0] - i * dt / q_cap, alpha * x[1] + r1 * (1.0 - alpha) * i];
        let f_jac = move |_x: &[f64]| Mat::new(2, 2, vec![1.0, 0.0, 0.0, alpha]);
        let h = move |x: &[f64]| vec![(a0 + a1 * x[0] + a2 * x[0] * x[0]) - x[1] - i * r0];
        let h_jac = move |x: &[f64]| Mat::new(1, 2, vec![a1 + 2.0 * a2 * x[0], -1.0]);

        self.ekf.predict(f, f_jac);
        self.ekf.update(&[v_meas], h, h_jac);
    }

    /// Current SoC estimate, reported in the physical range `[0, 1]`.
    ///
    /// The filter integrates an unconstrained internal state (clamping the
    /// Kalman state would corrupt the covariance feedback), but a *State of
    /// Charge* is a fraction of capacity and cannot leave `[0, 1]`; e.g.
    /// discharging a coulomb-counting model past empty drives the raw state
    /// negative. The public estimate is therefore saturated to the physical
    /// range.
    pub fn soc(&self) -> f64 {
        self.ekf.state()[0].clamp(0.0, 1.0)
    }

    /// Raw, unclamped internal SoC state (may leave `[0, 1]`).
    ///
    /// Exposed for diagnostics and for consumers that need the coulomb-counting
    /// quantity itself (e.g. an over-/under-charge margin) rather than the
    /// saturated physical SoC.
    pub fn soc_raw(&self) -> f64 {
        self.ekf.state()[0]
    }

    /// Current polarization voltage estimate `V₁`.
    pub fn v1(&self) -> f64 {
        self.ekf.state()[1]
    }

    /// Usable capacity currently assumed by the coulomb-counting model (As).
    pub fn capacity_as(&self) -> f64 {
        self.params.q_cap
    }

    /// Update the assumed usable capacity (As) — used by the dual estimator to
    /// feed the recursive SoH estimate back into SoC tracking.
    pub fn set_capacity(&mut self, q_as: f64) {
        if q_as > 0.0
        {
            self.params.q_cap = q_as;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell() -> CellParams {
        CellParams {
            q_cap: 2.0 * 3600.0, // 2 Ah
            r0: 0.05,
            r1: 0.02,
            c1: 2000.0,
            ocv: [3.3, 0.7, 0.2], // 3.3 V @ empty → 4.2 V @ full
        }
    }

    #[test]
    fn recovers_soc_from_a_wrong_initial_guess() {
        let p = cell();
        // True cell starts at SoC 0.80; EKF is told 0.50.
        let mut true_soc = 0.80;
        let mut true_v1 = 0.0;
        let mut ekf = BatteryEkf::new(p.clone(), 0.50);

        let dt = 1.0;
        let current = 2.0; // 1C discharge
        for _ in 0..600
        {
            // True cell evolution + terminal voltage.
            let alpha = (-dt / (p.r1 * p.c1)).exp();
            true_soc -= current * dt / p.q_cap;
            true_v1 = alpha * true_v1 + p.r1 * (1.0 - alpha) * current;
            let v_term = p.ocv(true_soc) - true_v1 - current * p.r0;
            ekf.step(current, dt, v_term);
        }
        // The voltage anchor pulls the estimate onto the true SoC.
        assert!(
            (ekf.soc() - true_soc).abs() < 0.03,
            "SoC est {} vs true {}",
            ekf.soc(),
            true_soc
        );
    }

    #[test]
    fn run_is_deterministic() {
        let run = || {
            let p = cell();
            let mut ekf = BatteryEkf::new(p.clone(), 0.5);
            let mut s = 0.8;
            let mut v1 = 0.0;
            for _ in 0..100
            {
                let alpha = (-1.0 / (p.r1 * p.c1)).exp();
                s -= 2.0 / p.q_cap;
                v1 = alpha * v1 + p.r1 * (1.0 - alpha) * 2.0;
                ekf.step(2.0, 1.0, p.ocv(s) - v1 - 2.0 * p.r0);
            }
            ekf.soc()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn ocv_polynomial_hits_known_values() {
        // ocv(s) = 3.3 + 0.7·s + 0.2·s²  ⇒ 3.3 @0, 3.7 @0.5, 4.2 @1.
        let p = cell();
        assert!((p.ocv(0.0) - 3.3).abs() < 1e-12);
        assert!((p.ocv(0.5) - 3.7).abs() < 1e-12);
        assert!((p.ocv(1.0) - 4.2).abs() < 1e-12);
    }

    #[test]
    fn docv_is_the_exact_derivative_of_ocv() {
        // d/ds (3.3 + 0.7s + 0.2s²) = 0.7 + 0.4s ⇒ 0.7 @0, 0.9 @0.5, 1.1 @1.
        let p = cell();
        assert!((p.docv(0.0) - 0.7).abs() < 1e-12);
        assert!((p.docv(0.5) - 0.9).abs() < 1e-12);
        assert!((p.docv(1.0) - 1.1).abs() < 1e-12);
        // Cross-check against a central finite difference of ocv itself.
        let h = 1e-6;
        for &s in &[0.1, 0.37, 0.8]
        {
            let fd = (p.ocv(s + h) - p.ocv(s - h)) / (2.0 * h);
            assert!(
                (p.docv(s) - fd).abs() < 1e-6,
                "docv({s}) {} vs FD {fd}",
                p.docv(s)
            );
        }
    }

    #[test]
    fn coulomb_count_is_exact_when_measurement_agrees() {
        // Feed v_meas EXACTLY equal to h(x_after_predict): the innovation is 0,
        // the Kalman correction vanishes, and soc() must equal the pure coulomb
        // count  soc0 − I·dt/Q  to machine precision.
        let p = cell();
        let (soc0, i, dt) = (0.80, 2.0, 1.0);
        let alpha = (-dt / (p.r1 * p.c1)).exp();
        let soc_pred = soc0 - i * dt / p.q_cap; // = 0.8 − 2/7200
        let v1_pred = alpha * 0.0 + p.r1 * (1.0 - alpha) * i;
        let v_match = p.ocv(soc_pred) - v1_pred - i * p.r0;

        let mut ekf = BatteryEkf::new(p.clone(), soc0);
        ekf.step(i, dt, v_match);

        assert!(
            (ekf.soc_raw() - soc_pred).abs() < 1e-12,
            "soc {} vs {soc_pred}",
            ekf.soc_raw()
        );
        assert!(
            (ekf.v1() - v1_pred).abs() < 1e-12,
            "v1 {} vs {v1_pred}",
            ekf.v1()
        );
    }

    #[test]
    fn charge_count_raises_soc_discharge_lowers_it() {
        // Sign check: one discharge step lowers SoC, one charge step raises it,
        // each by exactly |I|·dt/Q under a zero-innovation measurement.
        let p = cell();
        let dt = 1.0;
        let expect = 2.0 * dt / p.q_cap; // 2/7200

        // Discharge (+I): soc falls.
        let mut ekf = BatteryEkf::new(p.clone(), 0.5);
        let a = (-dt / (p.r1 * p.c1)).exp();
        let sp = 0.5 - 2.0 * dt / p.q_cap;
        let vp = p.r1 * (1.0 - a) * 2.0;
        ekf.step(2.0, dt, p.ocv(sp) - vp - 2.0 * p.r0);
        assert!((ekf.soc_raw() - (0.5 - expect)).abs() < 1e-12);

        // Charge (−I): soc rises by the same magnitude.
        let mut ekf2 = BatteryEkf::new(p.clone(), 0.5);
        let sp2 = 0.5 - (-2.0) * dt / p.q_cap;
        let vp2 = p.r1 * (1.0 - a) * (-2.0);
        ekf2.step(-2.0, dt, p.ocv(sp2) - vp2 - (-2.0) * p.r0);
        assert!((ekf2.soc_raw() - (0.5 + expect)).abs() < 1e-12);
    }

    #[test]
    fn reported_soc_is_clamped_to_unit_interval() {
        let p = cell();
        let dt = 1.0;

        // Over-discharge far past empty with zero-innovation measurements: the
        // raw coulomb state goes negative, but the reported SoC saturates at 0.
        let mut ekf = BatteryEkf::new(p.clone(), 0.05);
        let a = (-dt / (p.r1 * p.c1)).exp();
        let (mut hs, mut hv) = (0.05, 0.0);
        for _ in 0..1000
        {
            hs -= 2.0 * dt / p.q_cap;
            hv = a * hv + p.r1 * (1.0 - a) * 2.0;
            ekf.step(2.0, dt, p.ocv(hs) - hv - 2.0 * p.r0);
        }
        assert!(
            ekf.soc_raw() < 0.0,
            "raw state should be negative: {}",
            ekf.soc_raw()
        );
        assert_eq!(ekf.soc(), 0.0, "reported SoC must clamp to 0");

        // Over-charge far past full: raw state exceeds 1, reported SoC clamps to 1.
        let mut ekf2 = BatteryEkf::new(p.clone(), 0.95);
        let (mut hs2, mut hv2) = (0.95, 0.0);
        for _ in 0..1000
        {
            hs2 -= (-2.0) * dt / p.q_cap;
            hv2 = a * hv2 + p.r1 * (1.0 - a) * (-2.0);
            ekf2.step(-2.0, dt, p.ocv(hs2) - hv2 - (-2.0) * p.r0);
        }
        assert!(
            ekf2.soc_raw() > 1.0,
            "raw state should exceed 1: {}",
            ekf2.soc_raw()
        );
        assert_eq!(ekf2.soc(), 1.0, "reported SoC must clamp to 1");
    }

    #[test]
    fn capacity_accessor_and_guarded_setter() {
        let p = cell();
        let mut ekf = BatteryEkf::new(p.clone(), 0.5);
        assert!((ekf.capacity_as() - 2.0 * 3600.0).abs() < 1e-9);

        // A valid positive capacity is accepted.
        ekf.set_capacity(1.6 * 3600.0);
        assert!((ekf.capacity_as() - 1.6 * 3600.0).abs() < 1e-9);

        // Non-physical capacities are rejected (the previous value is kept).
        ekf.set_capacity(0.0);
        ekf.set_capacity(-100.0);
        assert!((ekf.capacity_as() - 1.6 * 3600.0).abs() < 1e-9);
    }

    #[test]
    fn smaller_capacity_means_faster_soc_drop() {
        // With a smaller usable capacity, the same charge moves SoC more: the
        // coulomb step I·dt/Q grows as Q shrinks. Zero-innovation feed isolates
        // the predict, so the comparison is exact.
        let p = cell();
        let dt = 1.0;
        let i = 2.0;

        let mut big = BatteryEkf::new(p.clone(), 0.8); // Q = 7200 As
        let mut small = BatteryEkf::new(p.clone(), 0.8);
        small.set_capacity(3600.0); // Q = 3600 As

        for ekf in [&mut big, &mut small]
        {
            let q = ekf.capacity_as();
            let a = (-dt / (p.r1 * p.c1)).exp();
            let sp = ekf.soc_raw() - i * dt / q;
            let vp = a * ekf.v1() + p.r1 * (1.0 - a) * i;
            ekf.step(i, dt, p.ocv(sp) - vp - i * p.r0);
        }
        let drop_big = 0.8 - big.soc_raw();
        let drop_small = 0.8 - small.soc_raw();
        assert!((drop_big - i * dt / 7200.0).abs() < 1e-12);
        assert!((drop_small - i * dt / 3600.0).abs() < 1e-12);
        assert!(
            (drop_small - 2.0 * drop_big).abs() < 1e-12,
            "half capacity ⇒ double drop"
        );
    }

    #[test]
    fn voltage_anchor_converges_when_charging_from_too_high_a_guess() {
        // The companion to `recovers_soc_from_a_wrong_initial_guess` for the
        // opposite current sign and the opposite initial error: a CHARGING cell
        // (I<0) whose EKF is told a SoC that is too HIGH. A wrong sign anywhere
        // in the measurement anchor would make this diverge instead of converge.
        let p = cell();
        let mut true_soc = 0.30;
        let mut true_v1 = 0.0;
        let mut ekf = BatteryEkf::new(p.clone(), 0.70); // told too high
        let dt = 1.0;
        let i = -2.0; // charge
        for _ in 0..500
        {
            let a = (-dt / (p.r1 * p.c1)).exp();
            true_soc -= i * dt / p.q_cap; // charging raises SoC
            true_v1 = a * true_v1 + p.r1 * (1.0 - a) * i;
            ekf.step(i, dt, p.ocv(true_soc) - true_v1 - i * p.r0);
        }
        assert!(
            (ekf.soc() - true_soc).abs() < 0.03,
            "charge anchor failed: est {} vs true {true_soc}",
            ekf.soc()
        );
    }
}
