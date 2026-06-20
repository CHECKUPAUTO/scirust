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

    /// Current SoC estimate.
    pub fn soc(&self) -> f64 {
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
}
