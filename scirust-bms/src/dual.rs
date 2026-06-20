//! Dual SoC + capacity (SoH) estimation.
//!
//! A SoC EKF that uses a *fixed* rated capacity drifts as the cell ages, because
//! its coulomb-counting term assumes a capacity the cell no longer has. The dual
//! estimator runs the [`BatteryEkf`] alongside the
//! [`RlsCapacity`]: each completed charge/discharge
//! segment is one capacity measurement (`Q = charge / ΔSoC`, with `ΔSoC` from the
//! voltage-anchored EKF), and the updated capacity is fed *back* into the EKF, so
//! SoC and SoH are tracked jointly. Deterministic.

use crate::capacity::RlsCapacity;
use crate::soc::{BatteryEkf, CellParams};
use serde::{Deserialize, Serialize};

/// Joint SoC + capacity estimator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualEstimator {
    ekf: BatteryEkf,
    rls: RlsCapacity,
    nominal_as: f64,
    seg_charge_as: f64,
    seg_start_soc: f64,
}

impl DualEstimator {
    /// Build from cell parameters, initial SoC, rated capacity (As) and the RLS
    /// forgetting factor.
    pub fn new(params: CellParams, soc0: f64, nominal_as: f64, rls_lambda: f64) -> Self {
        let ekf = BatteryEkf::new(params, soc0);
        let rls = RlsCapacity::new(nominal_as, rls_lambda, 1e6);
        Self {
            ekf,
            rls,
            nominal_as,
            seg_charge_as: 0.0,
            seg_start_soc: soc0,
        }
    }

    /// One sample within a segment: EKF SoC update + charge accumulation.
    pub fn step(&mut self, current: f64, dt: f64, v_meas: f64) {
        self.ekf.step(current, dt, v_meas);
        self.seg_charge_as += current * dt;
    }

    /// Close the current segment (a rest point): update the capacity estimate
    /// from the segment's charge and SoC change, and feed it back into the EKF.
    pub fn end_segment(&mut self) {
        let dsoc = (self.seg_start_soc - self.ekf.soc()).abs();
        if dsoc > 1e-3
        {
            let q = self.rls.update(dsoc, self.seg_charge_as.abs());
            self.ekf.set_capacity(q);
        }
        self.seg_start_soc = self.ekf.soc();
        self.seg_charge_as = 0.0;
    }

    /// Current SoC estimate.
    pub fn soc(&self) -> f64 {
        self.ekf.soc()
    }

    /// Estimated usable capacity (As).
    pub fn capacity_as(&self) -> f64 {
        self.rls.capacity_as()
    }

    /// Estimated State of Health (capacity / rated).
    pub fn soh(&self) -> f64 {
        self.rls.soh(self.nominal_as)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell() -> CellParams {
        CellParams {
            q_cap: 2.0 * 3600.0,
            r0: 0.05,
            r1: 0.02,
            c1: 2000.0,
            ocv: [3.3, 0.7, 0.2],
        }
    }

    #[test]
    fn jointly_recovers_soc_and_faded_capacity() {
        let nominal = 2.0 * 3600.0;
        let true_cap = 1.6 * 3600.0; // faded to 80% SoH
        let p = cell();
        // The estimator is told the (wrong) rated capacity.
        let mut dual = DualEstimator::new(p.clone(), 0.9, nominal, 0.95);

        let dt = 1.0;
        let mut true_soc = 0.9;
        let mut true_v1 = 0.0;
        // Several discharge segments separated by rests.
        for _seg in 0..8
        {
            for _ in 0..400
            {
                let current = 1.5; // discharge
                let alpha = (-dt / (p.r1 * p.c1)).exp();
                true_soc -= current * dt / true_cap; // TRUE capacity governs SoC
                true_v1 = alpha * true_v1 + p.r1 * (1.0 - alpha) * current;
                let v = p.ocv(true_soc) - true_v1 - current * p.r0;
                dual.step(current, dt, v);
            }
            dual.end_segment();
            // Recharge a touch so SoC stays in range.
            for _ in 0..200
            {
                let current = -1.5;
                let alpha = (-dt / (p.r1 * p.c1)).exp();
                true_soc -= current * dt / true_cap;
                true_v1 = alpha * true_v1 + p.r1 * (1.0 - alpha) * current;
                let v = p.ocv(true_soc) - true_v1 - current * p.r0;
                dual.step(current, dt, v);
            }
            dual.end_segment();
        }
        // Capacity / SoH recovered, and SoC still tracks the truth.
        assert!(
            (dual.soh() - 0.8).abs() < 0.03,
            "SoH {} (want ~0.80)",
            dual.soh()
        );
        assert!(
            (dual.soc() - true_soc).abs() < 0.03,
            "SoC {} vs {true_soc}",
            dual.soc()
        );
    }
}
