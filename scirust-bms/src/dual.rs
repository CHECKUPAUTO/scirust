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
        // The capacity measurement is Q = ∮I dt / ΔSoC, so ΔSoC must be the
        // coulomb-consistent change the EKF actually integrated — the raw,
        // unclamped state — not the saturated public SoC.
        let dsoc = (self.seg_start_soc - self.ekf.soc_raw()).abs();
        if dsoc > 1e-3
        {
            let q = self.rls.update(dsoc, self.seg_charge_as.abs());
            self.ekf.set_capacity(q);
        }
        self.seg_start_soc = self.ekf.soc_raw();
        self.seg_charge_as = 0.0;
    }

    /// Current SoC estimate (saturated to the physical range `[0, 1]`).
    pub fn soc(&self) -> f64 {
        self.ekf.soc()
    }

    /// Current polarization-voltage estimate `V₁` (V) of the underlying EKF.
    pub fn v1(&self) -> f64 {
        self.ekf.v1()
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

    /// Run a true 1-RC plant for `n` samples at constant `current`, feeding the
    /// noiseless terminal voltage into the estimator. Returns the updated truth.
    fn drive(
        dual: &mut DualEstimator,
        p: &CellParams,
        cap_true: f64,
        mut soc: f64,
        mut v1: f64,
        current: f64,
        n: usize,
    ) -> (f64, f64) {
        let dt = 1.0;
        let alpha = (-dt / (p.r1 * p.c1)).exp();
        for _ in 0..n
        {
            soc -= current * dt / cap_true; // TRUE capacity governs SoC
            v1 = alpha * v1 + p.r1 * (1.0 - alpha) * current;
            let v = p.ocv(soc) - v1 - current * p.r0;
            dual.step(current, dt, v);
        }
        (soc, v1)
    }

    #[test]
    fn soh_accessor_equals_capacity_over_nominal() {
        // dual.soh() must be exactly the internal capacity divided by the rated
        // capacity (clamped) — a self-consistency oracle independent of dynamics.
        let nominal = 2.0 * 3600.0;
        let p = cell();
        let mut dual = DualEstimator::new(p.clone(), 0.9, nominal, 0.95);
        let (mut s, mut v1) = (0.9, 0.0);
        let cap_true = 1.6 * 3600.0;
        (s, v1) = drive(&mut dual, &p, cap_true, s, v1, 1.5, 400);
        dual.end_segment();
        let _ = (s, v1);
        let expected = (dual.capacity_as() / nominal).clamp(0.0, 1.5);
        assert!((dual.soh() - expected).abs() < 1e-12);
    }

    #[test]
    fn capacity_estimate_tracks_true_faded_capacity() {
        // Start the EKF AT the truth (right SoC, right capacity) so its ΔSoC is
        // essentially exact; then Q = ∮I dt / ΔSoC must recover the true faded
        // capacity. The 80 %-SoH target (1.6 Ah) is derived from the plant, not
        // read from the estimator.
        let nominal = 2.0 * 3600.0;
        let cap_true = 1.6 * 3600.0; // 80 % SoH
        let p = cell();
        // The estimator is told the WRONG (rated) capacity; the segment-wise
        // feedback must drive it onto the true faded capacity. lambda < 1 lets
        // the running estimate forget the wrong initial guess.
        let mut dual = DualEstimator::new(p.clone(), 0.95, nominal, 0.9);
        let (mut s, mut v1) = (0.95, 0.0);
        for _ in 0..10
        {
            (s, v1) = drive(&mut dual, &p, cap_true, s, v1, 1.2, 300);
            dual.end_segment();
            (s, v1) = drive(&mut dual, &p, cap_true, s, v1, -1.2, 150); // partial recharge
            dual.end_segment();
        }
        let _ = (s, v1);
        assert!(
            (dual.capacity_as() - cap_true).abs() < 0.02 * cap_true,
            "capacity {} vs true {cap_true}",
            dual.capacity_as()
        );
        assert!((dual.soh() - 0.8).abs() < 0.02, "SoH {}", dual.soh());
    }

    #[test]
    fn segment_without_soc_change_does_not_update_capacity() {
        // A rest segment (no current, hence no SoC change) carries no capacity
        // information; the estimate must be left untouched.
        let nominal = 2.0 * 3600.0;
        let p = cell();
        let mut dual = DualEstimator::new(p.clone(), 0.7, nominal, 0.9);
        let before = dual.capacity_as();
        // Many zero-current samples fed a self-consistent voltage: the coulomb
        // term is exactly 0 (I = 0) and the measurement matches the state, so the
        // EKF SoC does not move and the segment's ΔSoC is below the 1e-3 guard.
        for _ in 0..50
        {
            let v = p.ocv(dual.soc()) - dual.v1();
            dual.step(0.0, 1.0, v);
        }
        dual.end_segment();
        assert_eq!(dual.capacity_as(), before);
    }

    #[test]
    fn reported_soc_stays_in_unit_range_under_deep_discharge() {
        // Even when discharged well past empty, the public SoC is clamped.
        let nominal = 2.0 * 3600.0;
        let p = cell();
        let mut dual = DualEstimator::new(p.clone(), 0.2, nominal, 0.95);
        let (mut s, mut v1) = (0.2, 0.0);
        // Discharge far past empty on a cell that is actually smaller than rated.
        (s, v1) = drive(&mut dual, &p, 1.0 * 3600.0, s, v1, 2.0, 1200);
        let _ = (s, v1);
        let soc = dual.soc();
        assert!((0.0..=1.0).contains(&soc), "SoC out of range: {soc}");
    }
}
