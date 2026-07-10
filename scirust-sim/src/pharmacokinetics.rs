//! Pharmacokinetic compartment models: first-order oral absorption into a
//! one-compartment body, and an intravenous two-compartment model. Both are
//! validated against their closed-form (sum-of-exponentials) solutions, and
//! the two-compartment area-under-the-curve against the exact `dose/k₁₀`
//! identity.
//!
//! Amounts are in dose units and rate constants in inverse time; a `volume`
//! of distribution converts a central amount to a plasma concentration.

use crate::engine::{SimError, System};

fn check_positive(name: &str, value: f64) -> Result<(), SimError> {
    if value.is_finite() && value > 0.0
    {
        Ok(())
    }
    else
    {
        Err(SimError::BadInput(format!(
            "{name} = {value} must be finite and positive"
        )))
    }
}

/// First-order oral absorption into a one-compartment body:
/// a gut depot empties into a central compartment that eliminates
/// first-order. State `y = [A_gut, A_central]`:
///
/// `A_gut' = -k_a·A_gut`, `A_central' = k_a·A_gut - k_e·A_central`.
///
/// Starting from `A_gut(0) = F·dose`, `A_central(0) = 0`, the central amount
/// follows the Bateman function
/// `A_central(t) = F·dose·k_a/(k_a-k_e)·(e^{-k_e·t} - e^{-k_a·t})`
/// (for `k_a ≠ k_e`), peaking at `t_max = ln(k_a/k_e)/(k_a-k_e)`.
#[derive(Debug, Clone, PartialEq)]
pub struct OralOneCompartment {
    k_a: f64,
    k_e: f64,
    volume: f64,
    bioavailability: f64,
    dose: f64,
}

impl OralOneCompartment {
    /// Create the model. Absorption `k_a`, elimination `k_e`, `volume` and
    /// `dose` must be finite and positive; `bioavailability` must lie in
    /// `(0, 1]`.
    pub fn new(
        k_a: f64,
        k_e: f64,
        volume: f64,
        bioavailability: f64,
        dose: f64,
    ) -> Result<Self, SimError> {
        check_positive("k_a", k_a)?;
        check_positive("k_e", k_e)?;
        check_positive("volume", volume)?;
        check_positive("dose", dose)?;
        if !(bioavailability.is_finite() && bioavailability > 0.0 && bioavailability <= 1.0)
        {
            return Err(SimError::BadInput(format!(
                "bioavailability = {bioavailability} must lie in (0, 1]"
            )));
        }
        Ok(OralOneCompartment {
            k_a,
            k_e,
            volume,
            bioavailability,
            dose,
        })
    }

    /// The initial state `[A_gut, A_central] = [F·dose, 0]`.
    pub fn initial_state(&self) -> [f64; 2] {
        [self.bioavailability * self.dose, 0.0]
    }

    /// Plasma concentration from a central amount, `A_central / volume`.
    pub fn concentration(&self, a_central: f64) -> f64 {
        a_central / self.volume
    }

    /// Time of the peak central amount, `ln(k_a/k_e)/(k_a-k_e)`, or `None`
    /// when `k_a = k_e` (the expression is singular there).
    pub fn peak_time(&self) -> Option<f64> {
        if self.k_a == self.k_e
        {
            return None;
        }
        Some((self.k_a / self.k_e).ln() / (self.k_a - self.k_e))
    }

    /// The closed-form state `[A_gut(t), A_central(t)]`, or `None` when
    /// `k_a = k_e` (removable singularity in the Bateman term).
    pub fn exact(&self, t: f64) -> Option<[f64; 2]> {
        if self.k_a == self.k_e
        {
            return None;
        }
        let a0 = self.bioavailability * self.dose;
        let a_gut = a0 * (-self.k_a * t).exp();
        let a_central =
            a0 * self.k_a / (self.k_a - self.k_e) * ((-self.k_e * t).exp() - (-self.k_a * t).exp());
        Some([a_gut, a_central])
    }
}

impl System for OralOneCompartment {
    fn dim(&self) -> usize {
        2
    }

    fn derivatives(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = -self.k_a * y[0];
        dydt[1] = self.k_a * y[0] - self.k_e * y[1];
    }
}

/// An intravenous two-compartment model: a central compartment (from which
/// the drug is eliminated, rate `k₁₀`) exchanges with a peripheral
/// compartment (rates `k₁₂` out, `k₂₁` back). State `y = [A_central,
/// A_peripheral]`:
///
/// `A_c' = -(k₁₀+k₁₂)·A_c + k₂₁·A_p`, `A_p' = k₁₂·A_c - k₂₁·A_p`.
///
/// After an IV bolus `A_c(0) = dose`, `A_p(0) = 0`, the central amount is the
/// biexponential `A_c(t) = dose·[(α-k₂₁)/(α-β)·e^{-α·t} + (k₂₁-β)/(α-β)·e^{-β·t}]`,
/// where `α > β` are the hybrid rate constants (the roots of
/// `s² + (k₁₀+k₁₂+k₂₁)·s + k₁₀·k₂₁`). The exact area under the central curve
/// is `dose/k₁₀`.
#[derive(Debug, Clone, PartialEq)]
pub struct TwoCompartmentIv {
    k10: f64,
    k12: f64,
    k21: f64,
    volume: f64,
    dose: f64,
}

impl TwoCompartmentIv {
    /// Create the model; all rate constants, the `volume` and the `dose` must
    /// be finite and positive.
    pub fn new(k10: f64, k12: f64, k21: f64, volume: f64, dose: f64) -> Result<Self, SimError> {
        check_positive("k10", k10)?;
        check_positive("k12", k12)?;
        check_positive("k21", k21)?;
        check_positive("volume", volume)?;
        check_positive("dose", dose)?;
        Ok(TwoCompartmentIv {
            k10,
            k12,
            k21,
            volume,
            dose,
        })
    }

    /// The initial state `[A_central, A_peripheral] = [dose, 0]`.
    pub fn initial_state(&self) -> [f64; 2] {
        [self.dose, 0.0]
    }

    /// Plasma concentration from a central amount, `A_central / volume`.
    pub fn concentration(&self, a_central: f64) -> f64 {
        a_central / self.volume
    }

    /// The hybrid rate constants `(α, β)` with `α ≥ β > 0`.
    pub fn hybrid_rates(&self) -> (f64, f64) {
        let sum = self.k10 + self.k12 + self.k21;
        let prod = self.k10 * self.k21;
        // The discriminant is (k10-k21)² + k12² + 2·k12·(k10+k21) ≥ 0.
        let root = (sum * sum - 4.0 * prod).max(0.0).sqrt();
        ((sum + root) / 2.0, (sum - root) / 2.0)
    }

    /// The exact area under the central-amount curve over `[0, ∞)`,
    /// `dose/k₁₀`.
    pub fn central_auc(&self) -> f64 {
        self.dose / self.k10
    }

    /// The closed-form central amount `A_central(t)`.
    pub fn exact_central(&self, t: f64) -> f64 {
        let (alpha, beta) = self.hybrid_rates();
        let denom = alpha - beta;
        if denom == 0.0
        {
            // Repeated root (measure-zero parameter set): fall back to the
            // limiting single-exponential form.
            return self.dose * (-alpha * t).exp();
        }
        self.dose
            * ((alpha - self.k21) / denom * (-alpha * t).exp()
                + (self.k21 - beta) / denom * (-beta * t).exp())
    }
}

impl System for TwoCompartmentIv {
    fn dim(&self) -> usize {
        2
    }

    fn derivatives(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = -(self.k10 + self.k12) * y[0] + self.k21 * y[1];
        dydt[1] = self.k12 * y[0] - self.k21 * y[1];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{simulate, simulate_adaptive};

    #[test]
    #[cfg_attr(miri, ignore)]
    fn oral_absorption_matches_the_bateman_closed_form() {
        let pk = OralOneCompartment::new(1.2, 0.25, 30.0, 0.8, 100.0).unwrap();
        let traj = simulate(&pk, &pk.initial_state(), 0.0, 24.0, 0.001).unwrap();
        for (t, row) in traj.t.iter().zip(traj.y.iter())
        {
            let exact = pk.exact(*t).unwrap();
            assert!((row[0] - exact[0]).abs() < 1e-7, "gut t = {t}");
            assert!((row[1] - exact[1]).abs() < 1e-7, "central t = {t}");
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn oral_concentration_peaks_at_the_analytic_t_max() {
        let pk = OralOneCompartment::new(1.2, 0.25, 30.0, 0.8, 100.0).unwrap();
        let t_max = pk.peak_time().unwrap();
        // Bracket the analytic peak: the central amount there exceeds its
        // values a little before and after.
        let at = |t: f64| pk.exact(t).unwrap()[1];
        assert!(at(t_max) > at(t_max - 0.1) && at(t_max) > at(t_max + 0.1));
        // The derivative of the central amount vanishes at t_max.
        let mut dydt = [0.0; 2];
        let state = pk.exact(t_max).unwrap();
        pk.derivatives(t_max, &state, &mut dydt);
        assert!(dydt[1].abs() < 1e-9, "central rate at t_max = {}", dydt[1]);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn two_compartment_central_matches_the_biexponential() {
        let pk = TwoCompartmentIv::new(0.3, 0.5, 0.2, 10.0, 100.0).unwrap();
        let (alpha, beta) = pk.hybrid_rates();
        assert!(alpha > beta && beta > 0.0);
        let traj = simulate(&pk, &pk.initial_state(), 0.0, 40.0, 0.001).unwrap();
        for (t, row) in traj.t.iter().zip(traj.y.iter())
        {
            assert!((row[0] - pk.exact_central(*t)).abs() < 1e-7, "t = {t}");
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn two_compartment_auc_matches_dose_over_k10() {
        // Integrate far past the slow phase with the adaptive solver and
        // trapezoid the central amount; the exact AUC is dose/k10.
        let pk = TwoCompartmentIv::new(0.3, 0.5, 0.2, 10.0, 100.0).unwrap();
        let traj = simulate_adaptive(&pk, &pk.initial_state(), 0.0, 250.0, 1e-9, 1e-12).unwrap();
        let mut auc = 0.0;
        for w in traj.t.windows(2).zip(traj.y.windows(2))
        {
            let (ts, ys) = w;
            auc += 0.5 * (ts[1] - ts[0]) * (ys[0][0] + ys[1][0]);
        }
        let exact = pk.central_auc();
        assert!((auc - exact).abs() < 1e-3 * exact, "AUC {auc} vs {exact}");
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn two_compartment_distributes_then_eliminates() {
        let pk = TwoCompartmentIv::new(0.3, 0.5, 0.2, 10.0, 100.0).unwrap();
        let traj = simulate(&pk, &pk.initial_state(), 0.0, 60.0, 0.005).unwrap();
        // Peripheral amount rises from zero then falls back toward zero.
        let peripheral = traj.column(1).unwrap();
        let peak = peripheral.iter().cloned().fold(0.0, f64::max);
        assert!(peak > 10.0 && *peripheral.first().unwrap() == 0.0);
        assert!(*peripheral.last().unwrap() < 0.5 * peak);
        // Total body amount only ever decreases: elimination is one-way.
        let total: Vec<f64> = traj.y.iter().map(|r| r[0] + r[1]).collect();
        assert!(
            total.windows(2).all(|w| w[1] <= w[0] + 1e-9),
            "total not monotone"
        );
        assert!(total.last().unwrap() < &(0.2 * total[0]));
    }

    #[test]
    fn constructors_and_closed_forms_reject_bad_inputs() {
        assert!(OralOneCompartment::new(0.0, 0.25, 30.0, 0.8, 100.0).is_err());
        assert!(OralOneCompartment::new(1.2, 0.25, 30.0, 1.5, 100.0).is_err());
        assert!(OralOneCompartment::new(1.2, 0.25, 30.0, 0.0, 100.0).is_err());
        assert!(OralOneCompartment::new(1.2, 0.25, -1.0, 0.8, 100.0).is_err());
        // k_a = k_e: the Bateman closed form and t_max are singular.
        let degenerate = OralOneCompartment::new(0.5, 0.5, 30.0, 1.0, 100.0).unwrap();
        assert!(degenerate.exact(1.0).is_none());
        assert!(degenerate.peak_time().is_none());
        assert!(TwoCompartmentIv::new(0.3, 0.5, 0.2, 10.0, 0.0).is_err());
        assert!(TwoCompartmentIv::new(-0.3, 0.5, 0.2, 10.0, 100.0).is_err());
        assert!(TwoCompartmentIv::new(0.3, 0.5, f64::NAN, 10.0, 100.0).is_err());
    }
}
