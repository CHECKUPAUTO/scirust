//! The inertial piloting chart (*carte de pilotage inertiel*).
//!
//! Where a Shewhart chart pilots the mean and the spread on two separate
//! charts, the inertial chart pilots the single quantity being toleranced —
//! the inertia `I = √(δ² + σ²)` — on one chart against the budget `I_max`.
//!
//! Each subgroup yields a point `Î = √(δ̂² + σ̂²)` (with `δ̂ = x̄ − T` and `σ̂` the
//! population standard deviation, so that `Î²` is unbiased for `I²`). The point
//! is compared to an **upper piloting limit** derived from `I_max` and a
//! sampling risk `α`:
//!
//! ```text
//! UPL(α) = I_max · √( χ²_{n; 1−α} / n ).
//! ```
//!
//! This is exact in the worst case for false alarms — a process sitting on the
//! cone boundary with its whole inertia in *dispersion* (`δ = 0`, `σ = I_max`)
//! gives `n·Î²/I_max² ~ χ²_n`, so `Î > UPL(α)` occurs with probability `α`.
//! (Any off-centre split has *smaller* spread in `Î`, so the limit stays
//! conservative.) A point above `UPL` signals the true inertia exceeds the
//! budget; the [`PilotingSignal`] then reports whether centering or dispersion
//! dominates and the re-centering move that would restore the target.

use crate::inertia::Inertia;
use crate::special::chi2_quantile;
use serde::{Deserialize, Serialize};

/// An inertial piloting chart: a target, an inertia budget and a subgroup
/// size.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PilotingChart {
    /// Target value `T` of the characteristic.
    pub target: f64,
    /// Maximum admissible inertia `I_max` (cone radius).
    pub i_max: f64,
    /// Subgroup size `n` used for each plotted point.
    pub subgroup: usize,
}

/// What piloting a subgroup recommends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PilotingAction {
    /// Inside the piloting limit — leave the process alone.
    LetRun,
    /// Off-centering dominates and the point breaches the limit — re-centre
    /// the process onto the target.
    Recenter,
    /// Dispersion dominates and the point breaches the limit — reduce
    /// variation (the shift alone will not bring the inertia back).
    ReduceDispersion,
}

/// The verdict for one subgroup.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PilotingSignal {
    /// Estimated inertia `Î` of the subgroup.
    pub inertia: f64,
    /// Estimated off-centering `δ̂ = x̄ − T`.
    pub off_centering: f64,
    /// Estimated dispersion `σ̂` (population).
    pub sigma: f64,
    /// Upper piloting limit the point was tested against.
    pub upper_limit: f64,
    /// Whether the point is within the piloting limit.
    pub in_control: bool,
    /// Recommended action.
    pub action: PilotingAction,
    /// Re-centering move `−δ̂` (add this to the process setting to hit target).
    pub recommended_shift: f64,
}

impl PilotingChart {
    /// A chart for target `target`, inertia budget `i_max` and subgroup size
    /// `subgroup`.
    pub fn new(target: f64, i_max: f64, subgroup: usize) -> Self {
        Self {
            target,
            i_max: i_max.abs(),
            subgroup,
        }
    }

    /// Estimated inertia of a subgroup, `Î = √((x̄ − T)² + σ̂²)`.
    pub fn inertia(&self, subgroup: &[f64]) -> Inertia {
        Inertia::from_sample(subgroup, self.target)
    }

    /// Upper piloting limit `UPL(α) = I_max · √(χ²_{n;1−α}/n)` for a subgroup
    /// size fixed by the chart. `alpha` is the tolerated false-alarm risk
    /// (e.g. `0.0027` for a 3σ-equivalent limit).
    pub fn upper_limit(&self, alpha: f64) -> f64 {
        let n = self.subgroup.max(1) as f64;
        self.i_max * (chi2_quantile(n, 1.0 - alpha) / n).sqrt()
    }

    /// Pilot a subgroup: estimate its inertia, test it against `UPL(alpha)`,
    /// and recommend an action.
    pub fn evaluate(&self, subgroup: &[f64], alpha: f64) -> PilotingSignal {
        let inertia = self.inertia(subgroup);
        let upper = self.upper_limit(alpha);
        let i = inertia.value();
        let in_control = i <= upper;
        let action = if in_control
        {
            PilotingAction::LetRun
        }
        else if inertia.off_centering_ratio() >= 0.5
        {
            PilotingAction::Recenter
        }
        else
        {
            PilotingAction::ReduceDispersion
        };
        PilotingSignal {
            inertia: i,
            off_centering: inertia.off_centering,
            sigma: inertia.sigma,
            upper_limit: upper,
            in_control,
            action,
            recommended_shift: -inertia.off_centering,
        }
    }

    /// Pilot a run of subgroups, returning one [`PilotingSignal`] each.
    pub fn evaluate_run(&self, subgroups: &[Vec<f64>], alpha: f64) -> Vec<PilotingSignal> {
        subgroups.iter().map(|g| self.evaluate(g, alpha)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn upper_limit_exceeds_budget_and_grows_with_confidence() {
        let chart = PilotingChart::new(10.0, 0.1, 5);
        let upl = chart.upper_limit(0.0027);
        // χ²_{5;0.9973}/5 > 1 ⇒ UPL > I_max, and it must exceed a looser limit.
        assert!(upl > chart.i_max);
        assert!(upl > chart.upper_limit(0.05));
    }

    #[test]
    fn upper_limit_matches_chi2_formula() {
        let chart = PilotingChart::new(0.0, 0.2, 8);
        let upl = chart.upper_limit(0.01);
        let want = 0.2 * (crate::special::chi2_quantile(8.0, 0.99) / 8.0).sqrt();
        assert_relative_eq!(upl, want, epsilon = 1e-12);
    }

    #[test]
    fn centered_tight_subgroup_lets_run() {
        let chart = PilotingChart::new(10.0, 0.2, 5);
        let sig = chart.evaluate(&[10.0, 10.02, 9.98, 10.01, 9.99], 0.0027);
        assert!(sig.in_control);
        assert_eq!(sig.action, PilotingAction::LetRun);
    }

    #[test]
    fn off_center_subgroup_asks_for_recentering() {
        let chart = PilotingChart::new(10.0, 0.05, 5);
        // Mean ≈ 10.2 with tiny spread ⇒ inertia dominated by off-centering.
        let sig = chart.evaluate(&[10.20, 10.21, 10.19, 10.20, 10.20], 0.0027);
        assert!(!sig.in_control);
        assert_eq!(sig.action, PilotingAction::Recenter);
        assert_relative_eq!(sig.recommended_shift, -(sig.off_centering), epsilon = 1e-12);
        assert!(sig.recommended_shift < 0.0); // pull the mean back down
    }

    #[test]
    fn dispersed_subgroup_asks_for_variation_reduction() {
        let chart = PilotingChart::new(10.0, 0.05, 5);
        // Centered but wildly spread ⇒ dispersion dominates.
        let sig = chart.evaluate(&[9.6, 10.4, 9.7, 10.3, 10.0], 0.0027);
        assert!(!sig.in_control);
        assert_eq!(sig.action, PilotingAction::ReduceDispersion);
    }

    #[test]
    fn evaluate_run_returns_one_signal_per_subgroup() {
        let chart = PilotingChart::new(0.0, 1.0, 4);
        let groups = vec![vec![0.1, -0.1, 0.0, 0.05], vec![0.2, 0.3, 0.25, 0.28]];
        assert_eq!(chart.evaluate_run(&groups, 0.0027).len(), 2);
    }
}
