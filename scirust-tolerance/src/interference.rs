//! Stress–strength interference and assembly-fit reliability.
//!
//! Tolerancing asks whether a *characteristic* stays in spec; assembly asks a
//! sharper question — will two mating parts actually **fit**, and how often will
//! they not? That is the stress–strength (load–capacity) interference model: with
//! a "strength" `S ∼ N(μ_S, σ_S²)` and a "stress" `L ∼ N(μ_L, σ_L²)`, the item
//! survives when `S > L`, so the reliability is the closed form
//!
//! ```text
//! R = P(S > L) = Φ(β) ,   β = (μ_S − μ_L) / √(σ_S² + σ_L²) ,
//! ```
//!
//! where `β` is the **reliability index** (the safety margin in standard-
//! deviations of the combined spread). The same algebra sizes an **interference
//! fit**: treat the clearance `C = hole − shaft ∼ N(μ_h − μ_s, σ_h² + σ_s²)` and
//! read off `P(C > 0)` for a clearance fit (or `P(C < 0)` for a designed press
//! fit) — the probability a random hole/shaft pair assembles as intended, which a
//! worst-case min/max stack cannot give.
//!
//! This complements [`crate::capability::nonconformity_ppm`] (one characteristic
//! vs fixed limits) by pitting **two random variables against each other**.

use crate::special::normal_cdf;
use serde::{Deserialize, Serialize};

/// Reliability index `β = (μ_S − μ_L)/√(σ_S² + σ_L²)` — the mean safety margin in
/// units of the combined standard deviation. Returns `+∞`/`−∞` when both spreads
/// vanish (a deterministic margin), by its sign.
pub fn reliability_index(
    mean_strength: f64,
    sd_strength: f64,
    mean_stress: f64,
    sd_stress: f64,
) -> f64 {
    let denom = (sd_strength * sd_strength + sd_stress * sd_stress).sqrt();
    let margin = mean_strength - mean_stress;
    if denom <= 0.0
    {
        return if margin > 0.0
        {
            f64::INFINITY
        }
        else if margin < 0.0
        {
            f64::NEG_INFINITY
        }
        else
        {
            0.0
        };
    }
    margin / denom
}

/// Stress–strength reliability `R = P(S > L) = Φ(β)` for independent normal
/// strength and stress. A zero combined spread returns 1 if `μ_S > μ_L`, else 0.
pub fn interference_reliability(
    mean_strength: f64,
    sd_strength: f64,
    mean_stress: f64,
    sd_stress: f64,
) -> f64 {
    let beta = reliability_index(mean_strength, sd_strength, mean_stress, sd_stress);
    if beta.is_infinite()
    {
        return if beta > 0.0 { 1.0 } else { 0.0 };
    }
    normal_cdf(beta)
}

/// Result of a clearance-fit analysis of a hole/shaft pair.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FitAnalysis {
    /// Mean clearance `μ_h − μ_s` (negative ⇒ mean interference / press fit).
    pub mean_clearance: f64,
    /// Clearance standard deviation `√(σ_h² + σ_s²)`.
    pub sd_clearance: f64,
    /// Probability of a clearance fit, `P(clearance > 0)`.
    pub prob_clearance: f64,
    /// Probability of interference, `P(clearance < 0) = 1 − prob_clearance`.
    pub prob_interference: f64,
    /// Reliability index of the clearance being positive, `μ_C/σ_C`.
    pub reliability_index: f64,
}

/// Analyse the fit of a random hole against a random shaft, each normal:
/// clearance `C = hole − shaft ∼ N(μ_h − μ_s, σ_h² + σ_s²)`, and the probability
/// it assembles with clearance (`C > 0`) or interference (`C < 0`). For a
/// designed press fit read `prob_interference` as the intended-fit probability.
pub fn clearance_fit(mean_hole: f64, sd_hole: f64, mean_shaft: f64, sd_shaft: f64) -> FitAnalysis {
    let mean_clearance = mean_hole - mean_shaft;
    let sd_clearance = (sd_hole * sd_hole + sd_shaft * sd_shaft).sqrt();
    let prob_clearance = interference_reliability(mean_hole, sd_hole, mean_shaft, sd_shaft);
    FitAnalysis {
        mean_clearance,
        sd_clearance,
        prob_clearance,
        prob_interference: 1.0 - prob_clearance,
        reliability_index: reliability_index(mean_hole, sd_hole, mean_shaft, sd_shaft),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn equal_means_give_half_reliability() {
        // μ_S = μ_L ⇒ β = 0 ⇒ R = 0.5, by symmetry.
        let r = interference_reliability(10.0, 1.0, 10.0, 2.0);
        assert_relative_eq!(r, 0.5, epsilon = 1e-12);
        assert_relative_eq!(
            reliability_index(10.0, 1.0, 10.0, 2.0),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn reliability_matches_combined_normal() {
        // β = (25−20)/√(3²+4²) = 5/5 = 1 ⇒ R = Φ(1).
        let r = interference_reliability(25.0, 3.0, 20.0, 4.0);
        assert_relative_eq!(
            reliability_index(25.0, 3.0, 20.0, 4.0),
            1.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(r, normal_cdf(1.0), epsilon = 1e-12);
    }

    #[test]
    fn deterministic_margin_is_certain() {
        // No spread, strength above stress ⇒ certain survival.
        assert_relative_eq!(
            interference_reliability(5.0, 0.0, 3.0, 0.0),
            1.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            interference_reliability(3.0, 0.0, 5.0, 0.0),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn clearance_fit_partitions_probability() {
        // Hole 10.0±0.02, shaft 9.95±0.02 ⇒ mean clearance 0.05, mostly clears.
        let f = clearance_fit(10.0, 0.02, 9.95, 0.02);
        assert_relative_eq!(f.mean_clearance, 0.05, epsilon = 1e-12);
        assert_relative_eq!(
            f.sd_clearance,
            (0.02_f64.powi(2) + 0.02_f64.powi(2)).sqrt(),
            epsilon = 1e-12
        );
        assert_relative_eq!(f.prob_clearance + f.prob_interference, 1.0, epsilon = 1e-12);
        assert!(f.prob_clearance > 0.95);
        // Interference (press) fit: shaft larger than hole on average.
        let press = clearance_fit(10.0, 0.02, 10.05, 0.02);
        assert!(press.prob_interference > 0.95);
    }
}
