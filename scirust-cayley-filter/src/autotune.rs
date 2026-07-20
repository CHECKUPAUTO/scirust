//! Held-out autotuning of the Cayley quasi-kernel threshold.

use scirust_core::transform_autotune::autotune_by;

use crate::projector::CayleyProjector;
use crate::scalar::{Sedenion, squared_norm};

const ENERGY_FLOOR: f64 = 1.0e-30;

/// One supervised denoising case.
#[derive(Clone, Debug)]
pub struct CayleyCase {
    pub multiplier: Sedenion,
    pub signal: Sedenion,
    pub noise: Sedenion,
}

impl CayleyCase {
    #[must_use]
    pub const fn new(multiplier: Sedenion, signal: Sedenion, noise: Sedenion) -> Self {
        Self {
            multiplier,
            signal,
            noise,
        }
    }
}

/// Result of SciRust's development/held-out threshold selection.
#[derive(Clone, Debug, PartialEq)]
pub struct CayleyAutotuneResult {
    pub chosen_threshold: Option<f64>,
    pub chosen_eval_score: f64,
    pub baseline_eval_score: f64,
    pub beats_baseline: bool,
    pub dev_scores: Vec<(f64, Option<f64>)>,
}

/// Selects a relative singular-value threshold on `dev`, then validates it
/// without refitting on `eval`.
///
/// The score rewards noise attenuation and penalizes signal distortion.
#[must_use]
pub fn autotune_threshold(
    dev: &[CayleyCase],
    eval: &[CayleyCase],
    candidates: &[f64],
    baseline_threshold: f64,
    distortion_weight: f64,
) -> CayleyAutotuneResult {
    assert!(
        distortion_weight.is_finite() && distortion_weight >= 0.0,
        "distortion weight must be finite and non-negative"
    );

    let score = |threshold: f64, _fit: &[CayleyCase], cases: &[CayleyCase]| -> Option<f64> {
        score_cases(cases, threshold, distortion_weight)
    };

    let baseline = |_fit: &[CayleyCase], cases: &[CayleyCase]| {
        score_cases(cases, baseline_threshold, distortion_weight).unwrap_or(f64::NEG_INFINITY)
    };

    let report = autotune_by(dev, eval, candidates, score, baseline);

    CayleyAutotuneResult {
        chosen_threshold: report.chosen,
        chosen_eval_score: report.chosen_eval_score,
        baseline_eval_score: report.baseline_eval_score,
        beats_baseline: report.beats_baseline,
        dev_scores: report.dev_scores,
    }
}

fn score_cases(cases: &[CayleyCase], threshold: f64, distortion_weight: f64) -> Option<f64> {
    if cases.is_empty()
    {
        return None;
    }

    let mut total = 0.0;

    for case in cases
    {
        let projector = CayleyProjector::new(case.multiplier, threshold).ok()?;

        let filtered_signal = projector.apply(&case.signal);
        let filtered_noise = projector.apply(&case.noise);

        let signal_energy = squared_norm(&case.signal);
        let input_noise_energy = squared_norm(&case.noise);
        let output_noise_energy = squared_norm(&filtered_noise);
        let distortion = squared_distance(&case.signal, &filtered_signal);

        let attenuation = 10.0
            * ((input_noise_energy + ENERGY_FLOOR) / (output_noise_energy + ENERGY_FLOOR)).log10();

        let distortion_db = 10.0 * (1.0 + distortion / (signal_energy + ENERGY_FLOOR)).log10();

        total += attenuation - distortion_weight * distortion_db;
    }

    Some(total / cases.len() as f64)
}

fn squared_distance(left: &Sedenion, right: &Sedenion) -> f64 {
    left.iter().zip(right).fold(0.0, |sum, (a, b)| {
        let difference = a - b;
        sum + difference * difference
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::{SEDENION_DIMENSION, basis_vector};

    const ZERO: Sedenion = [0.0; SEDENION_DIMENSION];

    #[test]
    fn autotune_is_deterministic() {
        let multiplier = basis_vector(0).expect("e0 exists");
        let signal = basis_vector(1).expect("e1 exists");
        let noise = basis_vector(2).expect("e2 exists");

        let case = CayleyCase::new(multiplier, signal, noise);
        let candidates = [0.0, 0.5, 1.0];

        let first = autotune_threshold(
            std::slice::from_ref(&case),
            std::slice::from_ref(&case),
            &candidates,
            0.0,
            200.0,
        );

        let second = autotune_threshold(
            std::slice::from_ref(&case),
            std::slice::from_ref(&case),
            &candidates,
            0.0,
            200.0,
        );

        assert_eq!(first, second);
        assert_eq!(first.chosen_threshold, Some(0.0));
        assert!(!first.beats_baseline);
    }

    #[test]
    fn exact_kernel_receives_a_large_score() {
        let mut multiplier = ZERO;
        multiplier[1] = 1.0;
        multiplier[10] = 1.0;

        let signal = basis_vector(0).expect("e0 exists");

        let mut noise = ZERO;
        noise[4] = 1.0;
        noise[15] = -1.0;

        let cases = [CayleyCase::new(multiplier, signal, noise)];

        let score = score_cases(&cases, 1.0e-12, 10.0).expect("valid score");

        assert!(score > 200.0, "score = {score}");
    }
}
