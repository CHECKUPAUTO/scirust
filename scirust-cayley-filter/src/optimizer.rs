//! Derivative-free optimization of the sedenion multiplier.

use scirust_solvers::{Tolerance, optimize::nelder_mead};

use crate::scalar::{SEDENION_DIMENSION, Sedenion, squared_norm};
use crate::soft::SoftCayleyFilter;

const ENERGY_FLOOR: f64 = 1.0e-30;
const INVALID_LOSS: f64 = 1.0e300;

/// One supervised signal/noise pair.
#[derive(Clone, Debug)]
pub struct MultiplierCase {
    pub signal: Sedenion,
    pub noise: Sedenion,
}

impl MultiplierCase {
    #[must_use]
    pub const fn new(signal: Sedenion, noise: Sedenion) -> Self {
        Self { signal, noise }
    }
}

/// Transparent score of one multiplier.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiplierScore {
    pub loss: f64,
    pub mean_noise_ratio: f64,
    pub mean_distortion_ratio: f64,
    pub rejected_dimension: usize,
}

/// Result returned by SciRust's Nelder-Mead optimizer.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiplierOptimizationResult {
    pub multiplier: Sedenion,
    pub score: MultiplierScore,
    pub iterations: usize,
    pub residual: f64,
}

/// Scores a multiplier after normalizing it to unit Euclidean norm.
///
/// Lower is better:
///
/// `loss = mean_noise_ratio + distortion_weight * mean_distortion_ratio`.
pub fn score_multiplier(
    cases: &[MultiplierCase],
    multiplier: &[f64],
    relative_scale: f64,
    distortion_weight: f64,
) -> Result<MultiplierScore, String> {
    if cases.is_empty()
    {
        return Err("at least one training case is required".into());
    }
    if !distortion_weight.is_finite() || distortion_weight < 0.0
    {
        return Err("distortion weight must be finite and non-negative".into());
    }

    let normalized = normalize_multiplier(multiplier)
        .ok_or_else(|| "multiplier must be a finite non-zero 16D vector".to_string())?;

    let filter =
        SoftCayleyFilter::new(normalized, relative_scale).map_err(|error| error.to_string())?;

    let mut noise_ratio = 0.0;
    let mut distortion_ratio = 0.0;

    for case in cases
    {
        let filtered_signal = filter.apply(&case.signal);
        let filtered_noise = filter.apply(&case.noise);

        noise_ratio += squared_norm(&filtered_noise) / squared_norm(&case.noise).max(ENERGY_FLOOR);

        distortion_ratio += squared_distance(&case.signal, &filtered_signal)
            / squared_norm(&case.signal).max(ENERGY_FLOOR);
    }

    let count = cases.len() as f64;
    let mean_noise_ratio = noise_ratio / count;
    let mean_distortion_ratio = distortion_ratio / count;

    Ok(MultiplierScore {
        loss: mean_noise_ratio + distortion_weight * mean_distortion_ratio,
        mean_noise_ratio,
        mean_distortion_ratio,
        rejected_dimension: filter.gains().iter().filter(|&&gain| gain <= 0.5).count(),
    })
}

/// Optimizes the 15 scale-independent multiplier coordinates.
///
/// The coordinate of largest absolute value in `initial` is selected
/// deterministically as the gauge pivot and fixed to `1`.
pub fn optimize_multiplier(
    cases: &[MultiplierCase],
    initial: Sedenion,
    relative_scale: f64,
    distortion_weight: f64,
    initial_step: f64,
    tolerance: Tolerance,
) -> Result<MultiplierOptimizationResult, String> {
    if !initial_step.is_finite() || initial_step <= 0.0
    {
        return Err("initial step must be finite and positive".into());
    }

    let normalized_initial = normalize_multiplier(&initial)
        .ok_or_else(|| "initial multiplier must be finite and non-zero".to_string())?;

    score_multiplier(
        cases,
        &normalized_initial,
        relative_scale,
        distortion_weight,
    )?;

    let pivot = largest_absolute_coordinate(&normalized_initial);
    let initial_coordinates = gauge_coordinates(&normalized_initial, pivot);

    let objective = |candidate: &[f64]| {
        multiplier_from_gauge(candidate, pivot)
            .and_then(|multiplier| {
                score_multiplier(cases, &multiplier, relative_scale, distortion_weight).ok()
            })
            .map_or(INVALID_LOSS, |score| score.loss)
    };

    let solution = nelder_mead(objective, initial_coordinates, initial_step, tolerance)
        .map_err(|error| error.to_string())?;

    let multiplier = multiplier_from_gauge(&solution.value, pivot)
        .ok_or_else(|| "optimizer returned invalid gauge coordinates".to_string())?;

    let score = score_multiplier(cases, &multiplier, relative_scale, distortion_weight)?;

    Ok(MultiplierOptimizationResult {
        multiplier,
        score,
        iterations: solution.info.iterations,
        residual: solution.info.residual,
    })
}

fn largest_absolute_coordinate(multiplier: &Sedenion) -> usize {
    let mut best_index = 0;
    let mut best_value = multiplier[0].abs();

    for (index, value) in multiplier.iter().enumerate().skip(1)
    {
        let absolute = value.abs();

        if absolute > best_value
        {
            best_index = index;
            best_value = absolute;
        }
    }

    best_index
}

fn gauge_coordinates(multiplier: &Sedenion, pivot: usize) -> Vec<f64> {
    let pivot_value = multiplier[pivot];

    multiplier
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (index != pivot).then_some(value / pivot_value))
        .collect()
}

fn multiplier_from_gauge(coordinates: &[f64], pivot: usize) -> Option<Sedenion> {
    if pivot >= SEDENION_DIMENSION
        || coordinates.len() != SEDENION_DIMENSION - 1
        || coordinates.iter().any(|value| !value.is_finite())
    {
        return None;
    }

    let mut multiplier = [0.0; SEDENION_DIMENSION];
    multiplier[pivot] = 1.0;

    let mut source = 0;

    for (index, value) in multiplier.iter_mut().enumerate()
    {
        if index != pivot
        {
            *value = coordinates[source];
            source += 1;
        }
    }

    normalize_multiplier(&multiplier)
}

fn normalize_multiplier(values: &[f64]) -> Option<Sedenion> {
    if values.len() != SEDENION_DIMENSION || values.iter().any(|value| !value.is_finite())
    {
        return None;
    }

    let norm = values.iter().map(|value| value * value).sum::<f64>().sqrt();
    if !norm.is_finite() || norm <= 1.0e-15
    {
        return None;
    }

    Some(core::array::from_fn(|index| values[index] / norm))
}

fn squared_distance(left: &Sedenion, right: &Sedenion) -> f64 {
    left.iter().zip(right).fold(0.0, |sum, (a, b)| {
        let difference = a - b;
        sum + difference * difference
    })
}

#[cfg(test)]
mod tests {
    use super::{
        MultiplierCase, gauge_coordinates, largest_absolute_coordinate, multiplier_from_gauge,
        normalize_multiplier, optimize_multiplier, score_multiplier,
    };
    use crate::scalar::{SEDENION_DIMENSION, Sedenion, basis_vector};
    use scirust_solvers::Tolerance;

    const ZERO: Sedenion = [0.0; SEDENION_DIMENSION];
    const SCALE: f64 = 1.0e-6;
    const DISTORTION_WEIGHT: f64 = 10.0;

    fn known_case() -> MultiplierCase {
        let signal = basis_vector(0).expect("e0 exists");

        let mut noise = ZERO;
        noise[4] = 1.0;
        noise[15] = -1.0;

        MultiplierCase::new(signal, noise)
    }

    fn known_zero_divisor() -> Sedenion {
        let mut multiplier = ZERO;
        multiplier[1] = 1.0;
        multiplier[10] = 1.0;
        multiplier
    }

    #[test]
    fn known_zero_divisor_beats_identity() {
        let case = known_case();

        let identity_score = score_multiplier(
            std::slice::from_ref(&case),
            &basis_vector(0).expect("e0 exists"),
            SCALE,
            DISTORTION_WEIGHT,
        )
        .expect("identity score");

        let cayley_score = score_multiplier(
            std::slice::from_ref(&case),
            &known_zero_divisor(),
            SCALE,
            DISTORTION_WEIGHT,
        )
        .expect("Cayley score");

        assert!(identity_score.loss > 0.99);
        assert!(cayley_score.loss < 1.0e-12);
        assert!(cayley_score.loss < identity_score.loss);
        assert!(cayley_score.mean_noise_ratio < 1.0e-16);
        assert!(cayley_score.mean_distortion_ratio < 1.0e-16);
    }

    #[test]
    fn multiplier_score_is_scale_invariant() {
        let case = known_case();
        let multiplier = known_zero_divisor();

        let scaled: Sedenion = core::array::from_fn(|index| 7.0 * multiplier[index]);

        let first = score_multiplier(
            std::slice::from_ref(&case),
            &multiplier,
            SCALE,
            DISTORTION_WEIGHT,
        )
        .expect("first score");

        let second = score_multiplier(
            std::slice::from_ref(&case),
            &scaled,
            SCALE,
            DISTORTION_WEIGHT,
        )
        .expect("second score");

        assert!((first.loss - second.loss).abs() < 1.0e-14);
        assert!((first.mean_noise_ratio - second.mean_noise_ratio).abs() < 1.0e-14);
        assert!((first.mean_distortion_ratio - second.mean_distortion_ratio).abs() < 1.0e-14);
    }

    #[test]
    fn invalid_score_inputs_are_rejected() {
        let case = known_case();

        assert!(score_multiplier(&[], &known_zero_divisor(), SCALE, DISTORTION_WEIGHT,).is_err());

        assert!(
            score_multiplier(std::slice::from_ref(&case), &ZERO, SCALE, DISTORTION_WEIGHT,)
                .is_err()
        );

        assert!(
            score_multiplier(
                std::slice::from_ref(&case),
                &known_zero_divisor(),
                SCALE,
                -1.0,
            )
            .is_err()
        );
    }

    #[test]
    fn invalid_optimizer_step_is_rejected() {
        let case = known_case();

        let result = optimize_multiplier(
            std::slice::from_ref(&case),
            known_zero_divisor(),
            SCALE,
            DISTORTION_WEIGHT,
            0.0,
            Tolerance::default(),
        );

        assert!(result.is_err());
    }

    #[test]
    fn gauge_round_trip_preserves_multiplier_direction() {
        let initial = [
            2.0, -1.0, 0.5, 0.0, 0.25, -0.75, 0.0, 0.0, 0.125, 0.0, 1.0, 0.0, -0.5, 0.0, 0.0, 0.25,
        ];

        let expected = normalize_multiplier(&initial).expect("valid multiplier");

        let pivot = largest_absolute_coordinate(&expected);
        let coordinates = gauge_coordinates(&expected, pivot);
        let reconstructed =
            multiplier_from_gauge(&coordinates, pivot).expect("valid gauge coordinates");

        for (left, right) in expected.iter().zip(reconstructed.iter())
        {
            assert!((left - right).abs() < 1.0e-14);
        }
    }

    #[test]
    fn gauge_pivot_ties_select_first_coordinate() {
        let multiplier = [
            1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        assert_eq!(largest_absolute_coordinate(&multiplier), 0);
    }
}
