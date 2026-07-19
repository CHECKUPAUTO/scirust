//! Experimental filtering and falsifiable quality measurements.

use crate::operator::LeftMultiplicationOperator;
use crate::scalar::{Sedenion, squared_norm};

/// Filter based on left multiplication `L_a(x) = a · x`.
#[derive(Clone, Debug, PartialEq)]
pub struct CayleyFilter {
    operator: LeftMultiplicationOperator,
}

/// Results obtained by filtering clean signal and noise separately.
#[derive(Clone, Debug, PartialEq)]
pub struct FilterEvaluation {
    filtered_signal: Sedenion,
    filtered_noise: Sedenion,
    filtered_observation: Sedenion,
    metrics: FilterMetrics,
}

/// Quantitative filter measurements.
///
/// Energies are squared Euclidean norms. Decibel values use power ratios.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilterMetrics {
    pub input_signal_energy: f64,
    pub input_noise_energy: f64,
    pub output_signal_energy: f64,
    pub output_noise_energy: f64,
    pub signal_distortion_energy: f64,
    pub noise_attenuation_db: Option<f64>,
    pub input_snr_db: Option<f64>,
    pub output_snr_db: Option<f64>,
}

impl CayleyFilter {
    /// Constructs a filter from its left multiplier.
    #[must_use]
    pub fn new(multiplier: Sedenion) -> Self {
        Self {
            operator: LeftMultiplicationOperator::new(multiplier),
        }
    }

    /// Applies the filter to one 16-dimensional sample.
    #[must_use]
    pub fn apply(&self, input: &Sedenion) -> Sedenion {
        self.operator.apply(input)
    }

    /// Evaluates signal preservation and noise rejection separately.
    #[must_use]
    pub fn evaluate(&self, signal: &Sedenion, noise: &Sedenion) -> FilterEvaluation {
        let observation = add(signal, noise);

        let filtered_signal = self.apply(signal);
        let filtered_noise = self.apply(noise);
        let filtered_observation = self.apply(&observation);

        let input_signal_energy = squared_norm(signal);
        let input_noise_energy = squared_norm(noise);
        let output_signal_energy = squared_norm(&filtered_signal);
        let output_noise_energy = squared_norm(&filtered_noise);

        let metrics = FilterMetrics {
            input_signal_energy,
            input_noise_energy,
            output_signal_energy,
            output_noise_energy,
            signal_distortion_energy: squared_distance(&filtered_signal, signal),
            noise_attenuation_db: attenuation_db(input_noise_energy, output_noise_energy),
            input_snr_db: power_ratio_db(input_signal_energy, input_noise_energy),
            output_snr_db: power_ratio_db(output_signal_energy, output_noise_energy),
        };

        FilterEvaluation {
            filtered_signal,
            filtered_noise,
            filtered_observation,
            metrics,
        }
    }

    /// Returns the underlying multiplication operator.
    #[must_use]
    pub const fn operator(&self) -> &LeftMultiplicationOperator {
        &self.operator
    }
}

impl FilterEvaluation {
    #[must_use]
    pub const fn filtered_signal(&self) -> &Sedenion {
        &self.filtered_signal
    }

    #[must_use]
    pub const fn filtered_noise(&self) -> &Sedenion {
        &self.filtered_noise
    }

    #[must_use]
    pub const fn filtered_observation(&self) -> &Sedenion {
        &self.filtered_observation
    }

    #[must_use]
    pub const fn metrics(&self) -> &FilterMetrics {
        &self.metrics
    }
}

fn add(left: &Sedenion, right: &Sedenion) -> Sedenion {
    core::array::from_fn(|index| left[index] + right[index])
}

fn squared_distance(left: &Sedenion, right: &Sedenion) -> f64 {
    left.iter()
        .zip(right.iter())
        .fold(0.0, |sum, (left_value, right_value)| {
            let difference = left_value - right_value;
            sum + difference * difference
        })
}

fn power_ratio_db(numerator_energy: f64, denominator_energy: f64) -> Option<f64> {
    match (numerator_energy, denominator_energy)
    {
        (0.0, 0.0) => None,
        (_, 0.0) => Some(f64::INFINITY),
        (0.0, _) => Some(f64::NEG_INFINITY),
        _ => Some(10.0 * (numerator_energy / denominator_energy).log10()),
    }
}

fn attenuation_db(input_energy: f64, output_energy: f64) -> Option<f64> {
    match (input_energy, output_energy)
    {
        (0.0, _) => None,
        (_, 0.0) => Some(f64::INFINITY),
        _ => Some(10.0 * (input_energy / output_energy).log10()),
    }
}

#[cfg(test)]
mod tests {
    use super::CayleyFilter;
    use crate::scalar::{SEDENION_DIMENSION, Sedenion, basis_vector, squared_norm};

    const ZERO: Sedenion = [0.0; SEDENION_DIMENSION];

    #[test]
    fn identity_filter_preserves_signal_and_noise() {
        let filter = CayleyFilter::new(basis_vector(0).expect("e0 exists"));

        let signal = basis_vector(2).expect("e2 exists");
        let noise = basis_vector(9).expect("e9 exists");
        let evaluation = filter.evaluate(&signal, &noise);

        assert_eq!(evaluation.filtered_signal(), &signal);
        assert_eq!(evaluation.filtered_noise(), &noise);
        assert_eq!(evaluation.metrics().signal_distortion_energy, 0.0);
        assert_eq!(evaluation.metrics().noise_attenuation_db, Some(0.0));
        assert_eq!(
            evaluation.metrics().input_snr_db,
            evaluation.metrics().output_snr_db
        );
    }

    #[test]
    fn known_kernel_noise_is_annihilated_exactly() {
        let mut multiplier = ZERO;
        multiplier[1] = 1.0;
        multiplier[10] = 1.0;

        let mut noise = ZERO;
        noise[4] = 1.0;
        noise[15] = -1.0;

        let signal = basis_vector(0).expect("e0 exists");
        let evaluation = CayleyFilter::new(multiplier).evaluate(&signal, &noise);

        assert_eq!(squared_norm(&noise), 2.0);
        assert_eq!(evaluation.filtered_noise(), &ZERO);
        assert_eq!(evaluation.metrics().output_noise_energy, 0.0);
        assert_eq!(
            evaluation.metrics().noise_attenuation_db,
            Some(f64::INFINITY)
        );
        assert_eq!(evaluation.metrics().output_snr_db, Some(f64::INFINITY));
    }

    #[test]
    fn filtered_observation_respects_linearity() {
        let mut multiplier = ZERO;
        multiplier[1] = 1.0;
        multiplier[10] = 1.0;

        let signal = basis_vector(3).expect("e3 exists");
        let noise = basis_vector(7).expect("e7 exists");

        let evaluation = CayleyFilter::new(multiplier).evaluate(&signal, &noise);

        let expected: Sedenion = core::array::from_fn(|index| {
            evaluation.filtered_signal()[index] + evaluation.filtered_noise()[index]
        });

        assert_eq!(evaluation.filtered_observation(), &expected);
    }

    #[test]
    fn zero_signal_and_zero_noise_have_undefined_snr() {
        let filter = CayleyFilter::new(basis_vector(0).expect("e0 exists"));
        let evaluation = filter.evaluate(&ZERO, &ZERO);

        assert_eq!(evaluation.metrics().input_snr_db, None);
        assert_eq!(evaluation.metrics().output_snr_db, None);
        assert_eq!(evaluation.metrics().noise_attenuation_db, None);
    }
}
