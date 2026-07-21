//! Falsifiable signal-preservation and noise-rejection metrics.

use crate::{SrccProjector, Vector16, squared_norm};

const ENERGY_FLOOR: f64 = 1.0e-30;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SrccEvaluation {
    pub input_signal_energy: f64,
    pub output_signal_energy: f64,
    pub input_noise_energy: f64,
    pub output_noise_energy: f64,
    pub signal_distortion_ratio: f64,
    pub residual_noise_ratio: f64,
}

impl SrccEvaluation {
    #[must_use]
    pub fn evaluate(projector: &SrccProjector, signal: &Vector16, noise: &Vector16) -> Self {
        let filtered_signal = projector.apply(signal);
        let filtered_noise = projector.apply(noise);

        let input_signal_energy = squared_norm(signal);
        let output_signal_energy = squared_norm(&filtered_signal);
        let input_noise_energy = squared_norm(noise);
        let output_noise_energy = squared_norm(&filtered_noise);

        let signal_error: Vector16 =
            core::array::from_fn(|index| filtered_signal[index] - signal[index]);

        let signal_distortion_ratio =
            squared_norm(&signal_error) / input_signal_energy.max(ENERGY_FLOOR);

        let residual_noise_ratio = output_noise_energy / input_noise_energy.max(ENERGY_FLOOR);

        Self {
            input_signal_energy,
            output_signal_energy,
            input_noise_energy,
            output_noise_energy,
            signal_distortion_ratio,
            residual_noise_ratio,
        }
    }

    #[must_use]
    pub fn loss(self, distortion_weight: f64) -> f64 {
        self.residual_noise_ratio + distortion_weight * self.signal_distortion_ratio
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LinearMap16, SRCC_DIMENSION, SrccConfig, basis_vector};

    fn transport(source: usize, target: usize, coefficient: f64) -> LinearMap16 {
        let mut map = [[0.0; SRCC_DIMENSION]; SRCC_DIMENSION];

        map[target][source] = coefficient;
        map
    }

    #[test]
    fn exact_closure_noise_is_rejected() {
        let projector = SrccProjector::build(
            &[basis_vector(1).unwrap()],
            &[transport(1, 2, 1.0), transport(1, 2, -1.0)],
            SrccConfig::default(),
        )
        .unwrap();

        let evaluation = SrccEvaluation::evaluate(
            &projector,
            &basis_vector(8).unwrap(),
            &basis_vector(2).unwrap(),
        );

        assert!(evaluation.residual_noise_ratio < 1.0e-24);
        assert!(evaluation.signal_distortion_ratio < 1.0e-24);
        assert!(evaluation.loss(10.0) < 1.0e-24);
    }

    #[test]
    fn rejected_signal_is_reported_as_distortion() {
        let projector = SrccProjector::build(
            &[basis_vector(1).unwrap()],
            &[transport(1, 2, 1.0), transport(1, 2, -1.0)],
            SrccConfig::default(),
        )
        .unwrap();

        let evaluation = SrccEvaluation::evaluate(
            &projector,
            &basis_vector(1).unwrap(),
            &basis_vector(8).unwrap(),
        );

        assert!((evaluation.signal_distortion_ratio - 1.0).abs() < 1.0e-15);
        assert!((evaluation.residual_noise_ratio - 1.0).abs() < 1.0e-15);
        assert!((evaluation.loss(10.0) - 11.0).abs() < 1.0e-15);
    }
}
