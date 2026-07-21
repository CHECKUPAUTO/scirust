//! End-to-end fitting of SRCC projectors from transport data.

use core::fmt;

use crate::{
    LinearMap16, SrccClosureError, SrccConfig, SrccDiscoveryError, SrccProjector,
    SrccTransportSample, Vector16, learn_interleaved_transport_views,
};

#[derive(Clone, Debug, PartialEq)]
pub struct SrccFitResult {
    pub transports: Vec<LinearMap16>,
    pub projector: SrccProjector,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SrccFitError {
    Discovery(SrccDiscoveryError),
    Closure(SrccClosureError),
}

impl fmt::Display for SrccFitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::Discovery(error) => error.fmt(formatter),
            Self::Closure(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for SrccFitError {}

impl From<SrccDiscoveryError> for SrccFitError {
    fn from(error: SrccDiscoveryError) -> Self {
        Self::Discovery(error)
    }
}

impl From<SrccClosureError> for SrccFitError {
    fn from(error: SrccClosureError) -> Self {
        Self::Closure(error)
    }
}

pub fn fit_srcc_projector(
    seeds: &[Vector16],
    samples: &[SrccTransportSample],
    view_count: usize,
    config: SrccConfig,
) -> Result<SrccFitResult, SrccFitError> {
    let transports = learn_interleaved_transport_views(samples, view_count, config.energy_floor)?;

    let projector = SrccProjector::build(seeds, &transports, config)?;

    Ok(SrccFitResult {
        transports,
        projector,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{basis_vector, squared_norm};

    #[test]
    fn fitted_projector_rejects_discovered_consensus() {
        let seed = basis_vector(1).unwrap();
        let target = basis_vector(2).unwrap();
        let negative = target.map(|value| -value);

        let samples = [
            SrccTransportSample::new(seed, target),
            SrccTransportSample::new(seed, negative),
            SrccTransportSample::new(seed, target),
            SrccTransportSample::new(seed, negative),
        ];

        let result = fit_srcc_projector(&[seed], &samples, 2, SrccConfig::default()).unwrap();

        assert_eq!(result.transports.len(), 2);
        assert_eq!(result.projector.rejected_dimension(), 2,);

        assert!(squared_norm(&result.projector.apply(&target)) < 1.0e-24);

        assert_eq!(result.projector.closure().certificates()[0].support, 2,);
    }

    #[test]
    fn fitting_is_deterministic() {
        let seed = basis_vector(3).unwrap();
        let target = basis_vector(7).unwrap();

        let samples = [
            SrccTransportSample::new(seed, target),
            SrccTransportSample::new(seed, target),
        ];

        let first = fit_srcc_projector(&[seed], &samples, 2, SrccConfig::default()).unwrap();

        let second = fit_srcc_projector(&[seed], &samples, 2, SrccConfig::default()).unwrap();

        assert_eq!(first, second);
    }
}
