//! Leave-one-out stability analysis for SRCC projectors.

use core::fmt;

use crate::{
    SRCC_DIMENSION, SrccConfig, SrccFitError, SrccProjector, SrccTransportSample, Vector16,
    fit_srcc_projector_from_views,
};

#[derive(Clone, Debug, PartialEq)]
pub struct SrccStabilityVariant {
    pub removed_view_index: usize,
    pub removed_sample_index: usize,
    pub rejected_dimension: usize,
    pub frobenius_distance: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SrccStabilityReport {
    pub full_projector: SrccProjector,
    pub variants: Vec<SrccStabilityVariant>,
    pub mean_frobenius_distance: f64,
    pub maximum_frobenius_distance: f64,
    pub stable_dimension_count: usize,
}

impl SrccStabilityReport {
    #[must_use]
    pub fn removal_count(&self) -> usize {
        self.variants.len()
    }

    #[must_use]
    pub fn dimension_stability_ratio(&self) -> f64 {
        self.stable_dimension_count as f64 / self.variants.len() as f64
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SrccStabilityError {
    NoRemovableSamples,
    Fit(SrccFitError),
}

impl fmt::Display for SrccStabilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::NoRemovableSamples =>
            {
                formatter.write_str("at least one explicit view must contain two samples")
            },
            Self::Fit(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for SrccStabilityError {}

impl From<SrccFitError> for SrccStabilityError {
    fn from(error: SrccFitError) -> Self {
        Self::Fit(error)
    }
}

pub fn evaluate_leave_one_out_stability(
    seeds: &[Vector16],
    views: &[&[SrccTransportSample]],
    config: SrccConfig,
) -> Result<SrccStabilityReport, SrccStabilityError> {
    let full = fit_srcc_projector_from_views(seeds, views, config)?;

    let full_dimension = full.projector.rejected_dimension();

    let mut variants = Vec::new();

    for (view_index, view) in views.iter().enumerate()
    {
        if view.len() <= 1
        {
            continue;
        }

        for sample_index in 0..view.len()
        {
            let mut reduced_views: Vec<Vec<SrccTransportSample>> =
                views.iter().map(|samples| samples.to_vec()).collect();

            reduced_views[view_index].remove(sample_index);

            let reduced_references: Vec<&[SrccTransportSample]> =
                reduced_views.iter().map(Vec::as_slice).collect();

            let reduced = fit_srcc_projector_from_views(seeds, &reduced_references, config)?;

            variants.push(SrccStabilityVariant {
                removed_view_index: view_index,
                removed_sample_index: sample_index,
                rejected_dimension: reduced.projector.rejected_dimension(),
                frobenius_distance: projector_frobenius_distance(
                    &full.projector,
                    &reduced.projector,
                ),
            });
        }
    }

    if variants.is_empty()
    {
        return Err(SrccStabilityError::NoRemovableSamples);
    }

    let distance_sum = variants
        .iter()
        .map(|variant| variant.frobenius_distance)
        .sum::<f64>();

    let maximum_frobenius_distance = variants
        .iter()
        .map(|variant| variant.frobenius_distance)
        .fold(0.0, f64::max);

    let stable_dimension_count = variants
        .iter()
        .filter(|variant| variant.rejected_dimension == full_dimension)
        .count();

    Ok(SrccStabilityReport {
        full_projector: full.projector,
        mean_frobenius_distance: distance_sum / variants.len() as f64,
        maximum_frobenius_distance,
        stable_dimension_count,
        variants,
    })
}

fn projector_frobenius_distance(left: &SrccProjector, right: &SrccProjector) -> f64 {
    let squared_distance = left
        .transform()
        .iter()
        .flatten()
        .zip(right.transform().iter().flatten())
        .fold(0.0, |sum, (left_value, right_value)| {
            let difference = left_value - right_value;

            sum + difference * difference
        });

    squared_distance.sqrt() / (SRCC_DIMENSION as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis_vector;

    #[test]
    fn duplicated_view_samples_are_stable() {
        let seed = basis_vector(1).unwrap();
        let target = basis_vector(2).unwrap();

        let first = [
            SrccTransportSample::new(seed, target),
            SrccTransportSample::new(seed, target),
        ];

        let second_target = target.map(|value| -value);

        let second = [
            SrccTransportSample::new(seed, second_target),
            SrccTransportSample::new(seed, second_target),
        ];

        let views = [first.as_slice(), second.as_slice()];

        let report =
            evaluate_leave_one_out_stability(&[seed], &views, SrccConfig::default()).unwrap();

        assert_eq!(report.removal_count(), 4);

        assert_eq!(report.stable_dimension_count, report.removal_count(),);

        assert_eq!(report.dimension_stability_ratio(), 1.0,);

        assert!(report.maximum_frobenius_distance < 1.0e-12);
    }

    #[test]
    fn unsupported_single_samples_are_detected() {
        let seed = basis_vector(1).unwrap();
        let first_target = basis_vector(2).unwrap();
        let second_target = basis_vector(3).unwrap();

        let first = [
            SrccTransportSample::new(seed, first_target),
            SrccTransportSample::new(seed, second_target),
        ];

        let second = first;

        let views = [first.as_slice(), second.as_slice()];

        let report =
            evaluate_leave_one_out_stability(&[seed], &views, SrccConfig::default()).unwrap();

        assert!(report.maximum_frobenius_distance > 1.0e-6);

        assert!(report.stable_dimension_count < report.removal_count());
    }

    #[test]
    fn singleton_views_have_no_removable_sample() {
        let seed = basis_vector(1).unwrap();
        let target = basis_vector(2).unwrap();

        let first = [SrccTransportSample::new(seed, target)];

        let second = [SrccTransportSample::new(seed, target.map(|value| -value))];

        let views = [first.as_slice(), second.as_slice()];

        assert_eq!(
            evaluate_leave_one_out_stability(&[seed], &views, SrccConfig::default(),),
            Err(SrccStabilityError::NoRemovableSamples,),
        );
    }
}
