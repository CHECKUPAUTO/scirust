//! Deterministic search over SRCC structural configurations.

use core::fmt;

use crate::{
    SrccCase, SrccConfig, SrccFitError, SrccGateDecision, SrccProjector, SrccScore,
    SrccSelectionError, SrccTransportSample, Vector16, fit_srcc_projector,
    fit_srcc_projector_from_views, select_srcc_train_dev,
};

#[derive(Clone, Debug, PartialEq)]
pub struct SrccSearchGrid {
    pub view_counts: Vec<usize>,
    pub resonance_thresholds: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SrccSearchCandidate {
    pub view_count: usize,
    pub resonance_threshold: f64,
    pub projector: SrccProjector,
    pub train_score: SrccScore,
    pub dev_score: SrccScore,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SrccSearchResult {
    pub selected: SrccSearchCandidate,
    pub candidates: Vec<SrccSearchCandidate>,
    pub identity_dev_score: SrccScore,
    pub decision: SrccGateDecision,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SrccSearchError {
    EmptyViewCounts,
    EmptyResonanceThresholds,
    Fit(SrccFitError),
    Selection(SrccSelectionError),
}

impl fmt::Display for SrccSearchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::EmptyViewCounts =>
            {
                formatter.write_str("view-count search grid must not be empty")
            },
            Self::EmptyResonanceThresholds =>
            {
                formatter.write_str("resonance-threshold grid must not be empty")
            },
            Self::Fit(error) => error.fmt(formatter),
            Self::Selection(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for SrccSearchError {}

impl From<SrccFitError> for SrccSearchError {
    fn from(error: SrccFitError) -> Self {
        Self::Fit(error)
    }
}

impl From<SrccSelectionError> for SrccSearchError {
    fn from(error: SrccSelectionError) -> Self {
        Self::Selection(error)
    }
}

pub fn search_srcc_structures(
    seeds: &[Vector16],
    samples: &[SrccTransportSample],
    grid: &SrccSearchGrid,
    base_config: SrccConfig,
    train: &[SrccCase],
    dev: &[SrccCase],
    distortion_weight: f64,
) -> Result<SrccSearchResult, SrccSearchError> {
    if grid.view_counts.is_empty()
    {
        return Err(SrccSearchError::EmptyViewCounts);
    }

    if grid.resonance_thresholds.is_empty()
    {
        return Err(SrccSearchError::EmptyResonanceThresholds);
    }

    let mut ordered_view_counts = grid.view_counts.clone();
    ordered_view_counts.sort_unstable();
    ordered_view_counts.dedup();

    let mut ordered_thresholds = grid.resonance_thresholds.clone();

    ordered_thresholds.sort_by(f64::total_cmp);
    ordered_thresholds.dedup_by(|left, right| left.total_cmp(right).is_eq());

    let mut specifications = Vec::new();
    let mut projectors = Vec::new();

    for view_count in ordered_view_counts
    {
        for &resonance_threshold in &ordered_thresholds
        {
            let config = SrccConfig {
                resonance_threshold,
                ..base_config
            };

            let fit = fit_srcc_projector(seeds, samples, view_count, config)?;

            specifications.push((view_count, resonance_threshold));

            projectors.push(fit.projector);
        }
    }

    let selection = select_srcc_train_dev(&projectors, train, dev, distortion_weight)?;

    let candidates: Vec<_> = selection
        .candidates
        .iter()
        .map(|candidate| {
            let (view_count, resonance_threshold) = specifications[candidate.candidate_index];

            SrccSearchCandidate {
                view_count,
                resonance_threshold,
                projector: candidate.projector.clone(),
                train_score: candidate.train_score,
                dev_score: candidate.dev_score,
            }
        })
        .collect();

    let selected = candidates[0].clone();

    Ok(SrccSearchResult {
        selected,
        candidates,
        identity_dev_score: selection.identity_dev_score,
        decision: selection.decision,
    })
}

pub fn search_srcc_structures_from_views(
    seeds: &[Vector16],
    views: &[&[SrccTransportSample]],
    resonance_thresholds: &[f64],
    base_config: SrccConfig,
    train: &[SrccCase],
    dev: &[SrccCase],
    distortion_weight: f64,
) -> Result<SrccSearchResult, SrccSearchError> {
    if resonance_thresholds.is_empty()
    {
        return Err(SrccSearchError::EmptyResonanceThresholds);
    }

    let mut ordered_thresholds = resonance_thresholds.to_vec();

    ordered_thresholds.sort_by(f64::total_cmp);
    ordered_thresholds.dedup_by(|left, right| left.total_cmp(right).is_eq());

    let mut projectors = Vec::with_capacity(ordered_thresholds.len());

    for &resonance_threshold in &ordered_thresholds
    {
        let config = SrccConfig {
            resonance_threshold,
            ..base_config
        };

        let fit = fit_srcc_projector_from_views(seeds, views, config)?;

        projectors.push(fit.projector);
    }

    let selection = select_srcc_train_dev(&projectors, train, dev, distortion_weight)?;

    let candidates: Vec<_> = selection
        .candidates
        .iter()
        .map(|candidate| {
            let resonance_threshold = ordered_thresholds[candidate.candidate_index];

            SrccSearchCandidate {
                view_count: views.len(),
                resonance_threshold,
                projector: candidate.projector.clone(),
                train_score: candidate.train_score,
                dev_score: candidate.dev_score,
            }
        })
        .collect();

    let selected = candidates[0].clone();

    Ok(SrccSearchResult {
        selected,
        candidates,
        identity_dev_score: selection.identity_dev_score,
        decision: selection.decision,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SrccTransportSample, basis_vector};

    fn samples() -> [SrccTransportSample; 3] {
        let source = basis_vector(1).unwrap();
        let e2 = basis_vector(2).unwrap();

        [
            SrccTransportSample::new(source, e2),
            SrccTransportSample::new(source, e2.map(|value| -value)),
            SrccTransportSample::new(source, basis_vector(3).unwrap()),
        ]
    }

    fn cases() -> [SrccCase; 1] {
        [SrccCase::new(
            basis_vector(8).unwrap(),
            basis_vector(2).unwrap(),
        )]
    }

    #[test]
    fn search_selects_supported_view_structure() {
        let seed = basis_vector(1).unwrap();
        let cases = cases();

        let result = search_srcc_structures(
            &[seed],
            &samples(),
            &SrccSearchGrid {
                view_counts: vec![2, 3],
                resonance_thresholds: vec![0.999],
            },
            SrccConfig::default(),
            &cases,
            &cases,
            10.0,
        )
        .unwrap();

        assert_eq!(result.candidates.len(), 2);
        assert_eq!(result.selected.view_count, 3);

        assert_eq!(result.decision, SrccGateDecision::Srcc,);

        assert!(result.selected.dev_score.loss < 1.0e-24);
    }

    #[test]
    fn search_is_invariant_to_grid_order() {
        let seed = basis_vector(1).unwrap();
        let cases = cases();

        let first = search_srcc_structures(
            &[seed],
            &samples(),
            &SrccSearchGrid {
                view_counts: vec![3, 2],
                resonance_thresholds: vec![1.0, 0.999],
            },
            SrccConfig::default(),
            &cases,
            &cases,
            10.0,
        )
        .unwrap();

        let second = search_srcc_structures(
            &[seed],
            &samples(),
            &SrccSearchGrid {
                view_counts: vec![2, 3],
                resonance_thresholds: vec![0.999, 1.0],
            },
            SrccConfig::default(),
            &cases,
            &cases,
            10.0,
        )
        .unwrap();

        assert_eq!(first, second);
    }

    #[test]
    fn explicit_view_search_selects_consensus() {
        let seed = basis_vector(1).unwrap();
        let target = basis_vector(2).unwrap();

        let positive = [SrccTransportSample::new(seed, target)];

        let negative = [SrccTransportSample::new(seed, target.map(|value| -value))];

        let views = [positive.as_slice(), negative.as_slice()];

        let cases = cases();

        let result = search_srcc_structures_from_views(
            &[seed],
            &views,
            &[1.0, 0.999],
            SrccConfig::default(),
            &cases,
            &cases,
            10.0,
        )
        .unwrap();

        assert_eq!(result.candidates.len(), 2);
        assert_eq!(result.selected.view_count, 2);

        assert_eq!(result.selected.resonance_threshold, 0.999,);

        assert_eq!(result.decision, SrccGateDecision::Srcc,);

        assert!(result.selected.dev_score.loss < 1.0e-24);
    }

    #[test]
    fn explicit_view_search_ignores_threshold_order() {
        let seed = basis_vector(1).unwrap();
        let target = basis_vector(2).unwrap();

        let positive = [SrccTransportSample::new(seed, target)];

        let negative = [SrccTransportSample::new(seed, target.map(|value| -value))];

        let views = [positive.as_slice(), negative.as_slice()];

        let cases = cases();

        let first = search_srcc_structures_from_views(
            &[seed],
            &views,
            &[1.0, 0.999],
            SrccConfig::default(),
            &cases,
            &cases,
            10.0,
        )
        .unwrap();

        let second = search_srcc_structures_from_views(
            &[seed],
            &views,
            &[0.999, 1.0],
            SrccConfig::default(),
            &cases,
            &cases,
            10.0,
        )
        .unwrap();

        assert_eq!(first, second);
    }

    #[test]
    fn empty_search_grids_are_rejected() {
        let seed = basis_vector(1).unwrap();
        let cases = cases();

        assert_eq!(
            search_srcc_structures(
                &[seed],
                &samples(),
                &SrccSearchGrid {
                    view_counts: vec![],
                    resonance_thresholds: vec![0.999],
                },
                SrccConfig::default(),
                &cases,
                &cases,
                10.0,
            ),
            Err(SrccSearchError::EmptyViewCounts),
        );

        assert_eq!(
            search_srcc_structures(
                &[seed],
                &samples(),
                &SrccSearchGrid {
                    view_counts: vec![2],
                    resonance_thresholds: vec![],
                },
                SrccConfig::default(),
                &cases,
                &cases,
                10.0,
            ),
            Err(SrccSearchError::EmptyResonanceThresholds,),
        );
    }
}
