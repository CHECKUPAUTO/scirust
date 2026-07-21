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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SrccStableSearchConfig {
    pub base_config: SrccConfig,
    pub distortion_weight: f64,
    pub maximum_frobenius_distance: f64,
    pub minimum_dimension_stability_ratio: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SrccStableSearchCandidate {
    pub candidate: SrccSearchCandidate,
    pub stability: crate::SrccStabilityReport,
    pub passes_stability_gate: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SrccStableSearchResult {
    pub selected: Option<SrccStableSearchCandidate>,
    pub candidates: Vec<SrccStableSearchCandidate>,
    pub identity_dev_score: SrccScore,
    pub decision: SrccGateDecision,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SrccStableSearchError {
    InvalidMaximumFrobeniusDistance,
    InvalidMinimumDimensionStabilityRatio,
    Search(SrccSearchError),
    Stability(crate::SrccStabilityError),
}

impl fmt::Display for SrccStableSearchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidMaximumFrobeniusDistance =>
            {
                formatter.write_str("maximum Frobenius distance must be finite and non-negative")
            },
            Self::InvalidMinimumDimensionStabilityRatio =>
            {
                formatter.write_str("minimum dimension stability ratio must belong to [0, 1]")
            },
            Self::Search(error) => error.fmt(formatter),
            Self::Stability(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for SrccStableSearchError {}

impl From<SrccSearchError> for SrccStableSearchError {
    fn from(error: SrccSearchError) -> Self {
        Self::Search(error)
    }
}

impl From<crate::SrccStabilityError> for SrccStableSearchError {
    fn from(error: crate::SrccStabilityError) -> Self {
        Self::Stability(error)
    }
}

pub fn search_stable_srcc_structures_from_views(
    seeds: &[Vector16],
    views: &[&[SrccTransportSample]],
    resonance_thresholds: &[f64],
    config: SrccStableSearchConfig,
    train: &[SrccCase],
    dev: &[SrccCase],
) -> Result<SrccStableSearchResult, SrccStableSearchError> {
    if !config.maximum_frobenius_distance.is_finite() || config.maximum_frobenius_distance < 0.0
    {
        return Err(SrccStableSearchError::InvalidMaximumFrobeniusDistance);
    }

    if !config.minimum_dimension_stability_ratio.is_finite()
        || !(0.0..=1.0).contains(&config.minimum_dimension_stability_ratio)
    {
        return Err(SrccStableSearchError::InvalidMinimumDimensionStabilityRatio);
    }

    let search = search_srcc_structures_from_views(
        seeds,
        views,
        resonance_thresholds,
        config.base_config,
        train,
        dev,
        config.distortion_weight,
    )?;

    let mut candidates = Vec::with_capacity(search.candidates.len());

    for candidate in search.candidates
    {
        let threshold_config = SrccConfig {
            resonance_threshold: candidate.resonance_threshold,
            ..config.base_config
        };

        let stability = crate::evaluate_leave_one_out_stability(seeds, views, threshold_config)?;

        let passes_stability_gate = stability.maximum_frobenius_distance
            <= config.maximum_frobenius_distance
            && stability.dimension_stability_ratio() >= config.minimum_dimension_stability_ratio;

        candidates.push(SrccStableSearchCandidate {
            candidate,
            stability,
            passes_stability_gate,
        });
    }

    let selected = candidates
        .iter()
        .find(|candidate| candidate.passes_stability_gate)
        .cloned();

    let decision = match &selected
    {
        Some(candidate) if candidate.candidate.dev_score.loss < search.identity_dev_score.loss =>
        {
            SrccGateDecision::Srcc
        },
        _ => SrccGateDecision::Identity,
    };

    Ok(SrccStableSearchResult {
        selected,
        candidates,
        identity_dev_score: search.identity_dev_score,
        decision,
    })
}

#[cfg(test)]
mod stable_search_tests {
    use super::*;
    use crate::{SrccTransportSample, basis_vector};

    fn stable_config() -> SrccStableSearchConfig {
        SrccStableSearchConfig {
            base_config: SrccConfig::default(),
            distortion_weight: 10.0,
            maximum_frobenius_distance: 1.0e-12,
            minimum_dimension_stability_ratio: 1.0,
        }
    }

    #[test]
    fn stable_candidate_passes_gate() {
        let seed = basis_vector(1).unwrap();
        let target = basis_vector(2).unwrap();

        let positive = [
            SrccTransportSample::new(seed, target),
            SrccTransportSample::new(seed, target),
        ];

        let negative_target = target.map(|value| -value);

        let negative = [
            SrccTransportSample::new(seed, negative_target),
            SrccTransportSample::new(seed, negative_target),
        ];

        let views = [positive.as_slice(), negative.as_slice()];

        let cases = [SrccCase::new(basis_vector(8).unwrap(), target)];

        let result = search_stable_srcc_structures_from_views(
            &[seed],
            &views,
            &[0.999],
            stable_config(),
            &cases,
            &cases,
        )
        .unwrap();

        assert_eq!(result.decision, SrccGateDecision::Srcc,);

        assert!(result.selected.as_ref().unwrap().passes_stability_gate);
    }

    #[test]
    fn unstable_candidate_falls_back_to_identity() {
        let seed = basis_vector(1).unwrap();
        let first_target = basis_vector(2).unwrap();
        let second_target = basis_vector(3).unwrap();

        let first = [
            SrccTransportSample::new(seed, first_target),
            SrccTransportSample::new(seed, second_target),
        ];

        let second = first;

        let views = [first.as_slice(), second.as_slice()];

        let scale = std::f64::consts::FRAC_1_SQRT_2;

        let mut noise = [0.0; crate::SRCC_DIMENSION];
        noise[2] = scale;
        noise[3] = scale;

        let cases = [SrccCase::new(basis_vector(8).unwrap(), noise)];

        let result = search_stable_srcc_structures_from_views(
            &[seed],
            &views,
            &[0.999],
            stable_config(),
            &cases,
            &cases,
        )
        .unwrap();

        assert_eq!(result.decision, SrccGateDecision::Identity,);

        assert!(result.selected.is_none());

        assert!(!result.candidates[0].passes_stability_gate);
    }

    #[test]
    fn invalid_stability_gate_is_rejected() {
        let seed = basis_vector(1).unwrap();
        let target = basis_vector(2).unwrap();

        let first = [
            SrccTransportSample::new(seed, target),
            SrccTransportSample::new(seed, target),
        ];

        let second = first;

        let views = [first.as_slice(), second.as_slice()];

        let cases = [SrccCase::new(basis_vector(8).unwrap(), target)];

        let mut config = stable_config();
        config.maximum_frobenius_distance = -1.0;

        assert_eq!(
            search_stable_srcc_structures_from_views(
                &[seed],
                &views,
                &[0.999],
                config,
                &cases,
                &cases,
            ),
            Err(SrccStableSearchError::InvalidMaximumFrobeniusDistance,),
        );

        let mut config = stable_config();
        config.minimum_dimension_stability_ratio = 1.1;

        assert_eq!(
            search_stable_srcc_structures_from_views(
                &[seed],
                &views,
                &[0.999],
                config,
                &cases,
                &cases,
            ),
            Err(SrccStableSearchError::InvalidMinimumDimensionStabilityRatio,),
        );
    }
}
