//! Deterministic train/development selection for SRCC projectors.

use core::fmt;

use crate::{SrccCase, SrccProjector, SrccScore, SrccScoringError, score_projector, squared_norm};

const ENERGY_FLOOR: f64 = 1.0e-30;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SrccGateDecision {
    Identity,
    Srcc,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelectedSrccCandidate {
    pub candidate_index: usize,
    pub projector: SrccProjector,
    pub train_score: SrccScore,
    pub dev_score: SrccScore,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SrccSelectionResult {
    pub selected: SelectedSrccCandidate,
    pub candidates: Vec<SelectedSrccCandidate>,
    pub identity_dev_score: SrccScore,
    pub decision: SrccGateDecision,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SrccSelectionError {
    EmptyCandidates,
    Scoring(SrccScoringError),
}

impl fmt::Display for SrccSelectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::EmptyCandidates => formatter.write_str("at least one SRCC candidate is required"),
            Self::Scoring(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for SrccSelectionError {}

impl From<SrccScoringError> for SrccSelectionError {
    fn from(error: SrccScoringError) -> Self {
        Self::Scoring(error)
    }
}

pub fn select_srcc_train_dev(
    projectors: &[SrccProjector],
    train: &[SrccCase],
    dev: &[SrccCase],
    distortion_weight: f64,
) -> Result<SrccSelectionResult, SrccSelectionError> {
    if projectors.is_empty()
    {
        return Err(SrccSelectionError::EmptyCandidates);
    }

    let identity_dev_score = score_identity(dev, distortion_weight)?;

    let mut candidates = Vec::with_capacity(projectors.len());

    for (candidate_index, projector) in projectors.iter().enumerate()
    {
        candidates.push(SelectedSrccCandidate {
            candidate_index,
            projector: projector.clone(),
            train_score: score_projector(train, projector, distortion_weight)?,
            dev_score: score_projector(dev, projector, distortion_weight)?,
        });
    }

    candidates.sort_by(|left, right| {
        left.dev_score
            .loss
            .total_cmp(&right.dev_score.loss)
            .then_with(|| left.train_score.loss.total_cmp(&right.train_score.loss))
            .then_with(|| left.candidate_index.cmp(&right.candidate_index))
    });

    let selected = candidates[0].clone();

    let decision = if selected.dev_score.loss < identity_dev_score.loss
    {
        SrccGateDecision::Srcc
    }
    else
    {
        SrccGateDecision::Identity
    };

    Ok(SrccSelectionResult {
        selected,
        candidates,
        identity_dev_score,
        decision,
    })
}

fn score_identity(
    cases: &[SrccCase],
    distortion_weight: f64,
) -> Result<SrccScore, SrccScoringError> {
    if cases.is_empty()
    {
        return Err(SrccScoringError::EmptyCases);
    }

    if !distortion_weight.is_finite() || distortion_weight < 0.0
    {
        return Err(SrccScoringError::InvalidDistortionWeight);
    }

    let residual_sum = cases.iter().fold(0.0, |sum, case| {
        let energy = squared_norm(&case.noise);

        sum + energy / energy.max(ENERGY_FLOOR)
    });

    let mean_residual_noise_ratio = residual_sum / cases.len() as f64;

    Ok(SrccScore {
        mean_residual_noise_ratio,
        mean_signal_distortion_ratio: 0.0,
        loss: mean_residual_noise_ratio,
        case_count: cases.len(),
    })
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

    fn projector(seed: usize, generated: usize) -> SrccProjector {
        SrccProjector::build(
            &[basis_vector(seed).unwrap()],
            &[
                transport(seed, generated, 1.0),
                transport(seed, generated, -1.0),
            ],
            SrccConfig::default(),
        )
        .unwrap()
    }

    #[test]
    fn development_set_selects_correct_structure() {
        let projectors = [projector(1, 2), projector(3, 4)];

        let train = [SrccCase::new(
            basis_vector(8).unwrap(),
            basis_vector(1).unwrap(),
        )];

        let dev = [SrccCase::new(
            basis_vector(9).unwrap(),
            basis_vector(4).unwrap(),
        )];

        let result = select_srcc_train_dev(&projectors, &train, &dev, 10.0).unwrap();

        assert_eq!(result.selected.candidate_index, 1);
        assert_eq!(result.decision, SrccGateDecision::Srcc);
        assert!(result.selected.dev_score.loss < 1.0e-24);
    }

    #[test]
    fn identity_is_selected_when_signal_is_destroyed() {
        let projectors = [projector(1, 2)];

        let cases = [SrccCase::new(
            basis_vector(1).unwrap(),
            basis_vector(8).unwrap(),
        )];

        let result = select_srcc_train_dev(&projectors, &cases, &cases, 10.0).unwrap();

        assert_eq!(result.decision, SrccGateDecision::Identity,);

        assert!(result.selected.dev_score.loss > result.identity_dev_score.loss);
    }

    #[test]
    fn empty_candidate_family_is_rejected() {
        let cases = [SrccCase::new(
            basis_vector(8).unwrap(),
            basis_vector(2).unwrap(),
        )];

        assert_eq!(
            select_srcc_train_dev(&[], &cases, &cases, 10.0,),
            Err(SrccSelectionError::EmptyCandidates),
        );
    }
}
