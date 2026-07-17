use std::collections::BTreeMap;

use crate::{
    Action, BranchingDistributionError, DistributionMathError, ExactRatio, State, StateError,
    TransitionSystem, distribution_overlap, uniform_branching_state_distribution,
};

/// Analyse exacte de la convergence distributionnelle après perturbation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BranchingRecoveryAnalysis {
    reference_state: State,
    perturbed_state: State,
    reference_distributions: Vec<BTreeMap<State, ExactRatio>>,
    perturbed_distributions: Vec<BTreeMap<State, ExactRatio>>,
    overlap_profile: Vec<ExactRatio>,
}

/// Erreurs de l'analyse de récupération branchante.
#[derive(Debug)]
pub enum BranchingRecoveryError<E> {
    Perturbation(StateError),
    ReferenceDistribution(BranchingDistributionError<E>),
    PerturbedDistribution(BranchingDistributionError<E>),
    DistributionMath(DistributionMathError),
}

impl BranchingRecoveryAnalysis {
    /// État non perturbé utilisé comme origine.
    #[must_use]
    pub const fn reference_state(&self) -> State {
        self.reference_state
    }

    /// État obtenu immédiatement après perturbation.
    #[must_use]
    pub const fn perturbed_state(&self) -> State {
        self.perturbed_state
    }

    /// Nombre de profondeurs prospectives analysées.
    #[must_use]
    pub fn horizon(&self) -> usize {
        self.overlap_profile.len()
    }

    /// Distributions non perturbées aux profondeurs `1..=horizon`.
    #[must_use]
    pub fn reference_distributions(&self) -> &[BTreeMap<State, ExactRatio>] {
        &self.reference_distributions
    }

    /// Distributions perturbées aux profondeurs `1..=horizon`.
    #[must_use]
    pub fn perturbed_distributions(&self) -> &[BTreeMap<State, ExactRatio>] {
        &self.perturbed_distributions
    }

    /// Recouvrement exact des deux distributions à chaque profondeur.
    ///
    /// Un recouvrement de `1` signifie que les distributions sont identiques.
    /// Un recouvrement de `0` signifie qu'elles sont disjointes.
    #[must_use]
    pub fn overlap_profile(&self) -> &[ExactRatio] {
        &self.overlap_profile
    }

    /// Recouvrement à la dernière profondeur analysée.
    #[must_use]
    pub fn final_overlap(&self) -> Option<ExactRatio> {
        self.overlap_profile.last().cloned()
    }

    /// Indique si les distributions sont identiques à l'horizon final.
    #[must_use]
    pub fn fully_recovered(&self) -> bool {
        self.final_overlap() == ExactRatio::new(1, 1)
    }
}

/// Compare exactement les distributions futures d'un état et de sa version
/// perturbée.
///
/// À chaque profondeur, chaque successeur distinct est choisi avec une
/// probabilité uniforme locale.
pub fn analyze_branching_recovery<S>(
    system: &S,
    reference_state: State,
    perturbation: Action,
    action: Action,
    horizon: usize,
) -> Result<BranchingRecoveryAnalysis, BranchingRecoveryError<S::Error>>
where
    S: TransitionSystem,
{
    let perturbed_state = perturbation
        .apply(reference_state)
        .map_err(BranchingRecoveryError::Perturbation)?;

    let mut reference_distributions = Vec::with_capacity(horizon);
    let mut perturbed_distributions = Vec::with_capacity(horizon);
    let mut overlap_profile = Vec::with_capacity(horizon);

    for depth in 1..=horizon
    {
        let reference_distribution =
            uniform_branching_state_distribution(system, reference_state, action, depth)
                .map_err(BranchingRecoveryError::ReferenceDistribution)?;

        let perturbed_distribution =
            uniform_branching_state_distribution(system, perturbed_state, action, depth)
                .map_err(BranchingRecoveryError::PerturbedDistribution)?;

        let overlap = distribution_overlap(&reference_distribution, &perturbed_distribution)
            .map_err(BranchingRecoveryError::DistributionMath)?;

        reference_distributions.push(reference_distribution);
        perturbed_distributions.push(perturbed_distribution);
        overlap_profile.push(overlap);
    }

    Ok(BranchingRecoveryAnalysis {
        reference_state,
        perturbed_state,
        reference_distributions,
        perturbed_distributions,
        overlap_profile,
    })
}

#[cfg(test)]
mod tests {
    use crate::{Action, ExactRatio, State, TableSystem, analyze_branching_recovery};

    #[test]
    fn detects_complete_reconvergence() {
        let zero = State::new(0b00, 2).expect("valid state");
        let one = State::new(0b01, 2).expect("valid state");
        let two = State::new(0b10, 2).expect("valid state");
        let three = State::new(0b11, 2).expect("valid state");

        let mut system = TableSystem::new(2).expect("valid system");

        system
            .insert(zero, Action::Noop, vec![two])
            .expect("valid transition");

        system
            .insert(one, Action::Noop, vec![three])
            .expect("valid transition");

        system
            .insert(two, Action::Noop, vec![zero])
            .expect("valid transition");

        system
            .insert(three, Action::Noop, vec![zero])
            .expect("valid transition");

        let analysis =
            analyze_branching_recovery(&system, zero, Action::Flip { node: 0 }, Action::Noop, 2)
                .expect("analysis succeeds");

        assert_eq!(
            analysis.overlap_profile(),
            &[
                ExactRatio::new(0, 1).expect("valid ratio"),
                ExactRatio::new(1, 1).expect("valid ratio"),
            ]
        );

        assert!(analysis.fully_recovered());
    }

    #[test]
    fn detects_persistent_separation() {
        let zero = State::new(0b00, 2).expect("valid state");
        let one = State::new(0b01, 2).expect("valid state");

        let mut system = TableSystem::new(2).expect("valid system");

        system
            .insert(zero, Action::Noop, vec![zero])
            .expect("valid transition");

        system
            .insert(one, Action::Noop, vec![one])
            .expect("valid transition");

        let analysis =
            analyze_branching_recovery(&system, zero, Action::Flip { node: 0 }, Action::Noop, 3)
                .expect("analysis succeeds");

        assert_eq!(
            analysis.overlap_profile(),
            &[
                ExactRatio::new(0, 1).expect("valid ratio"),
                ExactRatio::new(0, 1).expect("valid ratio"),
                ExactRatio::new(0, 1).expect("valid ratio"),
            ]
        );

        assert!(!analysis.fully_recovered());
    }

    #[test]
    fn measures_partial_distributional_overlap() {
        let zero = State::new(0b00, 2).expect("valid state");
        let one = State::new(0b01, 2).expect("valid state");
        let two = State::new(0b10, 2).expect("valid state");
        let three = State::new(0b11, 2).expect("valid state");

        let mut system = TableSystem::new(2).expect("valid system");

        system
            .insert(zero, Action::Noop, vec![two, three])
            .expect("valid transition");

        system
            .insert(one, Action::Noop, vec![three])
            .expect("valid transition");

        let analysis =
            analyze_branching_recovery(&system, zero, Action::Flip { node: 0 }, Action::Noop, 1)
                .expect("analysis succeeds");

        assert_eq!(analysis.final_overlap(), ExactRatio::new(1, 2));

        assert!(!analysis.fully_recovered());
    }

    #[test]
    fn supports_zero_horizon() {
        let zero = State::new(0b00, 2).expect("valid state");

        let system = TableSystem::new(2).expect("valid system");

        let analysis =
            analyze_branching_recovery(&system, zero, Action::Flip { node: 0 }, Action::Noop, 0)
                .expect("analysis succeeds");

        assert_eq!(analysis.horizon(), 0);
        assert_eq!(analysis.final_overlap(), None);
        assert!(!analysis.fully_recovered());
    }
}
