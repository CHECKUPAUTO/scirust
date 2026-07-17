use std::collections::BTreeSet;

use crate::{
    Action, OrbitAnalysis, OrbitError, State, StateError, TransitionSystem, analyze_orbit,
};

/// Résultat exact d'une perturbation appliquée à un système déterministe.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecoveryAnalysis {
    reference_state: State,
    perturbed_state: State,
    reference_orbit: OrbitAnalysis,
    perturbed_orbit: OrbitAnalysis,
    recovered: bool,
    recovery_time: Option<usize>,
}

/// Erreurs produites par l'analyse de récupération.
#[derive(Debug)]
pub enum RecoveryError<E> {
    Perturbation(StateError),
    ReferenceOrbit(OrbitError<E>),
    PerturbedOrbit(OrbitError<E>),
}

impl RecoveryAnalysis {
    /// État de l'attracteur sur lequel la perturbation a été appliquée.
    #[must_use]
    pub const fn reference_state(&self) -> State {
        self.reference_state
    }

    /// État obtenu immédiatement après la perturbation.
    #[must_use]
    pub const fn perturbed_state(&self) -> State {
        self.perturbed_state
    }

    /// Orbite non perturbée.
    #[must_use]
    pub const fn reference_orbit(&self) -> &OrbitAnalysis {
        &self.reference_orbit
    }

    /// Orbite issue de la perturbation.
    #[must_use]
    pub const fn perturbed_orbit(&self) -> &OrbitAnalysis {
        &self.perturbed_orbit
    }

    /// Indique si l'attracteur original est retrouvé.
    #[must_use]
    pub const fn recovered(&self) -> bool {
        self.recovered
    }

    /// Nombre d'étapes nécessaires pour rejoindre l'attracteur original.
    #[must_use]
    pub const fn recovery_time(&self) -> Option<usize> {
        self.recovery_time
    }
}

/// Analyse une perturbation appliquée au premier état de l'attracteur de
/// référence.
///
/// Deux attracteurs cycliques sont considérés identiques lorsque leurs
/// ensembles d'états sont identiques, indépendamment du point de départ
/// choisi dans le cycle.
pub fn analyze_recovery<S>(
    system: &S,
    initial: State,
    perturbation: Action,
    max_steps: usize,
) -> Result<RecoveryAnalysis, RecoveryError<S::Error>>
where
    S: TransitionSystem,
{
    let reference_orbit = analyze_orbit(system, initial, Action::Noop, max_steps)
        .map_err(RecoveryError::ReferenceOrbit)?;

    let reference_state = reference_orbit.attractor()[0];

    let perturbed_state = perturbation
        .apply(reference_state)
        .map_err(RecoveryError::Perturbation)?;

    let perturbed_orbit = analyze_orbit(system, perturbed_state, Action::Noop, max_steps)
        .map_err(RecoveryError::PerturbedOrbit)?;

    let reference_attractor: BTreeSet<State> =
        reference_orbit.attractor().iter().copied().collect();

    let perturbed_attractor: BTreeSet<State> =
        perturbed_orbit.attractor().iter().copied().collect();

    let recovered = reference_attractor == perturbed_attractor;

    let recovery_time = recovered.then(|| {
        perturbed_orbit
            .states()
            .iter()
            .position(|state| reference_attractor.contains(state))
            .expect("a recovered orbit must enter the reference attractor")
    });

    Ok(RecoveryAnalysis {
        reference_state,
        perturbed_state,
        reference_orbit,
        perturbed_orbit,
        recovered,
        recovery_time,
    })
}

#[cfg(test)]
mod tests {
    use crate::{Action, State, TableSystem, analyze_recovery};

    fn recovering_system() -> TableSystem {
        let zero = State::new(0b00, 2).expect("valid state");
        let one = State::new(0b01, 2).expect("valid state");
        let two = State::new(0b10, 2).expect("valid state");
        let three = State::new(0b11, 2).expect("valid state");

        let mut system = TableSystem::new(2).expect("valid system");

        for (source, target) in [(zero, one), (one, one), (two, one), (three, one)]
        {
            system
                .insert(source, Action::Noop, vec![target])
                .expect("valid transition");
        }

        system
    }

    fn switching_system() -> TableSystem {
        let zero = State::new(0b00, 2).expect("valid state");
        let one = State::new(0b01, 2).expect("valid state");
        let two = State::new(0b10, 2).expect("valid state");
        let three = State::new(0b11, 2).expect("valid state");

        let mut system = TableSystem::new(2).expect("valid system");

        for (source, target) in [(zero, one), (one, one), (two, two), (three, two)]
        {
            system
                .insert(source, Action::Noop, vec![target])
                .expect("valid transition");
        }

        system
    }

    #[test]
    fn detects_recovery_after_perturbation() {
        let initial = State::new(0b00, 2).expect("valid state");

        let analysis = analyze_recovery(&recovering_system(), initial, Action::Flip { node: 1 }, 8)
            .expect("recovery analysis succeeds");

        assert_eq!(analysis.reference_state().bits(), 0b01);
        assert_eq!(analysis.perturbed_state().bits(), 0b11);
        assert!(analysis.recovered());
        assert_eq!(analysis.recovery_time(), Some(1));
    }

    #[test]
    fn detects_switch_to_another_attractor() {
        let initial = State::new(0b00, 2).expect("valid state");

        let analysis = analyze_recovery(&switching_system(), initial, Action::Flip { node: 1 }, 8)
            .expect("recovery analysis succeeds");

        assert!(!analysis.recovered());
        assert_eq!(analysis.recovery_time(), None);
        assert_eq!(analysis.perturbed_orbit().attractor()[0].bits(), 0b10);
    }
}
