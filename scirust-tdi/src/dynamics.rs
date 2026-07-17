use std::collections::BTreeMap;

use crate::{Action, State, TransitionSystem};

/// Décomposition exacte d'une trajectoire déterministe en transitoire et cycle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrbitAnalysis {
    states: Vec<State>,
    transient_len: usize,
    period: usize,
}

/// Erreurs d'analyse d'une dynamique déterministe.
#[derive(Debug)]
pub enum OrbitError<E> {
    InitialWidthMismatch {
        expected: u8,
        actual: u8,
    },
    NonDeterministicTransition {
        state: State,
        successor_count: usize,
    },
    NoRecurrenceWithinLimit {
        max_steps: usize,
    },
    System(E),
}

impl OrbitAnalysis {
    /// États visités avant la première répétition.
    #[must_use]
    pub fn states(&self) -> &[State] {
        &self.states
    }

    /// Nombre d'états précédant l'attracteur.
    #[must_use]
    pub const fn transient_len(&self) -> usize {
        self.transient_len
    }

    /// Période exacte de l'attracteur.
    #[must_use]
    pub const fn period(&self) -> usize {
        self.period
    }

    /// États constituant l'attracteur.
    #[must_use]
    pub fn attractor(&self) -> &[State] {
        &self.states[self.transient_len..]
    }

    /// Indique si l'attracteur est un point fixe.
    #[must_use]
    pub const fn is_fixed_point(&self) -> bool {
        self.period == 1
    }
}

/// Analyse exactement une trajectoire déterministe jusqu'à sa première
/// récurrence.
pub fn analyze_orbit<S>(
    system: &S,
    initial: State,
    action: Action,
    max_steps: usize,
) -> Result<OrbitAnalysis, OrbitError<S::Error>>
where
    S: TransitionSystem,
{
    if initial.width() != system.width()
    {
        return Err(OrbitError::InitialWidthMismatch {
            expected: system.width(),
            actual: initial.width(),
        });
    }

    let mut first_seen = BTreeMap::<State, usize>::new();
    let mut states = Vec::new();
    let mut current = initial;

    for step in 0..=max_steps
    {
        if let Some(&cycle_start) = first_seen.get(&current)
        {
            return Ok(OrbitAnalysis {
                transient_len: cycle_start,
                period: step - cycle_start,
                states,
            });
        }

        if step == max_steps
        {
            break;
        }

        first_seen.insert(current, step);
        states.push(current);

        let successors = system
            .successors(current, action)
            .map_err(OrbitError::System)?;

        if successors.len() != 1
        {
            return Err(OrbitError::NonDeterministicTransition {
                state: current,
                successor_count: successors.len(),
            });
        }

        current = successors[0];
    }

    Err(OrbitError::NoRecurrenceWithinLimit { max_steps })
}

#[cfg(test)]
mod tests {
    use crate::{Action, OrbitError, State, TableSystem, analyze_orbit};

    #[test]
    fn detects_a_fixed_point() {
        let zero = State::new(0, 1).expect("valid state");

        let mut system = TableSystem::new(1).expect("valid system");
        system
            .insert(zero, Action::Noop, vec![zero])
            .expect("valid transition");

        let analysis = analyze_orbit(&system, zero, Action::Noop, 4).expect("analysis succeeds");

        assert_eq!(analysis.transient_len(), 0);
        assert_eq!(analysis.period(), 1);
        assert_eq!(analysis.attractor(), &[zero]);
        assert!(analysis.is_fixed_point());
    }

    #[test]
    fn detects_a_transient_followed_by_a_two_cycle() {
        let zero = State::new(0b00, 2).expect("valid state");
        let one = State::new(0b01, 2).expect("valid state");
        let two = State::new(0b10, 2).expect("valid state");

        let mut system = TableSystem::new(2).expect("valid system");
        system
            .insert(zero, Action::Noop, vec![one])
            .expect("valid transition");
        system
            .insert(one, Action::Noop, vec![two])
            .expect("valid transition");
        system
            .insert(two, Action::Noop, vec![one])
            .expect("valid transition");

        let analysis = analyze_orbit(&system, zero, Action::Noop, 8).expect("analysis succeeds");

        assert_eq!(analysis.states(), &[zero, one, two]);
        assert_eq!(analysis.transient_len(), 1);
        assert_eq!(analysis.period(), 2);
        assert_eq!(analysis.attractor(), &[one, two]);
        assert!(!analysis.is_fixed_point());
    }

    #[test]
    fn rejects_branching_transitions() {
        let zero = State::new(0b00, 2).expect("valid state");
        let one = State::new(0b01, 2).expect("valid state");
        let two = State::new(0b10, 2).expect("valid state");

        let mut system = TableSystem::new(2).expect("valid system");
        system
            .insert(zero, Action::Noop, vec![one, two])
            .expect("valid transition");

        assert!(matches!(
            analyze_orbit(&system, zero, Action::Noop, 4),
            Err(OrbitError::NonDeterministicTransition {
                state,
                successor_count: 2
            }) if state == zero
        ));
    }

    #[test]
    fn reports_an_insufficient_step_limit() {
        let zero = State::new(0b00, 2).expect("valid state");
        let one = State::new(0b01, 2).expect("valid state");

        let mut system = TableSystem::new(2).expect("valid system");
        system
            .insert(zero, Action::Noop, vec![one])
            .expect("valid transition");
        system
            .insert(one, Action::Noop, vec![zero])
            .expect("valid transition");

        assert!(matches!(
            analyze_orbit(&system, zero, Action::Noop, 1),
            Err(OrbitError::NoRecurrenceWithinLimit { max_steps: 1 })
        ));
    }
}
