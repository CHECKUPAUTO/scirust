use std::collections::BTreeMap;

use crate::{Action, State, TransitionSystem};

/// Résultat exact d'une exploration prospective.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReachabilityReport {
    initial: State,
    layers: Vec<BTreeMap<State, u128>>,
}

/// Erreurs d'exploration.
#[derive(Debug)]
pub enum ExploreError<E> {
    InitialWidthMismatch { expected: u8, actual: u8 },
    SuccessorWidthMismatch { expected: u8, actual: u8 },
    PathCountOverflow,
    System(E),
}

impl ReachabilityReport {
    /// État initial de l'exploration.
    #[must_use]
    pub const fn initial(&self) -> State {
        self.initial
    }

    /// Nombre de profondeurs explorées.
    #[must_use]
    pub fn horizon(&self) -> usize {
        self.layers.len()
    }

    /// États et multiplicités de chemins à la profondeur demandée.
    #[must_use]
    pub fn layer(&self, depth: usize) -> Option<&BTreeMap<State, u128>> {
        depth
            .checked_sub(1)
            .and_then(|index| self.layers.get(index))
    }

    /// Nombre d'états distincts accessibles à une profondeur.
    #[must_use]
    pub fn reachable_count(&self, depth: usize) -> Option<usize> {
        self.layer(depth).map(BTreeMap::len)
    }

    /// Nombre total de chemins admissibles à une profondeur.
    #[must_use]
    pub fn path_count(&self, depth: usize) -> Option<u128> {
        self.layer(depth).map(|layer| layer.values().copied().sum())
    }

    /// Nombre de chemins revenus à l'état initial.
    #[must_use]
    pub fn return_path_count(&self, depth: usize) -> Option<u128> {
        self.layer(depth)
            .map(|layer| layer.get(&self.initial).copied().unwrap_or(0))
    }

    /// Rapport exact `(chemins revenus, chemins totaux)`.
    #[must_use]
    pub fn return_ratio(&self, depth: usize) -> Option<(u128, u128)> {
        let total = self.path_count(depth)?;
        let returned = self.return_path_count(depth)?;
        Some((returned, total))
    }
}

/// Explore exhaustivement un système pour une séquence d'actions donnée.
pub fn explore<S>(
    system: &S,
    initial: State,
    actions: &[Action],
) -> Result<ReachabilityReport, ExploreError<S::Error>>
where
    S: TransitionSystem,
{
    if initial.width() != system.width()
    {
        return Err(ExploreError::InitialWidthMismatch {
            expected: system.width(),
            actual: initial.width(),
        });
    }

    let mut frontier = BTreeMap::from([(initial, 1_u128)]);
    let mut layers = Vec::with_capacity(actions.len());

    for &action in actions
    {
        let mut next = BTreeMap::<State, u128>::new();

        for (state, multiplicity) in frontier
        {
            let successors = system
                .successors(state, action)
                .map_err(ExploreError::System)?;

            for successor in successors
            {
                if successor.width() != system.width()
                {
                    return Err(ExploreError::SuccessorWidthMismatch {
                        expected: system.width(),
                        actual: successor.width(),
                    });
                }

                let entry = next.entry(successor).or_default();
                *entry = entry
                    .checked_add(multiplicity)
                    .ok_or(ExploreError::PathCountOverflow)?;
            }
        }

        layers.push(next.clone());
        frontier = next;
    }

    Ok(ReachabilityReport { initial, layers })
}

#[cfg(test)]
mod tests {
    use crate::{Action, ExploreError, State, TableSystem, explore};

    #[test]
    fn explores_a_branching_system_exactly() {
        let zero = State::new(0b00, 2).expect("valid state");
        let one = State::new(0b01, 2).expect("valid state");
        let two = State::new(0b10, 2).expect("valid state");
        let three = State::new(0b11, 2).expect("valid state");

        let mut system = TableSystem::new(2).expect("valid system");
        system
            .insert(zero, Action::Noop, vec![one, two])
            .expect("valid transition");
        system
            .insert(one, Action::Noop, vec![three])
            .expect("valid transition");
        system
            .insert(two, Action::Noop, vec![three])
            .expect("valid transition");

        let report =
            explore(&system, zero, &[Action::Noop, Action::Noop]).expect("exploration succeeds");

        assert_eq!(report.horizon(), 2);
        assert_eq!(report.reachable_count(1), Some(2));
        assert_eq!(report.path_count(1), Some(2));
        assert_eq!(report.reachable_count(2), Some(1));
        assert_eq!(report.path_count(2), Some(2));
        assert_eq!(report.return_ratio(2), Some((0, 2)));
    }

    #[test]
    fn counts_returning_paths() {
        let zero = State::new(0b00, 2).expect("valid state");
        let one = State::new(0b01, 2).expect("valid state");

        let mut system = TableSystem::new(2).expect("valid system");
        system
            .insert(zero, Action::Noop, vec![one])
            .expect("valid transition");
        system
            .insert(one, Action::Noop, vec![zero])
            .expect("valid transition");

        let report =
            explore(&system, zero, &[Action::Noop, Action::Noop]).expect("exploration succeeds");

        assert_eq!(report.return_ratio(1), Some((0, 1)));
        assert_eq!(report.return_ratio(2), Some((1, 1)));
    }

    #[test]
    fn rejects_an_initial_width_mismatch() {
        let system = TableSystem::new(2).expect("valid system");
        let initial = State::new(0, 3).expect("valid state");

        assert!(matches!(
            explore(&system, initial, &[]),
            Err(ExploreError::InitialWidthMismatch {
                expected: 2,
                actual: 3
            })
        ));
    }
}
