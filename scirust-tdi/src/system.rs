use std::collections::BTreeMap;

use crate::{Action, State};

/// Contrat minimal d'un système de transition fini.
///
/// Plusieurs successeurs permettent de représenter une dynamique ramifiée.
pub trait TransitionSystem {
    type Error;

    /// Nombre de constituants booléens du système.
    fn width(&self) -> u8;

    /// Calcule tous les états suivants admissibles.
    fn successors(&self, state: State, action: Action) -> Result<Vec<State>, Self::Error>;
}

/// Système fini défini explicitement par une table de transitions.
#[derive(Clone, Debug)]
pub struct TableSystem {
    width: u8,
    transitions: BTreeMap<(State, Action), Vec<State>>,
}

/// Erreurs produites par un système tabulaire.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TableSystemError {
    WidthOutOfRange { width: u8 },
    StateWidthMismatch { expected: u8, actual: u8 },
    SuccessorWidthMismatch { expected: u8, actual: u8 },
    EmptySuccessors,
    UndefinedTransition { state: State, action: Action },
}

impl TableSystem {
    /// Crée une table de transitions pour des états de largeur `width`.
    pub fn new(width: u8) -> Result<Self, TableSystemError> {
        if width == 0 || width > 64
        {
            return Err(TableSystemError::WidthOutOfRange { width });
        }

        Ok(Self {
            width,
            transitions: BTreeMap::new(),
        })
    }

    /// Ajoute une transition, en triant et dédupliquant ses successeurs.
    pub fn insert(
        &mut self,
        state: State,
        action: Action,
        mut successors: Vec<State>,
    ) -> Result<(), TableSystemError> {
        self.validate_state(state)?;

        if successors.is_empty()
        {
            return Err(TableSystemError::EmptySuccessors);
        }

        for successor in &successors
        {
            if successor.width() != self.width
            {
                return Err(TableSystemError::SuccessorWidthMismatch {
                    expected: self.width,
                    actual: successor.width(),
                });
            }
        }

        successors.sort_unstable();
        successors.dedup();
        self.transitions.insert((state, action), successors);

        Ok(())
    }

    fn validate_state(&self, state: State) -> Result<(), TableSystemError> {
        if state.width() != self.width
        {
            return Err(TableSystemError::StateWidthMismatch {
                expected: self.width,
                actual: state.width(),
            });
        }

        Ok(())
    }
}

impl TransitionSystem for TableSystem {
    type Error = TableSystemError;

    fn width(&self) -> u8 {
        self.width
    }

    fn successors(&self, state: State, action: Action) -> Result<Vec<State>, Self::Error> {
        self.validate_state(state)?;

        self.transitions
            .get(&(state, action))
            .cloned()
            .ok_or(TableSystemError::UndefinedTransition { state, action })
    }
}

#[cfg(test)]
mod tests {
    use crate::{Action, State, TableSystem, TableSystemError, TransitionSystem};

    #[test]
    fn stores_sorted_unique_successors() {
        let mut system = TableSystem::new(2).expect("valid width");
        let source = State::new(0, 2).expect("valid state");
        let first = State::new(1, 2).expect("valid state");
        let second = State::new(2, 2).expect("valid state");

        system
            .insert(source, Action::Noop, vec![second, first, second])
            .expect("valid transition");

        assert_eq!(
            system.successors(source, Action::Noop),
            Ok(vec![first, second])
        );
    }

    #[test]
    fn rejects_empty_successor_sets() {
        let mut system = TableSystem::new(2).expect("valid width");
        let source = State::new(0, 2).expect("valid state");

        assert_eq!(
            system.insert(source, Action::Noop, Vec::new()),
            Err(TableSystemError::EmptySuccessors)
        );
    }

    #[test]
    fn reports_undefined_transitions() {
        let system = TableSystem::new(2).expect("valid width");
        let source = State::new(0, 2).expect("valid state");

        assert_eq!(
            system.successors(source, Action::Noop),
            Err(TableSystemError::UndefinedTransition {
                state: source,
                action: Action::Noop
            })
        );
    }
}
