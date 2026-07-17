use crate::{State, StateError};

/// Intervention élémentaire appliquée à un système booléen.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Action {
    Noop,
    Flip { node: u8 },
    Clamp { node: u8, value: bool },
}

impl Action {
    /// Applique l'intervention à une configuration.
    pub fn apply(self, state: State) -> Result<State, StateError> {
        match self
        {
            Self::Noop => Ok(state),
            Self::Flip { node } => state.flip(node),
            Self::Clamp { node, value } => state.with_bit(node, value),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Action, State, StateError};

    #[test]
    fn applies_supported_actions() {
        let state = State::new(0b0101, 4).expect("valid state");

        assert_eq!(Action::Noop.apply(state), Ok(state));
        assert_eq!(
            Action::Flip { node: 1 }.apply(state).map(State::bits),
            Ok(0b0111)
        );
        assert_eq!(
            Action::Clamp {
                node: 2,
                value: false
            }
            .apply(state)
            .map(State::bits),
            Ok(0b0001)
        );
    }

    #[test]
    fn rejects_an_invalid_node() {
        let state = State::new(0, 4).expect("valid state");

        assert_eq!(
            Action::Flip { node: 4 }.apply(state),
            Err(StateError::NodeOutOfRange { node: 4, width: 4 })
        );
    }
}
