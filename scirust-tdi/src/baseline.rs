use std::collections::BTreeMap;

use crate::{Action, State, StateError, TransitionSystem};

const MAX_ENUMERATED_WIDTH: u8 = 20;

/// Erreurs produites par les mesures de référence exhaustives.
#[derive(Debug)]
pub enum BaselineError<E> {
    WidthTooLarge {
        width: u8,
        maximum: u8,
    },
    InvalidState(StateError),
    NonDeterministicTransition {
        state: State,
        successor_count: usize,
    },
    System(E),
}

/// Calcule exactement la distribution des blocs futurs pour une distribution
/// initiale uniforme sur tous les états.
///
/// Le bloc contient les états futurs et exclut l'état initial.
pub fn uniform_future_block_distribution<S>(
    system: &S,
    action: Action,
    horizon: usize,
) -> Result<BTreeMap<Vec<State>, u64>, BaselineError<S::Error>>
where
    S: TransitionSystem,
{
    let width = system.width();

    if width > MAX_ENUMERATED_WIDTH
    {
        return Err(BaselineError::WidthTooLarge {
            width,
            maximum: MAX_ENUMERATED_WIDTH,
        });
    }

    let state_count = 1_u64 << width;
    let mut distribution = BTreeMap::<Vec<State>, u64>::new();

    for bits in 0..state_count
    {
        let mut state = State::new(bits, width).map_err(BaselineError::InvalidState)?;
        let mut block = Vec::with_capacity(horizon);

        for _ in 0..horizon
        {
            let successors = system
                .successors(state, action)
                .map_err(BaselineError::System)?;

            if successors.len() != 1
            {
                return Err(BaselineError::NonDeterministicTransition {
                    state,
                    successor_count: successors.len(),
                });
            }

            state = successors[0];
            block.push(state);
        }

        let count = distribution.entry(block).or_default();
        *count = count
            .checked_add(1)
            .expect("the number of initial states fits in u64");
    }

    Ok(distribution)
}

/// Calcule l'entropie de Shannon en bits de la distribution uniforme induite
/// sur les blocs futurs.
pub fn uniform_future_block_entropy_bits<S>(
    system: &S,
    action: Action,
    horizon: usize,
) -> Result<f64, BaselineError<S::Error>>
where
    S: TransitionSystem,
{
    let distribution = uniform_future_block_distribution(system, action, horizon)?;

    let total: u64 = distribution.values().copied().sum();

    if total == 0
    {
        return Ok(0.0);
    }

    let total_f64 = total as f64;

    let entropy = distribution
        .values()
        .copied()
        .filter(|count| *count != 0)
        .map(|count| {
            let probability = count as f64 / total_f64;
            -probability * probability.log2()
        })
        .sum();

    Ok(entropy)
}

#[cfg(test)]
mod tests {
    use crate::{
        Action, State, TableSystem, TdiSignature, explore, uniform_future_block_entropy_bits,
    };

    fn cycle_four() -> TableSystem {
        let mut system = TableSystem::new(2).expect("valid width");

        for (source, target) in [(0, 1), (1, 2), (2, 3), (3, 0)]
        {
            system
                .insert(
                    State::new(source, 2).expect("valid source"),
                    Action::Noop,
                    vec![State::new(target, 2).expect("valid target")],
                )
                .expect("valid transition");
        }

        system
    }

    fn two_cycles() -> TableSystem {
        let mut system = TableSystem::new(2).expect("valid width");

        for (source, target) in [(0, 1), (1, 0), (2, 3), (3, 2)]
        {
            system
                .insert(
                    State::new(source, 2).expect("valid source"),
                    Action::Noop,
                    vec![State::new(target, 2).expect("valid target")],
                )
                .expect("valid transition");
        }

        system
    }

    #[test]
    fn adversarial_systems_share_block_entropy() {
        let first = cycle_four();
        let second = two_cycles();

        for horizon in 1..=8
        {
            let first_entropy = uniform_future_block_entropy_bits(&first, Action::Noop, horizon)
                .expect("entropy succeeds");

            let second_entropy = uniform_future_block_entropy_bits(&second, Action::Noop, horizon)
                .expect("entropy succeeds");

            assert_eq!(first_entropy, 2.0);
            assert_eq!(second_entropy, 2.0);
        }
    }

    #[test]
    fn tdi_return_profile_separates_adversarial_systems() {
        let initial = State::new(0, 2).expect("valid initial state");
        let actions = [Action::Noop, Action::Noop];

        let first_report = explore(&cycle_four(), initial, &actions).expect("exploration succeeds");

        let second_report =
            explore(&two_cycles(), initial, &actions).expect("exploration succeeds");

        let first_signature = TdiSignature::from_report(&first_report).expect("signature succeeds");

        let second_signature =
            TdiSignature::from_report(&second_report).expect("signature succeeds");

        assert_ne!(
            first_signature.return_profile(),
            second_signature.return_profile()
        );

        assert_eq!(
            first_signature.return_profile()[1].numerator(),
            &num_bigint::BigUint::from(0_u8)
        );

        assert_eq!(
            second_signature.return_profile()[1].numerator(),
            &num_bigint::BigUint::from(1_u8)
        );
    }
}
