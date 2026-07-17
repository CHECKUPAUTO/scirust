use std::collections::BTreeMap;

use crate::{Action, ExactRatio, State, TransitionSystem};

/// Erreurs des baselines prospectives pour dynamiques branchantes.
#[derive(Debug)]
pub enum BranchingBaselineError<E> {
    InitialWidthMismatch {
        expected: u8,
        actual: u8,
    },
    SuccessorWidthMismatch {
        expected: u8,
        actual: u8,
    },
    EmptySuccessorSet {
        state: State,
        depth: usize,
    },
    ProbabilityDenominatorOverflow {
        state: State,
        depth: usize,
        denominator: u128,
        successor_count: usize,
    },
    InvalidProbability,
    System(E),
}

/// Distribution exacte des chemins futurs lorsque chaque successeur admissible
/// est choisi avec une probabilité uniforme locale.
///
/// Chaque clé contient les états futurs et exclut l'état initial.
pub fn uniform_branching_path_distribution<S>(
    system: &S,
    initial: State,
    action: Action,
    horizon: usize,
) -> Result<BTreeMap<Vec<State>, ExactRatio>, BranchingBaselineError<S::Error>>
where
    S: TransitionSystem,
{
    if initial.width() != system.width()
    {
        return Err(BranchingBaselineError::InitialWidthMismatch {
            expected: system.width(),
            actual: initial.width(),
        });
    }

    if horizon == 0
    {
        return Ok(BTreeMap::from([(
            Vec::new(),
            ExactRatio::new(1, 1).ok_or(BranchingBaselineError::InvalidProbability)?,
        )]));
    }

    let mut frontier = vec![(initial, Vec::<State>::new(), 1_u128)];

    for depth in 1..=horizon
    {
        let mut next = Vec::new();

        for (state, path, denominator) in frontier
        {
            let mut successors = system
                .successors(state, action)
                .map_err(BranchingBaselineError::System)?;

            successors.sort_unstable();
            successors.dedup();

            if successors.is_empty()
            {
                return Err(BranchingBaselineError::EmptySuccessorSet { state, depth });
            }

            for successor in &successors
            {
                if successor.width() != system.width()
                {
                    return Err(BranchingBaselineError::SuccessorWidthMismatch {
                        expected: system.width(),
                        actual: successor.width(),
                    });
                }
            }

            let successor_count = successors.len();

            let next_denominator = denominator.checked_mul(successor_count as u128).ok_or(
                BranchingBaselineError::ProbabilityDenominatorOverflow {
                    state,
                    depth,
                    denominator,
                    successor_count,
                },
            )?;

            for successor in successors
            {
                let mut next_path = path.clone();
                next_path.push(successor);

                next.push((successor, next_path, next_denominator));
            }
        }

        frontier = next;
    }

    let mut distribution = BTreeMap::new();

    for (_, path, denominator) in frontier
    {
        let probability =
            ExactRatio::new(1, denominator).ok_or(BranchingBaselineError::InvalidProbability)?;

        distribution.insert(path, probability);
    }

    Ok(distribution)
}

/// Entropie de Shannon, en bits, de la distribution exacte des chemins futurs.
///
/// Les probabilités sont construites exactement. La conversion flottante est
/// limitée au calcul final du logarithme.
pub fn uniform_branching_path_entropy_bits<S>(
    system: &S,
    initial: State,
    action: Action,
    horizon: usize,
) -> Result<f64, BranchingBaselineError<S::Error>>
where
    S: TransitionSystem,
{
    let distribution = uniform_branching_path_distribution(system, initial, action, horizon)?;

    Ok(distribution
        .values()
        .map(|probability| {
            let value = probability.as_f64();

            if value == 0.0
            {
                0.0
            }
            else
            {
                -value * value.log2()
            }
        })
        .sum())
}

#[cfg(test)]
mod tests {
    use crate::{
        Action, ExactRatio, State, TableSystem, uniform_branching_path_distribution,
        uniform_branching_path_entropy_bits,
    };

    #[test]
    fn measures_a_binary_branch_exactly() {
        let zero = State::new(0b00, 2).expect("valid state");
        let one = State::new(0b01, 2).expect("valid state");
        let two = State::new(0b10, 2).expect("valid state");

        let mut system = TableSystem::new(2).expect("valid system");

        system
            .insert(zero, Action::Noop, vec![one, two])
            .expect("valid transition");

        let distribution = uniform_branching_path_distribution(&system, zero, Action::Noop, 1)
            .expect("distribution succeeds");

        assert_eq!(distribution.len(), 2);
        assert_eq!(distribution.get(&vec![one]), ExactRatio::new(1, 2).as_ref());
        assert_eq!(distribution.get(&vec![two]), ExactRatio::new(1, 2).as_ref());

        let entropy = uniform_branching_path_entropy_bits(&system, zero, Action::Noop, 1)
            .expect("entropy succeeds");

        assert_eq!(entropy, 1.0);
    }

    #[test]
    fn handles_non_uniform_branching_depths() {
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
            .insert(two, Action::Noop, vec![zero, three])
            .expect("valid transition");

        let distribution = uniform_branching_path_distribution(&system, zero, Action::Noop, 2)
            .expect("distribution succeeds");

        assert_eq!(distribution.len(), 3);
        assert_eq!(
            distribution.get(&vec![one, three]),
            ExactRatio::new(1, 2).as_ref()
        );
        assert_eq!(
            distribution.get(&vec![two, zero]),
            ExactRatio::new(1, 4).as_ref()
        );
        assert_eq!(
            distribution.get(&vec![two, three]),
            ExactRatio::new(1, 4).as_ref()
        );

        let entropy = uniform_branching_path_entropy_bits(&system, zero, Action::Noop, 2)
            .expect("entropy succeeds");

        assert_eq!(entropy, 1.5);
    }

    #[test]
    fn deterministic_paths_have_zero_entropy() {
        let zero = State::new(0b00, 2).expect("valid state");
        let one = State::new(0b01, 2).expect("valid state");

        let mut system = TableSystem::new(2).expect("valid system");

        system
            .insert(zero, Action::Noop, vec![one])
            .expect("valid transition");

        system
            .insert(one, Action::Noop, vec![zero])
            .expect("valid transition");

        let entropy = uniform_branching_path_entropy_bits(&system, zero, Action::Noop, 4)
            .expect("entropy succeeds");

        assert_eq!(entropy, 0.0);
    }

    #[test]
    fn zero_horizon_has_one_empty_path() {
        let zero = State::new(0, 1).expect("valid state");
        let system = TableSystem::new(1).expect("valid system");

        let distribution = uniform_branching_path_distribution(&system, zero, Action::Noop, 0)
            .expect("distribution succeeds");

        assert_eq!(
            distribution.get(&Vec::new()),
            ExactRatio::new(1, 1).as_ref()
        );
    }
}
