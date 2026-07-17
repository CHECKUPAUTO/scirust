use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use crate::{Action, ExactRatio, State, TransitionSystem};

/// Erreurs de propagation exacte d'une distribution branchante.
#[derive(Debug)]
pub enum BranchingDistributionError<E> {
    InitialWidthMismatch { expected: u8, actual: u8 },
    SuccessorWidthMismatch { expected: u8, actual: u8 },
    EmptySuccessorSet { state: State, depth: usize },
    ProbabilityOverflow { state: State, depth: usize },
    System(E),
}

/// Erreurs d'opérations sur des distributions rationnelles exactes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistributionMathError {
    ArithmeticOverflow,
}

/// Distribution exacte des états à une profondeur donnée.
///
/// Chaque successeur distinct d'un état est choisi avec une probabilité
/// uniforme locale.
pub fn uniform_branching_state_distribution<S>(
    system: &S,
    initial: State,
    action: Action,
    horizon: usize,
) -> Result<BTreeMap<State, ExactRatio>, BranchingDistributionError<S::Error>>
where
    S: TransitionSystem,
{
    if initial.width() != system.width()
    {
        return Err(BranchingDistributionError::InitialWidthMismatch {
            expected: system.width(),
            actual: initial.width(),
        });
    }

    let unit = ExactRatio::new(1, 1).expect("one is a valid exact probability");

    let mut frontier = BTreeMap::from([(initial, unit)]);

    for depth in 1..=horizon
    {
        let mut next = BTreeMap::<State, ExactRatio>::new();

        for (state, probability) in frontier
        {
            let mut successors = system
                .successors(state, action)
                .map_err(BranchingDistributionError::System)?;

            successors.sort_unstable();
            successors.dedup();

            if successors.is_empty()
            {
                return Err(BranchingDistributionError::EmptySuccessorSet { state, depth });
            }

            for successor in &successors
            {
                if successor.width() != system.width()
                {
                    return Err(BranchingDistributionError::SuccessorWidthMismatch {
                        expected: system.width(),
                        actual: successor.width(),
                    });
                }
            }

            let successor_count = successors.len() as u128;

            let branch_probability = probability
                .checked_div_u128(successor_count)
                .ok_or(BranchingDistributionError::ProbabilityOverflow { state, depth })?;

            for successor in successors
            {
                let updated = match next.get(&successor).cloned()
                {
                    Some(existing) => existing.checked_add(&branch_probability).ok_or(
                        BranchingDistributionError::ProbabilityOverflow {
                            state: successor,
                            depth,
                        },
                    )?,
                    None => branch_probability.clone(),
                };

                next.insert(successor, updated);
            }
        }

        frontier = next;
    }

    Ok(frontier)
}

/// Somme exacte des minima des probabilités de deux distributions.
///
/// Pour deux distributions normalisées, cette valeur est égale à
/// `1 - distance_de_variation_totale`.
pub fn distribution_overlap(
    left: &BTreeMap<State, ExactRatio>,
    right: &BTreeMap<State, ExactRatio>,
) -> Result<ExactRatio, DistributionMathError> {
    let states: BTreeSet<State> = left.keys().chain(right.keys()).copied().collect();

    let zero = ExactRatio::new(0, 1).expect("zero is a valid probability");

    let mut overlap = zero.clone();

    for state in states
    {
        let left_probability = left.get(&state).cloned().unwrap_or_else(|| zero.clone());

        let right_probability = right.get(&state).cloned().unwrap_or_else(|| zero.clone());

        let minimum = match left_probability
            .checked_cmp(&right_probability)
            .ok_or(DistributionMathError::ArithmeticOverflow)?
        {
            Ordering::Greater => right_probability,
            Ordering::Equal | Ordering::Less => left_probability,
        };

        overlap = overlap
            .checked_add(&minimum)
            .ok_or(DistributionMathError::ArithmeticOverflow)?;
    }

    Ok(overlap)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::{
        Action, ExactRatio, State, TableSystem, distribution_overlap,
        uniform_branching_state_distribution,
    };

    #[test]
    fn zero_horizon_is_a_point_mass() {
        let zero = State::new(0, 1).expect("valid state");
        let system = TableSystem::new(1).expect("valid system");

        let distribution = uniform_branching_state_distribution(&system, zero, Action::Noop, 0)
            .expect("distribution succeeds");

        assert_eq!(
            distribution,
            BTreeMap::from([(zero, ExactRatio::new(1, 1).expect("valid ratio"),)])
        );
    }

    #[test]
    fn propagates_and_merges_probabilities_exactly() {
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

        let depth_one = uniform_branching_state_distribution(&system, zero, Action::Noop, 1)
            .expect("distribution succeeds");

        assert_eq!(
            depth_one,
            BTreeMap::from([
                (one, ExactRatio::new(1, 2).expect("valid ratio"),),
                (two, ExactRatio::new(1, 2).expect("valid ratio"),),
            ])
        );

        let depth_two = uniform_branching_state_distribution(&system, zero, Action::Noop, 2)
            .expect("distribution succeeds");

        assert_eq!(
            depth_two,
            BTreeMap::from([(three, ExactRatio::new(1, 1).expect("valid ratio"),)])
        );
    }

    #[test]
    fn computes_exact_distribution_overlap() {
        let zero = State::new(0b00, 2).expect("valid state");
        let one = State::new(0b01, 2).expect("valid state");

        let left = BTreeMap::from([
            (zero, ExactRatio::new(1, 2).expect("valid ratio")),
            (one, ExactRatio::new(1, 2).expect("valid ratio")),
        ]);

        let right = BTreeMap::from([
            (zero, ExactRatio::new(1, 4).expect("valid ratio")),
            (one, ExactRatio::new(3, 4).expect("valid ratio")),
        ]);

        assert_eq!(
            distribution_overlap(&left, &right),
            Ok(ExactRatio::new(3, 4).expect("valid ratio"))
        );
    }

    #[test]
    fn disjoint_distributions_have_zero_overlap() {
        let zero = State::new(0b00, 2).expect("valid state");
        let one = State::new(0b01, 2).expect("valid state");

        let left = BTreeMap::from([(zero, ExactRatio::new(1, 1).expect("valid ratio"))]);

        let right = BTreeMap::from([(one, ExactRatio::new(1, 1).expect("valid ratio"))]);

        assert_eq!(
            distribution_overlap(&left, &right),
            Ok(ExactRatio::new(0, 1).expect("valid ratio"))
        );
    }
}
