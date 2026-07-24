//! Integration tests: the nested-Monte-Carlo EIG estimator (gap #1 tier 3),
//! exercised through the public API only — a caller with a discrete
//! hypothesis set and a prior, ending in a real `sos-planner` `Plan`.

use scirust_stats::{DiscreteDistribution, Poisson};
use sos_core::{HashAlgo, ObjectId};
use sos_planner::{Cost, GreedyPlanner, Planner, StopVerdict, UtilityPolicy};
use sos_scirust::NestedMcEigEstimator;

fn design(tag: &[u8]) -> ObjectId {
    ObjectId::compute(HashAlgo::default(), b"design", tag)
}

#[test]
fn a_discriminating_experiment_beats_a_useless_one_in_a_real_plan() {
    let est = NestedMcEigEstimator::uniform(2, 3000).unwrap();

    // "Useless" design: both hypotheses predict the same count distribution,
    // so no observation here can discriminate between them.
    let same_a = Poisson::new(10.0);
    let same_b = Poisson::new(10.0);
    let useless: Vec<&dyn DiscreteDistribution> = vec![&same_a, &same_b];
    let useless_candidate = est
        .candidate(design(b"useless"), &useless, Cost::new(1, 0, 0, 0), 1)
        .unwrap();

    // "Discriminating" design: near-disjoint supports, so an observation
    // almost always reveals which hypothesis is true.
    let apart_a = Poisson::new(0.01);
    let apart_b = Poisson::new(300.0);
    let discriminating: Vec<&dyn DiscreteDistribution> = vec![&apart_a, &apart_b];
    let discriminating_candidate = est
        .candidate(
            design(b"discriminating"),
            &discriminating,
            Cost::new(1, 0, 0, 0),
            1,
        )
        .unwrap();

    let plan = GreedyPlanner::new()
        .recommend(
            &[useless_candidate, discriminating_candidate],
            UtilityPolicy::EigPerCost,
            1,
        )
        .unwrap();

    assert_eq!(
        plan.verdict,
        StopVerdict::Recommend(design(b"discriminating"))
    );
}

#[test]
fn honestly_reports_information_exhausted_for_indistinguishable_hypotheses() {
    let est = NestedMcEigEstimator::uniform(3, 3000).unwrap();
    let a = Poisson::new(7.0);
    let b = Poisson::new(7.0);
    let c = Poisson::new(7.0);
    let models: Vec<&dyn DiscreteDistribution> = vec![&a, &b, &c];

    let candidate = est
        .candidate(design(b"identical"), &models, Cost::new(1, 0, 0, 0), 2)
        .unwrap();

    // A generous floor (100 millibits) that a near-zero-EIG design should
    // not clear, even accounting for Monte-Carlo noise.
    let plan = GreedyPlanner::new()
        .recommend(&[candidate], UtilityPolicy::EigPerCost, 100)
        .unwrap();
    assert_eq!(plan.verdict, StopVerdict::InformationExhausted);
}

#[test]
fn same_seed_is_bit_reproducible_end_to_end() {
    let est = NestedMcEigEstimator::uniform(2, 500).unwrap();
    let a = Poisson::new(4.0);
    let b = Poisson::new(9.0);
    let models: Vec<&dyn DiscreteDistribution> = vec![&a, &b];

    let x = est
        .candidate(design(b"d"), &models, Cost::new(2, 0, 0, 0), 55)
        .unwrap();
    let y = est
        .candidate(design(b"d"), &models, Cost::new(2, 0, 0, 0), 55)
        .unwrap();
    assert_eq!(x.eig, y.eig);
}
