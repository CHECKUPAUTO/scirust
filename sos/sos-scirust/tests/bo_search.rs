//! Integration tests: the Bayesian-optimization design search (gap #1 tier
//! 2), exercised through the public API only — a caller with a fitted GP, a
//! cost model, and a design box, ending in a real `sos-planner` `Plan`.

use scirust_gp::{GaussianProcess, Rbf};
use sos_core::{HashAlgo, ObjectId};
use sos_planner::{Cost, GreedyPlanner, Planner, StopVerdict, UtilityPolicy};
use sos_scirust::GpEigEstimator;

fn design(tag: &[u8]) -> ObjectId {
    ObjectId::compute(HashAlgo::default(), b"design", tag)
}

fn fit_sine_gp() -> GaussianProcess<Rbf> {
    let x: Vec<Vec<f64>> = (0..8).map(|i| vec![f64::from(i) * 0.5]).collect();
    let y: Vec<f64> = x.iter().map(|xi| xi[0].sin()).collect();
    let kernel = Rbf {
        lengthscale: 1.0,
        variance: 1.0,
    };
    GaussianProcess::fit(&x, &y, kernel, 1e-4).unwrap()
}

#[test]
fn bo_search_result_beats_a_replicated_training_point_in_a_real_plan() {
    let est = GpEigEstimator::new(fit_sine_gp(), 0.05).unwrap();

    // Search a box that reaches well past the training region [0, 3.5].
    let found = est.search_best_design(
        &[(0.0, 25.0)],
        &|_x| Cost::new(1, 0, 0, 0),
        UtilityPolicy::EigPerCost,
        40,
        10,
        123,
    );
    let bo_candidate = found.candidate(design(b"bo-best"));
    let replicate = est.candidate(design(b"replicate"), &[1.0], Cost::new(1, 0, 0, 0));

    let plan = GreedyPlanner::new()
        .recommend(&[bo_candidate, replicate], UtilityPolicy::EigPerCost, 1)
        .unwrap();

    assert_eq!(plan.verdict, StopVerdict::Recommend(design(b"bo-best")));
}

#[test]
fn same_seed_same_box_is_bit_reproducible() {
    let est = GpEigEstimator::new(fit_sine_gp(), 0.1).unwrap();
    let cost = |_x: &[f64]| Cost::new(2, 0, 0, 0);

    let a = est.search_best_design(&[(-5.0, 30.0)], &cost, UtilityPolicy::EigPerCost, 25, 6, 99);
    let b = est.search_best_design(&[(-5.0, 30.0)], &cost, UtilityPolicy::EigPerCost, 25, 6, 99);
    assert_eq!(a, b);
    assert_eq!(a.eig.level, sos_core::DeterminismLevel::L1);
}

#[test]
fn different_seeds_still_land_inside_the_box() {
    let est = GpEigEstimator::new(fit_sine_gp(), 0.1).unwrap();
    let cost = |_x: &[f64]| Cost::new(1, 0, 0, 0);
    let bounds = [(-3.0, 12.0)];

    for seed in [1, 2, 3, 4, 5]
    {
        let r = est.search_best_design(&bounds, &cost, UtilityPolicy::EigPerCost, 20, 5, seed);
        assert!(
            r.x[0] >= bounds[0].0 && r.x[0] <= bounds[0].1,
            "seed {seed}: x={} out of bounds",
            r.x[0]
        );
        assert_eq!(r.seed, seed);
    }
}
