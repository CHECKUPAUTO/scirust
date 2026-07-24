//! Integration tests: `sos-scirust`'s public API, exercised the way an
//! external caller would — through `sos-planner`'s unmodified ranking
//! machinery, never touching either crate's private internals.

use scirust_gp::{GaussianProcess, Rbf};
use sos_core::{HashAlgo, ObjectId};
use sos_planner::{Cost, GreedyPlanner, Planner, StopVerdict, UtilityPolicy};
use sos_scirust::GpEigEstimator;

fn design(tag: &[u8]) -> ObjectId {
    ObjectId::compute(HashAlgo::default(), b"design", tag)
}

/// A GP fit to noisy samples of a known function (`sin`), so "near training
/// data" and "far from training data" are unambiguous, real properties of
/// the fitted posterior — not hand-picked numbers.
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
fn recommends_the_least_explored_real_candidate() {
    let gp = fit_sine_gp();
    let est = GpEigEstimator::new(gp, 0.05).unwrap();

    // Three real candidate follow-up designs at equal cost: one right on top
    // of existing training data, one just past its edge, one far outside the
    // explored region entirely.
    let candidates = [
        est.candidate(design(b"replicate"), &[1.0], Cost::new(1, 0, 0, 0)),
        est.candidate(design(b"edge"), &[4.2], Cost::new(1, 0, 0, 0)),
        est.candidate(design(b"unexplored"), &[30.0], Cost::new(1, 0, 0, 0)),
    ];

    let plan = GreedyPlanner::new()
        .recommend(&candidates, UtilityPolicy::EigPerCost, 1)
        .unwrap();

    assert_eq!(plan.verdict, StopVerdict::Recommend(design(b"unexplored")));
    // The full ranking is monotone in distance from the training region.
    let ranked_ids: Vec<ObjectId> = plan.ranked.iter().map(|r| r.experiment).collect();
    assert_eq!(
        ranked_ids,
        vec![design(b"unexplored"), design(b"edge"), design(b"replicate"),]
    );
}

#[test]
fn honestly_reports_information_exhausted_when_every_candidate_is_already_explored() {
    let gp = fit_sine_gp();
    let est = GpEigEstimator::new(gp, 0.05).unwrap();

    // Every candidate replicates an existing training point — a real GP's
    // posterior variance there is tiny (bounded by the fit's own noise
    // floor), so real EIG stays well under a 500-millibit (0.5-bit) floor —
    // in sharp contrast to the ~2.2-bit EIG a design far from any training
    // data gets under the same observation noise (see the sibling test).
    let candidates = [
        est.candidate(design(b"a"), &[0.0], Cost::new(1, 0, 0, 0)),
        est.candidate(design(b"b"), &[1.0], Cost::new(1, 0, 0, 0)),
        est.candidate(design(b"c"), &[2.0], Cost::new(1, 0, 0, 0)),
    ];

    let plan = GreedyPlanner::new()
        .recommend(&candidates, UtilityPolicy::EigPerCost, 500)
        .unwrap();

    assert_eq!(plan.verdict, StopVerdict::InformationExhausted);
}

#[test]
fn every_estimate_is_exact_with_zero_standard_error() {
    // The closed-form path never produces a noisy estimate: every Estimate
    // sos-planner receives from this bridge is L3 with se_milli == 0, so
    // clears_floor's significance check is never spuriously defeated by
    // self-reported noise.
    let gp = fit_sine_gp();
    let est = GpEigEstimator::new(gp, 0.1).unwrap();

    for x in [0.0, 1.7, -3.0, 42.0]
    {
        let e = est.estimate(&[x]);
        assert_eq!(e.se_milli, 0);
        assert_eq!(e.level, sos_core::DeterminismLevel::L3);
        assert!(e.is_significant() || e.bits_milli == 0);
    }
}

#[test]
fn rejects_non_positive_observation_noise() {
    let gp = fit_sine_gp();
    assert!(GpEigEstimator::new(gp.clone(), 0.0).is_err());
    assert!(GpEigEstimator::new(gp, -0.1).is_err());
}
