use scirust_cache_policy::{
    DiscoveryConfig, calibrate_threshold, compare_on_holdout, discover_linear_policy,
    split_by_trajectory, synthetic_trace,
};

#[test]
fn trace_validation_is_fail_closed() {
    let mut row = synthetic_trace(1, 1, 7)[0];
    assert!(row.validate().is_ok());
    row.similarity = f64::NAN;
    assert!(row.validate().is_err());
    row.similarity = 0.5;
    row.similarity_delta = 1.1;
    assert!(row.validate().is_err());
}

#[test]
fn threshold_calibration_obeys_the_budget() {
    let rows = synthetic_trace(10, 32, 9);
    let weights = [5.0, 2.0, 1.0, 1.0, 1.0, 0.0, 1.0, -0.5];
    let (_, metrics) = calibrate_threshold(&rows, &weights, 0.10).unwrap();
    assert!(metrics.quality_loss_fraction <= 0.10 + 1e-12);
}

#[test]
fn trajectory_split_never_leaks_a_trajectory() {
    let rows = synthetic_trace(25, 8, 11);
    let (training, validation, test) = split_by_trajectory(&rows);
    assert!(training.iter().all(|row| row.trajectory_id % 5 <= 2));
    assert!(validation.iter().all(|row| row.trajectory_id % 5 == 3));
    assert!(test.iter().all(|row| row.trajectory_id % 5 == 4));
}

#[test]
fn discovery_is_deterministic_for_a_fixed_seed() {
    let rows = synthetic_trace(15, 24, 13);
    let (training, validation, _) = split_by_trajectory(&rows);
    let config = DiscoveryConfig {
        steps: 30,
        ..DiscoveryConfig::default()
    };
    let first = discover_linear_policy(&training, &validation, config).unwrap();
    let second = discover_linear_policy(&training, &validation, config).unwrap();
    assert_eq!(first, second);
}

#[test]
#[ignore = "research integration test; run with --release -- --ignored"]
fn discovered_policy_beats_similarity_only_on_the_nonlinear_oracle() {
    let rows = synthetic_trace(40, 64, 20_260_804);
    let (training, validation, test) = split_by_trajectory(&rows);
    let result = discover_linear_policy(
        &training,
        &validation,
        DiscoveryConfig {
            steps: 250,
            ..DiscoveryConfig::default()
        },
    )
    .unwrap();
    let comparison = compare_on_holdout(&result.policy, &test);
    assert!(comparison.relative_compute_improvement > 0.10, "{comparison:?}");
}
