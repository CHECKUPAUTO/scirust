use scirust_cache_policy::{
    DiscoveryConfig, calibrate_threshold, calibrate_threshold_robust, compare_on_holdout,
    discover_linear_policy, evaluate_policy_by_trajectory, split_by_trajectory,
    split_by_trajectory_fold, synthetic_trace,
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
fn robust_threshold_calibration_obeys_mean_and_tail_budgets() {
    let rows = synthetic_trace(15, 32, 9);
    let weights = [5.0, 2.0, 1.0, 1.0, 1.0, 0.0, 1.0, -0.5];
    let (_, aggregate, trajectory) =
        calibrate_threshold_robust(&rows, &weights, 0.10, 0.90).unwrap();
    assert!(aggregate.quality_loss_fraction <= 0.10 + 1e-12);
    assert!(trajectory.mean_quality_loss_fraction <= 0.10 + 1e-12);
    assert!(trajectory.tail_quality_loss_fraction <= 0.10 + 1e-12);
}

#[test]
fn trajectory_metrics_equalize_trajectory_weight() {
    let mut rows = synthetic_trace(2, 4, 17);
    rows.retain(|row| row.trajectory_id == 0 || row.step == 0);
    for row in &mut rows
    {
        row.stale_loss = 1.0;
        row.refresh_cost = 1.0;
    }
    let metrics = evaluate_policy_by_trajectory(&rows, 1.0, |row| row.trajectory_id == 0);
    assert_eq!(metrics.trajectories, 2);
    assert!((metrics.mean_refresh_rate - 0.5).abs() < 1e-12);
    assert!((metrics.mean_quality_loss_fraction - 0.5).abs() < 1e-12);
    assert!((metrics.tail_quality_loss_fraction - 1.0).abs() < 1e-12);
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
fn rotating_fold_split_never_leaks_a_trajectory() {
    let rows = synthetic_trace(25, 8, 11);
    for test_fold in 0..5
    {
        let (training, validation, test) = split_by_trajectory_fold(&rows, 5, test_fold).unwrap();
        let validation_fold = (test_fold + 4) % 5;
        assert!(
            training.iter().all(|row| row.trajectory_id % 5 != test_fold
                && row.trajectory_id % 5 != validation_fold)
        );
        assert!(
            validation
                .iter()
                .all(|row| row.trajectory_id % 5 == validation_fold)
        );
        assert!(test.iter().all(|row| row.trajectory_id % 5 == test_fold));
    }
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
fn robust_discovery_is_deterministic_for_a_fixed_seed() {
    let rows = synthetic_trace(15, 24, 13);
    let (training, validation, _) = split_by_trajectory(&rows);
    let config = DiscoveryConfig {
        steps: 30,
        trajectory_balanced: true,
        calibration_budget_fraction: 0.5,
        ..DiscoveryConfig::default()
    };
    let first = discover_linear_policy(&training, &validation, config).unwrap();
    let second = discover_linear_policy(&training, &validation, config).unwrap();
    assert_eq!(first, second);
}

#[test]
#[ignore = "research integration test; run with --release -- --ignored"]
fn discovered_policy_beats_similarity_only_on_the_nonlinear_oracle() {
    let rows = synthetic_trace(200, 64, 20_260_804);
    let (training, validation, test) = split_by_trajectory(&rows);
    let result = discover_linear_policy(
        &training,
        &validation,
        DiscoveryConfig {
            steps: 300,
            ..DiscoveryConfig::default()
        },
    )
    .unwrap();
    let comparison = compare_on_holdout(&result.policy, &test, 0.05);
    assert!(comparison.learned_meets_budget, "{comparison:?}");
    assert!(comparison.constrained_better, "{comparison:?}");
    assert!(comparison.pareto_dominates, "{comparison:?}");
    assert!(
        comparison.relative_compute_improvement > 0.10,
        "{comparison:?}"
    );
}
