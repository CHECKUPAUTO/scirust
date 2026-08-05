use scirust_cache_policy::trajectory::{
    TrajectoryDiscoveryConfig, discover_trajectory_policy,
};
use serde_json::json;
use std::io::Write;

fn write_synthetic_trajectory_dataset() -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "scirust-trajectory-policy-{}-{}.jsonl",
        std::process::id(),
        20_260_810_u64
    ));
    let mut file = std::fs::File::create(&path).unwrap();
    for prompt in 0..15usize {
        for ordinal in 1..=3usize {
            let unsafe_branch = ordinal == 2;
            let prediction_changed = unsafe_branch && prompt % 3 == 0;
            let response_invariant = !unsafe_branch;
            let decision_invariant = !unsafe_branch;
            let baseline_prediction = format!("{}", 100 + prompt);
            let branch_prediction = if prediction_changed {
                format!("{}", 200 + prompt)
            } else {
                baseline_prediction.clone()
            };
            let baseline_decisions = 40 + prompt;
            let branch_decisions = if decision_invariant {
                baseline_decisions
            } else {
                baseline_decisions + 2
            };
            let baseline_refresh_cost = baseline_decisions as f64 * 0.96;
            let branch_refresh_cost = if unsafe_branch {
                baseline_refresh_cost + 0.3
            } else {
                baseline_refresh_cost - 0.96
            };
            let skip_margin = 0.01 * ordinal as f64 + prompt as f64 * 0.0001;
            let refresh_cost = 27.0 / 28.0;
            let row = json!({
                "schema_version": 1,
                "split": "gsm8k_train",
                "dataset_index": prompt,
                "generation_seed": 20260809_u64 + prompt as u64,
                "candidate": {
                    "ordinal": ordinal,
                    "layer_id": 0,
                    "skip_margin": skip_margin,
                    "refresh_cost": refresh_cost,
                    "votes": if ordinal == 3 { 2 } else { 1 },
                    "features": {
                        "drift": if unsafe_branch { 0.12 } else { 0.003 * ordinal as f64 },
                        "worsening": if unsafe_branch { 0.08 } else { 0.001 },
                        "head_std": 0.01 * ordinal as f64,
                        "cache_age": 0.0625 * ordinal as f64,
                        "untracked_mass": if unsafe_branch { 0.2 } else { 0.02 },
                        "layer_fraction": 0.0,
                        "drift_age": if unsafe_branch { 0.02 } else { 0.0002 },
                        "refresh_cost": refresh_cost
                    }
                },
                "baseline": {
                    "correct": true,
                    "prediction": baseline_prediction,
                    "gold": format!("{}", 100 + prompt),
                    "elapsed_seconds": 4.0 + prompt as f64 * 0.01,
                    "decisions": baseline_decisions,
                    "refreshes": baseline_decisions,
                    "refresh_cost": baseline_refresh_cost
                },
                "single_skip": {
                    "correct": !prediction_changed,
                    "prediction": branch_prediction,
                    "gold": format!("{}", 100 + prompt),
                    "elapsed_seconds": if unsafe_branch { 4.2 } else { 3.9 },
                    "decisions": branch_decisions,
                    "refreshes": branch_decisions - 1,
                    "refresh_cost": branch_refresh_cost
                },
                "labels": {
                    "exact_response_invariant": response_invariant,
                    "prediction_invariant": !prediction_changed,
                    "correctness_invariant": !prediction_changed,
                    "quality_regression": prediction_changed,
                    "quality_improvement": false,
                    "decision_count_invariant": decision_invariant
                },
                "effects": {
                    "decision_delta": branch_decisions as i64 - baseline_decisions as i64,
                    "decision_ratio": branch_decisions as f64 / baseline_decisions as f64,
                    "latency_improvement": if unsafe_branch { -0.05 } else { 0.025 },
                    "refresh_cost_improvement": (baseline_refresh_cost - branch_refresh_cost)
                        / baseline_refresh_cost
                }
            });
            writeln!(file, "{}", serde_json::to_string(&row).unwrap()).unwrap();
        }
    }
    path
}

#[test]
fn trajectory_discovery_builds_fail_closed_report() {
    let path = write_synthetic_trajectory_dataset();
    let config = TrajectoryDiscoveryConfig {
        seed: 20_260_810,
        crf_epochs: 12,
        crf_learning_rate: 0.02,
        crf_l2_penalty: 0.001,
        nsga_population: 12,
        nsga_generations: 4,
        minimum_holdout_coverage: 0.0,
        symbolic: false,
    };
    let report = discover_trajectory_policy(&path, &config).unwrap();
    assert_eq!(
        report.status,
        "causal_sequential_trajectory_policy_development"
    );
    assert_eq!(report.source_dataset_sha256.len(), 64);
    assert_eq!(
        report.split.train.rows + report.split.validation.rows + report.split.holdout.rows,
        45
    );
    assert!(report.sequential.training_negative_log_likelihood.is_finite());
    assert!(report.gaussian_process.log_marginal_likelihood.is_finite());
    assert_eq!(report.nsga2.validation.quality_regressions_allowed, 0);
    assert_eq!(report.nsga2.validation.strict_unsafe_allowed, 0);
    assert!(!report.symbolic.enabled);
    std::fs::remove_file(path).unwrap();
}
