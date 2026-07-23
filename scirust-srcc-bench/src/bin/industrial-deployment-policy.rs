//! Deterministic benchmark for phase 4E.10 — the production decision and
//! rollback policy, the governance layer that ships (or refuses to ship) a
//! certified pipeline decision and rolls it back if live coverage degrades.
//!
//! Two parts:
//!
//! 1. **Pre-deployment gate.** For four synthetic certified-pipeline reports —
//!    a clear robust winner, a winner whose certificate targets too little
//!    coverage, an abstention, and a reject-all — we print the
//!    [`decide_deployment`] verdict. Only the well-certified winner ships.
//! 2. **Post-deployment rollback.** A deployed model is monitored on a live
//!    stream whose coverage is healthy at first and then drifts below the floor.
//!    We print where the [`RollbackMonitor`] warms up, stays healthy, and finally
//!    latches a rollback.
//!
//! Everything is seeded; the program is byte-identical across runs.

use scirust_srcc_bench::{
    CertifiedPipelineConfig, CertifiedPipelineReport, CoverageMode, DeploymentAction,
    DeploymentPolicy, EstimatorEvidence, EstimatorTournament, MonitorState, Orientation,
    RollbackMonitor, decide_deployment, run_certified_pipeline,
};

const TOURNAMENT_SEED: u64 = 0x00D0_9111_5EED;

fn residuals(magnitude: f64, n: usize, offset: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (((i + offset) * 7) % 11) as f64 / 11.0 - 0.5)
        .map(|u| u * magnitude)
        .collect()
}

fn tournament() -> EstimatorTournament {
    EstimatorTournament {
        orientation: Orientation::LowerIsBetter,
        min_improvement: 0.0,
        tie_margin: 0.0,
        quality_floor: None,
        resamples: 4000,
        level: 0.95,
        seed: TOURNAMENT_SEED,
    }
}

fn report_for(scenario: &str) -> CertifiedPipelineReport {
    let mut config = CertifiedPipelineConfig {
        tournament: tournament(),
        coverage_level: 0.9,
        coverage_mode: CoverageMode::Marginal,
    };
    match scenario
    {
        "clear_winner" =>
        {
            let incumbent =
                EstimatorEvidence::new("ols", residuals(8.0, 80, 0), residuals(8.0, 80, 1));
            let robust =
                EstimatorEvidence::new("robust", residuals(0.6, 80, 2), residuals(0.6, 80, 3));
            run_certified_pipeline(&incumbent, &[robust], &config).unwrap()
        },
        "under_certified" =>
        {
            // Same clear winner, but the certificate only targets 0.5 coverage.
            config.coverage_level = 0.5;
            let incumbent =
                EstimatorEvidence::new("ols", residuals(8.0, 80, 0), residuals(8.0, 80, 1));
            let robust =
                EstimatorEvidence::new("robust", residuals(0.6, 80, 2), residuals(0.6, 80, 3));
            run_certified_pipeline(&incumbent, &[robust], &config).unwrap()
        },
        "abstain" =>
        {
            let incumbent =
                EstimatorEvidence::new("ols", residuals(9.0, 80, 0), residuals(9.0, 80, 1));
            let a = EstimatorEvidence::new(
                "a",
                (0..80)
                    .map(|i| if i % 2 == 0 { 1.0 } else { 0.9 })
                    .collect(),
                residuals(1.0, 80, 5),
            );
            let b = EstimatorEvidence::new(
                "b",
                (0..80)
                    .map(|i| if i % 2 == 0 { 0.9 } else { 1.0 })
                    .collect(),
                residuals(1.0, 80, 6),
            );
            run_certified_pipeline(&incumbent, &[a, b], &config).unwrap()
        },
        "reject_all" =>
        {
            config.tournament.quality_floor = Some(1.0);
            let incumbent =
                EstimatorEvidence::new("ols", residuals(8.0, 80, 0), residuals(8.0, 80, 1));
            let robust =
                EstimatorEvidence::new("robust", residuals(6.0, 80, 2), residuals(6.0, 80, 3));
            run_certified_pipeline(&incumbent, &[robust], &config).unwrap()
        },
        other => panic!("unknown scenario: {other}"),
    }
}

fn main() {
    println!("# industrial_deployment_policy — phase 4E.10");
    let policy = DeploymentPolicy {
        minimum_certified_coverage: 0.85,
        rollback_coverage_floor: 0.8,
        rollback_window: 25,
        rollback_minimum_samples: 12,
    };
    println!(
        "# policy: deploy bar {:.2}; rollback floor {:.2} over a {}-obs window (warm-up {})",
        policy.minimum_certified_coverage,
        policy.rollback_coverage_floor,
        policy.rollback_window,
        policy.rollback_minimum_samples,
    );

    println!();
    println!("# pre-deployment gate");
    println!("# scenario          verdict");
    for scenario in ["clear_winner", "under_certified", "abstain", "reject_all"]
    {
        let report = report_for(scenario);
        let decision = decide_deployment(&report, &policy).expect("policy decides");
        let verdict = match &decision.action
        {
            DeploymentAction::DeployChallenger { name } => format!("DeployChallenger({name})"),
            DeploymentAction::HoldIncumbent => "HoldIncumbent".to_string(),
            DeploymentAction::BlockDeployment => "BlockDeployment".to_string(),
        };
        println!("{scenario:<18} {verdict}");
        println!("#   reason: {}", decision.reasons.join("; "));
    }

    println!();
    println!("# post-deployment rollback monitor (live coverage stream)");
    let mut monitor = RollbackMonitor::from_policy(&policy).expect("monitor builds");
    // 20 healthy steps (mostly covered), then a drift where coverage collapses.
    let stream: Vec<bool> = (0..60)
        .map(|t| {
            if t < 20
            {
                // ~92% covered: miss every 12th.
                t % 12 != 0
            }
            else
            {
                // Drift: only ~30% covered.
                t % 10 < 3
            }
        })
        .collect();

    let mut announced_healthy = false;
    let mut announced_rollback = false;
    for (t, &covered) in stream.iter().enumerate()
    {
        let state = monitor.observe(covered);
        match state
        {
            MonitorState::Healthy if !announced_healthy =>
            {
                announced_healthy = true;
                println!(
                    "#   step {t:>2}: HEALTHY (window coverage {:.3})",
                    monitor.window_coverage()
                );
            },
            MonitorState::Rollback if !announced_rollback =>
            {
                announced_rollback = true;
                println!(
                    "#   step {t:>2}: ROLLBACK latched (window coverage {:.3} below floor {:.2})",
                    monitor.window_coverage(),
                    policy.rollback_coverage_floor,
                );
            },
            _ =>
            {},
        }
    }
    println!(
        "#   final: triggered={} window_coverage={:.3}",
        monitor.triggered(),
        monitor.window_coverage()
    );
    println!(
        "# the gate ships only a well-certified winner; the monitor latches a rollback once live \
coverage sustains a breach — the safe default throughout is to keep what is running."
    );
}
