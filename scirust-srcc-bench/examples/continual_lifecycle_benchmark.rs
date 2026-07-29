//! Continual program, phase 6.1 — deterministic benchmark: one full life of a
//! governed model service, checked against hand-derived oracles.
//!
//! # What this demonstrates
//!
//! The whole loop that phases 4E.7 and 4E.10 left implicit, run end to end on
//! **real** certified decisions (the tournament and the deployment gate are
//! actually executed, not simulated): healthy service reaching its scheduled
//! retrain; a certified abstention closing that demand; drift raising its
//! alarm *while coverage still reads 100%* — the early warning arriving
//! before the damage; a blocked recovery attempt; the coverage latch rolling
//! back to the fallback and retiring the failed model; and finally a
//! certified promotion — the only exit the orchestrator accepts from a
//! rollback.
//!
//! # Reproducibility contract
//!
//! Logical ticks only — no wall clock, no RNG outside the tournament's fixed
//! seed. All scientific content goes to **stdout** in a fixed field order;
//! nothing else does. Two runs hashed with SHA-256 must be byte-identical.
//!
//! On any oracle mismatch this program prints a diagnostic to **stderr** and
//! exits with a non-zero status.

use scirust_srcc_bench::{
    CandidateOutcome, CertifiedPipelineConfig, ContinualConfig, ContinualOrchestrator,
    CoverageMode, DeploymentDecision, DeploymentPolicy, DriftMonitorConfig, DriftState,
    EstimatorEvidence, EstimatorTournament, Orientation, RetrainReason, ServicePhase,
    decide_deployment, run_certified_pipeline,
};

fn expect(condition: bool, description: String) {
    if !condition
    {
        eprintln!("ORACLE FAILURE: {description}");
        std::process::exit(1);
    }
}

fn policy() -> DeploymentPolicy {
    DeploymentPolicy {
        minimum_certified_coverage: 0.85,
        rollback_coverage_floor: 0.7,
        rollback_window: 12,
        rollback_minimum_samples: 6,
    }
}

fn config() -> ContinualConfig {
    ContinualConfig {
        retrain_cadence_ticks: 200,
        demand_cooldown_ticks: 15,
        policy: policy(),
        drift: DriftMonitorConfig {
            reference_scale: 1.0,
            alarm_ratio: 3.0,
            window: 10,
            minimum_samples: 5,
            consecutive_breaches_to_alarm: 4,
            consecutive_normals_to_clear: 6,
        },
    }
}

fn tournament() -> EstimatorTournament {
    EstimatorTournament {
        orientation: Orientation::LowerIsBetter,
        min_improvement: 0.0,
        tie_margin: 0.0,
        quality_floor: None,
        resamples: 2000,
        level: 0.95,
        seed: 0x00D0_9111,
    }
}

fn residuals(magnitude: f64, n: usize, offset: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (((i + offset) * 7) % 11) as f64 / 11.0 - 0.5)
        .map(|u| u * magnitude)
        .collect()
}

/// Runs the real certified pipeline and deployment gate: a firm winner.
fn certified_promotion(incumbent: &str, challenger: &str) -> DeploymentDecision {
    let incumbent = EstimatorEvidence::new(incumbent, residuals(8.0, 80, 0), residuals(8.0, 80, 1));
    let strong = EstimatorEvidence::new(challenger, residuals(0.6, 80, 2), residuals(0.6, 80, 3));
    let pipeline_config = CertifiedPipelineConfig {
        tournament: tournament(),
        coverage_level: 0.9,
        coverage_mode: CoverageMode::Marginal,
    };
    let report = run_certified_pipeline(&incumbent, &[strong], &pipeline_config).unwrap();
    decide_deployment(&report, &policy()).unwrap()
}

/// Runs the real certified pipeline and deployment gate: a genuine tie, so
/// the gate holds the incumbent.
fn certified_hold(incumbent: &str) -> DeploymentDecision {
    let incumbent = EstimatorEvidence::new(incumbent, residuals(9.0, 80, 0), residuals(9.0, 80, 1));
    let a = EstimatorEvidence::new(
        "challenger-a",
        (0..80)
            .map(|i| if i % 2 == 0 { 1.0 } else { 0.9 })
            .collect(),
        residuals(1.0, 80, 5),
    );
    let b = EstimatorEvidence::new(
        "challenger-b",
        (0..80)
            .map(|i| if i % 2 == 0 { 0.9 } else { 1.0 })
            .collect(),
        residuals(1.0, 80, 6),
    );
    let pipeline_config = CertifiedPipelineConfig {
        tournament: tournament(),
        coverage_level: 0.9,
        coverage_mode: CoverageMode::Marginal,
    };
    let report = run_certified_pipeline(&incumbent, &[a, b], &pipeline_config).unwrap();
    decide_deployment(&report, &policy()).unwrap()
}

fn describe_demand(demand: Option<RetrainReason>) -> String {
    demand.map_or("none".to_string(), |reason| format!("{reason:?}"))
}

fn report(scenario: &str, tick: u64, orchestrator: &ContinualOrchestrator) {
    let status = orchestrator.status();
    println!(
        "scenario={scenario} tick={tick} phase={:?} serving={} fallback={} demand={} \
         window_coverage={:.6} drift={:?} drift_ratio={:.6} consecutive_blocks={} audit_len={}",
        status.phase,
        status.serving,
        status.fallback.as_deref().unwrap_or("none"),
        describe_demand(status.open_demand),
        status.window_coverage,
        status.drift,
        status.drift_ratio,
        status.consecutive_blocks,
        orchestrator.audit_log().len(),
    );
}

fn main() {
    println!("# Continual program phase 6.1 deterministic lifecycle benchmark");
    println!(
        "# fields: scenario tick phase serving fallback demand window_coverage drift drift_ratio \
         consecutive_blocks audit_len"
    );
    println!(
        "# scope: the orchestrator sequences certified decisions over logical time; it does no \
         statistics, cannot manufacture a promotion, and Serving means only that no monitor \
         objects — never that the model is correct"
    );

    let mut orchestrator = ContinualOrchestrator::new(config(), "v1").unwrap();

    // ── 1. Healthy service reaches the scheduled cadence ─────────────────
    let mut scheduled_tick = 0;
    for tick in 1..=210
    {
        let outcome = orchestrator.observe(tick, true, 0.8).unwrap();
        if outcome.opened_demand == Some(RetrainReason::Scheduled)
        {
            scheduled_tick = tick;
        }
    }
    report("scheduled_demand_opens", scheduled_tick, &orchestrator);
    expect(
        scheduled_tick == 200,
        format!("the cadence of 200 must open the demand at tick 200, got {scheduled_tick}"),
    );

    // ── 2. A real certified abstention closes it ─────────────────────────
    let hold = certified_hold("v1");
    let outcome = orchestrator.candidate_evaluated(211, &hold).unwrap();
    report("certified_hold_closes_demand", 211, &orchestrator);
    expect(
        outcome
            == CandidateOutcome::HeldIncumbent {
                demand_still_open: false,
            },
        format!("a certified tie must hold and close, got {outcome:?}"),
    );

    // ── 3. A real certified winner is promoted at the next cadence ───────
    for tick in 212..=411
    {
        orchestrator.observe(tick, true, 0.8).unwrap();
    }
    let outcome = orchestrator
        .candidate_evaluated(412, &certified_promotion("v1", "v2"))
        .unwrap();
    report("certified_promotion", 412, &orchestrator);
    expect(
        outcome
            == CandidateOutcome::Promoted {
                now_serving: "v2".to_string(),
            },
        format!("the certified winner must be promoted, got {outcome:?}"),
    );
    expect(
        orchestrator.status().fallback.as_deref() == Some("v1"),
        "the previous champion must be retained as fallback".to_string(),
    );

    // ── 4. Drift warns while coverage still reads 100% ───────────────────
    for tick in 413..=427
    {
        orchestrator.observe(tick, true, 0.8).unwrap();
    }
    let mut drift_tick = 0;
    let mut tick = 428;
    while drift_tick == 0 && tick < 470
    {
        let outcome = orchestrator.observe(tick, true, 12.0).unwrap();
        if outcome.opened_demand == Some(RetrainReason::DriftAlarm)
        {
            drift_tick = tick;
        }
        tick += 1;
    }
    report("drift_alarm_before_damage", drift_tick, &orchestrator);
    println!(
        "# early_warning: the drift demand opened at tick {drift_tick} with window coverage \
         still {:.2} -- the alarm arrives before the failure, which is the whole point of \
         watching residual scale separately from coverage",
        orchestrator.status().window_coverage,
    );
    expect(
        drift_tick > 0 && orchestrator.status().window_coverage == 1.0,
        "the drift alarm must fire while coverage is still perfect".to_string(),
    );

    // ── 5. The demanded evaluation cannot certify anything yet ───────────
    let outcome = orchestrator
        .candidate_evaluated(drift_tick + 1, &certified_hold("v2"))
        .unwrap();
    report("recovery_attempt_held", drift_tick + 1, &orchestrator);
    expect(
        outcome
            == CandidateOutcome::HeldIncumbent {
                demand_still_open: false,
            },
        format!("a tie while Serving closes the demand, got {outcome:?}"),
    );

    // ── 6. Coverage collapses: the latch rolls back and retires v2 ───────
    let mut rolled_tick = 0;
    let mut tick = drift_tick + 2;
    while rolled_tick == 0 && tick < drift_tick + 60
    {
        if orchestrator.observe(tick, false, 12.0).unwrap().rolled_back
        {
            rolled_tick = tick;
        }
        tick += 1;
    }
    report("coverage_latch_rolls_back", rolled_tick, &orchestrator);
    println!(
        "# rollback_semantics: v2 is retired (not kept one crash away from serving again), the \
         fallback v1 serves under fresh warm-up monitors, and the demand stays open because \
         serving the fallback is a mitigation, not a recovery"
    );
    let status = orchestrator.status();
    expect(
        rolled_tick > 0
            && status.phase == ServicePhase::RolledBack
            && status.serving == "v1"
            && status.fallback.is_none()
            && status.open_demand == Some(RetrainReason::CoverageRollback),
        format!("rollback must serve the fallback and retire the failure, got {status:?}"),
    );

    // ── 7. Holds and blocks do not exit a rollback ───────────────────────
    let outcome = orchestrator
        .candidate_evaluated(rolled_tick + 1, &certified_hold("v1"))
        .unwrap();
    report("hold_cannot_exit_rollback", rolled_tick + 1, &orchestrator);
    expect(
        outcome
            == CandidateOutcome::HeldIncumbent {
                demand_still_open: true,
            },
        format!("a hold while rolled back must keep demanding, got {outcome:?}"),
    );

    // ── 8. Only a certified promotion exits ──────────────────────────────
    let outcome = orchestrator
        .candidate_evaluated(rolled_tick + 2, &certified_promotion("v1", "v3"))
        .unwrap();
    report(
        "certified_promotion_recovers",
        rolled_tick + 2,
        &orchestrator,
    );
    expect(
        outcome
            == CandidateOutcome::Promoted {
                now_serving: "v3".to_string(),
            },
        format!("the certified v3 must be promoted, got {outcome:?}"),
    );

    // ── 9. Service is healthy again, with the story on the record ────────
    for tick in rolled_tick + 3..=rolled_tick + 30
    {
        orchestrator.observe(tick, true, 0.8).unwrap();
    }
    report("healthy_again", rolled_tick + 30, &orchestrator);
    let status = orchestrator.status();
    expect(
        status.phase == ServicePhase::Serving
            && status.serving == "v3"
            && status.fallback.as_deref() == Some("v1")
            && status.window_coverage == 1.0
            && status.drift == DriftState::Quiet,
        format!("the service must be healthy after recovery, got {status:?}"),
    );
    println!(
        "# audit: {} transitions recorded, every one carrying its tick, phases and reason -- the \
         same event sequence replays to a byte-identical log",
        orchestrator.audit_log().len(),
    );

    println!("# all oracle checks passed");
}
