//! The continual loop, end to end: the orchestrator consuming *real*
//! deployment decisions produced by the certified pipeline (4E.7) and the
//! deployment gate (4E.10) — not hand-built ones — plus the honesty
//! properties that make the loop safe to leave running.

use scirust_srcc_bench::{
    CandidateOutcome, CertifiedPipelineConfig, ContinualConfig, ContinualError,
    ContinualOrchestrator, CoverageMode, DeploymentPolicy, DriftMonitorConfig, DriftState,
    EstimatorEvidence, EstimatorTournament, Orientation, RetrainReason, ServicePhase,
    decide_deployment, run_certified_pipeline,
};

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

/// A certified decision in which `challenger` firmly beats `incumbent`.
fn certified_promotion(
    incumbent: &str,
    challenger: &str,
) -> scirust_srcc_bench::DeploymentDecision {
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

/// A certified abstention: two challengers tie, the gate holds the incumbent.
fn certified_hold(incumbent: &str) -> scirust_srcc_bench::DeploymentDecision {
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

// ─── The full loop with real certificates ────────────────────────────────

#[test]
fn the_loop_runs_on_real_certified_decisions_not_hand_built_ones() {
    let mut orchestrator = ContinualOrchestrator::new(config(), "v1").unwrap();

    // Healthy service up to the cadence: the scheduled demand opens.
    for tick in 1..=200
    {
        orchestrator.observe(tick, true, 0.8).unwrap();
    }
    assert_eq!(
        orchestrator.status().open_demand,
        Some(RetrainReason::Scheduled)
    );

    // The certified pipeline abstains (real tie) -> the gate holds -> the
    // orchestrator closes the demand and keeps serving v1.
    let outcome = orchestrator
        .candidate_evaluated(201, &certified_hold("v1"))
        .unwrap();
    assert_eq!(
        outcome,
        CandidateOutcome::HeldIncumbent {
            demand_still_open: false
        }
    );
    assert_eq!(orchestrator.status().serving, "v1");

    // Later the pipeline certifies a genuine winner -> promotion.
    for tick in 202..=402
    {
        orchestrator.observe(tick, true, 0.8).unwrap();
    }
    assert_eq!(
        orchestrator.status().open_demand,
        Some(RetrainReason::Scheduled)
    );
    let decision = certified_promotion("v1", "v2");
    let outcome = orchestrator.candidate_evaluated(403, &decision).unwrap();
    assert_eq!(
        outcome,
        CandidateOutcome::Promoted {
            now_serving: "v2".to_string()
        }
    );
    let status = orchestrator.status();
    assert_eq!(status.phase, ServicePhase::Serving);
    assert_eq!(status.fallback.as_deref(), Some("v1"));
}

#[test]
fn the_full_arc_drift_warns_coverage_latches_and_a_certificate_recovers() {
    let mut orchestrator = ContinualOrchestrator::new(config(), "v1").unwrap();

    // Give the system a fallback via a first certified promotion.
    for tick in 1..=200
    {
        orchestrator.observe(tick, true, 0.8).unwrap();
    }
    orchestrator
        .candidate_evaluated(201, &certified_promotion("v1", "v2"))
        .unwrap();

    // Warm the fresh monitors, then inject drift: residual scores explode
    // while coverage still holds — the drift alarm is the early warning.
    for tick in 202..=215
    {
        orchestrator.observe(tick, true, 0.8).unwrap();
    }
    let mut drift_demand_tick = None;
    let mut tick = 216;
    while drift_demand_tick.is_none() && tick < 260
    {
        let outcome = orchestrator.observe(tick, true, 12.0).unwrap();
        if outcome.opened_demand == Some(RetrainReason::DriftAlarm)
        {
            drift_demand_tick = Some(tick);
        }
        tick += 1;
    }
    let drift_tick = drift_demand_tick.expect("drift must warn before coverage degrades");
    assert_eq!(orchestrator.status().window_coverage, 1.0);

    // The demanded evaluation is blocked (nothing certifiable yet).
    let blocked = orchestrator
        .candidate_evaluated(drift_tick + 1, &certified_hold("v2"))
        .unwrap();
    assert!(matches!(blocked, CandidateOutcome::HeldIncumbent { .. }));

    // Now coverage actually collapses: the latch rolls back to v1.
    let mut rolled_tick = None;
    let mut tick = drift_tick + 2;
    while rolled_tick.is_none() && tick < drift_tick + 60
    {
        if orchestrator.observe(tick, false, 12.0).unwrap().rolled_back
        {
            rolled_tick = Some(tick);
        }
        tick += 1;
    }
    rolled_tick.expect("sustained misses must latch the rollback");
    let status = orchestrator.status();
    assert_eq!(status.phase, ServicePhase::RolledBack);
    assert_eq!(status.serving, "v1");
    assert_eq!(status.fallback, None, "the failed v2 must be retired");
    assert_eq!(status.open_demand, Some(RetrainReason::CoverageRollback));

    // Only a certified promotion exits the rollback.
    let promoted = orchestrator
        .candidate_evaluated(tick, &certified_promotion("v1", "v3"))
        .unwrap();
    assert_eq!(
        promoted,
        CandidateOutcome::Promoted {
            now_serving: "v3".to_string()
        }
    );
    let status = orchestrator.status();
    assert_eq!(status.phase, ServicePhase::Serving);
    assert_eq!(status.fallback.as_deref(), Some("v1"));
    assert_eq!(status.drift, DriftState::Warming);

    // The audit log tells the whole story in order.
    let reasons: Vec<&str> = orchestrator
        .audit_log()
        .iter()
        .map(|record| record.reason.as_str())
        .collect();
    assert!(reasons.iter().any(|reason| reason.contains("Scheduled")));
    assert!(reasons.iter().any(|reason| reason.contains("DriftAlarm")));
    assert!(
        reasons
            .iter()
            .any(|reason| reason.contains("latched a rollback"))
    );
    assert!(
        reasons
            .iter()
            .any(|reason| reason.contains("promoted 'v3'"))
    );
}

// ─── Honesty properties ──────────────────────────────────────────────────

#[test]
fn a_promotion_cannot_bypass_the_demand_discipline() {
    let mut orchestrator = ContinualOrchestrator::new(config(), "v1").unwrap();
    orchestrator.observe(1, true, 0.8).unwrap();
    // A perfectly certified decision, delivered unsolicited, is refused.
    let decision = certified_promotion("v1", "v2");
    assert_eq!(
        orchestrator.candidate_evaluated(2, &decision),
        Err(ContinualError::UnsolicitedCandidate { tick: 2 })
    );
    assert_eq!(orchestrator.status().serving, "v1");
}

#[test]
fn replaying_the_same_event_sequence_reproduces_everything() {
    let run = || {
        let mut orchestrator = ContinualOrchestrator::new(config(), "v1").unwrap();
        for tick in 1..=200
        {
            orchestrator
                .observe(tick, tick % 11 != 0, 0.5 + ((tick % 5) as f64) * 0.2)
                .unwrap();
        }
        orchestrator
            .candidate_evaluated(201, &certified_promotion("v1", "v2"))
            .unwrap();
        for tick in 202..=300
        {
            orchestrator
                .observe(tick, tick % 7 != 0, 0.5 + ((tick % 3) as f64) * 0.3)
                .unwrap();
        }
        orchestrator
    };
    let first = run();
    let second = run();
    assert_eq!(first, second);
    assert_eq!(first.audit_log(), second.audit_log());
    assert_eq!(first.status(), second.status());
}

#[test]
fn ticks_may_repeat_but_never_regress() {
    let mut orchestrator = ContinualOrchestrator::new(config(), "v1").unwrap();
    orchestrator.observe(5, true, 0.5).unwrap();
    // Several observations on the same tick are legitimate (bursts).
    orchestrator.observe(5, true, 0.5).unwrap();
    assert_eq!(
        orchestrator.observe(4, true, 0.5),
        Err(ContinualError::TickRegression { last: 5, tick: 4 })
    );
}

#[test]
fn serving_is_not_a_correctness_claim_and_the_docs_say_so() {
    // The module documentation carries the boundary; this test pins the
    // observable half: a Serving status coexists with an alarmed drift
    // monitor (the world can be moving while coverage still holds), so
    // Serving cannot be read as "the model is right".
    let mut orchestrator = ContinualOrchestrator::new(config(), "v1").unwrap();
    for tick in 1..=20
    {
        orchestrator.observe(tick, true, 0.8).unwrap();
    }
    for tick in 21..=60
    {
        orchestrator.observe(tick, true, 12.0).unwrap();
    }
    let status = orchestrator.status();
    assert_eq!(status.phase, ServicePhase::Serving);
    assert_eq!(status.drift, DriftState::Alarmed);
    assert!(status.drift_ratio > 3.0);
}
