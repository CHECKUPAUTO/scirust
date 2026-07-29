//! Group-conditioned drift in the governed loop (phase 6.2), end to end: a
//! single stratum collapses, the conditioned orchestrator demands a retrain
//! that the pooled one never would, and a *real* certified decision — produced
//! by the 4E.7 pipeline and the 4E.10 gate, not hand-built — is what recovers
//! the service.

use scirust_srcc_bench::{
    CandidateOutcome, CertifiedPipelineConfig, ContinualConfig, ContinualOrchestrator,
    CoverageMode, DeploymentDecision, DeploymentPolicy, DriftMonitorConfig, DriftState,
    EstimatorEvidence, EstimatorTournament, GroupDriftConfig, GroupDriftState, Orientation,
    RetrainReason, ServicePhase, decide_deployment, run_certified_pipeline,
};

const GROUPS: [&str; 4] = ["east", "north", "south", "west"];

fn policy() -> DeploymentPolicy {
    DeploymentPolicy {
        minimum_certified_coverage: 0.85,
        rollback_coverage_floor: 0.7,
        rollback_window: 12,
        rollback_minimum_samples: 6,
    }
}

fn drift() -> DriftMonitorConfig {
    DriftMonitorConfig {
        reference_scale: 1.0,
        alarm_ratio: 3.0,
        window: 10,
        minimum_samples: 5,
        consecutive_breaches_to_alarm: 4,
        consecutive_normals_to_clear: 6,
    }
}

fn config() -> ContinualConfig {
    ContinualConfig {
        retrain_cadence_ticks: 2_000,
        demand_cooldown_ticks: 15,
        policy: policy(),
        drift: drift(),
    }
}

/// Four strata, each calibrated on its own scale. `west` naturally runs half
/// again hotter than the others — a difference that would skew a shared
/// reference and is exactly why conditioning carries per-group scales.
fn partition() -> GroupDriftConfig {
    GroupDriftConfig::uniform(
        drift(),
        &[("east", 1.0), ("north", 1.0), ("south", 1.0), ("west", 1.5)],
    )
    .unwrap()
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

/// The nominal residual score for a group in a healthy regime: `west` sits at
/// its own higher calibrated scale, everyone else at theirs.
fn healthy_score(group: &str) -> f64 {
    if group == "west" { 1.4 } else { 0.9 }
}

#[test]
fn a_single_stratum_collapse_is_caught_by_conditioning_and_missed_when_pooled() {
    let mut conditioned =
        ContinualOrchestrator::with_group_drift(config(), "v1", partition()).unwrap();
    let mut pooled = ContinualOrchestrator::new(config(), "v1").unwrap();

    // Warm both on a healthy stream.
    let mut tick = 1;
    for _ in 0..10
    {
        for group in GROUPS
        {
            conditioned
                .observe_in_group(tick, group, true, healthy_score(group))
                .unwrap();
            pooled.observe(tick, true, healthy_score(group)).unwrap();
            tick += 1;
        }
    }
    assert_eq!(conditioned.status().open_demand, None);
    assert_eq!(pooled.status().open_demand, None);

    // `south` collapses to four times its calibrated scale. Coverage still
    // holds everywhere — this is drift, not yet damage.
    let mut demand_tick = None;
    for _ in 0..25
    {
        for group in GROUPS
        {
            let score = if group == "south"
            {
                4.0
            }
            else
            {
                healthy_score(group)
            };
            let outcome = conditioned
                .observe_in_group(tick, group, true, score)
                .unwrap();
            pooled.observe(tick, true, score).unwrap();
            if demand_tick.is_none() && outcome.opened_demand == Some(RetrainReason::DriftAlarm)
            {
                demand_tick = Some(tick);
            }
            tick += 1;
        }
    }

    demand_tick.expect("conditioning must demand a retrain for the collapsed stratum");
    let monitor = conditioned.group_drift().unwrap();
    assert_eq!(monitor.state(), GroupDriftState::Alarmed);
    assert_eq!(monitor.alarmed_groups(), vec!["south"]);
    assert_eq!(conditioned.status().window_coverage, 1.0);

    // The pooled orchestrator saw the identical scores and never reacted: one
    // stratum in four cannot move a median. That is the blind spot.
    assert_eq!(pooled.status().open_demand, None);
    assert_eq!(pooled.status().drift, DriftState::Quiet);
    assert!(
        pooled.status().drift_ratio < 1.5,
        "the pooled median must sit far below its 3.0 threshold, not just under it — got {}",
        pooled.status().drift_ratio
    );
}

#[test]
fn only_a_real_certificate_closes_a_group_alarm_demand() {
    let mut orchestrator =
        ContinualOrchestrator::with_group_drift(config(), "v1", partition()).unwrap();
    let mut tick = 1;
    let mut demand_tick = None;
    while demand_tick.is_none() && tick < 400
    {
        for group in GROUPS
        {
            let score = if group == "north"
            {
                5.0
            }
            else
            {
                healthy_score(group)
            };
            let outcome = orchestrator
                .observe_in_group(tick, group, true, score)
                .unwrap();
            if outcome.opened_demand == Some(RetrainReason::DriftAlarm)
            {
                demand_tick = Some(tick);
            }
            tick += 1;
        }
    }
    demand_tick.expect("the alarmed stratum must open a demand");

    // The certified pipeline runs for real and returns a firm winner. The
    // evaluation lands at the current tick, not at the demand's: the inner
    // loop keeps streaming after the alarm, and ticks never regress.
    let outcome = orchestrator
        .candidate_evaluated(tick, &certified_promotion("v1", "v2"))
        .unwrap();
    assert_eq!(
        outcome,
        CandidateOutcome::Promoted {
            now_serving: "v2".to_string()
        }
    );

    let status = orchestrator.status();
    assert_eq!(status.phase, ServicePhase::Serving);
    assert_eq!(status.fallback.as_deref(), Some("v1"));
    // The successor inherits no evidence: every stratum is warming again.
    let monitor = orchestrator.group_drift().unwrap();
    assert_eq!(monitor.state(), GroupDriftState::Warming);
    assert_eq!(monitor.warming_groups().len(), 4);
}

#[test]
fn a_starved_stratum_never_reads_as_healthy() {
    let mut orchestrator =
        ContinualOrchestrator::with_group_drift(config(), "v1", partition()).unwrap();
    // Only `east` sends traffic. The aggregate is Quiet — which is true, and
    // radically incomplete.
    for tick in 1..=60
    {
        orchestrator
            .observe_in_group(tick, "east", true, 0.9)
            .unwrap();
    }
    let monitor = orchestrator.group_drift().unwrap();
    assert_eq!(monitor.state(), GroupDriftState::Quiet);
    assert_eq!(monitor.warming_groups(), vec!["north", "south", "west"]);
    assert_eq!(monitor.observations("north"), Some(0));
    assert_eq!(monitor.group_state("east"), Some(DriftState::Quiet));
}

#[test]
fn replaying_a_grouped_event_sequence_reproduces_everything() {
    let run = || {
        let mut orchestrator =
            ContinualOrchestrator::with_group_drift(config(), "v1", partition()).unwrap();
        let mut tick = 1;
        for round in 0..40
        {
            for (offset, group) in GROUPS.iter().enumerate()
            {
                let score = healthy_score(group) + 0.3 * ((round + offset) % 5) as f64;
                orchestrator
                    .observe_in_group(tick, group, tick % 13 != 0, score)
                    .unwrap();
                tick += 1;
            }
        }
        orchestrator
    };
    let first = run();
    let second = run();
    assert_eq!(first, second);
    assert_eq!(first.audit_log(), second.audit_log());
    assert_eq!(
        first.group_drift().unwrap().reports(),
        second.group_drift().unwrap().reports()
    );
}
