//! Continual program, phase 6.2 — deterministic benchmark: what conditioning
//! sees that pooling cannot, checked against hand-derived oracles.
//!
//! # What this demonstrates
//!
//! Two orchestrators are run **side by side on a byte-identical score
//! stream**. One pools every residual into a single rolling median; the other
//! conditions on the declared partition. The pooled one is the phase 6.1
//! design, unchanged and working exactly as specified — and it never reacts,
//! because one stratum in four cannot move a median. The conditioned one
//! demands a retrain and names the stratum.
//!
//! That is the whole point, and it is uncomfortable rather than triumphant:
//! the breakdown property that makes a median trustworthy against
//! contamination is the same property that makes it blind to a minority
//! subpopulation whose model has genuinely failed. Robustness is not a free
//! good; it buys resistance to noise by spending sensitivity to minorities.
//!
//! The run also shows what conditioning does **not** fix: a stratum with no
//! traffic stays `Warming` forever, and an aggregate that reads `Quiet` while
//! three of four groups have never been observed is telling the truth and
//! saying almost nothing.
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
    EstimatorEvidence, EstimatorTournament, GroupDriftConfig, GroupDriftState, Orientation,
    RetrainReason, ServicePhase, decide_deployment, run_certified_pipeline,
};

const GROUPS: [&str; 4] = ["east", "north", "south", "west"];

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
        retrain_cadence_ticks: 5_000,
        demand_cooldown_ticks: 15,
        policy: policy(),
        drift: drift(),
    }
}

/// Four strata on their own calibrated scales: `west` naturally runs half
/// again hotter, which a shared reference would either mask or misreport.
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

fn healthy_score(group: &str) -> f64 {
    if group == "west" { 1.4 } else { 0.9 }
}

fn describe_groups(groups: &[&str]) -> String {
    if groups.is_empty()
    {
        "none".to_string()
    }
    else
    {
        groups.join("+")
    }
}

fn report_pair(
    scenario: &str,
    tick: u64,
    conditioned: &ContinualOrchestrator,
    pooled: &ContinualOrchestrator,
) {
    let monitor = conditioned
        .group_drift()
        .expect("conditioned by construction");
    let conditioned_status = conditioned.status();
    let pooled_status = pooled.status();
    println!(
        "scenario={scenario} tick={tick} conditioned_state={:?} alarmed={} warming={} \
         conditioned_demand={} pooled_drift={:?} pooled_ratio={:.6} pooled_demand={} \
         window_coverage={:.6}",
        monitor.state(),
        describe_groups(&monitor.alarmed_groups()),
        describe_groups(&monitor.warming_groups()),
        conditioned_status
            .open_demand
            .map_or("none".to_string(), |reason| format!("{reason:?}")),
        pooled_status.drift,
        pooled_status.drift_ratio,
        pooled_status
            .open_demand
            .map_or("none".to_string(), |reason| format!("{reason:?}")),
        conditioned_status.window_coverage,
    );
}

fn main() {
    println!("# Continual program phase 6.2 deterministic group-drift benchmark");
    println!(
        "# fields: scenario tick conditioned_state alarmed warming conditioned_demand \
         pooled_drift pooled_ratio pooled_demand window_coverage"
    );
    println!(
        "# scope: both orchestrators consume a byte-identical score stream; the pooled one is \
         phase 6.1 unchanged and correct — its silence is a property of the median, not a bug"
    );
    println!(
        "# strata: east/north/south calibrated at scale 1.0, west at 1.5 — each judged against \
         its own reference, never a shared one"
    );

    let mut conditioned =
        ContinualOrchestrator::with_group_drift(config(), "v1", partition()).unwrap();
    let mut pooled = ContinualOrchestrator::new(config(), "v1").unwrap();
    let mut tick = 1;

    // ── 1. A healthy stream leaves both monitors quiet ────────────────────
    for _ in 0..12
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
    report_pair(
        "healthy_stream_quiet_everywhere",
        tick - 1,
        &conditioned,
        &pooled,
    );
    expect(
        conditioned.group_drift().unwrap().state() == GroupDriftState::Quiet
            && conditioned
                .group_drift()
                .unwrap()
                .alarmed_groups()
                .is_empty()
            && pooled.status().drift == DriftState::Quiet,
        "a healthy stream must alarm nowhere".to_string(),
    );

    // ── 2. One stratum in four collapses ──────────────────────────────────
    let mut demand_tick = 0;
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
            if demand_tick == 0 && outcome.opened_demand == Some(RetrainReason::DriftAlarm)
            {
                demand_tick = tick;
            }
            tick += 1;
        }
    }
    report_pair("one_stratum_collapses", tick - 1, &conditioned, &pooled);
    println!(
        "# head_to_head: conditioning demanded a retrain at tick {demand_tick} naming 'south', \
         while the pooled median sat at {:.6}x its reference and never reacted -- the minority \
         that cannot move a median is exactly the minority a median cannot protect",
        pooled.status().drift_ratio,
    );
    expect(
        demand_tick > 0
            && conditioned.group_drift().unwrap().alarmed_groups() == vec!["south"]
            && conditioned.status().open_demand == Some(RetrainReason::DriftAlarm),
        "conditioning must catch and name the collapsed stratum".to_string(),
    );
    expect(
        pooled.status().open_demand.is_none()
            && pooled.status().drift == DriftState::Quiet
            && pooled.status().drift_ratio < 3.0,
        format!(
            "the pooled monitor must stay quiet on the same stream, got {:?} at ratio {:.6}",
            pooled.status().drift,
            pooled.status().drift_ratio
        ),
    );
    expect(
        conditioned.status().window_coverage == 1.0,
        "the alarm must arrive while coverage is still perfect".to_string(),
    );

    // ── 3. Coverage is untouched: this is drift, not yet damage ───────────
    println!(
        "# early_warning: window coverage is still {:.2} on both sides -- conditioning moves the \
         warning earlier, it does not manufacture a failure",
        pooled.status().window_coverage,
    );

    // ── 4. Only a real certificate closes the demand ──────────────────────
    let outcome = conditioned
        .candidate_evaluated(tick, &certified_promotion("v1", "v2"))
        .unwrap();
    report_pair("certified_promotion_recovers", tick, &conditioned, &pooled);
    expect(
        outcome
            == CandidateOutcome::Promoted {
                now_serving: "v2".to_string(),
            },
        format!("the certified winner must be promoted, got {outcome:?}"),
    );
    expect(
        conditioned.status().phase == ServicePhase::Serving
            && conditioned.status().fallback.as_deref() == Some("v1"),
        "promotion must retain the previous champion as fallback".to_string(),
    );

    // ── 5. The successor inherits nothing ─────────────────────────────────
    expect(
        conditioned.group_drift().unwrap().state() == GroupDriftState::Warming
            && conditioned.group_drift().unwrap().warming_groups().len() == 4,
        "every stratum must start the successor's record warming".to_string(),
    );
    println!(
        "# reset_semantics: all 4 strata return to warming -- carrying a failed model's evidence \
         into its successor would be stale authority, and the warm-up blindness is the price"
    );

    // ── 6. A starved stratum is unmonitored, not healthy ──────────────────
    let mut starved = ContinualOrchestrator::with_group_drift(config(), "v1", partition()).unwrap();
    for starved_tick in 1..=60
    {
        starved
            .observe_in_group(starved_tick, "east", true, 0.9)
            .unwrap();
    }
    report_pair("starved_strata_are_not_healthy", 60, &starved, &pooled);
    println!(
        "# silence: the aggregate reads Quiet on the strength of one stratum out of four; the \
         other three have never been observed and say nothing, which is not the same as saying \
         they are fine"
    );
    expect(
        starved.group_drift().unwrap().state() == GroupDriftState::Quiet
            && starved.group_drift().unwrap().warming_groups() == vec!["north", "south", "west"]
            && starved.group_drift().unwrap().observations("north") == Some(0),
        "a starved stratum must be reported as unmonitored".to_string(),
    );

    // ── 7. Per-group reference scales, side by side ───────────────────────
    for group_report in starved.group_drift().unwrap().reports()
    {
        println!(
            "group={} state={:?} ratio={:.6} observations={}",
            group_report.name, group_report.state, group_report.ratio, group_report.observations,
        );
    }
    expect(
        starved.group_drift().unwrap().reports().len() == 4,
        "every declared stratum must appear in the report".to_string(),
    );

    println!("# all oracle checks passed");
}
