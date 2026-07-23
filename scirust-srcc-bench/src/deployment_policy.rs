//! Production decision and rollback policy (phase 4E.10).
//!
//! The certified pipeline (4E.7) turns leakage-free evidence into a decision plus
//! a coverage certificate. Shipping that decision safely needs one more layer:
//! a **pre-deployment gate** that only promotes a challenger when the certificate
//! actually clears a production bar, and a **post-deployment monitor** that rolls
//! the deployment back if the model's *live* coverage degrades (drift the offline
//! certificate could not foresee). This module is that governance layer, and it
//! is deliberately conservative — the safe default is to keep what is running.
//!
//! - [`decide_deployment`] maps a [`CertifiedPipelineReport`] to a
//!   [`DeploymentAction`]: deploy the challenger only on a *firm* `Select` whose
//!   certificate meets the required coverage; an abstention (`Tie` /
//!   `Inconclusive`) or a `HoldIncumbent` keeps the incumbent; a `RejectAll`, or a
//!   `Select` whose certified coverage falls short, **blocks** the deployment.
//! - [`RollbackMonitor`] tracks the deployed model's per-observation coverage over
//!   a rolling window and **latches** a rollback once the window coverage sits
//!   below a floor (with a warm-up so a cold start cannot trigger it). A rollback
//!   is one-way: once tripped it stays tripped until a human resets it.
//!
//! All deterministic; no RNG.

use core::fmt;
use std::collections::VecDeque;

use crate::certified_pipeline::CertifiedPipelineReport;
use crate::tournament::TournamentDecision;

/// A production deployment / rollback policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeploymentPolicy {
    /// The challenger's certificate must target **and** empirically meet at least
    /// this coverage on the held-out test fold to be deployable, in `(0, 1)`.
    pub minimum_certified_coverage: f64,
    /// Live window coverage at or below this floor (after warm-up) latches a
    /// rollback, in `(0, 1)`.
    pub rollback_coverage_floor: f64,
    /// Rolling-window size for live monitoring.
    pub rollback_window: usize,
    /// Minimum observations before the monitor may trigger (warm-up).
    pub rollback_minimum_samples: usize,
}

impl DeploymentPolicy {
    fn validate(&self) -> Result<(), DeploymentPolicyError> {
        if !in_open_unit(self.minimum_certified_coverage)
        {
            return Err(DeploymentPolicyError::InvalidCoverage {
                value: self.minimum_certified_coverage,
            });
        }
        if !in_open_unit(self.rollback_coverage_floor)
        {
            return Err(DeploymentPolicyError::InvalidCoverage {
                value: self.rollback_coverage_floor,
            });
        }
        if self.rollback_window == 0
        {
            return Err(DeploymentPolicyError::ZeroWindow);
        }
        if self.rollback_minimum_samples == 0
            || self.rollback_minimum_samples > self.rollback_window
        {
            return Err(DeploymentPolicyError::InvalidWarmup {
                minimum_samples: self.rollback_minimum_samples,
                window: self.rollback_window,
            });
        }
        Ok(())
    }
}

fn in_open_unit(value: f64) -> bool {
    value.is_finite() && value > 0.0 && value < 1.0
}

/// The pre-deployment verdict.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeploymentAction {
    /// Promote this challenger to production.
    DeployChallenger {
        /// The challenger's name.
        name: String,
    },
    /// Keep the incumbent running (no defensible, certified improvement).
    HoldIncumbent,
    /// Do not deploy anything — the evidence is unsafe (rejected, or certified
    /// coverage falls short of the production bar).
    BlockDeployment,
}

/// A pre-deployment decision with its justification.
#[derive(Clone, Debug, PartialEq)]
pub struct DeploymentDecision {
    /// The action to take.
    pub action: DeploymentAction,
    /// Human-readable justification.
    pub reasons: Vec<String>,
}

/// Typed policy errors.
#[derive(Clone, Debug, PartialEq)]
pub enum DeploymentPolicyError {
    /// A coverage value was not in the open interval `(0, 1)`.
    InvalidCoverage {
        /// The rejected value.
        value: f64,
    },
    /// The rolling window was zero.
    ZeroWindow,
    /// The warm-up was zero or exceeded the window.
    InvalidWarmup {
        /// The requested warm-up.
        minimum_samples: usize,
        /// The window size.
        window: usize,
    },
}

impl fmt::Display for DeploymentPolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidCoverage { value } =>
            {
                write!(
                    formatter,
                    "coverage {value} must lie in the open interval (0, 1)"
                )
            },
            Self::ZeroWindow => formatter.write_str("the rollback window must be positive"),
            Self::InvalidWarmup {
                minimum_samples,
                window,
            } => write!(
                formatter,
                "warm-up {minimum_samples} must be in 1..={window}"
            ),
        }
    }
}

impl std::error::Error for DeploymentPolicyError {}

/// Maps a certified pipeline report to a production deployment action.
///
/// # Errors
///
/// [`DeploymentPolicyError`] when the policy's coverage or window parameters are
/// out of range.
pub fn decide_deployment(
    report: &CertifiedPipelineReport,
    policy: &DeploymentPolicy,
) -> Result<DeploymentDecision, DeploymentPolicyError> {
    policy.validate()?;
    let bar = policy.minimum_certified_coverage;

    let decision = match &report.selection.decision
    {
        TournamentDecision::Select { winner } =>
        {
            // A firm selection: deploy only if its certificate clears the bar.
            match &report.coverage
            {
                Some(cert) if cert.level >= bar && cert.empirical_test_coverage >= bar =>
                {
                    DeploymentDecision {
                        action: DeploymentAction::DeployChallenger {
                            name: winner.clone(),
                        },
                        reasons: vec![format!(
                            "'{winner}' firmly selected; certified coverage {:.3} (target {:.3}) \
                             meets the production bar {bar:.3}",
                            cert.empirical_test_coverage, cert.level
                        )],
                    }
                },
                Some(cert) => DeploymentDecision {
                    action: DeploymentAction::BlockDeployment,
                    reasons: vec![format!(
                        "'{winner}' selected but certified coverage {:.3} (target {:.3}) is \
                             below the production bar {bar:.3} — blocked",
                        cert.empirical_test_coverage, cert.level
                    )],
                },
                None => DeploymentDecision {
                    action: DeploymentAction::BlockDeployment,
                    reasons: vec![
                        "a challenger was selected but carries no coverage certificate — blocked"
                            .to_string(),
                    ],
                },
            }
        },
        TournamentDecision::HoldIncumbent => DeploymentDecision {
            action: DeploymentAction::HoldIncumbent,
            reasons: vec![
                "no challenger was a defensible improvement — incumbent retained".to_string(),
            ],
        },
        TournamentDecision::Tie { contenders } => DeploymentDecision {
            action: DeploymentAction::HoldIncumbent,
            reasons: vec![format!(
                "selection abstained (tie among {contenders:?}) — incumbent retained pending a \
                     secondary criterion"
            )],
        },
        TournamentDecision::Inconclusive => DeploymentDecision {
            action: DeploymentAction::HoldIncumbent,
            reasons: vec![
                "selection abstained (inconclusive) — incumbent retained pending more evidence"
                    .to_string(),
            ],
        },
        TournamentDecision::RejectAll => DeploymentDecision {
            action: DeploymentAction::BlockDeployment,
            reasons: vec!["no candidate met the quality floor — deployment blocked".to_string()],
        },
    };

    Ok(decision)
}

/// The live-monitoring verdict for one observation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MonitorState {
    /// Still inside the warm-up window; not yet actionable.
    Warming,
    /// Window coverage is at or above the floor.
    Healthy,
    /// Window coverage fell below the floor — roll back (latched).
    Rollback,
}

/// A rolling-window live coverage monitor that latches a rollback on a sustained
/// breach.
#[derive(Clone, Debug, PartialEq)]
pub struct RollbackMonitor {
    floor: f64,
    capacity: usize,
    minimum_samples: usize,
    window: VecDeque<bool>,
    covered: usize,
    triggered: bool,
}

impl RollbackMonitor {
    /// Builds a monitor from a policy's rollback parameters.
    ///
    /// # Errors
    ///
    /// [`DeploymentPolicyError`] when the policy parameters are out of range.
    pub fn from_policy(policy: &DeploymentPolicy) -> Result<Self, DeploymentPolicyError> {
        policy.validate()?;
        Ok(Self {
            floor: policy.rollback_coverage_floor,
            capacity: policy.rollback_window,
            minimum_samples: policy.rollback_minimum_samples,
            window: VecDeque::with_capacity(policy.rollback_window),
            covered: 0,
            triggered: false,
        })
    }

    /// Records one live outcome (`true` = the interval covered the realized value)
    /// and returns the monitor state. Once a rollback is latched it stays latched.
    pub fn observe(&mut self, covered: bool) -> MonitorState {
        if self.window.len() == self.capacity
            && let Some(oldest) = self.window.pop_front()
            && oldest
        {
            self.covered -= 1;
        }
        self.window.push_back(covered);
        if covered
        {
            self.covered += 1;
        }

        if self.triggered
        {
            return MonitorState::Rollback;
        }
        if self.window.len() < self.minimum_samples
        {
            return MonitorState::Warming;
        }
        if self.window_coverage() < self.floor
        {
            self.triggered = true;
            MonitorState::Rollback
        }
        else
        {
            MonitorState::Healthy
        }
    }

    /// The current rolling-window coverage (`1.0` before any observation).
    pub fn window_coverage(&self) -> f64 {
        if self.window.is_empty()
        {
            return 1.0;
        }
        self.covered as f64 / self.window.len() as f64
    }

    /// Whether a rollback has been latched.
    pub fn triggered(&self) -> bool {
        self.triggered
    }

    /// Observations seen so far in the window.
    pub fn window_len(&self) -> usize {
        self.window.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certified_pipeline::{
        CertifiedPipelineConfig, CoverageMode, EstimatorEvidence, run_certified_pipeline,
    };
    use crate::promotion::Orientation;
    use crate::tournament::EstimatorTournament;

    fn policy() -> DeploymentPolicy {
        DeploymentPolicy {
            minimum_certified_coverage: 0.85,
            rollback_coverage_floor: 0.8,
            rollback_window: 20,
            rollback_minimum_samples: 10,
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

    /// A clear-winner report: a robust challenger dominates OLS.
    fn winning_report(level: f64) -> CertifiedPipelineReport {
        let incumbent = EstimatorEvidence::new("ols", residuals(8.0, 80, 0), residuals(8.0, 80, 1));
        let robust = EstimatorEvidence::new("robust", residuals(0.6, 80, 2), residuals(0.6, 80, 3));
        let config = CertifiedPipelineConfig {
            tournament: tournament(),
            coverage_level: level,
            coverage_mode: CoverageMode::Marginal,
        };
        run_certified_pipeline(&incumbent, &[robust], &config).unwrap()
    }

    #[test]
    fn deploys_a_firm_selection_that_clears_the_bar() {
        let report = winning_report(0.9);
        let decision = decide_deployment(&report, &policy()).unwrap();
        assert_eq!(
            decision.action,
            DeploymentAction::DeployChallenger {
                name: "robust".to_string()
            }
        );
    }

    #[test]
    fn blocks_a_selection_whose_coverage_is_below_the_bar() {
        // A certificate targeting only 0.5 cannot clear a 0.85 production bar.
        let report = winning_report(0.5);
        let decision = decide_deployment(&report, &policy()).unwrap();
        assert_eq!(decision.action, DeploymentAction::BlockDeployment);
        assert!(decision.reasons[0].contains("below the production bar"));
    }

    #[test]
    fn abstention_holds_the_incumbent() {
        // Two challengers that both beat OLS but tie each other -> abstain.
        let incumbent = EstimatorEvidence::new("ols", residuals(9.0, 80, 0), residuals(9.0, 80, 1));
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
        let config = CertifiedPipelineConfig {
            tournament: tournament(),
            coverage_level: 0.9,
            coverage_mode: CoverageMode::Marginal,
        };
        let report = run_certified_pipeline(&incumbent, &[a, b], &config).unwrap();
        assert!(matches!(
            report.selection.decision,
            TournamentDecision::Tie { .. } | TournamentDecision::Inconclusive
        ));
        let decision = decide_deployment(&report, &policy()).unwrap();
        assert_eq!(decision.action, DeploymentAction::HoldIncumbent);
    }

    #[test]
    fn reject_all_blocks_deployment() {
        let incumbent = EstimatorEvidence::new("ols", residuals(8.0, 80, 0), residuals(8.0, 80, 1));
        let robust = EstimatorEvidence::new("robust", residuals(6.0, 80, 2), residuals(6.0, 80, 3));
        let mut config = CertifiedPipelineConfig {
            tournament: tournament(),
            coverage_level: 0.9,
            coverage_mode: CoverageMode::Marginal,
        };
        config.tournament.quality_floor = Some(1.0); // nobody meets it
        let report = run_certified_pipeline(&incumbent, &[robust], &config).unwrap();
        assert_eq!(report.selection.decision, TournamentDecision::RejectAll);
        let decision = decide_deployment(&report, &policy()).unwrap();
        assert_eq!(decision.action, DeploymentAction::BlockDeployment);
    }

    #[test]
    fn monitor_stays_healthy_then_rolls_back_on_a_breach() {
        let mut monitor = RollbackMonitor::from_policy(&policy()).unwrap();
        // 15 healthy observations (all covered): after warm-up (10) -> Healthy.
        let mut last = MonitorState::Warming;
        for _ in 0..15
        {
            last = monitor.observe(true);
        }
        assert_eq!(last, MonitorState::Healthy);
        assert!(!monitor.triggered());

        // Now a run of misses drives window coverage below the 0.8 floor.
        let mut rolled = false;
        for _ in 0..20
        {
            if monitor.observe(false) == MonitorState::Rollback
            {
                rolled = true;
            }
        }
        assert!(rolled, "sustained misses must trigger a rollback");
        assert!(monitor.triggered());
        // Latched: even a recovery keeps it rolled back.
        assert_eq!(monitor.observe(true), MonitorState::Rollback);
    }

    #[test]
    fn monitor_warms_up_before_it_can_trigger() {
        let mut monitor = RollbackMonitor::from_policy(&policy()).unwrap();
        // Even all-misses cannot trigger before minimum_samples (10) is reached.
        for _ in 0..9
        {
            assert_eq!(monitor.observe(false), MonitorState::Warming);
        }
        assert!(!monitor.triggered());
        // The 10th miss reaches warm-up with 0% coverage -> rollback.
        assert_eq!(monitor.observe(false), MonitorState::Rollback);
    }

    #[test]
    fn invalid_policy_is_a_typed_error() {
        let mut bad = policy();
        bad.minimum_certified_coverage = 1.5;
        assert_eq!(
            decide_deployment(&winning_report(0.9), &bad),
            Err(DeploymentPolicyError::InvalidCoverage { value: 1.5 })
        );

        let mut zero_window = policy();
        zero_window.rollback_window = 0;
        assert_eq!(
            RollbackMonitor::from_policy(&zero_window),
            Err(DeploymentPolicyError::ZeroWindow)
        );

        let mut bad_warmup = policy();
        bad_warmup.rollback_minimum_samples = 100; // > window
        assert_eq!(
            RollbackMonitor::from_policy(&bad_warmup),
            Err(DeploymentPolicyError::InvalidWarmup {
                minimum_samples: 100,
                window: 20
            })
        );
    }
}
