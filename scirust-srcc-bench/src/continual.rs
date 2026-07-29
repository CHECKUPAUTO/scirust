//! Continual-service orchestration: the governed retrain–certify–promote loop
//! (continual program, phase 6.1).
//!
//! Phase 4E.10 ended with the two governance primitives — a pre-deployment
//! gate ([`crate::decide_deployment`]) and a live rollback latch
//! ([`crate::RollbackMonitor`]) — but nothing owning the *loop*: who notices
//! that a retrain is due, what a drift alarm is worth, what is served after a
//! rollback, and under what conditions the system may ever claim to be
//! healthy again. This module is that owner.
//!
//! # What the orchestrator is, and refuses to be
//!
//! It is a **deterministic state machine over logical time**. Ticks are
//! caller-supplied `u64`s; there is no wall clock, no thread, no I/O and no
//! RNG here, which is what keeps every run replayable — the same event
//! sequence produces byte-identical state and audit log. "Real time" is the
//! caller's business: feed events as they happen and the orchestrator
//! sequences the decisions.
//!
//! It also does **no statistics**. It never retrains, never evaluates, and
//! cannot manufacture a promotion: the only way a new model enters service is
//! a [`DeploymentDecision`] produced upstream by the certified pipeline
//! (4E.7) and the deployment gate (4E.10). The orchestrator's contribution is
//! *when* to demand that evaluation and *what to do* with its outcome — the
//! judgement stays with the certificate.
//!
//! # The loop
//!
//! ```text
//!            observe(tick, covered, residual_score)
//!                 │
//!   coverage latch│drift alarm edge│cadence due
//!        ▼        ▼                ▼
//!   rollback    open retrain demand (reason-ranked)
//!        │        │
//!        └────────┴──► caller runs certified pipeline + deployment gate
//!                          │
//!                 candidate_evaluated(tick, decision)
//!                          │
//!         DeployChallenger │ HoldIncumbent / BlockDeployment
//!                  ▼       ▼
//!            promotion   demand closed (Serving) or kept open (RolledBack/
//!            (fresh      Degraded — a hold does not certify recovery)
//!            monitors)
//! ```
//!
//! # Semantics worth stating precisely
//!
//! - **A retrain demand is opened** for three reasons, ranked
//!   `CoverageRollback > DriftAlarm > Scheduled`. A higher-ranked reason
//!   upgrades an already-open demand; a lower-ranked one never downgrades it.
//!   Scheduled demands respect the cadence; drift demands respect the
//!   cooldown; a coverage rollback respects neither — it is a safety event.
//! - **Rollback serves the fallback, and trusts it only with a warm-up.** The
//!   model that latched is *retired*, never kept as a fallback (a model that
//!   demonstrably failed must not be one crash away from serving again). The
//!   fallback gets fresh monitors — and the demand stays open, because
//!   serving the fallback is a mitigation, not a recovery: drift that broke
//!   the champion may break the fallback too, and if it does, the system
//!   moves to [`ServicePhase::Degraded`] and says so rather than pretending
//!   it has somewhere safe left to go.
//! - **Only a certified promotion exits `RolledBack` or `Degraded`.** A
//!   `HoldIncumbent` or `BlockDeployment` while rolled back records the
//!   failed attempt and keeps the demand open. This is deliberately stricter
//!   than the bare 4E.10 monitor (which leaves the reset to a human): here
//!   the exit criterion is explicit and machine-checkable — a new
//!   certificate.
//! - **Promotion resets the monitors.** The new model's live record starts
//!   empty, protected by the warm-up, and the previous serving model becomes
//!   the fallback — unless it is the one that latched a rollback, in which
//!   case it is retired and the fallback slot stays empty.
//!
//! # What none of this certifies
//!
//! The orchestrator sequences certificates; it does not strengthen them. A
//! promotion means the candidate cleared the 4E.7/4E.10 bar on the evidence
//! supplied at that tick — with every caveat those layers document (coverage
//! is marginal or group-conditional, exchangeability is assumed between
//! calibration and serving, and live degradation is exactly what the monitors
//! are for). `Serving` means "no monitor objects", never "the model is
//! correct".

use core::fmt;

use crate::deployment_policy::{
    DeploymentAction, DeploymentDecision, DeploymentPolicy, DeploymentPolicyError, MonitorState,
    RollbackMonitor,
};
use crate::drift_monitor::{DriftMonitor, DriftMonitorConfig, DriftMonitorError, DriftState};

/// Configuration for a [`ContinualOrchestrator`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContinualConfig {
    /// Open a `Scheduled` retrain demand once this many ticks have elapsed
    /// since the last cycle anchor (start, promotion, or demand close); must
    /// be `> 0`.
    pub retrain_cadence_ticks: u64,
    /// Minimum ticks between closing a demand and opening the next
    /// `DriftAlarm` or `Scheduled` demand. Zero disables the cooldown. A
    /// `CoverageRollback` ignores it.
    pub demand_cooldown_ticks: u64,
    /// The deployment policy whose rollback parameters drive the live
    /// coverage monitor.
    pub policy: DeploymentPolicy,
    /// The drift monitor configuration for the serving model's residual
    /// scores.
    pub drift: DriftMonitorConfig,
}

/// Why a retrain demand is open. Ordered by rank: a coverage rollback
/// outranks a drift alarm, which outranks the schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetrainReason {
    /// The cadence elapsed with nothing else wrong.
    Scheduled,
    /// The drift monitor raised a sustained alarm.
    DriftAlarm,
    /// The live coverage monitor latched a rollback.
    CoverageRollback,
}

impl RetrainReason {
    fn rank(self) -> u8 {
        match self
        {
            Self::Scheduled => 0,
            Self::DriftAlarm => 1,
            Self::CoverageRollback => 2,
        }
    }
}

/// Where the service currently stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServicePhase {
    /// A model is serving and no monitor objects.
    Serving,
    /// The champion latched a rollback; the fallback is serving under fresh
    /// monitors and a retrain demand is open.
    RolledBack,
    /// A rollback latched with no fallback available: the failed model keeps
    /// serving because nothing safer exists, and this state says so
    /// explicitly.
    Degraded,
}

/// What one [`ContinualOrchestrator::observe`] call did.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservationOutcome {
    /// The live coverage monitor's verdict for this observation.
    pub coverage: MonitorState,
    /// The drift monitor's state after this observation.
    pub drift: DriftState,
    /// A demand opened (or upgraded) by this observation, if any.
    pub opened_demand: Option<RetrainReason>,
    /// Whether this observation executed a rollback transition.
    pub rolled_back: bool,
}

/// What one [`ContinualOrchestrator::candidate_evaluated`] call did.
#[derive(Clone, Debug, PartialEq)]
pub enum CandidateOutcome {
    /// The candidate was promoted and is now serving.
    Promoted {
        /// The model now serving.
        now_serving: String,
    },
    /// The incumbent was held. `demand_still_open` is `true` when the service
    /// is rolled back or degraded — holding does not certify recovery.
    HeldIncumbent {
        /// Whether the retrain demand remains open.
        demand_still_open: bool,
    },
    /// The deployment was blocked. The consecutive-block count is reported so
    /// a caller can escalate; the orchestrator itself keeps demanding.
    Blocked {
        /// Consecutive blocked evaluations since the last promotion.
        consecutive_blocks: usize,
        /// Whether the retrain demand remains open.
        demand_still_open: bool,
    },
}

/// A snapshot of the orchestrator's externally relevant state.
#[derive(Clone, Debug, PartialEq)]
pub struct ServiceStatus {
    /// The current phase.
    pub phase: ServicePhase,
    /// The model currently serving.
    pub serving: String,
    /// The fallback model, if one is held.
    pub fallback: Option<String>,
    /// The open retrain demand, if any.
    pub open_demand: Option<RetrainReason>,
    /// Consecutive blocked evaluations since the last promotion.
    pub consecutive_blocks: usize,
    /// The live rolling-window coverage of the serving model.
    pub window_coverage: f64,
    /// The drift monitor's current state.
    pub drift: DriftState,
    /// The window median as a multiple of the calibrated reference scale.
    pub drift_ratio: f64,
}

/// One entry of the deterministic audit log.
#[derive(Clone, Debug, PartialEq)]
pub struct TransitionRecord {
    /// The logical tick at which this happened.
    pub tick: u64,
    /// The phase before.
    pub phase_before: ServicePhase,
    /// The phase after.
    pub phase_after: ServicePhase,
    /// What happened, in one human-readable sentence.
    pub reason: String,
}

/// Typed orchestrator errors.
#[derive(Clone, Debug, PartialEq)]
pub enum ContinualError {
    /// The retrain cadence was zero.
    ZeroCadence,
    /// The initial model name was empty.
    EmptyModelName,
    /// An event arrived with a tick earlier than one already processed.
    TickRegression {
        /// The last tick processed.
        last: u64,
        /// The offending tick.
        tick: u64,
    },
    /// A candidate evaluation arrived with no retrain demand open. Accepting
    /// it would let a promotion bypass the demand discipline, so it is a
    /// typed error rather than a silent promotion path.
    UnsolicitedCandidate {
        /// The tick of the unsolicited evaluation.
        tick: u64,
    },
    /// The deployment policy was invalid.
    Policy(DeploymentPolicyError),
    /// The drift configuration or an observed score was invalid.
    Drift(DriftMonitorError),
}

impl fmt::Display for ContinualError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::ZeroCadence => formatter.write_str("the retrain cadence must be positive"),
            Self::EmptyModelName => formatter.write_str("the model name must not be empty"),
            Self::TickRegression { last, tick } =>
            {
                write!(
                    formatter,
                    "tick {tick} precedes the last processed tick {last}; events must arrive in \
                     non-decreasing tick order"
                )
            },
            Self::UnsolicitedCandidate { tick } =>
            {
                write!(
                    formatter,
                    "candidate evaluation at tick {tick} arrived with no retrain demand open"
                )
            },
            Self::Policy(error) => write!(formatter, "deployment policy: {error}"),
            Self::Drift(error) => write!(formatter, "drift monitor: {error}"),
        }
    }
}

impl std::error::Error for ContinualError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self
        {
            Self::Policy(error) => Some(error),
            Self::Drift(error) => Some(error),
            _ => None,
        }
    }
}

impl From<DeploymentPolicyError> for ContinualError {
    fn from(error: DeploymentPolicyError) -> Self {
        Self::Policy(error)
    }
}

impl From<DriftMonitorError> for ContinualError {
    fn from(error: DriftMonitorError) -> Self {
        Self::Drift(error)
    }
}

/// The deterministic continual-service orchestrator. See the module docs for
/// the exact loop semantics.
#[derive(Clone, Debug, PartialEq)]
pub struct ContinualOrchestrator {
    config: ContinualConfig,
    phase: ServicePhase,
    serving: String,
    fallback: Option<String>,
    coverage: RollbackMonitor,
    drift: DriftMonitor,
    open_demand: Option<RetrainReason>,
    consecutive_blocks: usize,
    last_tick: u64,
    cycle_anchor: u64,
    last_demand_close: Option<u64>,
    log: Vec<TransitionRecord>,
}

impl ContinualOrchestrator {
    /// Starts orchestration with `initial_model` serving from tick 0 and no
    /// fallback.
    ///
    /// # Errors
    ///
    /// [`ContinualError`] when the cadence is zero, the model name is empty,
    /// or the policy / drift configuration is invalid.
    pub fn new(config: ContinualConfig, initial_model: &str) -> Result<Self, ContinualError> {
        if config.retrain_cadence_ticks == 0
        {
            return Err(ContinualError::ZeroCadence);
        }
        if initial_model.trim().is_empty()
        {
            return Err(ContinualError::EmptyModelName);
        }
        let coverage = RollbackMonitor::from_policy(&config.policy)?;
        let drift = DriftMonitor::new(config.drift)?;
        Ok(Self {
            config,
            phase: ServicePhase::Serving,
            serving: initial_model.to_string(),
            fallback: None,
            coverage,
            drift,
            open_demand: None,
            consecutive_blocks: 0,
            last_tick: 0,
            cycle_anchor: 0,
            last_demand_close: None,
            log: Vec::new(),
        })
    }

    /// Feeds one live observation of the serving model: whether its interval
    /// covered the realized value, and its residual score.
    ///
    /// # Errors
    ///
    /// [`ContinualError::TickRegression`] for out-of-order ticks and
    /// [`ContinualError::Drift`] for an invalid score; in both cases the
    /// orchestrator's state is left untouched.
    pub fn observe(
        &mut self,
        tick: u64,
        covered: bool,
        residual_score: f64,
    ) -> Result<ObservationOutcome, ContinualError> {
        self.check_tick(tick)?;
        // The drift monitor validates the score without mutating on error, and
        // the coverage monitor has not been touched yet — so an invalid score
        // leaves the whole orchestrator exactly as it was.
        let drift_state = self.drift.observe(residual_score)?;
        self.last_tick = tick;

        let was_latched = self.coverage.triggered();
        let coverage_state = self.coverage.observe(covered);
        let mut rolled_back = false;
        let mut opened = None;

        if coverage_state == MonitorState::Rollback && !was_latched
        {
            rolled_back = true;
            self.execute_rollback(tick);
            opened = self.open_demand_at(tick, RetrainReason::CoverageRollback);
        }
        else if drift_state == DriftState::Alarmed
            && self.open_demand.is_none()
            && self.cooldown_elapsed(tick)
        {
            opened = self.open_demand_at(tick, RetrainReason::DriftAlarm);
        }
        else if self.open_demand.is_none()
            && tick.saturating_sub(self.cycle_anchor) >= self.config.retrain_cadence_ticks
            && self.cooldown_elapsed(tick)
        {
            opened = self.open_demand_at(tick, RetrainReason::Scheduled);
        }

        Ok(ObservationOutcome {
            coverage: coverage_state,
            drift: drift_state,
            opened_demand: opened,
            rolled_back,
        })
    }

    /// Consumes the deployment gate's verdict on a demanded evaluation.
    ///
    /// # Errors
    ///
    /// [`ContinualError::TickRegression`] for out-of-order ticks and
    /// [`ContinualError::UnsolicitedCandidate`] when no demand is open — a
    /// promotion must never bypass the demand discipline.
    pub fn candidate_evaluated(
        &mut self,
        tick: u64,
        decision: &DeploymentDecision,
    ) -> Result<CandidateOutcome, ContinualError> {
        self.check_tick(tick)?;
        if self.open_demand.is_none()
        {
            return Err(ContinualError::UnsolicitedCandidate { tick });
        }
        self.last_tick = tick;

        match &decision.action
        {
            DeploymentAction::DeployChallenger { name } =>
            {
                let before = self.phase;
                // A model that latched a rollback is retired, never kept one
                // crash away from serving again; otherwise the previous
                // serving model becomes the fallback.
                let retiring_failed_model = self.phase == ServicePhase::Degraded;
                let previous = std::mem::replace(&mut self.serving, name.clone());
                self.fallback = if retiring_failed_model
                {
                    None
                }
                else
                {
                    Some(previous)
                };
                self.phase = ServicePhase::Serving;
                self.open_demand = None;
                self.consecutive_blocks = 0;
                self.cycle_anchor = tick;
                self.last_demand_close = Some(tick);
                self.reset_monitors();
                self.record(
                    tick,
                    before,
                    format!(
                        "promoted '{}' on a certified deployment decision; previous model {}",
                        self.serving,
                        if retiring_failed_model
                        {
                            "retired after its rollback".to_string()
                        }
                        else
                        {
                            format!(
                                "'{}' retained as fallback",
                                self.fallback.as_deref().unwrap_or("<none>")
                            )
                        }
                    ),
                );
                Ok(CandidateOutcome::Promoted {
                    now_serving: self.serving.clone(),
                })
            },
            DeploymentAction::HoldIncumbent =>
            {
                let still_open = self.close_demand_unless_unhealthy(tick, "hold");
                Ok(CandidateOutcome::HeldIncumbent {
                    demand_still_open: still_open,
                })
            },
            DeploymentAction::BlockDeployment =>
            {
                self.consecutive_blocks += 1;
                let still_open = self.close_demand_unless_unhealthy(tick, "block");
                Ok(CandidateOutcome::Blocked {
                    consecutive_blocks: self.consecutive_blocks,
                    demand_still_open: still_open,
                })
            },
        }
    }

    /// The current externally relevant state.
    pub fn status(&self) -> ServiceStatus {
        ServiceStatus {
            phase: self.phase,
            serving: self.serving.clone(),
            fallback: self.fallback.clone(),
            open_demand: self.open_demand,
            consecutive_blocks: self.consecutive_blocks,
            window_coverage: self.coverage.window_coverage(),
            drift: self.drift.state(),
            drift_ratio: self.drift.ratio(),
        }
    }

    /// The deterministic audit log, in event order.
    pub fn audit_log(&self) -> &[TransitionRecord] {
        &self.log
    }

    fn check_tick(&self, tick: u64) -> Result<(), ContinualError> {
        if tick < self.last_tick
        {
            return Err(ContinualError::TickRegression {
                last: self.last_tick,
                tick,
            });
        }
        Ok(())
    }

    fn cooldown_elapsed(&self, tick: u64) -> bool {
        match self.last_demand_close
        {
            None => true,
            Some(closed) => tick.saturating_sub(closed) >= self.config.demand_cooldown_ticks,
        }
    }

    /// Opens (or upgrades) the demand; returns the reason iff something
    /// actually changed.
    fn open_demand_at(&mut self, tick: u64, reason: RetrainReason) -> Option<RetrainReason> {
        match self.open_demand
        {
            Some(existing) if existing.rank() >= reason.rank() => None,
            existing =>
            {
                self.open_demand = Some(reason);
                let verb = if existing.is_some()
                {
                    "upgraded"
                }
                else
                {
                    "opened"
                };
                self.record(
                    tick,
                    self.phase,
                    format!("{verb} retrain demand: {reason:?}"),
                );
                Some(reason)
            },
        }
    }

    fn execute_rollback(&mut self, tick: u64) {
        let before = self.phase;
        match self.fallback.take()
        {
            Some(fallback) =>
            {
                let failed = std::mem::replace(&mut self.serving, fallback);
                self.phase = ServicePhase::RolledBack;
                self.reset_monitors();
                self.record(
                    tick,
                    before,
                    format!(
                        "live coverage latched a rollback: '{failed}' retired, fallback '{}' now \
                         serving under fresh monitors",
                        self.serving
                    ),
                );
            },
            None =>
            {
                self.phase = ServicePhase::Degraded;
                self.record(
                    tick,
                    before,
                    format!(
                        "live coverage latched a rollback for '{}' and no fallback exists: \
                         serving continues DEGRADED",
                        self.serving
                    ),
                );
            },
        }
    }

    /// Closes the demand only in a healthy phase; while rolled back or
    /// degraded a hold/block records the attempt and keeps demanding, because
    /// only a certified promotion may claim recovery.
    fn close_demand_unless_unhealthy(&mut self, tick: u64, verdict: &str) -> bool {
        if self.phase == ServicePhase::Serving
        {
            self.open_demand = None;
            self.last_demand_close = Some(tick);
            self.cycle_anchor = tick;
            self.record(
                tick,
                self.phase,
                format!("evaluation returned {verdict}: demand closed, incumbent continues"),
            );
            false
        }
        else
        {
            self.record(
                tick,
                self.phase,
                format!(
                    "evaluation returned {verdict} while {:?}: demand stays open — only a \
                     certified promotion exits this phase",
                    self.phase
                ),
            );
            true
        }
    }

    fn reset_monitors(&mut self) {
        // Both constructors re-validate configurations already validated in
        // `new`, so these cannot fail.
        self.coverage = RollbackMonitor::from_policy(&self.config.policy)
            .expect("policy was validated at construction");
        self.drift = DriftMonitor::new(self.config.drift)
            .expect("drift config was validated at construction");
    }

    fn record(&mut self, tick: u64, phase_before: ServicePhase, reason: String) {
        self.log.push(TransitionRecord {
            tick,
            phase_before,
            phase_after: self.phase,
            reason,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ContinualConfig {
        ContinualConfig {
            retrain_cadence_ticks: 100,
            demand_cooldown_ticks: 10,
            policy: DeploymentPolicy {
                minimum_certified_coverage: 0.85,
                rollback_coverage_floor: 0.7,
                rollback_window: 10,
                rollback_minimum_samples: 5,
            },
            drift: DriftMonitorConfig {
                reference_scale: 1.0,
                alarm_ratio: 3.0,
                window: 6,
                minimum_samples: 3,
                consecutive_breaches_to_alarm: 3,
                consecutive_normals_to_clear: 4,
            },
        }
    }

    fn deploy(name: &str) -> DeploymentDecision {
        DeploymentDecision {
            action: DeploymentAction::DeployChallenger {
                name: name.to_string(),
            },
            reasons: vec!["test".to_string()],
        }
    }

    fn hold() -> DeploymentDecision {
        DeploymentDecision {
            action: DeploymentAction::HoldIncumbent,
            reasons: vec!["test".to_string()],
        }
    }

    fn block() -> DeploymentDecision {
        DeploymentDecision {
            action: DeploymentAction::BlockDeployment,
            reasons: vec!["test".to_string()],
        }
    }

    #[test]
    fn a_healthy_stream_opens_only_the_scheduled_demand_at_the_cadence() {
        let mut orchestrator = ContinualOrchestrator::new(config(), "v1").unwrap();
        let mut openings = Vec::new();
        for tick in 1..=150
        {
            let outcome = orchestrator.observe(tick, true, 0.5).unwrap();
            if let Some(reason) = outcome.opened_demand
            {
                openings.push((tick, reason));
            }
        }
        assert_eq!(openings, vec![(100, RetrainReason::Scheduled)]);
        assert_eq!(orchestrator.status().phase, ServicePhase::Serving);
    }

    #[test]
    fn a_hold_closes_the_scheduled_demand_and_restarts_the_cadence() {
        let mut orchestrator = ContinualOrchestrator::new(config(), "v1").unwrap();
        for tick in 1..=100
        {
            orchestrator.observe(tick, true, 0.5).unwrap();
        }
        assert_eq!(
            orchestrator.status().open_demand,
            Some(RetrainReason::Scheduled)
        );
        let outcome = orchestrator.candidate_evaluated(101, &hold()).unwrap();
        assert_eq!(
            outcome,
            CandidateOutcome::HeldIncumbent {
                demand_still_open: false
            }
        );
        // The next scheduled demand is a full cadence after the close.
        let mut next_opening = None;
        for tick in 102..=260
        {
            if let Some(reason) = orchestrator.observe(tick, true, 0.5).unwrap().opened_demand
            {
                next_opening = Some((tick, reason));
                break;
            }
        }
        assert_eq!(next_opening, Some((201, RetrainReason::Scheduled)));
    }

    #[test]
    fn a_drift_alarm_opens_a_demand_before_coverage_degrades() {
        let mut orchestrator = ContinualOrchestrator::new(config(), "v1").unwrap();
        for tick in 1..=20
        {
            orchestrator.observe(tick, true, 0.5).unwrap();
        }
        // Residual scores explode while coverage is still fine.
        let mut opened_at = None;
        for tick in 21..=40
        {
            let outcome = orchestrator.observe(tick, true, 20.0).unwrap();
            if outcome.opened_demand == Some(RetrainReason::DriftAlarm)
            {
                opened_at = Some(tick);
                break;
            }
        }
        let tick = opened_at.expect("the drift alarm must open a demand");
        assert!(tick < 40, "the alarm should fire promptly, got {tick}");
        assert_eq!(orchestrator.status().phase, ServicePhase::Serving);
        assert_eq!(orchestrator.status().window_coverage, 1.0);
    }

    #[test]
    fn a_coverage_latch_with_a_fallback_rolls_back_and_retires_the_failure() {
        let mut orchestrator = ContinualOrchestrator::new(config(), "v1").unwrap();
        for tick in 1..=100
        {
            orchestrator.observe(tick, true, 0.5).unwrap();
        }
        orchestrator
            .candidate_evaluated(101, &deploy("v2"))
            .unwrap();
        assert_eq!(orchestrator.status().fallback.as_deref(), Some("v1"));

        // v2 collapses: sustained misses latch the rollback.
        let mut rolled_at = None;
        for tick in 102..=140
        {
            if orchestrator.observe(tick, false, 0.5).unwrap().rolled_back
            {
                rolled_at = Some(tick);
                break;
            }
        }
        rolled_at.expect("sustained misses must latch");
        let status = orchestrator.status();
        assert_eq!(status.phase, ServicePhase::RolledBack);
        assert_eq!(status.serving, "v1");
        assert_eq!(status.fallback, None, "the failed model must be retired");
        assert_eq!(status.open_demand, Some(RetrainReason::CoverageRollback));
        assert_eq!(
            status.window_coverage, 1.0,
            "the fallback starts with fresh monitors"
        );
    }

    #[test]
    fn a_latch_without_fallback_is_degraded_not_hidden() {
        let mut orchestrator = ContinualOrchestrator::new(config(), "v1").unwrap();
        let mut degraded_at = None;
        for tick in 1..=40
        {
            orchestrator.observe(tick, false, 0.5).unwrap();
            if orchestrator.status().phase == ServicePhase::Degraded
            {
                degraded_at = Some(tick);
                break;
            }
        }
        degraded_at.expect("a latch with no fallback must go Degraded");
        let status = orchestrator.status();
        assert_eq!(status.serving, "v1", "nothing safer exists to serve");
        assert_eq!(status.open_demand, Some(RetrainReason::CoverageRollback));
    }

    #[test]
    fn hold_and_block_do_not_exit_a_rollback_but_a_promotion_does() {
        let mut orchestrator = ContinualOrchestrator::new(config(), "v1").unwrap();
        for tick in 1..=100
        {
            orchestrator.observe(tick, true, 0.5).unwrap();
        }
        orchestrator
            .candidate_evaluated(101, &deploy("v2"))
            .unwrap();
        // Stop at the latch: feeding further misses would (correctly) latch
        // the fallback's own fresh monitor too and cascade to Degraded.
        let mut tick = 102;
        while !orchestrator.observe(tick, false, 0.5).unwrap().rolled_back
        {
            tick += 1;
        }
        assert_eq!(orchestrator.status().phase, ServicePhase::RolledBack);

        let held = orchestrator.candidate_evaluated(141, &hold()).unwrap();
        assert_eq!(
            held,
            CandidateOutcome::HeldIncumbent {
                demand_still_open: true
            }
        );
        let blocked = orchestrator.candidate_evaluated(142, &block()).unwrap();
        assert_eq!(
            blocked,
            CandidateOutcome::Blocked {
                consecutive_blocks: 1,
                demand_still_open: true
            }
        );
        assert_eq!(orchestrator.status().phase, ServicePhase::RolledBack);

        let promoted = orchestrator
            .candidate_evaluated(143, &deploy("v3"))
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
        assert_eq!(status.consecutive_blocks, 0);
    }

    #[test]
    fn promotion_out_of_degraded_retires_the_failed_model() {
        let mut orchestrator = ContinualOrchestrator::new(config(), "v1").unwrap();
        for tick in 1..=40
        {
            orchestrator.observe(tick, false, 0.5).unwrap();
        }
        assert_eq!(orchestrator.status().phase, ServicePhase::Degraded);
        orchestrator.candidate_evaluated(41, &deploy("v2")).unwrap();
        let status = orchestrator.status();
        assert_eq!(status.phase, ServicePhase::Serving);
        assert_eq!(status.serving, "v2");
        assert_eq!(
            status.fallback, None,
            "a model that latched a rollback must not become the fallback"
        );
    }

    #[test]
    fn an_unsolicited_candidate_is_a_typed_error() {
        let mut orchestrator = ContinualOrchestrator::new(config(), "v1").unwrap();
        assert_eq!(
            orchestrator.candidate_evaluated(5, &deploy("v2")),
            Err(ContinualError::UnsolicitedCandidate { tick: 5 })
        );
        assert_eq!(orchestrator.status().serving, "v1");
    }

    #[test]
    fn tick_regressions_are_typed_errors_that_leave_state_untouched() {
        let mut orchestrator = ContinualOrchestrator::new(config(), "v1").unwrap();
        orchestrator.observe(10, true, 0.5).unwrap();
        let before = orchestrator.clone();
        assert_eq!(
            orchestrator.observe(9, true, 0.5),
            Err(ContinualError::TickRegression { last: 10, tick: 9 })
        );
        assert_eq!(orchestrator, before);
    }

    #[test]
    fn an_invalid_score_leaves_the_orchestrator_untouched() {
        let mut orchestrator = ContinualOrchestrator::new(config(), "v1").unwrap();
        orchestrator.observe(1, true, 0.5).unwrap();
        let before = orchestrator.clone();
        assert!(matches!(
            orchestrator.observe(2, true, f64::NAN),
            Err(ContinualError::Drift(
                DriftMonitorError::InvalidScore { .. }
            ))
        ));
        assert_eq!(orchestrator, before);
    }

    #[test]
    fn a_rollback_outranks_an_open_scheduled_demand() {
        let mut orchestrator = ContinualOrchestrator::new(config(), "v1").unwrap();
        for tick in 1..=100
        {
            orchestrator.observe(tick, true, 0.5).unwrap();
        }
        assert_eq!(
            orchestrator.status().open_demand,
            Some(RetrainReason::Scheduled)
        );
        // Give it a fallback so the latch rolls back rather than degrading.
        orchestrator
            .candidate_evaluated(101, &deploy("v2"))
            .unwrap();
        for tick in 102..=101 + 100
        {
            orchestrator.observe(tick, true, 0.5).unwrap();
        }
        assert_eq!(
            orchestrator.status().open_demand,
            Some(RetrainReason::Scheduled)
        );
        for tick in 202..=240
        {
            orchestrator.observe(tick, false, 0.5).unwrap();
        }
        assert_eq!(
            orchestrator.status().open_demand,
            Some(RetrainReason::CoverageRollback),
            "the safety reason must upgrade the open demand"
        );
    }

    #[test]
    fn the_cooldown_suppresses_an_immediate_drift_reopen() {
        let mut orchestrator = ContinualOrchestrator::new(config(), "v1").unwrap();
        // Reach a drift alarm, evaluate, hold — then the alarm persists but
        // the cooldown must suppress an instant reopen.
        for tick in 1..=10
        {
            orchestrator.observe(tick, true, 0.5).unwrap();
        }
        let mut alarm_tick = 0;
        for tick in 11..=40
        {
            if orchestrator
                .observe(tick, true, 20.0)
                .unwrap()
                .opened_demand
                == Some(RetrainReason::DriftAlarm)
            {
                alarm_tick = tick;
                break;
            }
        }
        assert!(alarm_tick > 0);
        // While the phase is Serving, a hold closes even a drift demand.
        orchestrator
            .candidate_evaluated(alarm_tick + 1, &hold())
            .unwrap();
        // Drift is still alarmed, but the cooldown (10 ticks) gates reopening.
        let mut reopened_at = None;
        for tick in alarm_tick + 2..alarm_tick + 30
        {
            if orchestrator
                .observe(tick, true, 20.0)
                .unwrap()
                .opened_demand
                == Some(RetrainReason::DriftAlarm)
            {
                reopened_at = Some(tick);
                break;
            }
        }
        let reopened = reopened_at.expect("a persistent alarm must eventually reopen");
        assert!(
            reopened >= alarm_tick + 1 + 10,
            "reopen at {reopened} violated the cooldown after the close at {}",
            alarm_tick + 1
        );
    }

    #[test]
    fn identical_event_sequences_produce_identical_audit_logs() {
        let run = || {
            let mut orchestrator = ContinualOrchestrator::new(config(), "v1").unwrap();
            for tick in 1..=100
            {
                orchestrator.observe(tick, true, 0.5).unwrap();
            }
            orchestrator
                .candidate_evaluated(101, &deploy("v2"))
                .unwrap();
            for tick in 102..=140
            {
                orchestrator
                    .observe(tick, tick % 3 == 0, 0.5 + (tick % 7) as f64)
                    .unwrap();
            }
            orchestrator
        };
        let first = run();
        let second = run();
        assert_eq!(first, second);
        assert_eq!(first.audit_log(), second.audit_log());
        assert!(!first.audit_log().is_empty());
    }

    #[test]
    fn invalid_construction_is_a_typed_error() {
        let mut bad = config();
        bad.retrain_cadence_ticks = 0;
        assert_eq!(
            ContinualOrchestrator::new(bad, "v1"),
            Err(ContinualError::ZeroCadence)
        );
        assert_eq!(
            ContinualOrchestrator::new(config(), "  "),
            Err(ContinualError::EmptyModelName)
        );
        let mut bad_policy = config();
        bad_policy.policy.rollback_window = 0;
        assert!(matches!(
            ContinualOrchestrator::new(bad_policy, "v1"),
            Err(ContinualError::Policy(DeploymentPolicyError::ZeroWindow))
        ));
        let mut bad_drift = config();
        bad_drift.drift.alarm_ratio = 0.5;
        assert!(matches!(
            ContinualOrchestrator::new(bad_drift, "v1"),
            Err(ContinualError::Drift(
                DriftMonitorError::InvalidAlarmRatio { .. }
            ))
        ));
    }
}
