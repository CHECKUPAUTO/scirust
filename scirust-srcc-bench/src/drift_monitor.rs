//! Streaming drift monitoring on residual scores (continual program, phase 6.1).
//!
//! The rollback monitor (phase 4E.10) watches one thing — live *coverage* — and
//! latches, because a coverage collapse is a safety event with no honest way
//! back short of a newly certified model. Drift is a different kind of signal:
//! it is the **early warning** that the world the model was calibrated on is
//! moving, often before coverage visibly degrades, and it can subside. So this
//! monitor is deliberately *not* latched: it raises an alarm under hysteresis
//! and clears it the same way, and the layer above (the continual
//! orchestrator) decides what an alarm edge is worth.
//!
//! # What is measured
//!
//! One non-negative **residual score** per observation — typically `|y − ŷ|`,
//! or any monotone proxy the caller trusts. The monitor compares the rolling
//! **median** of the last `window` scores against a frozen
//! `reference_scale` fixed at calibration time. The median is the robust
//! choice this program has used since phase 721: a minority of wild scores
//! cannot fake or mask a drift signal on their own.
//!
//! # Hysteresis, precisely
//!
//! An observation *breaches* when `window median ≥ alarm_ratio ×
//! reference_scale` (evaluated only once the warm-up is met). The state
//! machine is:
//!
//! - `Warming` until `minimum_samples` scores have been seen;
//! - `Quiet` → `Alarmed` after `consecutive_breaches_to_alarm` breaches in a
//!   row;
//! - `Alarmed` → `Quiet` after `consecutive_normals_to_clear` non-breaches in
//!   a row.
//!
//! Both thresholds are counts of *consecutive* observations, so one odd
//! window cannot flip the state in either direction.
//!
//! # What an alarm is not
//!
//! An alarm says the recent residual scale sits above the calibrated
//! reference by the configured factor — nothing more. It does not say *why*
//! (drift, sensor fault, regime change and workload mix all look alike here),
//! does not localize the cause, and its clearing does not certify recovery.
//! Those judgements belong to the certified evaluation the orchestrator
//! demands when it sees the alarm edge.
//!
//! All deterministic; no RNG, no clock.

use core::fmt;
use std::collections::VecDeque;

/// Configuration for a [`DriftMonitor`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DriftMonitorConfig {
    /// The frozen residual scale from calibration; must be finite and `> 0`.
    /// The monitor never re-estimates it — a reference that adapted to the
    /// stream would eventually absorb the very drift it is meant to expose.
    pub reference_scale: f64,
    /// A window median at or above `alarm_ratio × reference_scale` counts as
    /// a breach; must be finite and `> 1`.
    pub alarm_ratio: f64,
    /// Rolling-window size for the median; must be `> 0`.
    pub window: usize,
    /// Observations required before the monitor is actionable, in
    /// `1..=window`.
    pub minimum_samples: usize,
    /// Consecutive breaches required to raise the alarm; must be `> 0`.
    pub consecutive_breaches_to_alarm: usize,
    /// Consecutive non-breaches required to clear it; must be `> 0`.
    pub consecutive_normals_to_clear: usize,
}

impl DriftMonitorConfig {
    fn validate(&self) -> Result<(), DriftMonitorError> {
        if !self.reference_scale.is_finite() || self.reference_scale <= 0.0
        {
            return Err(DriftMonitorError::InvalidReferenceScale {
                value: self.reference_scale,
            });
        }
        if !self.alarm_ratio.is_finite() || self.alarm_ratio <= 1.0
        {
            return Err(DriftMonitorError::InvalidAlarmRatio {
                value: self.alarm_ratio,
            });
        }
        if self.window == 0
        {
            return Err(DriftMonitorError::ZeroWindow);
        }
        if self.minimum_samples == 0 || self.minimum_samples > self.window
        {
            return Err(DriftMonitorError::InvalidWarmup {
                minimum_samples: self.minimum_samples,
                window: self.window,
            });
        }
        if self.consecutive_breaches_to_alarm == 0
        {
            return Err(DriftMonitorError::ZeroHysteresis {
                which: "consecutive_breaches_to_alarm",
            });
        }
        if self.consecutive_normals_to_clear == 0
        {
            return Err(DriftMonitorError::ZeroHysteresis {
                which: "consecutive_normals_to_clear",
            });
        }
        Ok(())
    }
}

/// The monitor's state after an observation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DriftState {
    /// Not enough observations to be actionable.
    Warming,
    /// No sustained breach.
    Quiet,
    /// A sustained breach is in effect.
    Alarmed,
}

/// Typed drift-monitor errors.
#[derive(Clone, Debug, PartialEq)]
pub enum DriftMonitorError {
    /// The reference scale was not finite and positive.
    InvalidReferenceScale {
        /// The rejected value.
        value: f64,
    },
    /// The alarm ratio was not finite and `> 1`.
    InvalidAlarmRatio {
        /// The rejected value.
        value: f64,
    },
    /// The window was zero.
    ZeroWindow,
    /// The warm-up was zero or exceeded the window.
    InvalidWarmup {
        /// The requested warm-up.
        minimum_samples: usize,
        /// The window size.
        window: usize,
    },
    /// A hysteresis count was zero.
    ZeroHysteresis {
        /// Which count was zero.
        which: &'static str,
    },
    /// An observed score was not finite and non-negative.
    InvalidScore {
        /// The rejected value.
        value: f64,
    },
}

impl fmt::Display for DriftMonitorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidReferenceScale { value } =>
            {
                write!(
                    formatter,
                    "reference scale {value} must be finite and positive"
                )
            },
            Self::InvalidAlarmRatio { value } =>
            {
                write!(
                    formatter,
                    "alarm ratio {value} must be finite and greater than 1"
                )
            },
            Self::ZeroWindow => formatter.write_str("the drift window must be positive"),
            Self::InvalidWarmup {
                minimum_samples,
                window,
            } => write!(
                formatter,
                "warm-up {minimum_samples} must be in 1..={window}"
            ),
            Self::ZeroHysteresis { which } =>
            {
                write!(formatter, "{which} must be positive")
            },
            Self::InvalidScore { value } =>
            {
                write!(
                    formatter,
                    "residual score {value} must be finite and non-negative"
                )
            },
        }
    }
}

impl std::error::Error for DriftMonitorError {}

/// A rolling-median drift monitor with symmetric hysteresis.
#[derive(Clone, Debug, PartialEq)]
pub struct DriftMonitor {
    config: DriftMonitorConfig,
    window: VecDeque<f64>,
    state: DriftState,
    consecutive_breaches: usize,
    consecutive_normals: usize,
}

impl DriftMonitor {
    /// Builds a monitor from a validated configuration.
    ///
    /// # Errors
    ///
    /// [`DriftMonitorError`] when any configuration parameter is out of
    /// range.
    pub fn new(config: DriftMonitorConfig) -> Result<Self, DriftMonitorError> {
        config.validate()?;
        Ok(Self {
            config,
            window: VecDeque::with_capacity(config.window),
            state: DriftState::Warming,
            consecutive_breaches: 0,
            consecutive_normals: 0,
        })
    }

    /// Records one residual score and returns the monitor state.
    ///
    /// # Errors
    ///
    /// [`DriftMonitorError::InvalidScore`] for a non-finite or negative
    /// score; the monitor's state is left untouched in that case.
    pub fn observe(&mut self, score: f64) -> Result<DriftState, DriftMonitorError> {
        if !score.is_finite() || score < 0.0
        {
            return Err(DriftMonitorError::InvalidScore { value: score });
        }
        if self.window.len() == self.config.window
        {
            self.window.pop_front();
        }
        self.window.push_back(score);

        if self.window.len() < self.config.minimum_samples
        {
            self.state = DriftState::Warming;
            return Ok(self.state);
        }

        let breach = self.window_median() >= self.config.alarm_ratio * self.config.reference_scale;
        if breach
        {
            self.consecutive_breaches += 1;
            self.consecutive_normals = 0;
        }
        else
        {
            self.consecutive_normals += 1;
            self.consecutive_breaches = 0;
        }

        self.state = match self.state
        {
            DriftState::Warming | DriftState::Quiet =>
            {
                if self.consecutive_breaches >= self.config.consecutive_breaches_to_alarm
                {
                    DriftState::Alarmed
                }
                else
                {
                    DriftState::Quiet
                }
            },
            DriftState::Alarmed =>
            {
                if self.consecutive_normals >= self.config.consecutive_normals_to_clear
                {
                    DriftState::Quiet
                }
                else
                {
                    DriftState::Alarmed
                }
            },
        };
        Ok(self.state)
    }

    /// The rolling-window median (`0.0` before any observation). Even-length
    /// windows take the mean of the two central order statistics; ordering
    /// uses `f64::total_cmp`, so the value is a deterministic function of the
    /// window contents.
    pub fn window_median(&self) -> f64 {
        if self.window.is_empty()
        {
            return 0.0;
        }
        let mut sorted: Vec<f64> = self.window.iter().copied().collect();
        sorted.sort_by(f64::total_cmp);
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 1
        {
            sorted[mid]
        }
        else
        {
            f64::midpoint(sorted[mid - 1], sorted[mid])
        }
    }

    /// The current window median expressed as a multiple of the reference
    /// scale.
    pub fn ratio(&self) -> f64 {
        self.window_median() / self.config.reference_scale
    }

    /// The current state without observing anything.
    pub fn state(&self) -> DriftState {
        self.state
    }

    /// Observations currently in the window.
    pub fn window_len(&self) -> usize {
        self.window.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> DriftMonitorConfig {
        DriftMonitorConfig {
            reference_scale: 1.0,
            alarm_ratio: 2.0,
            window: 8,
            minimum_samples: 4,
            consecutive_breaches_to_alarm: 3,
            consecutive_normals_to_clear: 5,
        }
    }

    #[test]
    fn a_clean_stream_never_alarms() {
        let mut monitor = DriftMonitor::new(config()).unwrap();
        let mut last = DriftState::Warming;
        for i in 0..50
        {
            last = monitor.observe(0.8 + 0.1 * ((i % 3) as f64)).unwrap();
        }
        assert_eq!(last, DriftState::Quiet);
        assert!(monitor.ratio() < 2.0);
    }

    #[test]
    fn warm_up_gates_the_first_verdicts() {
        let mut monitor = DriftMonitor::new(config()).unwrap();
        for _ in 0..3
        {
            assert_eq!(monitor.observe(10.0).unwrap(), DriftState::Warming);
        }
        // The 4th observation reaches the warm-up; verdicts start counting
        // from here, not before.
        assert_ne!(monitor.observe(10.0).unwrap(), DriftState::Warming);
    }

    #[test]
    fn the_alarm_needs_exactly_the_configured_run_of_breaches() {
        let mut monitor = DriftMonitor::new(config()).unwrap();
        for _ in 0..8
        {
            monitor.observe(0.5).unwrap();
        }
        // With an 8-window of 0.5s, the median first breaches on the 4th
        // wild score (window [0.5,0.5,0.5,0.5,50,50,50,50] -> median 25.25),
        // verified by hand; pushes 4, 5, 6 are then the three consecutive
        // breaches the configuration requires.
        assert_eq!(monitor.observe(50.0).unwrap(), DriftState::Quiet);
        assert_eq!(monitor.observe(50.0).unwrap(), DriftState::Quiet);
        assert_eq!(monitor.observe(50.0).unwrap(), DriftState::Quiet);
        assert_eq!(monitor.observe(50.0).unwrap(), DriftState::Quiet); // breach 1
        assert_eq!(monitor.observe(50.0).unwrap(), DriftState::Quiet); // breach 2
        assert_eq!(monitor.observe(50.0).unwrap(), DriftState::Alarmed); // breach 3
    }

    #[test]
    fn one_odd_window_cannot_flip_the_state_either_way() {
        let mut monitor = DriftMonitor::new(config()).unwrap();
        for _ in 0..8
        {
            monitor.observe(0.5).unwrap();
        }
        // A single wild score cannot move the median past the ratio, and even
        // if it could, one breach is below the run length.
        assert_eq!(monitor.observe(1_000.0).unwrap(), DriftState::Quiet);
        assert_eq!(monitor.observe(0.5).unwrap(), DriftState::Quiet);

        // Drive to Alarmed, then check one quiet observation does not clear.
        for _ in 0..16
        {
            monitor.observe(50.0).unwrap();
        }
        assert_eq!(monitor.state(), DriftState::Alarmed);
        assert_eq!(monitor.observe(0.1).unwrap(), DriftState::Alarmed);
    }

    #[test]
    fn the_alarm_clears_after_the_configured_run_of_normals() {
        let mut monitor = DriftMonitor::new(config()).unwrap();
        for _ in 0..16
        {
            monitor.observe(50.0).unwrap();
        }
        assert_eq!(monitor.state(), DriftState::Alarmed);
        // Non-breaching medians require the window itself to recover first
        // (the median must drop below ratio × reference), then 5 consecutive
        // normal verdicts.
        let mut states = Vec::new();
        for _ in 0..16
        {
            states.push(monitor.observe(0.1).unwrap());
        }
        assert_eq!(*states.last().unwrap(), DriftState::Quiet);
        assert!(
            states.contains(&DriftState::Alarmed),
            "clearing must not be instantaneous"
        );
    }

    #[test]
    fn invalid_scores_are_typed_errors_and_do_not_corrupt_state() {
        let mut monitor = DriftMonitor::new(config()).unwrap();
        monitor.observe(1.0).unwrap();
        let before = monitor.clone();
        // NaN != NaN, so the NaN case is matched structurally rather than by
        // equality on the payload.
        assert!(matches!(
            monitor.observe(f64::NAN),
            Err(DriftMonitorError::InvalidScore { value }) if value.is_nan()
        ));
        assert_eq!(
            monitor.observe(-1.0),
            Err(DriftMonitorError::InvalidScore { value: -1.0 })
        );
        assert_eq!(monitor, before);
    }

    #[test]
    fn invalid_configuration_is_a_typed_error() {
        let mut bad = config();
        bad.reference_scale = 0.0;
        assert_eq!(
            DriftMonitor::new(bad),
            Err(DriftMonitorError::InvalidReferenceScale { value: 0.0 })
        );

        let mut bad = config();
        bad.alarm_ratio = 1.0;
        assert_eq!(
            DriftMonitor::new(bad),
            Err(DriftMonitorError::InvalidAlarmRatio { value: 1.0 })
        );

        let mut bad = config();
        bad.minimum_samples = 9;
        assert_eq!(
            DriftMonitor::new(bad),
            Err(DriftMonitorError::InvalidWarmup {
                minimum_samples: 9,
                window: 8
            })
        );

        let mut bad = config();
        bad.consecutive_normals_to_clear = 0;
        assert_eq!(
            DriftMonitor::new(bad),
            Err(DriftMonitorError::ZeroHysteresis {
                which: "consecutive_normals_to_clear"
            })
        );
    }

    #[test]
    fn identical_streams_produce_identical_monitors() {
        let stream: Vec<f64> = (0..40).map(|i| ((i * 13) % 17) as f64 / 4.0).collect();
        let mut first = DriftMonitor::new(config()).unwrap();
        let mut second = DriftMonitor::new(config()).unwrap();
        for &score in &stream
        {
            first.observe(score).unwrap();
            second.observe(score).unwrap();
        }
        assert_eq!(first, second);
        assert_eq!(first.window_median(), second.window_median());
    }
}
