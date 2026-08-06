//! Structured lifecycle events an execution emits.
//!
//! No event here is fabricated. [`RunEvent::Progress`] carries the fraction
//! of the *time span* an integration has actually covered, measured from the
//! `t` `scirust-sim` reports after each accepted step — not a guess, not a
//! timer, and not a spinner dressed up as a percentage. Capabilities whose
//! solver cannot report intermediate steps (the adaptive stiff path, see
//! `crate::adapters::robertson`) emit no `Progress` at all rather than an
//! invented one.

use crate::result::RunWarning;

/// One structured event during an execution.
#[derive(Debug, Clone, PartialEq)]
pub enum RunEvent {
    /// Execution began.
    Started,
    /// Integration has reached time `t`, having covered `fraction` of the
    /// requested span (`0.0..=1.0`).
    ///
    /// Emitted at most `20` times per run — enough to drive a progress bar,
    /// few enough that reporting never costs more than the work being
    /// reported on.
    Progress {
        /// Fraction of the time span completed, in `0.0..=1.0`.
        fraction: f64,
        /// The simulation time reached, in the axis's unit.
        t: f64,
    },
    /// A non-fatal warning was raised.
    Warning(RunWarning),
    /// Execution finished successfully.
    Completed,
    /// Execution was cancelled.
    Cancelled,
    /// Execution failed.
    Failed(String),
}

/// Receives [`RunEvent`]s as an execution progresses.
pub trait EventSink {
    /// Record one event.
    fn emit(&mut self, event: RunEvent);
}

/// An [`EventSink`] that discards every event — for callers (like today's
/// CLI) that only care about the final [`crate::RunResult`].
#[derive(Debug, Clone, Copy, Default)]
pub struct NullEventSink;

impl EventSink for NullEventSink {
    fn emit(&mut self, _event: RunEvent) {}
}

/// An [`EventSink`] that maps a sub-run's progress into a slice of the whole
/// run's, and forwards everything else unchanged.
///
/// # Why this exists
///
/// Two capabilities integrate more than one trajectory —
/// `sim.mechanics.double_pendulum` runs a perturbed shadow to measure its own
/// divergence, and `sim.electrical.van_der_pol` runs a second start to show
/// the limit cycle does not depend on where you begin. Each call to
/// `simulate_cancellable` reports its own fraction from zero to one, so
/// forwarding both to the same sink makes the bar reach the end and start
/// again.
///
/// The first workaround was to hand the second run a [`NullEventSink`]. That
/// removes the reset and replaces it with a bar that reaches one hundred
/// percent at the halfway point and then sits there — a fraction the reader
/// cannot trust, which is the thing progress reporting exists not to be.
///
/// So the sub-run is given a window instead. `Started` and `Completed` are
/// **swallowed**: they are the whole run's events, not a phase's, and a
/// caller that saw two `Completed` would reasonably believe the run had
/// finished when the first arrived.
pub struct SubRangeSink<'a> {
    inner: &'a mut dyn EventSink,
    lo: f64,
    hi: f64,
}

impl<'a> SubRangeSink<'a> {
    /// Map this sub-run's `0.0..=1.0` onto `lo..=hi` of the whole.
    pub fn new(inner: &'a mut dyn EventSink, lo: f64, hi: f64) -> Self {
        SubRangeSink { inner, lo, hi }
    }
}

impl EventSink for SubRangeSink<'_> {
    fn emit(&mut self, event: RunEvent) {
        match event
        {
            RunEvent::Progress { fraction, t } => self.inner.emit(RunEvent::Progress {
                fraction: self.lo + fraction.clamp(0.0, 1.0) * (self.hi - self.lo),
                t,
            }),
            // A phase beginning or ending is not the run beginning or ending.
            RunEvent::Started | RunEvent::Completed =>
            {},
            // Cancellation, failure and warnings are the run's, and travel.
            other => self.inner.emit(other),
        }
    }
}

/// An [`EventSink`] that records every event it receives, in order — for
/// tests, and for any future caller (a desktop lifecycle panel) that needs
/// to inspect the whole sequence.
#[derive(Debug, Clone, Default)]
pub struct CollectingEventSink {
    events: Vec<RunEvent>,
}

impl CollectingEventSink {
    /// A fresh, empty sink.
    pub fn new() -> Self {
        CollectingEventSink::default()
    }

    /// Every event recorded so far, in emission order.
    pub fn events(&self) -> &[RunEvent] {
        &self.events
    }
}

impl EventSink for CollectingEventSink {
    fn emit(&mut self, event: RunEvent) {
        self.events.push(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::result::WarningCategory;

    #[test]
    fn null_sink_discards_everything() {
        let mut sink = NullEventSink;
        sink.emit(RunEvent::Started);
        sink.emit(RunEvent::Completed);
        // Nothing to assert beyond "did not panic" — that is the point.
    }

    #[test]
    fn collecting_sink_preserves_order() {
        let mut sink = CollectingEventSink::new();
        sink.emit(RunEvent::Started);
        sink.emit(RunEvent::Warning(RunWarning {
            category: WarningCategory::Numerical,
            message: "example".to_string(),
        }));
        sink.emit(RunEvent::Completed);
        assert_eq!(sink.events().len(), 3);
        assert_eq!(sink.events()[0], RunEvent::Started);
        assert_eq!(sink.events()[2], RunEvent::Completed);
    }
}
