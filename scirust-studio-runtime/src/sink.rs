//! Structured lifecycle events an execution emits.
//!
//! No adapter in this crate emits a fake fractional `Progress` — Phase 2A's
//! integrations are single blocking calls with no intermediate callback
//! from `scirust-sim`, so there is nothing genuine to report between
//! `Started` and `Completed`/`Failed`/`Cancelled`. A future chunked or
//! worker-driven execution (Phase 2B) can emit real `Progress` events
//! without changing this enum's shape.

use crate::result::RunWarning;

/// One structured event during an execution.
#[derive(Debug, Clone, PartialEq)]
pub enum RunEvent {
    /// Execution began.
    Started,
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
