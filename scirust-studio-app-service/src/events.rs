//! Bounded application event fan-out.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::job::JobState;
use crate::worker::WorkerExitClass;

/// Something the application wants the interface to know about.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum AppEvent {
    /// A worker process is up and has completed the handshake.
    WorkerStarted {
        /// The worker's crate version.
        worker_version: String,
        /// Where it was launched from.
        path: String,
    },
    /// The worker process is gone.
    WorkerExited {
        /// Why.
        class: WorkerExitClass,
    },
    /// A job was accepted.
    JobStarted {
        /// The job.
        job_id: String,
        /// The capability being run.
        capability_id: String,
        /// Whether genuine progress will be reported for it.
        supports_progress: bool,
    },
    /// A job's state changed.
    JobState {
        /// The job.
        job_id: String,
        /// Its new state.
        state: JobState,
    },
    /// A job raised a non-fatal warning.
    JobWarning {
        /// The job.
        job_id: String,
        /// The warning.
        warning: scirust_studio_runtime::RunWarning,
    },
    /// Something worth showing in the activity log that is not job- or
    /// worker-lifecycle.
    Diagnostic {
        /// Severity.
        level: DiagnosticLevel,
        /// The message.
        message: String,
    },
}

/// How prominent a diagnostic is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticLevel {
    /// Routine progress through a workflow.
    Info,
    /// A check that passed.
    Pass,
    /// Something the user should notice.
    Warning,
    /// Something that failed.
    Error,
}

/// A bounded, lossy-at-the-oldest event buffer.
///
/// Bounded because a runaway producer must not be able to exhaust memory
/// through the event channel. Lossy at the *oldest* end because the newest
/// events are the ones a user is looking at, and dropping *silently* is what
/// would be unacceptable — [`EventBus::drain`] reports how many were
/// discarded so the interface can say so rather than showing a gap.
#[derive(Debug, Clone)]
pub struct EventBus {
    inner: Arc<Mutex<BusState>>,
}

#[derive(Debug)]
struct BusState {
    events: VecDeque<AppEvent>,
    capacity: usize,
    dropped: u64,
}

/// A batch of events, plus how many were lost before it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventBatch {
    /// The events, oldest first.
    pub events: Vec<AppEvent>,
    /// How many events were discarded because the buffer was full since the
    /// previous drain. Non-zero means the interface fell behind.
    pub dropped: u64,
}

impl EventBus {
    /// A bus holding at most `capacity` events.
    pub fn new(capacity: usize) -> Self {
        EventBus {
            inner: Arc::new(Mutex::new(BusState {
                events: VecDeque::new(),
                capacity: capacity.max(1),
                dropped: 0,
            })),
        }
    }

    /// Publish an event, discarding the oldest if the buffer is full.
    pub fn publish(&self, event: AppEvent) {
        // A poisoned lock means another thread panicked while holding it.
        // Losing an event is the right trade here: the alternative is
        // propagating a panic out of a worker-reader thread and taking the
        // application down over a log line.
        let Ok(mut state) = self.inner.lock()
        else
        {
            return;
        };
        if state.events.len() >= state.capacity
        {
            state.events.pop_front();
            state.dropped += 1;
        }
        state.events.push_back(event);
    }

    /// Take everything buffered, resetting the dropped counter.
    pub fn drain(&self) -> EventBatch {
        let Ok(mut state) = self.inner.lock()
        else
        {
            return EventBatch {
                events: Vec::new(),
                dropped: 0,
            };
        };
        let events = state.events.drain(..).collect();
        let dropped = std::mem::take(&mut state.dropped);
        EventBatch { events, dropped }
    }

    /// How many events are waiting.
    pub fn len(&self) -> usize {
        self.inner.lock().map(|s| s.events.len()).unwrap_or(0)
    }

    /// Whether nothing is waiting.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(n: usize) -> AppEvent {
        AppEvent::Diagnostic {
            level: DiagnosticLevel::Info,
            message: format!("event {n}"),
        }
    }

    #[test]
    fn events_come_back_in_order() {
        let bus = EventBus::new(10);
        for i in 0..5
        {
            bus.publish(info(i));
        }
        let batch = bus.drain();
        assert_eq!(batch.events.len(), 5);
        assert_eq!(batch.dropped, 0);
        assert_eq!(batch.events[0], info(0));
        assert_eq!(batch.events[4], info(4));
    }

    #[test]
    fn draining_empties_the_bus() {
        let bus = EventBus::new(10);
        bus.publish(info(1));
        assert!(!bus.is_empty());
        bus.drain();
        assert!(bus.is_empty());
        assert!(bus.drain().events.is_empty());
    }

    /// The bound is the point: overflowing must cost the oldest events and
    /// must be *reported*, never silent.
    #[test]
    fn overflow_drops_the_oldest_and_says_how_many() {
        let bus = EventBus::new(3);
        for i in 0..10
        {
            bus.publish(info(i));
        }
        let batch = bus.drain();
        assert_eq!(batch.events.len(), 3);
        assert_eq!(batch.dropped, 7);
        assert_eq!(batch.events[0], info(7), "the newest are kept");
        assert_eq!(batch.events[2], info(9));
    }

    #[test]
    fn the_dropped_counter_resets_after_a_drain() {
        let bus = EventBus::new(2);
        for i in 0..5
        {
            bus.publish(info(i));
        }
        assert_eq!(bus.drain().dropped, 3);
        bus.publish(info(99));
        assert_eq!(bus.drain().dropped, 0);
    }

    #[test]
    fn a_zero_capacity_still_holds_one_event() {
        let bus = EventBus::new(0);
        bus.publish(info(1));
        assert_eq!(bus.drain().events.len(), 1);
    }

    #[test]
    fn the_bus_is_shared_across_clones_and_threads() {
        let bus = EventBus::new(1000);
        let mut handles = Vec::new();
        for t in 0..4
        {
            let bus = bus.clone();
            handles.push(std::thread::spawn(move || {
                for i in 0..50
                {
                    bus.publish(info(t * 100 + i));
                }
            }));
        }
        for h in handles
        {
            h.join().unwrap();
        }
        assert_eq!(bus.drain().events.len(), 200);
    }

    #[test]
    fn events_round_trip_through_json_with_stable_tags() {
        let events = vec![
            AppEvent::WorkerStarted {
                worker_version: "0.1.0".into(),
                path: "/x/worker".into(),
            },
            AppEvent::WorkerExited {
                class: WorkerExitClass::Requested,
            },
            AppEvent::JobStarted {
                job_id: "j1".into(),
                capability_id: "c".into(),
                supports_progress: true,
            },
            AppEvent::JobState {
                job_id: "j1".into(),
                state: JobState::RunningIndeterminate,
            },
            info(1),
        ];
        for e in events
        {
            let json = serde_json::to_string(&e).unwrap();
            let back: AppEvent = serde_json::from_str(&json).unwrap();
            assert_eq!(back, e);
            assert!(json.contains("\"event\""), "{json}");
        }
    }
}
