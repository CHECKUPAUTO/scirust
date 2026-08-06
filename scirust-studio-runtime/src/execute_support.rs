//! Shared execution helpers: cancellable integration and genuine progress
//! reporting.
//!
//! Phase 2A checked [`ExecutionControl`] exactly once, before handing the
//! whole time span to `scirust-sim` as a single blocking call — so a run that
//! had already started could not be stopped. Phase 2B replaces that with
//! `scirust_sim::simulate_observed` / `simulate_second_order_observed`, which
//! call back after every accepted step. Those functions run the *same* loop
//! as the unobserved ones (`simulate` is a wrapper around
//! `simulate_observed`), and `scirust-sim`'s own
//! `observed_rk4_run_is_bit_identical_to_the_unobserved_one` test pins that
//! down — so making a run cancellable does not move a single digit of a run
//! that is allowed to finish.
//!
//! The cancellation flag is read once per accepted step. An RK4 step costs
//! four right-hand-side evaluations of real floating-point work, so one
//! atomic load beside it is not worth batching away, and checking every step
//! is what makes cancellation feel immediate rather than "immediate to
//! within a batch".

use scirust_sim::{
    ObservedRun, SecondOrderSystem, SimError, StepAction, System, Trajectory, simulate_observed,
    simulate_second_order_observed,
};

use crate::adapter::ExecutionError;
use crate::control::ExecutionControl;
use crate::sink::{EventSink, RunEvent};

/// The most [`RunEvent::Progress`] events a single run will emit.
///
/// Progress exists so a caller can draw a bar and a desktop client can show
/// that a long run is alive — not so it can receive one message per step. A
/// million-step run emitting a million progress events across a pipe would
/// cost far more than the integration it is reporting on, so the emitter
/// only fires when the completed fraction crosses the next
/// `1 / MAX_PROGRESS_EVENTS` boundary.
pub(crate) const MAX_PROGRESS_EVENTS: usize = 20;

/// The time span and step an integration covers.
///
/// Bundled rather than passed as three bare `f64`s so a caller cannot
/// silently swap `t0` and `t_end`, and so the helpers below stay readable.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TimeSpan {
    /// Start time.
    pub t0: f64,
    /// End time.
    pub t_end: f64,
    /// Fixed step size.
    pub h: f64,
}

/// Emits [`RunEvent::Progress`] at most [`MAX_PROGRESS_EVENTS`] times, when
/// the fraction of the time span completed crosses the next boundary.
struct ProgressEmitter {
    t0: f64,
    span: f64,
    next_boundary: f64,
}

impl ProgressEmitter {
    fn new(t0: f64, t_end: f64) -> Self {
        let span = t_end - t0;
        ProgressEmitter {
            t0,
            span,
            next_boundary: 1.0 / MAX_PROGRESS_EVENTS as f64,
        }
    }

    /// Report that integration has reached `t`, emitting an event if a
    /// boundary was crossed.
    fn observe(&mut self, t: f64, sink: &mut dyn EventSink) {
        // A zero or non-finite span cannot happen for a validated scenario
        // (`solver.end` must exceed `solver.start`, both finite), but a
        // guard here is cheaper than a division that could produce the
        // NaN fraction we spend the rest of the crate keeping out.
        // `is_finite` rejects NaN, so the span comparison below is a
        // plain, totally-ordered one.
        if !self.span.is_finite() || self.span <= 0.0
        {
            return;
        }
        let fraction = ((t - self.t0) / self.span).clamp(0.0, 1.0);
        if fraction >= self.next_boundary
        {
            // Skip past every boundary this step jumped over, so a coarse
            // step never produces a burst of back-to-back events.
            let step = 1.0 / MAX_PROGRESS_EVENTS as f64;
            while self.next_boundary <= fraction
            {
                self.next_boundary += step;
            }
            sink.emit(RunEvent::Progress { fraction, t });
        }
    }
}

/// Turn a stopped-early [`ObservedRun`] into the cancellation error, and a
/// completed one into its trajectory.
fn finish(
    run: Result<ObservedRun, SimError>,
    sink: &mut dyn EventSink,
) -> Result<Trajectory, ExecutionError> {
    match run
    {
        Ok(run) if run.completed => Ok(run.trajectory),
        Ok(_) =>
        {
            sink.emit(RunEvent::Cancelled);
            Err(ExecutionError::Cancelled)
        },
        Err(e) =>
        {
            sink.emit(RunEvent::Failed(e.to_string()));
            Err(ExecutionError::Numerical(e.to_string()))
        },
    }
}

/// Integrate `system` with fixed-step RK4, checking `control` after every
/// accepted step and reporting genuine progress to `sink`.
///
/// Returns [`ExecutionError::Cancelled`] if cancellation was requested
/// mid-run, and [`ExecutionError::Numerical`] if `scirust-sim` reported a
/// failure — emitting the matching [`RunEvent`] in both cases, so a caller
/// watching only the event stream sees the same outcome as one inspecting
/// the return value.
pub(crate) fn simulate_cancellable<S: System>(
    system: &S,
    y0: &[f64],
    span: TimeSpan,
    control: &ExecutionControl,
    sink: &mut dyn EventSink,
) -> Result<Trajectory, ExecutionError> {
    if control.is_cancelled()
    {
        sink.emit(RunEvent::Cancelled);
        return Err(ExecutionError::Cancelled);
    }
    let mut progress = ProgressEmitter::new(span.t0, span.t_end);
    let run = simulate_observed(system, y0, span.t0, span.t_end, span.h, |t, _| {
        progress.observe(t, sink);
        if control.is_cancelled()
        {
            StepAction::Stop
        }
        else
        {
            StepAction::Continue
        }
    });
    finish(run, sink)
}

/// [`simulate_cancellable`] for mechanical systems integrated with
/// symplectic Euler.
pub(crate) fn simulate_second_order_cancellable<S: SecondOrderSystem>(
    system: &S,
    q0: &[f64],
    v0: &[f64],
    span: TimeSpan,
    control: &ExecutionControl,
    sink: &mut dyn EventSink,
) -> Result<Trajectory, ExecutionError> {
    if control.is_cancelled()
    {
        sink.emit(RunEvent::Cancelled);
        return Err(ExecutionError::Cancelled);
    }
    let mut progress = ProgressEmitter::new(span.t0, span.t_end);
    let run =
        simulate_second_order_observed(system, q0, v0, span.t0, span.t_end, span.h, |t, _| {
            progress.observe(t, sink);
            if control.is_cancelled()
            {
                StepAction::Stop
            }
            else
            {
                StepAction::Continue
            }
        });
    finish(run, sink)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::{CollectingEventSink, NullEventSink};
    use scirust_sim::simulate;

    /// `y' = -y`.
    struct Decay;

    impl System for Decay {
        fn dim(&self) -> usize {
            1
        }

        fn derivatives(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
            dydt[0] = -y[0];
        }
    }

    #[test]
    fn an_uncancelled_run_matches_the_plain_integrator_exactly() {
        let plain = simulate(&Decay, &[1.0], 0.0, 5.0, 0.01).unwrap();
        let observed = simulate_cancellable(
            &Decay,
            &[1.0],
            TimeSpan {
                t0: 0.0,
                t_end: 5.0,
                h: 0.01,
            },
            &ExecutionControl::new(),
            &mut NullEventSink,
        )
        .unwrap();
        assert_eq!(plain.t, observed.t);
        assert_eq!(plain.y, observed.y);
    }

    #[test]
    fn a_control_cancelled_before_the_run_stops_immediately() {
        let control = ExecutionControl::new();
        control.cancel();
        let mut sink = CollectingEventSink::new();
        let err = simulate_cancellable(
            &Decay,
            &[1.0],
            TimeSpan {
                t0: 0.0,
                t_end: 5.0,
                h: 0.01,
            },
            &control,
            &mut sink,
        )
        .unwrap_err();
        assert_eq!(err, ExecutionError::Cancelled);
        assert_eq!(sink.events(), &[RunEvent::Cancelled]);
    }

    /// Cancels the run the moment it reports its second progress event —
    /// so the cancellation lands genuinely *mid-integration*, at a point
    /// the test can assert on, with no sleeps and no thread racing.
    struct CancelOnSecondProgress {
        control: ExecutionControl,
        progress_seen: usize,
        events: Vec<RunEvent>,
    }

    impl EventSink for CancelOnSecondProgress {
        fn emit(&mut self, event: RunEvent) {
            if let RunEvent::Progress { .. } = event
            {
                self.progress_seen += 1;
                if self.progress_seen == 2
                {
                    self.control.cancel();
                }
            }
            self.events.push(event);
        }
    }

    #[test]
    fn cancelling_mid_run_stops_the_integration_and_reports_cancelled() {
        let control = ExecutionControl::new();
        let mut sink = CancelOnSecondProgress {
            control: control.clone(),
            progress_seen: 0,
            events: Vec::new(),
        };

        // Long enough that finishing before the second progress event is
        // impossible: progress fires at 5% of the span, cancellation at 10%.
        let err = simulate_cancellable(
            &Decay,
            &[1.0],
            TimeSpan {
                t0: 0.0,
                t_end: 100.0,
                h: 0.001,
            },
            &control,
            &mut sink,
        )
        .unwrap_err();

        assert_eq!(err, ExecutionError::Cancelled);
        assert!(sink.events.contains(&RunEvent::Cancelled));
        // It really did stop early rather than run to completion and then
        // report cancellation: the last progress fraction is nowhere near 1.
        let last_fraction = sink
            .events
            .iter()
            .filter_map(|e| match e
            {
                RunEvent::Progress { fraction, .. } => Some(*fraction),
                _ => None,
            })
            .next_back()
            .expect("progress was reported");
        assert!(
            last_fraction < 0.5,
            "cancelled at {last_fraction} of the span, expected well before halfway"
        );
    }

    #[test]
    fn progress_events_are_bounded_monotonic_and_within_range() {
        let mut sink = CollectingEventSink::new();
        simulate_cancellable(
            &Decay,
            &[1.0],
            TimeSpan {
                t0: 0.0,
                t_end: 5.0,
                h: 0.001,
            },
            &ExecutionControl::new(),
            &mut sink,
        )
        .unwrap();

        let fractions: Vec<f64> = sink
            .events()
            .iter()
            .filter_map(|e| match e
            {
                RunEvent::Progress { fraction, .. } => Some(*fraction),
                _ => None,
            })
            .collect();

        assert!(
            !fractions.is_empty(),
            "a 5000-step run should report progress"
        );
        assert!(
            fractions.len() <= MAX_PROGRESS_EVENTS,
            "emitted {} progress events, more than the {MAX_PROGRESS_EVENTS} cap",
            fractions.len()
        );
        for pair in fractions.windows(2)
        {
            assert!(pair[1] > pair[0], "progress must increase: {fractions:?}");
        }
        for f in &fractions
        {
            assert!((0.0..=1.0).contains(f), "fraction out of range: {f}");
        }
    }

    #[test]
    fn a_short_run_never_reports_a_fraction_above_one() {
        // Two steps over the whole span: the second lands exactly on t_end.
        let mut sink = CollectingEventSink::new();
        simulate_cancellable(
            &Decay,
            &[1.0],
            TimeSpan {
                t0: 0.0,
                t_end: 1.0,
                h: 0.5,
            },
            &ExecutionControl::new(),
            &mut sink,
        )
        .unwrap();
        for e in sink.events()
        {
            if let RunEvent::Progress { fraction, .. } = e
            {
                assert!((0.0..=1.0).contains(fraction), "{fraction}");
            }
        }
    }

    #[test]
    fn a_numerical_failure_emits_failed_and_reports_numerical() {
        // A step far larger than the span is rejected by the engine's own
        // input validation; the helper must surface it as Numerical, not
        // swallow it into a partial trajectory.
        let mut sink = CollectingEventSink::new();
        let err = simulate_cancellable(
            &Decay,
            &[1.0],
            TimeSpan {
                t0: 0.0,
                t_end: 1.0,
                h: -1.0,
            },
            &ExecutionControl::new(),
            &mut sink,
        )
        .unwrap_err();
        assert!(matches!(err, ExecutionError::Numerical(_)), "{err:?}");
        assert!(
            sink.events()
                .iter()
                .any(|e| matches!(e, RunEvent::Failed(_)))
        );
    }
}
