//! Running one capability job on its own thread.

use std::sync::mpsc::Sender;

use scirust_studio_ipc::{FailureKind, RequestId};
use scirust_studio_runtime::{
    EventSink, ExecutionControl, ExecutionError, RunEvent, RunResult, find_adapter,
};

/// What a finished job produced.
#[derive(Debug)]
pub enum JobOutcome {
    /// The job produced a result.
    Completed(Box<RunResult>),
    /// The job was cancelled before producing a result.
    Cancelled,
    /// The job failed.
    Failed {
        /// How it failed.
        kind: FailureKind,
        /// Why, already formatted.
        message: String,
    },
}

/// Everything the worker's single event loop can be woken by.
///
/// Both the stdin reader and every job thread funnel into one channel of
/// these, so the main thread has a single place to select on and remains
/// the *only* writer to stdout. That is what keeps a job's progress events
/// from interleaving mid-line with a reply to an unrelated request.
#[derive(Debug)]
pub enum Incoming {
    /// A well-formed message arrived from the client.
    Client(Box<scirust_studio_ipc::ClientMessage>),
    /// The client closed its side cleanly.
    ClientEof,
    /// The client's stream is unusable.
    ClientProtocolError(String),
    /// A running job emitted a lifecycle event.
    Event(RequestId, RunEvent),
    /// A running job finished.
    Finished(RequestId, JobOutcome),
}

/// An [`EventSink`] that forwards every event to the worker's event loop,
/// tagged with the request it belongs to.
struct ChannelSink {
    request_id: RequestId,
    tx: Sender<Incoming>,
}

impl EventSink for ChannelSink {
    fn emit(&mut self, event: RunEvent) {
        // A send failure means the event loop is gone, which happens only
        // while the process is tearing down. Dropping the event is right:
        // there is no longer anyone to tell, and a panic here would turn an
        // orderly shutdown into a crash.
        let _ = self.tx.send(Incoming::Event(self.request_id, event));
    }
}

/// Validate and run `scenario_toml`, reporting events and the final outcome
/// on `tx`.
///
/// Runs on a dedicated thread so the worker's event loop stays responsive —
/// that responsiveness is precisely what makes a `Cancel` arriving mid-run
/// able to do anything.
pub fn run_job(
    request_id: RequestId,
    scenario_toml: &str,
    control: ExecutionControl,
    tx: Sender<Incoming>,
) {
    let outcome = execute(request_id, scenario_toml, &control, &tx);
    let _ = tx.send(Incoming::Finished(request_id, outcome));
}

fn execute(
    request_id: RequestId,
    scenario_toml: &str,
    control: &ExecutionControl,
    tx: &Sender<Incoming>,
) -> JobOutcome {
    let scenario = match scirust_studio_schema::parse_toml(scenario_toml)
    {
        Ok(s) => s,
        Err(e) =>
        {
            return JobOutcome::Failed {
                kind: FailureKind::Validation,
                message: e.to_string(),
            };
        },
    };

    let registry = scirust_studio_runtime::build_registry();
    let known: Vec<&str> = registry.iter().map(|d| d.id.0).collect();
    let schema_errors = scirust_studio_schema::validate(&scenario, Some(&known));
    if !schema_errors.is_empty()
    {
        return JobOutcome::Failed {
            kind: FailureKind::Validation,
            message: schema_errors
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join("\n"),
        };
    }

    let Some(adapter) = find_adapter(&scenario.capability.id)
    else
    {
        // Generic validation already checked the id against the registry,
        // so reaching here means the registry and the adapter table
        // disagree — a bug in this build, not in the input.
        return JobOutcome::Failed {
            kind: FailureKind::Internal,
            message: format!(
                "capability `{}` is catalogued but has no adapter",
                scenario.capability.id
            ),
        };
    };

    let validated = match adapter.validate(&scenario)
    {
        Ok(v) => v,
        Err(report) =>
        {
            return JobOutcome::Failed {
                kind: FailureKind::Validation,
                message: report.to_string(),
            };
        },
    };

    let mut sink = ChannelSink {
        request_id,
        tx: tx.clone(),
    };
    match adapter.execute(&validated, control, &mut sink)
    {
        Ok(result) => JobOutcome::Completed(Box::new(result)),
        Err(ExecutionError::Cancelled) => JobOutcome::Cancelled,
        Err(ExecutionError::Numerical(message)) => JobOutcome::Failed {
            kind: FailureKind::Numerical,
            message,
        },
        Err(ExecutionError::InvalidModelState(message)) => JobOutcome::Failed {
            kind: FailureKind::Validation,
            message,
        },
        Err(ExecutionError::Internal(message)) => JobOutcome::Failed {
            kind: FailureKind::Internal,
            message,
        },
    }
}
