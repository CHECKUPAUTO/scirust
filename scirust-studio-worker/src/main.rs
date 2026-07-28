//! `scirust-studio-worker` — the out-of-process capability worker.
//!
//! Phase 2B of the SciRust Studio effort (see
//! `docs/studio/adr/0003-worker-process-and-ipc.md`). The worker speaks
//! `scirust-studio-ipc` over stdin/stdout and exists for three reasons a
//! same-process execution cannot provide:
//!
//! 1. **A crash is survivable.** A capability that aborts the process takes
//!    the worker down, not the application that spawned it.
//! 2. **Cancellation always works.** Cooperative cancellation covers the
//!    fixed-step solvers; for anything that cannot be interrupted between
//!    steps — the adaptive stiff path — the supervisor can kill the process,
//!    which is a guarantee no in-process design offers.
//! 3. **The desktop shell never blocks.** Long runs happen somewhere else
//!    entirely.
//!
//! ## Concurrency
//!
//! Exactly one job runs at a time; further `Run` requests queue in arrival
//! order. Adapters are CPU-bound, so running several at once on the same
//! cores would finish none of them sooner while making progress reporting
//! meaningless. A queued job can still be cancelled before it starts.
//!
//! ## Why one writer
//!
//! The stdin reader and every job thread funnel into a single channel, and
//! only the main loop writes to stdout. Message order is therefore
//! deterministic and no line can interleave with another.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod job;

use std::collections::HashMap;
use std::collections::VecDeque;
use std::io::{BufReader, Write};
use std::sync::mpsc::{Sender, channel};

use scirust_studio_ipc::{
    ClientMessage, DEFAULT_MAX_MESSAGE_BYTES, FailureKind, IpcError, PROTOCOL_VERSION, RequestId,
    WorkerMessage, read_message, write_message,
};
use scirust_studio_runtime::{ExecutionControl, RunEvent};

use job::{Incoming, JobOutcome, run_job};

fn main() -> std::process::ExitCode {
    match serve()
    {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) =>
        {
            // stderr is the worker's only channel for something that is not
            // a protocol message; a supervisor captures it for diagnosis.
            eprintln!("scirust-studio-worker: {e}");
            std::process::ExitCode::FAILURE
        },
    }
}

/// A queued but not-yet-started job.
struct QueuedJob {
    request_id: RequestId,
    scenario_toml: String,
}

fn serve() -> Result<(), IpcError> {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    let (tx, rx) = channel::<Incoming>();
    spawn_stdin_reader(tx.clone());

    let mut handshaken = false;
    let mut running: Option<(RequestId, ExecutionControl)> = None;
    let mut queue: VecDeque<QueuedJob> = VecDeque::new();
    let mut cancelled_before_start: HashMap<RequestId, ()> = HashMap::new();
    let mut shutting_down = false;

    while let Ok(incoming) = rx.recv()
    {
        match incoming
        {
            Incoming::ClientEof =>
            {
                // The client vanished. Stop whatever is running rather than
                // computing a result nobody will ever read.
                if let Some((_, control)) = &running
                {
                    control.cancel();
                }
                break;
            },
            Incoming::ClientProtocolError(message) =>
            {
                write_message(&mut out, &WorkerMessage::ProtocolError { message })?;
                if let Some((_, control)) = &running
                {
                    control.cancel();
                }
                break;
            },
            Incoming::Client(message) =>
            {
                let message = *message;
                if !handshaken
                {
                    match handshake(&mut out, &message)?
                    {
                        true => handshaken = true,
                        false => break,
                    }
                    continue;
                }
                match message
                {
                    ClientMessage::Hello { .. } =>
                    {
                        write_message(
                            &mut out,
                            &WorkerMessage::ProtocolError {
                                message: "hello was already sent for this session".to_string(),
                            },
                        )?;
                        break;
                    },
                    ClientMessage::Catalog { request_id } =>
                    {
                        let catalog_json = scirust_studio_runtime::build_registry()
                            .to_json()
                            .map_err(|e| IpcError::Malformed(e.to_string()))?;
                        write_message(
                            &mut out,
                            &WorkerMessage::Catalog {
                                request_id,
                                catalog_json,
                            },
                        )?;
                    },
                    ClientMessage::Validate {
                        request_id,
                        scenario_toml,
                    } =>
                    {
                        let reply = validate_only(request_id, &scenario_toml);
                        write_message(&mut out, &reply)?;
                    },
                    ClientMessage::Run {
                        request_id,
                        scenario_toml,
                    } =>
                    {
                        if shutting_down
                        {
                            write_message(
                                &mut out,
                                &WorkerMessage::Failed {
                                    request_id,
                                    kind: FailureKind::Internal,
                                    message: "the worker is shutting down".to_string(),
                                },
                            )?;
                        }
                        else if running.is_some()
                        {
                            queue.push_back(QueuedJob {
                                request_id,
                                scenario_toml,
                            });
                        }
                        else
                        {
                            running = Some(start_job(request_id, scenario_toml, tx.clone()));
                        }
                    },
                    ClientMessage::Cancel { request_id } =>
                    {
                        if let Some((active, control)) = &running
                        {
                            if *active == request_id
                            {
                                control.cancel();
                            }
                        }
                        // A queued job can be cancelled before it ever
                        // starts; it must still get exactly one terminal
                        // message, like any other request.
                        if let Some(pos) = queue.iter().position(|j| j.request_id == request_id)
                        {
                            queue.remove(pos);
                            cancelled_before_start.insert(request_id, ());
                            write_message(&mut out, &WorkerMessage::Cancelled { request_id })?;
                        }
                        // Cancelling an unknown or already-finished request
                        // is deliberately silent — a client racing a
                        // completion must not be punished for it.
                    },
                    ClientMessage::Shutdown =>
                    {
                        shutting_down = true;
                        // Drain the queue first: those jobs will never run,
                        // and leaving a request with no terminal message
                        // would strand a client waiting on it.
                        while let Some(queued) = queue.pop_front()
                        {
                            write_message(
                                &mut out,
                                &WorkerMessage::Cancelled {
                                    request_id: queued.request_id,
                                },
                            )?;
                        }
                        match &running
                        {
                            Some((_, control)) => control.cancel(),
                            None =>
                            {
                                write_message(&mut out, &WorkerMessage::ShuttingDown)?;
                                break;
                            },
                        }
                    },
                }
            },
            Incoming::Event(request_id, event) =>
            {
                if let Some(message) = event_to_message(request_id, event)
                {
                    write_message(&mut out, &message)?;
                }
            },
            Incoming::Finished(request_id, outcome) =>
            {
                if cancelled_before_start.remove(&request_id).is_none()
                {
                    write_message(&mut out, &outcome_to_message(request_id, outcome))?;
                }
                running = None;

                if shutting_down
                {
                    write_message(&mut out, &WorkerMessage::ShuttingDown)?;
                    break;
                }
                if let Some(next) = queue.pop_front()
                {
                    running = Some(start_job(next.request_id, next.scenario_toml, tx.clone()));
                }
            },
        }
    }

    out.flush()?;
    Ok(())
}

/// Handle the opening message. Returns whether the session may continue.
fn handshake<W: Write>(out: &mut W, message: &ClientMessage) -> Result<bool, IpcError> {
    let ClientMessage::Hello { protocol_version } = message
    else
    {
        write_message(
            out,
            &WorkerMessage::ProtocolError {
                message: "the first message must be `hello`".to_string(),
            },
        )?;
        return Ok(false);
    };

    if let Err(e) = scirust_studio_ipc::check_protocol_version(*protocol_version)
    {
        write_message(
            out,
            &WorkerMessage::ProtocolError {
                message: e.to_string(),
            },
        )?;
        return Ok(false);
    }

    write_message(
        out,
        &WorkerMessage::Ready {
            protocol_version: PROTOCOL_VERSION,
            worker_version: env!("CARGO_PKG_VERSION").to_string(),
        },
    )?;
    Ok(true)
}

fn start_job(
    request_id: RequestId,
    scenario_toml: String,
    tx: Sender<Incoming>,
) -> (RequestId, ExecutionControl) {
    let control = ExecutionControl::new();
    let job_control = control.clone();
    std::thread::spawn(move || {
        run_job(request_id, &scenario_toml, job_control, tx);
    });
    (request_id, control)
}

/// Validate without running, replying with exactly one terminal message.
fn validate_only(request_id: RequestId, scenario_toml: &str) -> WorkerMessage {
    let scenario = match scirust_studio_schema::parse_toml(scenario_toml)
    {
        Ok(s) => s,
        Err(e) =>
        {
            return WorkerMessage::Failed {
                request_id,
                kind: FailureKind::Validation,
                message: e.to_string(),
            };
        },
    };

    let registry = scirust_studio_runtime::build_registry();
    let known: Vec<&str> = registry.iter().map(|d| d.id.0).collect();
    let errors = scirust_studio_schema::validate(&scenario, Some(&known));
    if !errors.is_empty()
    {
        return WorkerMessage::Failed {
            request_id,
            kind: FailureKind::Validation,
            message: errors
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join("\n"),
        };
    }

    match scirust_studio_runtime::find_adapter(&scenario.capability.id)
    {
        Some(adapter) => match adapter.validate(&scenario)
        {
            Ok(_) => WorkerMessage::Validated {
                request_id,
                capability_id: scenario.capability.id.clone(),
            },
            Err(report) => WorkerMessage::Failed {
                request_id,
                kind: FailureKind::Validation,
                message: report.to_string(),
            },
        },
        None => WorkerMessage::Failed {
            request_id,
            kind: FailureKind::Internal,
            message: format!(
                "capability `{}` is catalogued but has no adapter",
                scenario.capability.id
            ),
        },
    }
}

/// Map a runtime event onto the wire.
///
/// `Started`, `Completed`, `Cancelled` and `Failed` are deliberately
/// dropped: the outcome message the job thread sends when it finishes is
/// the single authoritative terminal message for a request, and echoing the
/// same fact twice would let a client see a completion before the result
/// that belongs to it.
fn event_to_message(request_id: RequestId, event: RunEvent) -> Option<WorkerMessage> {
    match event
    {
        RunEvent::Progress { fraction, t } => Some(WorkerMessage::Progress {
            request_id,
            fraction,
            t,
        }),
        RunEvent::Warning(warning) => Some(WorkerMessage::Warning {
            request_id,
            warning,
        }),
        RunEvent::Started | RunEvent::Completed | RunEvent::Cancelled | RunEvent::Failed(_) => None,
    }
}

fn outcome_to_message(request_id: RequestId, outcome: JobOutcome) -> WorkerMessage {
    match outcome
    {
        JobOutcome::Completed(result) => WorkerMessage::Completed { request_id, result },
        JobOutcome::Cancelled => WorkerMessage::Cancelled { request_id },
        JobOutcome::Failed { kind, message } => WorkerMessage::Failed {
            request_id,
            kind,
            message,
        },
    }
}

/// Read client messages on a dedicated thread so the event loop never blocks
/// on stdin while a job is running — without this, a `Cancel` could not be
/// noticed until the run it was cancelling had already finished.
fn spawn_stdin_reader(tx: Sender<Incoming>) {
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        let mut reader = BufReader::new(stdin.lock());
        loop
        {
            match read_message::<_, ClientMessage>(&mut reader, DEFAULT_MAX_MESSAGE_BYTES)
            {
                Ok(Some(message)) =>
                {
                    if tx.send(Incoming::Client(Box::new(message))).is_err()
                    {
                        return;
                    }
                },
                Ok(None) =>
                {
                    let _ = tx.send(Incoming::ClientEof);
                    return;
                },
                Err(e) =>
                {
                    let _ = tx.send(Incoming::ClientProtocolError(e.to_string()));
                    return;
                },
            }
        }
    });
}
