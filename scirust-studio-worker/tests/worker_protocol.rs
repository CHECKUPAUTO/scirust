//! End-to-end tests against the **real** worker binary.
//!
//! Every test here spawns the compiled `scirust-studio-worker` executable as
//! a genuine child process and talks to it over real pipes. Nothing is
//! stubbed and no worker logic is re-implemented in the test: if the
//! handshake, the framing, the job lifecycle, or the cancellation path were
//! broken, these would fail. That is the point — an in-process test of the
//! same functions would prove the protocol works between two halves of one
//! program, which is not the thing the worker exists to do.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use scirust_studio_ipc::{ClientMessage, FailureKind, PROTOCOL_VERSION, RequestId, WorkerMessage};

const SIR_TUTORIAL: &str = include_str!("../../docs/studio/tutorials/sir_epidemic.scirust.toml");

/// A spawned worker plus its pipes.
struct Worker {
    child: Child,
    /// An `Option` so a test can close the client's side of the pipe
    /// without moving out of a type that implements `Drop`.
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
}

impl Worker {
    /// Spawn the real binary Cargo just built.
    fn spawn() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_scirust-studio-worker"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("the worker binary must be built before this test runs");
        let stdin = child.stdin.take().expect("piped");
        let stdout = BufReader::new(child.stdout.take().expect("piped"));
        Worker {
            child,
            stdin: Some(stdin),
            stdout,
        }
    }

    fn send(&mut self, message: &ClientMessage) {
        let line = serde_json::to_string(message).expect("encodes");
        self.send_raw(&line);
    }

    /// Send a raw line, bypassing the message types — for testing what the
    /// worker does with input it should reject.
    fn send_raw(&mut self, line: &str) {
        let stdin = self.stdin.as_mut().expect("the client pipe is still open");
        writeln!(stdin, "{line}").expect("worker accepts input");
        stdin.flush().expect("flush");
    }

    /// Read the next message, failing the test if the worker closed instead.
    fn recv(&mut self) -> WorkerMessage {
        self.try_recv()
            .expect("the worker closed its output before replying")
    }

    fn try_recv(&mut self) -> Option<WorkerMessage> {
        let mut line = String::new();
        let read = self.stdout.read_line(&mut line).expect("readable");
        if read == 0
        {
            return None;
        }
        Some(
            serde_json::from_str(line.trim_end())
                .unwrap_or_else(|e| panic!("worker sent an undecodable line {line:?}: {e}")),
        )
    }

    /// Complete the opening handshake.
    fn handshake(&mut self) {
        self.send(&ClientMessage::Hello {
            protocol_version: PROTOCOL_VERSION,
        });
        match self.recv()
        {
            WorkerMessage::Ready {
                protocol_version, ..
            } => assert_eq!(protocol_version, PROTOCOL_VERSION),
            other => panic!("expected Ready, got {other:?}"),
        }
    }

    /// Read messages until one satisfies `predicate`, returning it.
    ///
    /// A failure or protocol error that the caller was *not* waiting for
    /// aborts the test immediately. Without this the caller would keep
    /// reading, the worker would have nothing left to say, and a genuine
    /// bug would show up as a hung test rather than a failing one.
    fn recv_until(&mut self, mut predicate: impl FnMut(&WorkerMessage) -> bool) -> WorkerMessage {
        loop
        {
            let message = self.recv();
            if predicate(&message)
            {
                return message;
            }
            match &message
            {
                WorkerMessage::Failed { kind, message, .. } =>
                {
                    panic!("worker reported an unexpected {kind:?} failure: {message}")
                },
                WorkerMessage::ProtocolError { message } =>
                {
                    panic!("worker reported a protocol error: {message}")
                },
                _ =>
                {},
            }
        }
    }

    fn wait(mut self) -> std::process::ExitStatus {
        // Closing stdin is the client going away, which the worker must
        // treat as a reason to stop.
        self.stdin.take();
        self.child.wait().expect("the worker exits")
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        // Never leave a worker running if a test panicked part-way through.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A scenario long enough that it cannot finish between the test seeing its
/// first progress event and sending a cancellation, but still inside
/// `scirust-sim`'s 10 000 000-step budget — past that the engine rejects the
/// run outright and there would be no progress to cancel.
///
/// 4 000 s at a 0.001 s step is 4 000 000 steps; the first progress event
/// fires at 5% of the span, leaving 95% of the work for the cancellation to
/// interrupt. The tests that use it all cancel early, so none of them
/// actually integrates the full span.
fn long_running_scenario() -> String {
    let scenario = SIR_TUTORIAL
        .replace(
            r#"end = { value = 60.0, unit = "s" }"#,
            r#"end = { value = 4000.0, unit = "s" }"#,
        )
        .replace(
            r#"step = { value = 0.05, unit = "s" }"#,
            r#"step = { value = 0.001, unit = "s" }"#,
        );
    assert!(
        scenario.contains("4000.0") && scenario.contains("0.001"),
        "the tutorial's solver block changed shape; this substitution no longer applies"
    );
    scenario
}

#[test]
fn the_worker_completes_the_handshake() {
    let mut worker = Worker::spawn();
    worker.handshake();
    worker.send(&ClientMessage::Shutdown);
    assert!(matches!(worker.recv(), WorkerMessage::ShuttingDown));
    assert!(worker.wait().success());
}

#[test]
fn a_request_before_hello_is_a_protocol_error() {
    let mut worker = Worker::spawn();
    worker.send(&ClientMessage::Catalog {
        request_id: RequestId(1),
    });
    match worker.recv()
    {
        WorkerMessage::ProtocolError { message } =>
        {
            assert!(message.contains("hello"), "{message}");
        },
        other => panic!("expected ProtocolError, got {other:?}"),
    }
}

#[test]
fn a_mismatched_protocol_version_is_refused() {
    let mut worker = Worker::spawn();
    worker.send(&ClientMessage::Hello {
        protocol_version: PROTOCOL_VERSION + 99,
    });
    match worker.recv()
    {
        WorkerMessage::ProtocolError { message } =>
        {
            assert!(message.contains("protocol version"), "{message}");
        },
        other => panic!("expected ProtocolError, got {other:?}"),
    }
}

#[test]
fn a_malformed_line_is_a_protocol_error_not_a_crash() {
    let mut worker = Worker::spawn();
    worker.handshake();
    worker.send_raw("{ this is not a protocol message }");
    match worker.recv()
    {
        WorkerMessage::ProtocolError { .. } =>
        {},
        other => panic!("expected ProtocolError, got {other:?}"),
    }
    // It exits cleanly rather than aborting.
    assert!(worker.wait().success());
}

#[test]
fn the_catalog_lists_every_registered_capability() {
    let mut worker = Worker::spawn();
    worker.handshake();
    worker.send(&ClientMessage::Catalog {
        request_id: RequestId(1),
    });

    let catalog_json = match worker.recv()
    {
        WorkerMessage::Catalog {
            request_id,
            catalog_json,
        } =>
        {
            assert_eq!(request_id, RequestId(1), "responses must be correlated");
            catalog_json
        },
        other => panic!("expected Catalog, got {other:?}"),
    };

    let parsed: serde_json::Value = serde_json::from_str(&catalog_json).expect("valid JSON");
    let entries = parsed.as_array().expect("the catalogue is an array");
    // The worker must report exactly what this build actually registers,
    // not a number written down separately and left to drift.
    assert_eq!(
        entries.len(),
        scirust_studio_runtime::build_registry().len()
    );
    let ids: Vec<&str> = entries
        .iter()
        .map(|e| e["id"].as_str().expect("each entry has an id"))
        .collect();
    assert!(ids.contains(&"sim.epidemiology.sir"), "{ids:?}");
}

#[test]
fn a_valid_scenario_validates() {
    let mut worker = Worker::spawn();
    worker.handshake();
    worker.send(&ClientMessage::Validate {
        request_id: RequestId(7),
        scenario_toml: SIR_TUTORIAL.to_string(),
    });
    match worker.recv()
    {
        WorkerMessage::Validated {
            request_id,
            capability_id,
        } =>
        {
            assert_eq!(request_id, RequestId(7));
            assert_eq!(capability_id, "sim.epidemiology.sir");
        },
        other => panic!("expected Validated, got {other:?}"),
    }
}

#[test]
fn an_invalid_scenario_fails_validation_without_running() {
    let mut worker = Worker::spawn();
    worker.handshake();
    worker.send(&ClientMessage::Validate {
        request_id: RequestId(8),
        scenario_toml: SIR_TUTORIAL
            .replace(r#"beta = { value = 0.6, unit = "1/s" }"#, "")
            .to_string(),
    });
    match worker.recv()
    {
        WorkerMessage::Failed {
            request_id,
            kind,
            message,
        } =>
        {
            assert_eq!(request_id, RequestId(8));
            assert_eq!(kind, FailureKind::Validation);
            assert!(message.contains("beta"), "{message}");
        },
        other => panic!("expected Failed, got {other:?}"),
    }
}

#[test]
fn unparseable_toml_fails_validation() {
    let mut worker = Worker::spawn();
    worker.handshake();
    worker.send(&ClientMessage::Validate {
        request_id: RequestId(9),
        scenario_toml: "this is not toml = = =".to_string(),
    });
    match worker.recv()
    {
        WorkerMessage::Failed { kind, .. } => assert_eq!(kind, FailureKind::Validation),
        other => panic!("expected Failed, got {other:?}"),
    }
}

/// The load-bearing test: a run through the worker process must produce the
/// same numbers as the same run in-process. If the worker were a second,
/// divergent execution path this would catch it.
#[test]
fn a_run_through_the_worker_matches_an_in_process_run() {
    let mut worker = Worker::spawn();
    worker.handshake();
    worker.send(&ClientMessage::Run {
        request_id: RequestId(1),
        scenario_toml: SIR_TUTORIAL.to_string(),
    });

    let mut progress_fractions = Vec::new();
    let completed = loop
    {
        match worker.recv()
        {
            WorkerMessage::Progress {
                request_id,
                fraction,
                ..
            } =>
            {
                assert_eq!(request_id, RequestId(1));
                progress_fractions.push(fraction);
            },
            WorkerMessage::Completed { request_id, result } =>
            {
                assert_eq!(request_id, RequestId(1));
                break *result;
            },
            other => panic!("unexpected {other:?}"),
        }
    };

    assert!(
        !progress_fractions.is_empty(),
        "the run should have reported progress"
    );
    for pair in progress_fractions.windows(2)
    {
        assert!(pair[1] > pair[0], "progress must increase");
    }

    // Now the same scenario, in this process, through the same adapter.
    let scenario = scirust_studio_schema::parse_toml(SIR_TUTORIAL).unwrap();
    let adapter = scirust_studio_runtime::find_adapter(&scenario.capability.id).unwrap();
    let validated = adapter.validate(&scenario).unwrap();
    let in_process = adapter
        .execute(
            &validated,
            &scirust_studio_runtime::ExecutionControl::new(),
            &mut scirust_studio_runtime::NullEventSink,
        )
        .unwrap();

    // Everything except wall-clock provenance must match exactly.
    assert_eq!(completed.capability_id, in_process.capability_id);
    assert_eq!(completed.summary, in_process.summary);
    assert_eq!(completed.series, in_process.series);
    assert_eq!(completed.metrics, in_process.metrics);
    assert_eq!(completed.verifications, in_process.verifications);
}

#[test]
fn a_run_that_is_cancelled_mid_flight_reports_cancelled() {
    let mut worker = Worker::spawn();
    worker.handshake();
    worker.send(&ClientMessage::Run {
        request_id: RequestId(1),
        scenario_toml: long_running_scenario(),
    });

    // Waiting for a progress event proves the run had genuinely started
    // before the cancellation was sent — this is a mid-run cancellation,
    // not a pre-execution one.
    let first = worker.recv_until(|m| matches!(m, WorkerMessage::Progress { .. }));
    assert!(matches!(first, WorkerMessage::Progress { .. }));

    worker.send(&ClientMessage::Cancel {
        request_id: RequestId(1),
    });

    let terminal = worker.recv_until(|m| {
        matches!(
            m,
            WorkerMessage::Cancelled { .. }
                | WorkerMessage::Completed { .. }
                | WorkerMessage::Failed { .. }
        )
    });
    match terminal
    {
        WorkerMessage::Cancelled { request_id } => assert_eq!(request_id, RequestId(1)),
        other => panic!("expected Cancelled, got {other:?}"),
    }
}

/// After a cancellation the worker must still be usable — cancelling a job
/// is not a session-ending event.
#[test]
fn the_worker_still_serves_requests_after_a_cancellation() {
    let mut worker = Worker::spawn();
    worker.handshake();
    worker.send(&ClientMessage::Run {
        request_id: RequestId(1),
        scenario_toml: long_running_scenario(),
    });
    worker.recv_until(|m| matches!(m, WorkerMessage::Progress { .. }));
    worker.send(&ClientMessage::Cancel {
        request_id: RequestId(1),
    });
    worker.recv_until(|m| matches!(m, WorkerMessage::Cancelled { .. }));

    worker.send(&ClientMessage::Catalog {
        request_id: RequestId(2),
    });
    match worker.recv()
    {
        WorkerMessage::Catalog { request_id, .. } => assert_eq!(request_id, RequestId(2)),
        other => panic!("expected Catalog, got {other:?}"),
    }
}

#[test]
fn cancelling_an_unknown_request_is_silently_ignored() {
    let mut worker = Worker::spawn();
    worker.handshake();
    worker.send(&ClientMessage::Cancel {
        request_id: RequestId(999),
    });
    // No reply is expected; the next request must still be answered.
    worker.send(&ClientMessage::Catalog {
        request_id: RequestId(1),
    });
    match worker.recv()
    {
        WorkerMessage::Catalog { request_id, .. } => assert_eq!(request_id, RequestId(1)),
        other => panic!("expected Catalog, got {other:?}"),
    }
}

/// Two runs submitted back to back must both settle, each with exactly one
/// terminal message, correlated by its own request id.
#[test]
fn a_second_run_is_queued_and_runs_after_the_first() {
    let mut worker = Worker::spawn();
    worker.handshake();
    worker.send(&ClientMessage::Run {
        request_id: RequestId(1),
        scenario_toml: SIR_TUTORIAL.to_string(),
    });
    worker.send(&ClientMessage::Run {
        request_id: RequestId(2),
        scenario_toml: SIR_TUTORIAL.to_string(),
    });

    let mut terminal_order = Vec::new();
    while terminal_order.len() < 2
    {
        match worker.recv()
        {
            WorkerMessage::Completed { request_id, .. } => terminal_order.push(request_id),
            WorkerMessage::Cancelled { request_id } | WorkerMessage::Failed { request_id, .. } =>
            {
                panic!("run {request_id} should have completed")
            },
            WorkerMessage::Progress { .. } | WorkerMessage::Warning { .. } =>
            {},
            other => panic!("unexpected {other:?}"),
        }
    }
    assert_eq!(
        terminal_order,
        vec![RequestId(1), RequestId(2)],
        "queued jobs must run in arrival order"
    );
}

#[test]
fn a_queued_run_can_be_cancelled_before_it_starts() {
    let mut worker = Worker::spawn();
    worker.handshake();
    worker.send(&ClientMessage::Run {
        request_id: RequestId(1),
        scenario_toml: long_running_scenario(),
    });
    worker.send(&ClientMessage::Run {
        request_id: RequestId(2),
        scenario_toml: SIR_TUTORIAL.to_string(),
    });
    // Make sure the first is genuinely running, so the second is queued.
    worker.recv_until(|m| matches!(m, WorkerMessage::Progress { .. }));

    worker.send(&ClientMessage::Cancel {
        request_id: RequestId(2),
    });
    let cancelled = worker.recv_until(|m| matches!(m, WorkerMessage::Cancelled { .. }));
    match cancelled
    {
        WorkerMessage::Cancelled { request_id } => assert_eq!(request_id, RequestId(2)),
        other => panic!("unexpected {other:?}"),
    }

    // The still-running first job is untouched by the second's cancellation.
    worker.send(&ClientMessage::Cancel {
        request_id: RequestId(1),
    });
    worker.recv_until(
        |m| matches!(m, WorkerMessage::Cancelled { request_id } if *request_id == RequestId(1)),
    );
}

#[test]
fn a_failing_run_reports_the_failure_kind() {
    let mut worker = Worker::spawn();
    worker.handshake();
    worker.send(&ClientMessage::Run {
        request_id: RequestId(1),
        scenario_toml: SIR_TUTORIAL.replace(
            r#"id = "sim.epidemiology.sir""#,
            r#"id = "sim.epidemiology.no_such_model""#,
        ),
    });
    match worker.recv()
    {
        WorkerMessage::Failed { kind, .. } => assert_eq!(kind, FailureKind::Validation),
        other => panic!("expected Failed, got {other:?}"),
    }
}

#[test]
fn shutdown_cancels_a_running_job_and_exits_cleanly() {
    let mut worker = Worker::spawn();
    worker.handshake();
    worker.send(&ClientMessage::Run {
        request_id: RequestId(1),
        scenario_toml: long_running_scenario(),
    });
    worker.recv_until(|m| matches!(m, WorkerMessage::Progress { .. }));

    worker.send(&ClientMessage::Shutdown);

    let mut saw_cancelled = false;
    let mut saw_shutting_down = false;
    while let Some(message) = worker.try_recv()
    {
        match message
        {
            WorkerMessage::Cancelled { request_id } =>
            {
                assert_eq!(request_id, RequestId(1));
                saw_cancelled = true;
            },
            WorkerMessage::ShuttingDown =>
            {
                saw_shutting_down = true;
                break;
            },
            WorkerMessage::Progress { .. } =>
            {},
            other => panic!("unexpected {other:?}"),
        }
    }
    assert!(
        saw_cancelled,
        "the in-flight job must be told it was cancelled"
    );
    assert!(saw_shutting_down);
    assert!(worker.wait().success());
}

/// Closing the client's side must stop the worker rather than leaving it
/// computing a result nobody will read.
#[test]
fn closing_the_client_pipe_stops_the_worker() {
    let mut worker = Worker::spawn();
    worker.handshake();
    worker.send(&ClientMessage::Run {
        request_id: RequestId(1),
        scenario_toml: long_running_scenario(),
    });
    worker.recv_until(|m| matches!(m, WorkerMessage::Progress { .. }));

    // `wait` drops stdin first, which is the client going away.
    assert!(worker.wait().success());
}
