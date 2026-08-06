//! Supervising the out-of-process worker.
//!
//! Phase 2B deliberately stopped at "a well-behaved worker process with a
//! tested lifecycle" and left *when to restart one* to the application,
//! because that decision belongs with whoever owns the window. This is that
//! decision, implemented.
//!
//! The rules, and why:
//!
//! - **One worker, one active job.** Adapters are CPU-bound; a second
//!   concurrent run finishes neither sooner. The worker's own FIFO queue
//!   still exists, but the application never uses it.
//! - **A dead worker is never a silent rerun.** If the process exits while a
//!   job is running, that job becomes [`crate::JobState::Interrupted`] and
//!   stays that way. Re-running someone's scientific calculation without
//!   being asked would produce a second result they did not request and
//!   cannot distinguish from the first.
//! - **Restart is for the *next* job.** The supervisor will start a fresh
//!   worker on request, so the application stays usable, but never
//!   resurrects the interrupted work.
//! - **stderr is bounded.** A worker looping on a warning must not be able
//!   to exhaust the application's memory through its diagnostic channel.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex};

use scirust_studio_ipc::{
    ClientMessage, DEFAULT_MAX_MESSAGE_BYTES, PROTOCOL_VERSION, RequestId, WorkerMessage,
    read_message, write_message,
};

use crate::error::AppServiceError;

/// How the worker executable is found and launched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerLaunchConfig {
    /// Candidate paths, tried in order. The first one that is a regular file
    /// wins.
    ///
    /// A list rather than a single path because a development build, an
    /// installed bundle and a test harness each keep the worker somewhere
    /// different, and the alternative — searching `PATH` — would let an
    /// unrelated executable of the same name be launched.
    pub candidates: Vec<PathBuf>,
    /// How many bytes of the worker's stderr to retain for diagnostics.
    pub stderr_capacity: usize,
}

impl WorkerLaunchConfig {
    /// The file name the worker executable has on this platform.
    pub fn executable_name() -> &'static str {
        if cfg!(windows)
        {
            "scirust-studio-worker.exe"
        }
        else
        {
            "scirust-studio-worker"
        }
    }

    /// Look for the worker beside the currently-running executable, which is
    /// where a bundled application keeps its sidecar.
    pub fn beside_current_executable() -> Self {
        let mut candidates = Vec::new();
        if let Ok(exe) = std::env::current_exe()
        {
            if let Some(dir) = exe.parent()
            {
                candidates.push(dir.join(Self::executable_name()));
            }
        }
        WorkerLaunchConfig {
            candidates,
            stderr_capacity: 64 * 1024,
        }
    }

    /// An explicit path, for a test harness or a development run.
    pub fn at(path: impl Into<PathBuf>) -> Self {
        WorkerLaunchConfig {
            candidates: vec![path.into()],
            stderr_capacity: 64 * 1024,
        }
    }

    /// The first candidate that exists as a regular file.
    pub fn resolve(&self) -> Result<PathBuf, AppServiceError> {
        self.candidates
            .iter()
            .find(|p| p.is_file())
            .cloned()
            .ok_or_else(|| AppServiceError::WorkerNotFound {
                searched: self.candidates.clone(),
            })
    }
}

/// A bounded record of what the worker wrote to stderr.
#[derive(Debug, Default)]
pub struct StderrLog {
    lines: VecDeque<String>,
    bytes: usize,
    capacity: usize,
    dropped: u64,
}

impl StderrLog {
    fn new(capacity: usize) -> Self {
        StderrLog {
            capacity,
            ..Default::default()
        }
    }

    fn push(&mut self, line: String) {
        self.bytes += line.len();
        self.lines.push_back(line);
        while self.bytes > self.capacity && self.lines.len() > 1
        {
            if let Some(old) = self.lines.pop_front()
            {
                self.bytes -= old.len();
                self.dropped += 1;
            }
        }
    }

    /// Everything retained, oldest first.
    pub fn lines(&self) -> Vec<String> {
        self.lines.iter().cloned().collect()
    }

    /// How many lines were discarded to stay within the limit.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }
}

/// Why the worker process is gone.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerExitClass {
    /// It was asked to shut down and did.
    Requested,
    /// It exited on its own while the application still expected it.
    Unexpected {
        /// The exit status, rendered.
        status: String,
    },
    /// The supervisor terminated it — the only way to stop a computation
    /// that cannot be interrupted cooperatively.
    Terminated,
}

/// A running worker process and the channel to it.
#[derive(Debug)]
pub struct WorkerProcess {
    child: Child,
    stdin: Option<ChildStdin>,
    /// The worker's crate version, from the handshake.
    pub worker_version: String,
    /// Bounded stderr capture, shared with the reader thread.
    pub stderr: Arc<Mutex<StderrLog>>,
    path: PathBuf,
}

impl WorkerProcess {
    /// The executable this process was launched from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Send a message to the worker.
    pub fn send(&mut self, message: &ClientMessage) -> Result<(), AppServiceError> {
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| AppServiceError::WorkerUnavailable {
                reason: "the channel to the worker is closed".to_string(),
            })?;
        write_message(stdin, message).map_err(|e| AppServiceError::WorkerChannel {
            reason: e.to_string(),
        })
    }

    /// Close the worker's input, which it treats as the client going away.
    pub fn close_input(&mut self) {
        self.stdin.take();
    }

    /// Terminate the process immediately.
    ///
    /// The escape hatch for a computation that cannot be stopped
    /// cooperatively — the adaptive stiff solver offers no per-step callback
    /// to interrupt, so for that capability this is what "cancel" means.
    pub fn terminate(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    /// The operating-system process id of the running worker.
    ///
    /// Exposed for diagnostics — "the engine is running as PID 1234" is the
    /// kind of thing a support panel should be able to show — and so a test
    /// can target exactly this process rather than matching on a command
    /// line, which would also match the harness that launched it.
    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Whether the process has exited, without blocking.
    pub fn has_exited(&mut self) -> Option<std::process::ExitStatus> {
        self.child.try_wait().ok().flatten()
    }
}

/// Spawn the worker and complete the protocol handshake.
///
/// Returns the live process plus a reader over its stdout, which the caller
/// drives on its own thread. Handshaking here rather than later means a
/// version-mismatched worker is discovered at bootstrap, when the
/// application can still present a coherent "reinstall" message, instead of
/// halfway through someone's first run.
pub fn spawn_and_handshake(
    config: &WorkerLaunchConfig,
) -> Result<(WorkerProcess, BufReader<std::process::ChildStdout>), AppServiceError> {
    let path = config.resolve()?;

    let mut child = Command::new(&path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| AppServiceError::WorkerLaunchFailed {
            path: path.clone(),
            reason: e.to_string(),
        })?;

    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| AppServiceError::WorkerLaunchFailed {
            path: path.clone(),
            reason: "the worker's standard input was not available".to_string(),
        })?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| AppServiceError::WorkerLaunchFailed {
            path: path.clone(),
            reason: "the worker's standard output was not available".to_string(),
        })?;
    let stderr = child.stderr.take();

    let log = Arc::new(Mutex::new(StderrLog::new(config.stderr_capacity)));
    if let Some(stderr) = stderr
    {
        let log = Arc::clone(&log);
        std::thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines().map_while(Result::ok)
            {
                if let Ok(mut log) = log.lock()
                {
                    log.push(line);
                }
            }
        });
    }

    let mut reader = BufReader::new(stdout);

    write_message(
        &mut stdin,
        &ClientMessage::Hello {
            protocol_version: PROTOCOL_VERSION,
        },
    )
    .map_err(|e| AppServiceError::WorkerChannel {
        reason: e.to_string(),
    })?;

    let reply =
        read_message::<_, WorkerMessage>(&mut reader, DEFAULT_MAX_MESSAGE_BYTES).map_err(|e| {
            AppServiceError::WorkerChannel {
                reason: e.to_string(),
            }
        })?;

    let worker_version = match reply
    {
        Some(WorkerMessage::Ready {
            protocol_version,
            worker_version,
        }) =>
        {
            if protocol_version != PROTOCOL_VERSION
            {
                let _ = child.kill();
                let _ = child.wait();
                return Err(AppServiceError::WorkerVersionMismatch {
                    ours: PROTOCOL_VERSION,
                    theirs: Some(protocol_version),
                    detail: "the worker accepted the handshake but announced another version"
                        .to_string(),
                });
            }
            worker_version
        },
        Some(WorkerMessage::ProtocolError { message }) =>
        {
            let _ = child.kill();
            let _ = child.wait();
            return Err(AppServiceError::WorkerVersionMismatch {
                ours: PROTOCOL_VERSION,
                theirs: None,
                detail: message,
            });
        },
        Some(other) =>
        {
            let _ = child.kill();
            let _ = child.wait();
            return Err(AppServiceError::WorkerChannel {
                reason: format!("expected a handshake reply, got {other:?}"),
            });
        },
        None =>
        {
            let _ = child.wait();
            let detail = log.lock().map(|l| l.lines().join("\n")).unwrap_or_default();
            return Err(AppServiceError::WorkerChannel {
                reason: if detail.is_empty()
                {
                    "the worker closed its output without replying".to_string()
                }
                else
                {
                    format!("the worker exited during the handshake: {detail}")
                },
            });
        },
    };

    Ok((
        WorkerProcess {
            child,
            stdin: Some(stdin),
            worker_version,
            stderr: log,
            path,
        },
        reader,
    ))
}

/// Read one worker message, or `None` at end of stream.
pub fn read_worker_message(
    reader: &mut BufReader<std::process::ChildStdout>,
) -> Result<Option<WorkerMessage>, AppServiceError> {
    read_message(reader, DEFAULT_MAX_MESSAGE_BYTES).map_err(|e| AppServiceError::WorkerChannel {
        reason: e.to_string(),
    })
}

/// Allocates the request ids the application sends.
#[derive(Debug, Default)]
pub struct RequestIds {
    next: u64,
}

impl RequestIds {
    /// The next unused id.
    pub fn next(&mut self) -> RequestId {
        self.next += 1;
        RequestId(self.next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_executable_name_matches_the_platform() {
        let name = WorkerLaunchConfig::executable_name();
        if cfg!(windows)
        {
            assert!(name.ends_with(".exe"), "{name}");
        }
        else
        {
            assert!(!name.ends_with(".exe"), "{name}");
        }
        assert!(name.starts_with("scirust-studio-worker"));
    }

    #[test]
    fn resolving_reports_every_place_it_looked() {
        let config = WorkerLaunchConfig {
            candidates: vec![PathBuf::from("/no/such/a"), PathBuf::from("/no/such/b")],
            stderr_capacity: 1024,
        };
        let err = config.resolve().unwrap_err();
        match err
        {
            AppServiceError::WorkerNotFound { searched } =>
            {
                assert_eq!(searched.len(), 2);
            },
            other => panic!("expected WorkerNotFound, got {other:?}"),
        }
    }

    #[test]
    fn resolving_picks_the_first_existing_candidate() {
        let temp = tempfile::tempdir().unwrap();
        let real = temp.path().join("worker");
        std::fs::write(&real, b"not really an executable").unwrap();

        let config = WorkerLaunchConfig {
            candidates: vec![PathBuf::from("/no/such/thing"), real.clone()],
            stderr_capacity: 1024,
        };
        assert_eq!(config.resolve().unwrap(), real);
    }

    /// A directory named like the worker must not be mistaken for it.
    #[test]
    fn resolving_ignores_a_directory() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path().join("scirust-studio-worker");
        std::fs::create_dir(&dir).unwrap();
        let config = WorkerLaunchConfig {
            candidates: vec![dir],
            stderr_capacity: 1024,
        };
        assert!(config.resolve().is_err());
    }

    #[test]
    fn the_stderr_log_stays_within_its_budget() {
        let mut log = StderrLog::new(100);
        for i in 0..500
        {
            log.push(format!("line {i} with some padding to take up room"));
        }
        let retained: usize = log.lines().iter().map(|l| l.len()).sum();
        assert!(retained <= 100 || log.lines().len() == 1, "{retained}");
        assert!(log.dropped() > 0, "it must record what it discarded");
    }

    #[test]
    fn the_stderr_log_keeps_the_most_recent_lines() {
        let mut log = StderrLog::new(64);
        log.push("oldest".to_string());
        for i in 0..20
        {
            log.push(format!("newer {i}"));
        }
        let lines = log.lines();
        assert!(!lines.contains(&"oldest".to_string()));
        assert_eq!(lines.last().unwrap(), "newer 19");
    }

    #[test]
    fn request_ids_increase_and_never_repeat() {
        let mut ids = RequestIds::default();
        let a = ids.next();
        let b = ids.next();
        let c = ids.next();
        assert!(a < b && b < c);
        assert_eq!(a, RequestId(1));
    }

    #[test]
    fn exit_classes_round_trip_through_json() {
        for class in [
            WorkerExitClass::Requested,
            WorkerExitClass::Unexpected {
                status: "signal 9".into(),
            },
            WorkerExitClass::Terminated,
        ]
        {
            let json = serde_json::to_string(&class).unwrap();
            let back: WorkerExitClass = serde_json::from_str(&json).unwrap();
            assert_eq!(back, class);
        }
    }

    #[test]
    fn launching_a_missing_worker_fails_without_panicking() {
        let config = WorkerLaunchConfig::at("/definitely/not/here/worker");
        assert!(matches!(
            spawn_and_handshake(&config),
            Err(AppServiceError::WorkerNotFound { .. })
        ));
    }

    /// A file that exists but is not a runnable program must produce a
    /// launch failure naming the path, not a panic.
    #[test]
    fn launching_a_non_executable_file_reports_a_launch_failure() {
        let temp = tempfile::tempdir().unwrap();
        let fake = temp.path().join("scirust-studio-worker");
        std::fs::write(&fake, b"#not a program").unwrap();

        match spawn_and_handshake(&WorkerLaunchConfig::at(&fake))
        {
            Err(AppServiceError::WorkerLaunchFailed { path, .. }) => assert_eq!(path, fake),
            // A platform may successfully spawn it and have it die at once,
            // which surfaces as a channel failure instead. Both are correct
            // and neither may panic.
            Err(AppServiceError::WorkerChannel { .. }) =>
            {},
            other => panic!("expected a launch or channel failure, got {other:?}"),
        }
    }
}
