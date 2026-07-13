//! Échappatoire générique : exécute le binaire CLI `scirust` (voir
//! `scirust-cli`) comme sous-processus, pour exposer d'un coup toutes ses
//! commandes (`linsolve`, `solve`, `diff`, `integrate`, `ode`, `certify`,
//! `conformal`, `evo`, `analyze`, ...) sans réimplémenter chacune comme un
//! outil MCP dédié. Préférer un outil dédié (ex. `linalg_eigen_symmetric`)
//! quand il existe : il renvoie du JSON structuré au lieu de texte à
//! reparser, et documente son schéma d'entrée précisément.
//!
//! Résolution du binaire, dans l'ordre : `SCIRUST_BIN` (chemin explicite),
//! puis `scirust` sur `PATH`, puis `cargo run -p scirust-cli --` en dernier
//! recours (lent, mais fonctionne depuis un checkout source sans install
//! préalable).

use crate::registry::McpTool;
use command_group::{CommandGroup, GroupChild};
use serde_json::json;
use std::io::Read;
use std::path::{Component, Path};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const MAX_ARGS: usize = 64;
const MAX_ARG_BYTES: usize = 4096;
const MAX_OUTPUT_BYTES: usize = 64 * 1024;
const CLI_TIMEOUT: Duration = Duration::from_secs(60);
const REAP_GRACE: Duration = Duration::from_secs(2);
const PROCESS_POLL_INTERVAL: Duration = Duration::from_millis(10);
type BoundedOutput = (Option<i32>, Vec<u8>, Vec<u8>);
const SECRET_ENV_VARS: &[&str] = &[
    "SCIRUST_DISCOVERY_KEY",
    "SCIRUST_EXCHANGE_SECRET",
    "SCIRUST_WALLET_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
];

fn is_on_path(bin: &str) -> bool {
    let executable = format!("{bin}{}", std::env::consts::EXE_SUFFIX);
    std::env::var_os("PATH")
        .map(|paths| std::env::split_paths(&paths).any(|dir| dir.join(&executable).is_file()))
        .unwrap_or(false)
}

fn development_root() -> std::path::PathBuf {
    std::env::var_os("SCIAGENT_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf()
        })
}

fn resolve_command(args: &[String]) -> Command {
    if let Ok(bin) = std::env::var("SCIRUST_BIN")
    {
        let mut cmd = Command::new(bin);
        cmd.args(args);
        cmd.current_dir(development_root());
        return cmd;
    }
    if is_on_path("scirust")
    {
        let mut cmd = Command::new("scirust");
        cmd.args(args);
        cmd.current_dir(development_root());
        return cmd;
    }
    let mut cmd = Command::new("cargo");
    cmd.args(["run", "--quiet", "-p", "scirust-cli", "--"]);
    cmd.args(args);
    cmd.current_dir(development_root());
    cmd
}

fn safe_argument(argument: &str) -> bool {
    argument.len() <= MAX_ARG_BYTES
        && !Path::new(argument).is_absolute()
        && !Path::new(argument)
            .components()
            .any(|component| component == Component::ParentDir)
}

struct PipeDrain {
    bytes: Arc<Mutex<Vec<u8>>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl PipeDrain {
    fn is_finished(&self) -> bool {
        self.thread
            .as_ref()
            .is_none_or(std::thread::JoinHandle::is_finished)
    }

    fn snapshot(&self) -> Vec<u8> {
        self.bytes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    fn finish(mut self) -> Vec<u8> {
        if let Some(thread) = self.thread.take()
        {
            let _ = thread.join();
        }
        self.snapshot()
    }
}

fn drain_pipe<R: Read + Send + 'static>(mut pipe: R) -> PipeDrain {
    let bytes = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&bytes);
    let thread = std::thread::spawn(move || {
        let mut chunk = [0u8; 8192];
        while let Ok(count) = pipe.read(&mut chunk)
        {
            if count == 0
            {
                break;
            }
            let mut kept = captured
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let remaining = MAX_OUTPUT_BYTES.saturating_sub(kept.len());
            kept.extend_from_slice(&chunk[..count.min(remaining)]);
        }
    });
    PipeDrain {
        bytes,
        thread: Some(thread),
    }
}

fn spawn_process_group(command: &mut Command) -> std::io::Result<GroupChild> {
    #[cfg(windows)]
    {
        // A Job Object contains every descendant and kill-on-close is a final
        // safeguard if error handling ever returns before an explicit kill.
        command.group().kill_on_drop(true).spawn()
    }
    #[cfg(not(windows))]
    {
        command.group_spawn()
    }
}

/// Keep ownership of a stubborn POSIX process group after the request has
/// returned. On Windows, dropping the Job Object is itself the reliable
/// kill-on-close fallback. Reader threads are retained with the reaper so a
/// descendant can never make the request thread wait indefinitely on a pipe.
fn defer_group_cleanup(child: GroupChild, stdout: PipeDrain, stderr: PipeDrain) {
    std::thread::spawn(move || {
        #[cfg(windows)]
        drop(child);

        #[cfg(not(windows))]
        {
            let mut child = child;
            loop
            {
                let _ = child.kill();
                if matches!(child.try_wait(), Ok(Some(_)))
                {
                    drop(child);
                    break;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }

        let _ = stdout.finish();
        let _ = stderr.finish();
    });
}

/// Terminate the entire process group and reap it for a bounded grace period.
/// If the OS keeps rejecting termination or reaping, cleanup continues on a
/// background owner instead of either blocking this request or abandoning the
/// descendants.
fn terminate_and_reap(
    mut child: GroupChild,
    stdout: PipeDrain,
    stderr: PipeDrain,
) -> (Option<ExitStatus>, Vec<u8>, Vec<u8>) {
    let deadline = Instant::now() + REAP_GRACE;
    let mut status = None;
    let mut next_kill = Instant::now();
    loop
    {
        let now = Instant::now();
        if now >= next_kill
        {
            let _ = child.kill();
            next_kill = now + Duration::from_millis(100);
        }
        if status.is_none()
        {
            if let Ok(current) = child.try_wait()
            {
                status = current;
            }
        }
        if status.is_some() && stdout.is_finished() && stderr.is_finished()
        {
            drop(child);
            return (status, stdout.finish(), stderr.finish());
        }
        if now >= deadline
        {
            let stdout_bytes = stdout.snapshot();
            let stderr_bytes = stderr.snapshot();
            defer_group_cleanup(child, stdout, stderr);
            return (status, stdout_bytes, stderr_bytes);
        }
        std::thread::sleep(PROCESS_POLL_INTERVAL);
    }
}

fn run_bounded(command: Command) -> Result<BoundedOutput, String> {
    run_bounded_with_timeout(command, CLI_TIMEOUT)
}

fn run_bounded_with_timeout(
    mut command: Command,
    timeout: Duration,
) -> Result<BoundedOutput, String> {
    for variable in SECRET_ENV_VARS
    {
        command.env_remove(variable);
    }
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = spawn_process_group(&mut command).map_err(|e| e.to_string())?;
    let stdout = drain_pipe(child.inner().stdout.take().expect("stdout was piped"));
    let stderr = drain_pipe(child.inner().stderr.take().expect("stderr was piped"));
    let deadline = Instant::now() + timeout;
    let mut exit_status = None;
    loop
    {
        if exit_status.is_none()
        {
            match child.try_wait()
            {
                Ok(status) => exit_status = status,
                Err(error) =>
                {
                    let _ = terminate_and_reap(child, stdout, stderr);
                    return Err(error.to_string());
                },
            }
        }
        // A process-group leader may exit before its descendants. Do not join
        // the capture threads until every inherited pipe handle has closed.
        if exit_status.is_some() && stdout.is_finished() && stderr.is_finished()
        {
            break;
        }
        if Instant::now() >= deadline
        {
            let _ = terminate_and_reap(child, stdout, stderr);
            return Err(format!(
                "scirust CLI timed out after {} seconds",
                timeout.as_secs()
            ));
        }
        std::thread::sleep(PROCESS_POLL_INTERVAL);
    }
    drop(child);
    let status = exit_status.expect("completed process group has an exit status");
    Ok((status.code(), stdout.finish(), stderr.finish()))
}

pub fn cli_tool() -> McpTool {
    McpTool {
        name: "scirust_cli".to_string(),
        description: "Run any `scirust` CLI subcommand (run `scirust_cli` with args=[\"help\"] \
            to list them) — linsolve, solve, diff, integrate, ode, certify, conformal, evo, \
            analyze, and more. Input: `args`, the argument list without the leading `scirust`. \
            Returns the captured exit code, stdout, and stderr."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "args": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "e.g. [\"linsolve\", \"2,1;1,3\", \"3,5\"]",
                }
            },
            "required": ["args"],
        }),
        handler: Box::new(|args| {
            let arg_list: Vec<String> = args
                .get("args")
                .and_then(|v| v.as_array())
                .ok_or("missing `args` array")?
                .iter()
                .map(|v| {
                    v.as_str()
                        .map(|s| s.to_string())
                        .ok_or_else(|| "`args` entries must be strings".to_string())
                })
                .collect::<Result<_, _>>()?;
            if arg_list.len() > MAX_ARGS
            {
                return Err(format!("refused more than {MAX_ARGS} CLI arguments"));
            }
            if let Some(argument) = arg_list.iter().find(|argument| !safe_argument(argument))
            {
                return Err(format!(
                    "refused absolute, parent-relative, or oversized CLI argument: `{argument}`"
                ));
            }
            let (exit_code, stdout, stderr) = run_bounded(resolve_command(&arg_list))
                .map_err(|e| format!("failed to run the scirust CLI: {e}"))?;
            Ok(json!({
                "exit_code": exit_code,
                "stdout": String::from_utf8_lossy(&stdout),
                "stderr": String::from_utf8_lossy(&stderr),
            }))
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    const PROCESS_TREE_TEST_ROLE: &str = "SCIRUST_MCP_PROCESS_TREE_TEST_ROLE";

    fn process_tree_helper_name() -> String {
        let module = module_path!();
        let module = module
            .split_once("::")
            .map(|(_, rest)| rest)
            .unwrap_or(module);
        format!("{module}::process_tree_helper")
    }

    // The leader must deliberately exit without waiting: this reproduces the
    // orphaned-descendant regression that the process group is meant to fix.
    #[allow(clippy::zombie_processes)]
    #[test]
    fn process_tree_helper() {
        match std::env::var(PROCESS_TREE_TEST_ROLE).as_deref()
        {
            Ok("leader") =>
            {
                Command::new(std::env::current_exe().expect("test executable"))
                    .args(["--exact", &process_tree_helper_name(), "--nocapture"])
                    .env(PROCESS_TREE_TEST_ROLE, "descendant")
                    .stdout(Stdio::inherit())
                    .stderr(Stdio::inherit())
                    .spawn()
                    .expect("spawn descendant");
                println!("descendant-spawned");
                std::io::stdout().flush().expect("flush helper output");
            },
            Ok("descendant") =>
            {
                println!("descendant-ready");
                std::io::stdout().flush().expect("flush helper output");
                std::thread::sleep(Duration::from_secs(30));
            },
            _ =>
            {},
        }
    }

    #[test]
    fn timeout_terminates_descendants_that_hold_capture_pipes() {
        let mut command = Command::new(std::env::current_exe().expect("test executable"));
        command
            .args(["--exact", &process_tree_helper_name(), "--nocapture"])
            .env(PROCESS_TREE_TEST_ROLE, "leader");

        let started = Instant::now();
        let error = run_bounded_with_timeout(command, Duration::from_secs(2))
            .expect_err("the process tree should time out");

        assert!(error.contains("timed out"), "unexpected error: {error}");
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "capture readers remained blocked after the timeout"
        );
    }

    #[test]
    fn rejects_missing_args() {
        let tool = cli_tool();
        assert!((tool.handler)(json!({})).is_err());
    }

    #[test]
    fn rejects_non_string_args() {
        let tool = cli_tool();
        assert!((tool.handler)(json!({ "args": [1, 2] })).is_err());
    }

    #[test]
    fn rejects_paths_that_can_escape_development_workspace() {
        let tool = cli_tool();
        assert!((tool.handler)(json!({ "args": ["analyze", "../secret"] })).is_err());
        let absolute = if cfg!(windows)
        {
            "C:\\Windows\\win.ini"
        }
        else
        {
            "/etc/passwd"
        };
        assert!((tool.handler)(json!({ "args": ["analyze", absolute] })).is_err());
    }
}
