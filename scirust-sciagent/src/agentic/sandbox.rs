mod landlock;

use super::enforcement::{ExecutionConstraints, GOVERNANCE_FAILURE_PREFIX, apply_to_command};
use super::sandbox_approval::SandboxPermission;
use super::tools::Tool;
use command_group::{CommandGroup, GroupChild};
use std::collections::HashMap;
use std::ffi::OsString;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const MAX_TOOL_OUTPUT_BYTES: usize = 64 * 1024;
const TOOL_TIMEOUT: Duration = Duration::from_secs(30);
const BWRAP_PROBE_TIMEOUT: Duration = Duration::from_secs(5);
const REAP_GRACE: Duration = Duration::from_secs(2);
const PROCESS_POLL_INTERVAL: Duration = Duration::from_millis(10);
const SANDBOX_UNAVAILABLE: &str = "SANDBOX_UNAVAILABLE";
const BWRAP_RUNNER_FAILURE_SIGNATURE: &str = "bwrap: ";
const INTERNAL_SANDBOX_OVERRIDE: &str = "__sciagent_sandbox_override";
#[cfg(test)]
const BWRAP_DENIAL_SIGNATURE: &str = "read-only file system";
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

/// File-effect policy vocabulary shared with the DeepSeek harness.
///
/// Network and process visibility deliberately stay outside this enum. The
/// sandbox seam governs only filesystem effects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SandboxMode {
    ReadOnly,
    WorkspaceWrite,
    DangerFullAccess,
}

impl SandboxMode {
    fn parse(value: Option<&str>) -> Result<Self, String> {
        match value
            .unwrap_or("workspace-write")
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "read-only" => Ok(Self::ReadOnly),
            "workspace-write" => Ok(Self::WorkspaceWrite),
            "danger-full-access" => Ok(Self::DangerFullAccess),
            other => Err(format!(
                "Invalid SCIAGENT_SANDBOX_MODE {other:?}; expected read-only, workspace-write, or danger-full-access"
            )),
        }
    }

    fn from_permission(permission: SandboxPermission) -> Self {
        match permission
        {
            SandboxPermission::ReadOnly => Self::ReadOnly,
            SandboxPermission::WorkspaceWrite => Self::WorkspaceWrite,
            SandboxPermission::DangerFullAccess => Self::DangerFullAccess,
        }
    }

    fn permission(self) -> SandboxPermission {
        match self
        {
            Self::ReadOnly => SandboxPermission::ReadOnly,
            Self::WorkspaceWrite => SandboxPermission::WorkspaceWrite,
            Self::DangerFullAccess => SandboxPermission::DangerFullAccess,
        }
    }

    fn is_confined(self) -> bool {
        !matches!(self, Self::DangerFullAccess)
    }

    fn label(self) -> &'static str {
        match self
        {
            Self::ReadOnly => "read-only",
            Self::WorkspaceWrite => "workspace-write",
            Self::DangerFullAccess => "danger-full-access",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SandboxEnforcement {
    Full,
    Partial,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SandboxBackend {
    Direct,
    Bubblewrap(PathBuf),
    Landlock(landlock::Support),
}

#[derive(Debug, Clone)]
struct SandboxConfig {
    mode: SandboxMode,
    bwrap: Option<PathBuf>,
}

impl SandboxConfig {
    fn from_env_with_override(requested: Option<SandboxPermission>) -> Result<Self, String> {
        let configured =
            SandboxMode::parse(std::env::var("SCIAGENT_SANDBOX_MODE").ok().as_deref())?;
        let mode = resolve_effective_mode(configured, requested)?;
        let bwrap = if cfg!(target_os = "linux")
        {
            std::env::var_os("SCIAGENT_BWRAP")
                .map(PathBuf::from)
                .or_else(find_bwrap_on_path)
        }
        else
        {
            None
        };
        Ok(Self { mode, bwrap })
    }

    fn backend(&self, root: &Path) -> Result<(SandboxBackend, Option<SandboxEnforcement>), String> {
        if !self.mode.is_confined()
        {
            return select_backend(self.mode, None, None, cfg!(target_os = "linux"));
        }
        if !cfg!(target_os = "linux")
        {
            return select_backend(self.mode, None, None, false);
        }

        let bwrap = self
            .bwrap
            .clone()
            .filter(|candidate| bubblewrap_usable(candidate, root));
        let landlock = if bwrap.is_none()
        {
            landlock::probe()
        }
        else
        {
            None
        };
        select_backend(self.mode, bwrap, landlock, true)
    }
}

fn sandbox_unavailable(mode: SandboxMode, detail: impl AsRef<str>) -> String {
    format!(
        "[{SANDBOX_UNAVAILABLE}] sandbox mode {:?} was requested but no usable Linux sandbox backend is available; refusing to run the command unconfined. {}",
        mode.label(),
        detail.as_ref()
    )
}

fn select_backend(
    mode: SandboxMode,
    bwrap: Option<PathBuf>,
    landlock: Option<landlock::Support>,
    linux: bool,
) -> Result<(SandboxBackend, Option<SandboxEnforcement>), String> {
    if !mode.is_confined()
    {
        return Ok((SandboxBackend::Direct, None));
    }
    if !linux
    {
        return Err(sandbox_unavailable(
            mode,
            "Confined process sandboxes currently require Linux.",
        ));
    }
    if let Some(bwrap) = bwrap
    {
        return Ok((
            SandboxBackend::Bubblewrap(bwrap),
            Some(SandboxEnforcement::Full),
        ));
    }
    if let Some(support) = landlock
    {
        return Ok((SandboxBackend::Landlock(support), Some(support.enforcement)));
    }
    Err(sandbox_unavailable(
        mode,
        "bubblewrap is absent or failed its functional probe, and Landlock is unsupported or disabled by this kernel.",
    ))
}

fn find_bwrap_on_path() -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|directory| directory.join("bwrap"))
        .find(|candidate| candidate.is_file())
}

fn bubblewrap_usable(bwrap: &Path, root: &Path) -> bool {
    let mut command =
        bubblewrap_command(bwrap, SandboxMode::ReadOnly, root, root, "true", &[], false);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let Ok(mut child) = spawn_process_group(&mut command)
    else
    {
        return false;
    };
    let deadline = Instant::now() + BWRAP_PROBE_TIMEOUT;
    loop
    {
        match child.try_wait()
        {
            Ok(Some(status)) =>
            {
                drop(child);
                return status.success();
            },
            Ok(None) =>
            {},
            Err(_) =>
            {
                let _ = child.kill();
                return false;
            },
        }
        if Instant::now() >= deadline
        {
            let _ = child.kill();
            let _ = child.inner().wait();
            return false;
        }
        std::thread::sleep(PROCESS_POLL_INTERVAL);
    }
}

pub(crate) fn configured_permission() -> Result<SandboxPermission, String> {
    SandboxMode::parse(std::env::var("SCIAGENT_SANDBOX_MODE").ok().as_deref())
        .map(SandboxMode::permission)
}

pub(crate) fn tool_supports_sandbox(name: &str) -> bool {
    matches!(name, "search" | "grep" | "build" | "test" | "status")
}

pub(crate) fn install_one_shot_override(
    params: &mut HashMap<String, String>,
    permission: SandboxPermission,
) {
    params.insert(
        INTERNAL_SANDBOX_OVERRIDE.to_string(),
        permission.label().to_string(),
    );
}

fn take_one_shot_override(
    params: &mut HashMap<String, String>,
) -> Result<Option<SandboxPermission>, String> {
    params
        .remove(INTERNAL_SANDBOX_OVERRIDE)
        .map(|value| SandboxPermission::parse(&value).map_err(|error| error.to_string()))
        .transpose()
}

/// Pull declared resource governance out of the call parameters. Malformed
/// governance payloads refuse the call — never silently ignored limits.
fn take_constraints(
    params: &mut HashMap<String, String>,
) -> Result<Option<ExecutionConstraints>, String> {
    let limits = params.remove(crate::agentic::tool_runtime::RESOURCE_LIMITS_METADATA);
    let egress = params.remove(crate::agentic::tool_runtime::EGRESS_POLICY_METADATA);
    ExecutionConstraints::from_metadata(limits.as_deref(), egress.as_deref())
}

fn resolve_effective_mode(
    configured: SandboxMode,
    requested: Option<SandboxPermission>,
) -> Result<SandboxMode, String> {
    let Some(requested) = requested
    else
    {
        return Ok(configured);
    };
    if !configured.permission().can_escalate_to(requested)
    {
        return Err(format!(
            "Refused non-widening sandbox override from {:?} to {:?}",
            configured.label(),
            requested.label()
        ));
    }
    Ok(SandboxMode::from_permission(requested))
}

pub(crate) fn install_process_sandbox(tools: &mut [Tool]) {
    for tool in tools
    {
        tool.execute = match tool.name
        {
            "search" => sandboxed_search,
            "grep" => sandboxed_grep,
            "build" => sandboxed_build,
            "test" => sandboxed_test,
            "status" => sandboxed_status,
            _ => continue,
        };
    }
}

fn canonical_workspace_root() -> Result<PathBuf, String> {
    std::fs::canonicalize(super::tools::workspace_root())
        .map_err(|error| format!("Cannot resolve the configured workspace root: {error}"))
}

fn resolve_workspace_path(requested: &str) -> Result<PathBuf, String> {
    let root = canonical_workspace_root()?;
    let requested = if requested.is_empty()
    {
        root.clone()
    }
    else
    {
        let path = Path::new(requested);
        if path.is_absolute()
        {
            path.to_path_buf()
        }
        else
        {
            root.join(path)
        }
    };
    let resolved = std::fs::canonicalize(&requested)
        .map_err(|error| format!("Cannot resolve `{}`: {error}", requested.display()))?;
    if !resolved.starts_with(&root)
    {
        return Err(format!(
            "Refused path outside workspace `{}`",
            root.display()
        ));
    }
    Ok(resolved)
}

fn valid_crate_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
}

fn sandboxed_search(mut params: HashMap<String, String>) -> String {
    let requested = match take_one_shot_override(&mut params)
    {
        Ok(requested) => requested,
        Err(error) => return error,
    };
    let constraints = match take_constraints(&mut params)
    {
        Ok(constraints) => constraints,
        Err(error) => return error,
    };
    search_workspace(&params, "10", requested, constraints.as_ref())
}

fn sandboxed_grep(mut params: HashMap<String, String>) -> String {
    let requested = match take_one_shot_override(&mut params)
    {
        Ok(requested) => requested,
        Err(error) => return error,
    };
    let constraints = match take_constraints(&mut params)
    {
        Ok(constraints) => constraints,
        Err(error) => return error,
    };
    search_workspace(&params, "15", requested, constraints.as_ref())
}

fn search_workspace(
    params: &HashMap<String, String>,
    max_count: &str,
    requested: Option<SandboxPermission>,
    constraints: Option<&ExecutionConstraints>,
) -> String {
    let pattern = params.get("pattern").map(String::as_str).unwrap_or("");
    if pattern.is_empty()
    {
        return "Missing pattern".to_string();
    }
    if pattern.len() > 1024
    {
        return "Refused search pattern longer than 1024 bytes".to_string();
    }
    let path = match resolve_workspace_path(params.get("path").map(String::as_str).unwrap_or(""))
    {
        Ok(path) => path,
        Err(error) => return error,
    };
    let root = match canonical_workspace_root()
    {
        Ok(root) => root,
        Err(error) => return error,
    };

    let rg_args = vec![
        OsString::from("-n"),
        OsString::from("--max-count"),
        OsString::from(max_count),
        OsString::from("--max-filesize"),
        OsString::from("1M"),
        OsString::from("--max-columns"),
        OsString::from("512"),
        OsString::from("--glob"),
        OsString::from("!target/**"),
        OsString::from("--"),
        OsString::from(pattern),
        path.clone().into_os_string(),
    ];
    match run_sandboxed("rg", &rg_args, &root, requested, constraints)
    {
        Ok(output) if output.timed_out => "Search timed out after 30 seconds".to_string(),
        Ok(output) if output.success => String::from_utf8_lossy(&output.stdout).into_owned(),
        _ =>
        {
            let grep_args = vec![
                OsString::from("-rn"),
                OsString::from("--max-count"),
                OsString::from(max_count),
                OsString::from("--exclude-dir=target"),
                OsString::from("--"),
                OsString::from(pattern),
                path.into_os_string(),
            ];
            match run_sandboxed("grep", &grep_args, &root, requested, constraints)
            {
                Ok(output) if output.timed_out => "Search timed out after 30 seconds".to_string(),
                Ok(output) if output.success =>
                {
                    String::from_utf8_lossy(&output.stdout).into_owned()
                },
                Ok(output) => format!("No matches: {}", String::from_utf8_lossy(&output.stderr)),
                Err(error) => format!("Failed to run search: {error}"),
            }
        },
    }
}

fn sandboxed_build(mut params: HashMap<String, String>) -> String {
    let requested = match take_one_shot_override(&mut params)
    {
        Ok(requested) => requested,
        Err(error) => return error,
    };
    let constraints = match take_constraints(&mut params)
    {
        Ok(constraints) => constraints,
        Err(error) => return error,
    };
    let crate_name = params.get("crate").map(String::as_str).unwrap_or("");
    if !valid_crate_name(crate_name)
    {
        return "Invalid crate name".to_string();
    }
    let root = match canonical_workspace_root()
    {
        Ok(root) => root,
        Err(error) => return error,
    };
    let args = [
        "check",
        "--locked",
        "-p",
        crate_name,
        "--message-format=short",
    ]
    .into_iter()
    .map(OsString::from)
    .collect::<Vec<_>>();
    match run_sandboxed("cargo", &args, &root, requested, constraints.as_ref())
    {
        Ok(output) if output.timed_out => "Build timed out after 30 seconds".to_string(),
        Ok(output) if output.success => format!("{crate_name} builds successfully"),
        Ok(output) => format!("Build errors:\n{}", String::from_utf8_lossy(&output.stderr)),
        Err(error) => format!("Failed to run cargo: {error}"),
    }
}

fn sandboxed_test(mut params: HashMap<String, String>) -> String {
    let requested = match take_one_shot_override(&mut params)
    {
        Ok(requested) => requested,
        Err(error) => return error,
    };
    let constraints = match take_constraints(&mut params)
    {
        Ok(constraints) => constraints,
        Err(error) => return error,
    };
    let crate_name = params.get("crate").map(String::as_str).unwrap_or("");
    if !valid_crate_name(crate_name)
    {
        return "Invalid crate name".to_string();
    }
    let mut args = vec![
        OsString::from("test"),
        OsString::from("--locked"),
        OsString::from("-p"),
        OsString::from(crate_name),
        OsString::from("--message-format=short"),
    ];
    if let Some(filter) = params.get("test")
    {
        if filter.len() > 256
        {
            return "Refused test filter longer than 256 bytes".to_string();
        }
        args.push(OsString::from("--"));
        args.push(OsString::from(filter));
    }
    let root = match canonical_workspace_root()
    {
        Ok(root) => root,
        Err(error) => return error,
    };
    match run_sandboxed("cargo", &args, &root, requested, constraints.as_ref())
    {
        Ok(output) if output.timed_out => "Tests timed out after 30 seconds".to_string(),
        Ok(output) if output.success =>
        {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let passed = stdout
                .lines()
                .find(|line| line.contains("test result"))
                .unwrap_or("unknown");
            format!("Tests passed: {passed}")
        },
        Ok(output) => format!(
            "Test failures:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ),
        Err(error) => format!("Failed to run tests: {error}"),
    }
}

fn sandboxed_status(mut params: HashMap<String, String>) -> String {
    let requested = match take_one_shot_override(&mut params)
    {
        Ok(requested) => requested,
        Err(error) => return error,
    };
    let constraints = match take_constraints(&mut params)
    {
        Ok(constraints) => constraints,
        Err(error) => return error,
    };
    let root = match canonical_workspace_root()
    {
        Ok(root) => root,
        Err(error) => return error,
    };
    let args = [OsString::from("status"), OsString::from("--short")];
    match run_sandboxed("git", &args, &root, requested, constraints.as_ref())
    {
        Ok(output) if output.timed_out => "Git status timed out after 30 seconds".to_string(),
        Ok(output) => String::from_utf8_lossy(&output.stdout).into_owned(),
        Err(error) => format!("Git error: {error}"),
    }
}

struct LimitedOutput {
    success: bool,
    timed_out: bool,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
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
            let remaining = MAX_TOOL_OUTPUT_BYTES.saturating_sub(kept.len());
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
        command.group().kill_on_drop(true).spawn()
    }
    #[cfg(not(windows))]
    {
        command.group_spawn()
    }
}

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

fn run_sandboxed(
    program: &str,
    args: &[OsString],
    cwd: &Path,
    requested: Option<SandboxPermission>,
    constraints: Option<&ExecutionConstraints>,
) -> Result<LimitedOutput, String> {
    run_sandboxed_with_config(
        program,
        args,
        cwd,
        SandboxConfig::from_env_with_override(requested)?,
        TOOL_TIMEOUT,
        constraints,
    )
}

fn run_sandboxed_with_config(
    program: &str,
    args: &[OsString],
    cwd: &Path,
    config: SandboxConfig,
    timeout: Duration,
    constraints: Option<&ExecutionConstraints>,
) -> Result<LimitedOutput, String> {
    let root = canonical_workspace_root()?;
    let (backend, _enforcement) = config.backend(&root)?;
    let confined_mode = config.mode.is_confined();
    let backend_label = match &backend
    {
        SandboxBackend::Direct => "direct",
        SandboxBackend::Bubblewrap(_) => "bubblewrap",
        SandboxBackend::Landlock(_) => "landlock",
    };
    // Declared wall-time governance narrows the runtime's own kill deadline;
    // it never extends past the global cap.
    let deadline_timeout = constraints
        .map(|constraints| constraints.effective_wall_time(timeout))
        .unwrap_or(timeout);
    let net_isolated = constraints.is_some_and(|constraints| {
        constraints.egress.enforce && constraints.egress.allow.is_empty()
    });
    let mut command = match &backend
    {
        SandboxBackend::Direct =>
        {
            let mut command = Command::new(program);
            command.args(args).current_dir(cwd);
            command
        },
        SandboxBackend::Bubblewrap(bwrap) =>
        {
            bubblewrap_command(bwrap, config.mode, &root, cwd, program, args, net_isolated)
        },
        SandboxBackend::Landlock(support) =>
        {
            let mut command = Command::new(program);
            command.args(args).current_dir(cwd);
            landlock::configure_command(&mut command, config.mode, &root, *support).map_err(
                |error| sandbox_unavailable(config.mode, format!("Landlock setup failed: {error}")),
            )?;
            command
        },
    };
    if let Some(constraints) = constraints
    {
        apply_to_command(&mut command, constraints).map_err(|error| {
            format!(
                "{GOVERNANCE_FAILURE_PREFIX} declared resource limits could not be installed on the {backend_label} spawn path; refusing to run unbounded. {error}"
            )
        })?;
    }

    for variable in SECRET_ENV_VARS
    {
        command.env_remove(variable);
    }
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = spawn_process_group(&mut command).map_err(|error| {
        let message = error.to_string();

        if message.contains(GOVERNANCE_FAILURE_PREFIX)
        {
            message
        }
        else if confined_mode
        {
            sandbox_unavailable(
                config.mode,
                format!("{backend_label} spawn failed: {message}"),
            )
        }
        else
        {
            message
        }
    })?;
    let stdout = drain_pipe(child.inner().stdout.take().expect("stdout was piped"));
    let stderr = drain_pipe(child.inner().stderr.take().expect("stderr was piped"));
    let deadline = Instant::now() + deadline_timeout;
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
        if exit_status.is_some() && stdout.is_finished() && stderr.is_finished()
        {
            break;
        }
        if Instant::now() >= deadline
        {
            let (_status, stdout, stderr) = terminate_and_reap(child, stdout, stderr);
            return Ok(LimitedOutput {
                success: false,
                timed_out: true,
                stdout,
                stderr,
            });
        }
        std::thread::sleep(PROCESS_POLL_INTERVAL);
    }
    drop(child);
    let status = exit_status.expect("completed process group has an exit status");
    let stdout = stdout.finish();
    let stderr = stderr.finish();
    if matches!(backend, SandboxBackend::Bubblewrap(_))
        && confined_mode
        && !status.success()
        && bwrap_runner_failed(&stderr)
    {
        return Err(sandbox_unavailable(
            config.mode,
            format!(
                "bubblewrap refused the profile after its probe: {}",
                String::from_utf8_lossy(&stderr).trim()
            ),
        ));
    }
    Ok(LimitedOutput {
        success: status.success(),
        timed_out: false,
        stdout,
        stderr,
    })
}

fn bwrap_runner_failed(stderr: &[u8]) -> bool {
    String::from_utf8_lossy(stderr).lines().any(|line| {
        line.to_ascii_lowercase()
            .contains(BWRAP_RUNNER_FAILURE_SIGNATURE)
    })
}

fn bubblewrap_profile_args(mode: SandboxMode, root: &Path, net_isolated: bool) -> Vec<OsString> {
    let mut args = [
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--die-with-parent",
    ]
    .into_iter()
    .map(OsString::from)
    .collect::<Vec<_>>();
    if matches!(mode, SandboxMode::WorkspaceWrite)
    {
        args.extend([OsString::from("--tmpfs"), OsString::from("/tmp")]);
        args.push(OsString::from("--bind"));
        args.push(root.as_os_str().to_owned());
        args.push(root.as_os_str().to_owned());
    }
    // Kernel-level network isolation for governed calls: the payload runs in
    // a network namespace with no interfaces, so deny-all egress holds even
    // for syscalls our seccomp filter does not inspect.
    if net_isolated
    {
        args.push(OsString::from("--unshare-net"));
    }
    args
}

fn bubblewrap_command(
    bwrap: &Path,
    mode: SandboxMode,
    root: &Path,
    cwd: &Path,
    program: &str,
    args: &[OsString],
    net_isolated: bool,
) -> Command {
    debug_assert!(mode.is_confined());
    let mut command = Command::new(bwrap);
    command
        .args(bubblewrap_profile_args(mode, root, net_isolated))
        .arg("--chdir")
        .arg(cwd)
        .arg("--")
        .arg(program)
        .args(args);
    command
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;

    fn landlock_support(abi: u32, enforcement: SandboxEnforcement) -> landlock::Support {
        landlock::Support { abi, enforcement }
    }

    #[test]
    fn sandbox_mode_parser_matches_harness_vocabulary() {
        assert_eq!(
            SandboxMode::parse(None).unwrap(),
            SandboxMode::WorkspaceWrite
        );
        assert_eq!(
            SandboxMode::parse(Some("read-only")).unwrap(),
            SandboxMode::ReadOnly
        );
        assert_eq!(
            SandboxMode::parse(Some("workspace-write")).unwrap(),
            SandboxMode::WorkspaceWrite
        );
        assert_eq!(
            SandboxMode::parse(Some("danger-full-access")).unwrap(),
            SandboxMode::DangerFullAccess
        );
        assert!(SandboxMode::parse(Some("auto")).is_err());
        assert!(SandboxMode::parse(Some("strict")).is_err());
    }

    #[test]
    fn one_shot_override_is_strictly_widening() {
        assert_eq!(
            resolve_effective_mode(
                SandboxMode::ReadOnly,
                Some(SandboxPermission::WorkspaceWrite),
            )
            .unwrap(),
            SandboxMode::WorkspaceWrite
        );
        assert_eq!(
            resolve_effective_mode(
                SandboxMode::WorkspaceWrite,
                Some(SandboxPermission::DangerFullAccess),
            )
            .unwrap(),
            SandboxMode::DangerFullAccess
        );
        assert!(
            resolve_effective_mode(
                SandboxMode::WorkspaceWrite,
                Some(SandboxPermission::ReadOnly),
            )
            .is_err()
        );
        assert!(
            resolve_effective_mode(
                SandboxMode::WorkspaceWrite,
                Some(SandboxPermission::WorkspaceWrite),
            )
            .is_err()
        );
    }

    #[test]
    fn internal_override_round_trip_is_not_public_tool_metadata() {
        let mut params = HashMap::new();
        install_one_shot_override(&mut params, SandboxPermission::DangerFullAccess);
        assert_eq!(
            take_one_shot_override(&mut params).unwrap(),
            Some(SandboxPermission::DangerFullAccess)
        );
        assert!(params.is_empty());
    }

    #[test]
    fn confined_modes_fail_closed_without_backend() {
        for mode in [SandboxMode::ReadOnly, SandboxMode::WorkspaceWrite]
        {
            let error = select_backend(mode, None, None, true).unwrap_err();
            assert!(error.contains(SANDBOX_UNAVAILABLE));
        }
        let error = select_backend(SandboxMode::ReadOnly, None, None, false).unwrap_err();
        assert!(error.contains(SANDBOX_UNAVAILABLE));
    }

    #[test]
    fn danger_full_access_is_the_only_direct_mode() {
        assert_eq!(
            select_backend(SandboxMode::DangerFullAccess, None, None, false).unwrap(),
            (SandboxBackend::Direct, None)
        );
        assert_eq!(
            select_backend(SandboxMode::DangerFullAccess, None, None, true).unwrap(),
            (SandboxBackend::Direct, None)
        );
    }

    #[test]
    fn bubblewrap_is_preferred_and_reports_full_enforcement() {
        let landlock = landlock_support(4, SandboxEnforcement::Partial);
        assert_eq!(
            select_backend(
                SandboxMode::WorkspaceWrite,
                Some(PathBuf::from("/usr/bin/bwrap")),
                Some(landlock),
                true,
            )
            .unwrap(),
            (
                SandboxBackend::Bubblewrap(PathBuf::from("/usr/bin/bwrap")),
                Some(SandboxEnforcement::Full)
            )
        );
    }

    #[test]
    fn landlock_is_the_linux_fallback_and_preserves_enforcement_strength() {
        let partial = landlock_support(4, SandboxEnforcement::Partial);
        assert_eq!(
            select_backend(SandboxMode::ReadOnly, None, Some(partial), true).unwrap(),
            (
                SandboxBackend::Landlock(partial),
                Some(SandboxEnforcement::Partial)
            )
        );
        let full = landlock_support(5, SandboxEnforcement::Full);
        assert_eq!(
            select_backend(SandboxMode::WorkspaceWrite, None, Some(full), true).unwrap(),
            (
                SandboxBackend::Landlock(full),
                Some(SandboxEnforcement::Full)
            )
        );
    }

    #[test]
    fn read_only_profile_matches_harness_mount_contract() {
        let args = bubblewrap_profile_args(SandboxMode::ReadOnly, Path::new("/workspace"), false)
            .into_iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert_eq!(
            args,
            [
                "--ro-bind",
                "/",
                "/",
                "--dev",
                "/dev",
                "--proc",
                "/proc",
                "--die-with-parent",
            ]
        );
    }

    #[test]
    fn workspace_write_profile_adds_only_private_tmp_and_workspace_bind() {
        let args =
            bubblewrap_profile_args(SandboxMode::WorkspaceWrite, Path::new("/workspace"), false)
                .into_iter()
                .map(|arg| arg.to_string_lossy().into_owned())
                .collect::<Vec<_>>();
        assert_eq!(
            args,
            [
                "--ro-bind",
                "/",
                "/",
                "--dev",
                "/dev",
                "--proc",
                "/proc",
                "--die-with-parent",
                "--tmpfs",
                "/tmp",
                "--bind",
                "/workspace",
                "/workspace",
            ]
        );
    }

    #[test]
    fn bubblewrap_wrapper_preserves_exact_command_argv() {
        let root = Path::new("/workspace");
        let command = bubblewrap_command(
            Path::new("/usr/bin/bwrap"),
            SandboxMode::ReadOnly,
            root,
            root,
            "git",
            &[OsString::from("status"), OsString::from("--short")],
            false,
        );
        assert_eq!(command.get_program(), OsStr::new("/usr/bin/bwrap"));
        let args = command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert!(
            args.windows(2)
                .any(|window| window == ["--chdir", "/workspace"])
        );
        assert!(args.ends_with(&[
            "--".to_string(),
            "git".to_string(),
            "status".to_string(),
            "--short".to_string(),
        ]));
    }

    #[test]
    fn bwrap_failure_and_denial_dialects_stay_distinct() {
        assert!(bwrap_runner_failed(
            b"bwrap: Creating new namespace failed\n"
        ));
        assert!(!bwrap_runner_failed(
            format!("touch: /etc/x: {BWRAP_DENIAL_SIGNATURE}\n").as_bytes()
        ));
    }

    #[test]
    fn governed_deny_all_egress_unshares_the_network_namespace() {
        let args =
            bubblewrap_profile_args(SandboxMode::WorkspaceWrite, Path::new("/workspace"), true);
        assert!(args.iter().any(|arg| arg == OsStr::new("--unshare-net")));
        // Exactly one isolation flag, appended last.
        assert_eq!(args.last(), Some(&OsString::from("--unshare-net")));
    }

    #[test]
    fn constraints_metadata_is_extracted_and_removed() {
        let mut params = HashMap::new();
        assert!(take_constraints(&mut params).unwrap().is_none());
        assert!(params.is_empty(), "extraction must not leave residue");

        let mut params = HashMap::new();
        params.insert(
            crate::agentic::tool_runtime::RESOURCE_LIMITS_METADATA.to_string(),
            r#"{"max_memory_bytes": 65536}"#.to_string(),
        );
        let constraints = take_constraints(&mut params).unwrap().expect("limits only");
        assert_eq!(constraints.limits.max_memory_bytes, Some(65536));
        // Missing egress half falls back to enforced deny-all.
        assert!(constraints.egress.enforce);
        assert!(constraints.egress.allow.is_empty());
        assert!(
            !params.contains_key(crate::agentic::tool_runtime::RESOURCE_LIMITS_METADATA),
            "governance metadata must not leak into tool parameters"
        );
    }

    #[test]
    fn malformed_governance_payloads_refuse_the_call() {
        let mut params = HashMap::new();
        params.insert(
            crate::agentic::tool_runtime::RESOURCE_LIMITS_METADATA.to_string(),
            "{not json".to_string(),
        );
        let error = take_constraints(&mut params).expect_err("malformed limits must refuse");
        assert!(error.contains("malformed resource_limits"), "{error}");

        let mut params = HashMap::new();
        params.insert(
            crate::agentic::tool_runtime::EGRESS_POLICY_METADATA.to_string(),
            r#"{"allow": [{"host": 42}]}"#.to_string(),
        );
        let error = take_constraints(&mut params).expect_err("malformed egress must refuse");
        assert!(error.contains("malformed egress_policy"), "{error}");
    }

    #[test]
    fn governed_wall_time_fails_closed_without_tree_lifecycle_enforcement() {
        let constraints = ExecutionConstraints {
            limits: super::super::budgets::ResourceLimits {
                wall_time_seconds: Some(1),
                ..Default::default()
            },
            ..ExecutionConstraints::default()
        };

        let error = constraints
            .ensure_enforceable(crate::agentic::enforcement::probed_backend())
            .expect_err("wall-time must refuse until descendants cannot escape lifecycle control");

        assert!(error.contains("wall-time"), "{error}");
    }
}
