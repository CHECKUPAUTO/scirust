//! Linux Landlock fallback for SciAgent process-backed tools.
//!
//! Bubblewrap remains the strongest Linux backend because it can provide a
//! private `/tmp` and a read-only mount namespace.  When an executable bwrap
//! is not available, this module can enforce the filesystem write boundary in
//! the child itself with Landlock ABI >= 3.  ABI 3 is the minimum accepted
//! here because it is the first ABI that can restrict `truncate(2)` and
//! `O_TRUNC`; accepting an older ABI would make `read-only` misleading.
//!
//! Landlock does not mediate every metadata operation (for example chmod/chown
//! on kernels where those operations are outside Landlock's filesystem access
//! vocabulary), so this rung is deliberately a fallback rather than being
//! advertised as equivalent to the bubblewrap mount namespace.  Confined
//! modes still fail closed if neither backend can enforce its rung.

use super::tools::Tool;
#[cfg(target_os = "linux")]
use command_group::{CommandGroup, GroupChild};
use std::collections::HashMap;
use std::ffi::OsString;
#[cfg(target_os = "linux")]
use std::io::Read;
use std::path::{Path, PathBuf};
#[cfg(target_os = "linux")]
use std::process::{Command, ExitStatus, Stdio};
#[cfg(target_os = "linux")]
use std::sync::{Arc, Mutex};
use std::time::Duration;
#[cfg(target_os = "linux")]
use std::time::Instant;

#[cfg(target_os = "linux")]
const MAX_TOOL_OUTPUT_BYTES: usize = 64 * 1024;
const TOOL_TIMEOUT: Duration = Duration::from_secs(30);
#[cfg(target_os = "linux")]
const REAP_GRACE: Duration = Duration::from_secs(2);
#[cfg(target_os = "linux")]
const PROCESS_POLL_INTERVAL: Duration = Duration::from_millis(10);
const SANDBOX_UNAVAILABLE: &str = "SANDBOX_UNAVAILABLE";
const MIN_LANDLOCK_ABI: i32 = 3;
#[cfg(target_os = "linux")]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SandboxMode {
    ReadOnly,
    WorkspaceWrite,
    DangerFullAccess,
}

impl SandboxMode {
    fn from_env() -> Result<Self, String> {
        match std::env::var("SCIAGENT_SANDBOX_MODE")
            .ok()
            .unwrap_or_else(|| "danger-full-access".to_string())
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

    fn confined(self) -> bool {
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

/// Replace the process-backed callbacks with the Landlock rung only when a
/// confined mode is requested, bubblewrap is not executable, and the running
/// kernel exposes an ABI strong enough to cover truncate semantics.
///
/// The normal bubblewrap installer runs first.  This function is therefore a
/// selective override, not an independent direct-execution path.
pub(crate) fn install_fallback_if_needed(tools: &mut [Tool]) {
    if !should_use_landlock_fallback()
    {
        return;
    }
    for tool in tools
    {
        tool.execute = match tool.name
        {
            "search" => landlocked_search,
            "grep" => landlocked_grep,
            "build" => landlocked_build,
            "test" => landlocked_test,
            "status" => landlocked_status,
            _ => continue,
        };
    }
}

fn should_use_landlock_fallback() -> bool {
    let Ok(mode) = SandboxMode::from_env()
    else
    {
        return false;
    };
    if !mode.confined() || !cfg!(target_os = "linux") || bubblewrap_is_executable()
    {
        return false;
    }
    landlock_abi().is_some_and(|abi| abi >= MIN_LANDLOCK_ABI)
}

fn fallback_policy(mode: SandboxMode, bwrap_executable: bool, abi: Option<i32>) -> bool {
    mode.confined()
        && cfg!(target_os = "linux")
        && !bwrap_executable
        && abi.is_some_and(|abi| abi >= MIN_LANDLOCK_ABI)
}

#[cfg(target_os = "linux")]
fn bubblewrap_is_executable() -> bool {
    use std::os::unix::fs::PermissionsExt;

    configured_bwrap().is_some_and(|path| {
        std::fs::metadata(path)
            .map(|metadata| metadata.is_file() && metadata.permissions().mode() & 0o111 != 0)
            .unwrap_or(false)
    })
}

#[cfg(not(target_os = "linux"))]
fn bubblewrap_is_executable() -> bool {
    false
}

#[cfg(target_os = "linux")]
fn configured_bwrap() -> Option<PathBuf> {
    std::env::var_os("SCIAGENT_BWRAP")
        .map(PathBuf::from)
        .or_else(|| {
            let path = std::env::var_os("PATH")?;
            std::env::split_paths(&path)
                .map(|directory| directory.join("bwrap"))
                .find(|candidate| candidate.is_file())
        })
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

fn landlocked_search(params: HashMap<String, String>) -> String {
    search_workspace(&params, "10")
}

fn landlocked_grep(params: HashMap<String, String>) -> String {
    search_workspace(&params, "15")
}

fn search_workspace(params: &HashMap<String, String>, max_count: &str) -> String {
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
    match run_landlocked("rg", &rg_args, &root)
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
            match run_landlocked("grep", &grep_args, &root)
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

fn landlocked_build(params: HashMap<String, String>) -> String {
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
    match run_landlocked("cargo", &args, &root)
    {
        Ok(output) if output.timed_out => "Build timed out after 30 seconds".to_string(),
        Ok(output) if output.success => format!("{crate_name} builds successfully"),
        Ok(output) => format!("Build errors:\n{}", String::from_utf8_lossy(&output.stderr)),
        Err(error) => format!("Failed to run cargo: {error}"),
    }
}

fn landlocked_test(params: HashMap<String, String>) -> String {
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
    match run_landlocked("cargo", &args, &root)
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

fn landlocked_status(_params: HashMap<String, String>) -> String {
    let root = match canonical_workspace_root()
    {
        Ok(root) => root,
        Err(error) => return error,
    };
    let args = [OsString::from("status"), OsString::from("--short")];
    match run_landlocked("git", &args, &root)
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

#[cfg(target_os = "linux")]
struct PipeDrain {
    bytes: Arc<Mutex<Vec<u8>>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

#[cfg(target_os = "linux")]
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

#[cfg(target_os = "linux")]
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

#[cfg(target_os = "linux")]
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

#[cfg(target_os = "linux")]
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

#[cfg(target_os = "linux")]
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

fn run_landlocked(program: &str, args: &[OsString], cwd: &Path) -> Result<LimitedOutput, String> {
    let mode = SandboxMode::from_env()?;
    if !mode.confined()
    {
        return Err(format!(
            "[{SANDBOX_UNAVAILABLE}] Landlock fallback cannot execute {:?} unconfined; rebuild the runtime after selecting danger-full-access",
            mode.label()
        ));
    }
    run_landlocked_with_timeout(program, args, cwd, mode, TOOL_TIMEOUT)
}

#[cfg(target_os = "linux")]
fn run_landlocked_with_timeout(
    program: &str,
    args: &[OsString],
    cwd: &Path,
    mode: SandboxMode,
    timeout: Duration,
) -> Result<LimitedOutput, String> {
    use std::os::unix::process::CommandExt;

    let root = canonical_workspace_root()?;
    let ruleset = linux_landlock::prepare(&root, mode).map_err(|error| {
        format!(
            "[{SANDBOX_UNAVAILABLE}] Landlock could not enforce {:?}: {error}",
            mode.label()
        )
    })?;
    let ruleset_fd = linux_landlock::raw_fd(&ruleset);

    let mut command = Command::new(program);
    command.args(args).current_dir(cwd);
    if matches!(mode, SandboxMode::WorkspaceWrite)
    {
        let private_tmp = root.join("target/.sciagent-tmp");
        std::fs::create_dir_all(&private_tmp).map_err(|error| {
            format!(
                "[{SANDBOX_UNAVAILABLE}] cannot prepare workspace-local TMPDIR `{}`: {error}",
                private_tmp.display()
            )
        })?;
        command.env("TMPDIR", private_tmp);
    }
    for variable in SECRET_ENV_VARS
    {
        command.env_remove(variable);
    }
    command.stdout(Stdio::piped()).stderr(Stdio::piped());

    // SAFETY: the hook performs only prctl(2) and landlock_restrict_self(2)
    // against a ruleset prepared in the parent.  It does not allocate, open
    // paths, or take locks after fork.
    unsafe {
        command.pre_exec(move || linux_landlock::restrict_current_process(ruleset_fd));
    }

    let mut child = spawn_process_group(&mut command).map_err(|error| error.to_string())?;
    drop(ruleset);
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
    Ok(LimitedOutput {
        success: status.success(),
        timed_out: false,
        stdout: stdout.finish(),
        stderr: stderr.finish(),
    })
}

#[cfg(not(target_os = "linux"))]
fn run_landlocked_with_timeout(
    _program: &str,
    _args: &[OsString],
    _cwd: &Path,
    mode: SandboxMode,
    _timeout: Duration,
) -> Result<LimitedOutput, String> {
    Err(format!(
        "[{SANDBOX_UNAVAILABLE}] Landlock fallback for {:?} is Linux-only",
        mode.label()
    ))
}

#[cfg(target_os = "linux")]
fn landlock_abi() -> Option<i32> {
    linux_landlock::abi().ok()
}

#[cfg(not(target_os = "linux"))]
fn landlock_abi() -> Option<i32> {
    None
}

#[cfg(target_os = "linux")]
mod linux_landlock {
    use super::{MIN_LANDLOCK_ABI, SandboxMode};
    use std::ffi::c_void;
    use std::fs::{File, OpenOptions};
    use std::io;
    use std::mem::size_of;
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
    use std::os::raw::{c_int, c_long};
    use std::os::unix::fs::OpenOptionsExt;
    use std::path::Path;

    const SYS_LANDLOCK_CREATE_RULESET: c_long = 444;
    const SYS_LANDLOCK_ADD_RULE: c_long = 445;
    const SYS_LANDLOCK_RESTRICT_SELF: c_long = 446;

    const LANDLOCK_CREATE_RULESET_VERSION: u32 = 1;
    const LANDLOCK_RULE_PATH_BENEATH: u32 = 1;
    const PR_SET_NO_NEW_PRIVS: c_int = 38;
    const F_SETFD: c_int = 2;
    const FD_CLOEXEC: c_int = 1;
    const O_PATH: c_int = 0o10000000;

    const ACCESS_WRITE_FILE: u64 = 1 << 1;
    const ACCESS_REMOVE_DIR: u64 = 1 << 4;
    const ACCESS_REMOVE_FILE: u64 = 1 << 5;
    const ACCESS_MAKE_CHAR: u64 = 1 << 6;
    const ACCESS_MAKE_DIR: u64 = 1 << 7;
    const ACCESS_MAKE_REG: u64 = 1 << 8;
    const ACCESS_MAKE_SOCK: u64 = 1 << 9;
    const ACCESS_MAKE_FIFO: u64 = 1 << 10;
    const ACCESS_MAKE_BLOCK: u64 = 1 << 11;
    const ACCESS_MAKE_SYM: u64 = 1 << 12;
    const ACCESS_REFER: u64 = 1 << 13;
    const ACCESS_TRUNCATE: u64 = 1 << 14;

    #[repr(C)]
    struct RulesetAttr {
        handled_access_fs: u64,
    }

    #[repr(C, packed)]
    struct PathBeneathAttr {
        allowed_access: u64,
        parent_fd: c_int,
    }

    extern "C" {
        fn syscall(number: c_long, ...) -> c_long;
        fn prctl(option: c_int, ...) -> c_int;
        fn fcntl(fd: c_int, cmd: c_int, ...) -> c_int;
    }

    pub(super) fn abi() -> io::Result<i32> {
        #[cfg(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "riscv64"
        ))]
        {
            // SAFETY: the version-query form requires a null attr and size 0.
            let result = unsafe {
                syscall(
                    SYS_LANDLOCK_CREATE_RULESET,
                    std::ptr::null::<RulesetAttr>(),
                    0usize,
                    LANDLOCK_CREATE_RULESET_VERSION,
                )
            };
            if result < 0
            {
                Err(io::Error::last_os_error())
            }
            else
            {
                Ok(result as i32)
            }
        }
        #[cfg(not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "riscv64"
        )))]
        {
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Landlock syscall numbers are not validated for this architecture",
            ))
        }
    }

    fn handled_write_access(abi: i32) -> io::Result<u64> {
        if abi < MIN_LANDLOCK_ABI
        {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "kernel exposes Landlock ABI {abi}, but ABI {MIN_LANDLOCK_ABI}+ is required to restrict truncate semantics"
                ),
            ));
        }
        let mut access = ACCESS_WRITE_FILE
            | ACCESS_REMOVE_DIR
            | ACCESS_REMOVE_FILE
            | ACCESS_MAKE_CHAR
            | ACCESS_MAKE_DIR
            | ACCESS_MAKE_REG
            | ACCESS_MAKE_SOCK
            | ACCESS_MAKE_FIFO
            | ACCESS_MAKE_BLOCK
            | ACCESS_MAKE_SYM;
        if abi >= 2
        {
            access |= ACCESS_REFER;
        }
        if abi >= 3
        {
            access |= ACCESS_TRUNCATE;
        }
        Ok(access)
    }

    pub(super) fn prepare(root: &Path, mode: SandboxMode) -> io::Result<OwnedFd> {
        let abi = abi()?;
        let handled = handled_write_access(abi)?;
        let attr = RulesetAttr {
            handled_access_fs: handled,
        };
        // SAFETY: attr points to a valid RulesetAttr and the size matches the
        // single-field ABI used since Landlock v1.
        let fd = unsafe {
            syscall(
                SYS_LANDLOCK_CREATE_RULESET,
                &attr as *const RulesetAttr,
                size_of::<RulesetAttr>(),
                0u32,
            )
        };
        if fd < 0
        {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: successful create_ruleset returns a new owned descriptor.
        let ruleset = unsafe { OwnedFd::from_raw_fd(fd as RawFd) };
        // Keep the descriptor available to pre_exec, but ensure it disappears
        // automatically after the target program is exec'd.
        // SAFETY: ruleset is a live descriptor and F_SETFD takes an int flag.
        if unsafe { fcntl(ruleset.as_raw_fd(), F_SETFD, FD_CLOEXEC) } < 0
        {
            return Err(io::Error::last_os_error());
        }

        if matches!(mode, SandboxMode::WorkspaceWrite)
        {
            add_path_rule(&ruleset, root, handled)?;
        }
        Ok(ruleset)
    }

    fn add_path_rule(ruleset: &OwnedFd, path: &Path, allowed_access: u64) -> io::Result<()> {
        let parent: File = OpenOptions::new()
            .read(true)
            .custom_flags(O_PATH)
            .open(path)?;
        let attr = PathBeneathAttr {
            allowed_access,
            parent_fd: parent.as_raw_fd(),
        };
        // SAFETY: the packed attribute matches linux/landlock.h and both file
        // descriptors remain live for the duration of the syscall.
        let result = unsafe {
            syscall(
                SYS_LANDLOCK_ADD_RULE,
                ruleset.as_raw_fd(),
                LANDLOCK_RULE_PATH_BENEATH,
                &attr as *const PathBeneathAttr as *const c_void,
                0u32,
            )
        };
        if result < 0
        {
            Err(io::Error::last_os_error())
        }
        else
        {
            Ok(())
        }
    }

    pub(super) fn raw_fd(ruleset: &OwnedFd) -> RawFd {
        ruleset.as_raw_fd()
    }

    pub(super) fn restrict_current_process(ruleset_fd: RawFd) -> io::Result<()> {
        // SAFETY: PR_SET_NO_NEW_PRIVS is process-local and takes scalar values.
        if unsafe { prctl(PR_SET_NO_NEW_PRIVS, 1usize, 0usize, 0usize, 0usize) } < 0
        {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: ruleset_fd is inherited from the parent and remains valid
        // until spawn returns; restrict_self takes no extra structure.
        let result = unsafe { syscall(SYS_LANDLOCK_RESTRICT_SELF, ruleset_fd, 0u32) };
        if result < 0
        {
            Err(io::Error::last_os_error())
        }
        else
        {
            Ok(())
        }
    }

    #[cfg(test)]
    pub(super) fn write_access_for_test(abi: i32) -> io::Result<u64> {
        handled_write_access(abi)
    }

    #[cfg(test)]
    pub(super) const REFER_FOR_TEST: u64 = ACCESS_REFER;
    #[cfg(test)]
    pub(super) const TRUNCATE_FOR_TEST: u64 = ACCESS_TRUNCATE;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_never_replaces_danger_full_access() {
        assert!(!fallback_policy(
            SandboxMode::DangerFullAccess,
            false,
            Some(MIN_LANDLOCK_ABI)
        ));
    }

    #[test]
    fn executable_bwrap_keeps_the_primary_backend() {
        assert!(!fallback_policy(
            SandboxMode::ReadOnly,
            true,
            Some(MIN_LANDLOCK_ABI)
        ));
    }

    #[test]
    fn landlock_requires_abi_three_or_newer() {
        assert!(!fallback_policy(
            SandboxMode::ReadOnly,
            false,
            Some(MIN_LANDLOCK_ABI - 1)
        ));
        assert!(fallback_policy(
            SandboxMode::ReadOnly,
            false,
            Some(MIN_LANDLOCK_ABI)
        ));
        assert!(fallback_policy(
            SandboxMode::WorkspaceWrite,
            false,
            Some(MIN_LANDLOCK_ABI + 1)
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn abi_three_write_set_covers_refer_and_truncate() {
        let access = linux_landlock::write_access_for_test(3).unwrap();
        assert_ne!(access & linux_landlock::REFER_FOR_TEST, 0);
        assert_ne!(access & linux_landlock::TRUNCATE_FOR_TEST, 0);
        assert!(linux_landlock::write_access_for_test(2).is_err());
    }
}
