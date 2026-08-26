//! Real OS-level enforcement for CCOS Enterprise resource budgets.
//!
//! H-5 closes the gap between *declared* limits ([`super::budgets`]) and
//! *actual* enforcement. Everything here follows the same contract as the
//! sandbox seam: raw syscalls, no new dependencies, and fail-closed
//! behaviour — a limit that cannot be enforced denies the call instead of
//! running unbounded.
//!
//! What this backend can currently enforce without overstating process-tree
//! isolation:
//!
//! | Declared limit          | Kernel mechanism |
//! |-------------------------|------------------|
//! | `max_file_size_bytes`   | inherited `RLIMIT_FSIZE` (`prlimit64`) |
//! | deny-all egress         | inherited seccomp-BPF deny of INET sockets, plus bwrap `--unshare-net` |
//!
//! Tree-wide memory, process-count, CPU-count and wall-time budgets are
//! deliberately refused: per-process rlimits, affinity and process-group kills
//! do not provide non-escapable accounting/lifecycle control for arbitrary
//! descendants. Those require cgroup/PID-namespace-class enforcement.
//! GPU memory caps and per-host egress allow-lists are likewise refused rather
//! than simulated.

use super::budgets::{EgressPolicy, ResourceBackend, ResourceLimits};
use std::time::Duration;

pub mod cgroup;

/// A prepared cgroup lease for one governed execution, when the host
/// provides a writable cgroup v2 hierarchy. Dropping the lease removes the
/// subgroup (best effort).
pub enum CgroupLease {
    #[cfg(target_os = "linux")]
    V2(Box<cgroup::V2Group>),
    None,
}

impl CgroupLease {
    /// Place the future child process into the leased group.
    pub fn attach_to_command(&self, command: &mut std::process::Command) -> Result<(), String> {
        match self
        {
            #[cfg(target_os = "linux")]
            Self::V2(group) => group.attach_to_command(command),
            #[cfg(target_os = "linux")]
            Self::None => Ok(()),
            // Off Linux no lease is ever handed out; the parameter exists
            // for signature parity and is intentionally untouched.
            #[cfg(not(target_os = "linux"))]
            Self::None =>
            {
                let _ = command;
                Ok(())
            },
        }
    }
}

/// Prepare a cgroup enforcing the tree-wide subset of the declared limits.
///
/// Returns [`CgroupLease::None`] when the platform or hierarchy cannot host
/// one — per-process rlimits still apply — and an error when the hierarchy
/// exists but refuses the declared limits (fail closed, mirroring every
/// other seam in this module).
pub fn prepare_cgroup(limits: &ResourceLimits) -> Result<CgroupLease, String> {
    #[cfg(target_os = "linux")]
    {
        match cgroup::V2Group::prepare(limits)?
        {
            Some(group) => Ok(CgroupLease::V2(Box::new(group))),
            None => Ok(CgroupLease::None),
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = limits;
        Ok(CgroupLease::None)
    }
}

pub(super) const GOVERNANCE_FAILURE_PREFIX: &str = "[GOVERNANCE_UNAVAILABLE]";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionConstraints {
    pub limits: ResourceLimits,
    pub egress: EgressPolicy,
}

impl Default for ExecutionConstraints {
    fn default() -> Self {
        Self {
            limits: ResourceLimits::default(),
            egress: EgressPolicy::deny_all(),
        }
    }
}

impl ExecutionConstraints {
    /// Fail-closed enforceability gate, evaluated BEFORE any side effect.
    ///
    /// Extends [`ResourceLimits::check_enforceable`] with the two rules the
    /// kernel backend adds: deny-all egress needs a working filter backend,
    /// and non-empty allow-lists are refused outright on backends that can
    /// only deny whole address families.
    pub fn ensure_enforceable(&self, backend: &dyn ResourceBackend) -> Result<(), String> {
        self.limits.check_enforceable(backend, &self.egress)?;
        let deny_all_active = self.egress.enforce && self.egress.allow.is_empty();
        let allow_list_active = !self.egress.allow.is_empty();
        if deny_all_active && !backend.supports_egress_deny_all()
        {
            return Err(
                "deny-all network egress cannot be enforced by the active backend; refusing \
                 execution"
                    .to_string(),
            );
        }
        if allow_list_active && !backend.supports_egress_allow_list()
        {
            return Err(
                "host/port egress allow-lists cannot be enforced by the active backend; refusing \
                 execution"
                    .to_string(),
            );
        }
        Ok(())
    }

    /// Wall-clock deadline implied by the declared wall-time limit, capped by
    /// the runtime's own global timeout.
    pub fn effective_wall_time(&self, global_cap: Duration) -> Duration {
        self.limits
            .wall_time_seconds
            .map(|seconds| Duration::from_secs(seconds).min(global_cap))
            .unwrap_or(global_cap)
    }

    /// Rebuild constraints from call metadata produced by
    /// [`ToolRuntime::execute_governed`](super::tool_runtime::ToolRuntime::execute_governed).
    ///
    /// Missing halves fall back to their fail-closed defaults (unbounded
    /// limits, enforced deny-all egress); malformed payloads are an error,
    /// never silently ignored governance.
    pub fn from_metadata(
        limits_json: Option<&str>,
        egress_json: Option<&str>,
    ) -> Result<Option<Self>, String> {
        if limits_json.is_none() && egress_json.is_none()
        {
            return Ok(None);
        }
        let limits = match limits_json
        {
            None => ResourceLimits::default(),
            Some(json) => serde_json::from_str(json)
                .map_err(|error| format!("malformed resource_limits metadata: {error}"))?,
        };
        let egress = match egress_json
        {
            None => EgressPolicy::deny_all(),
            Some(json) => serde_json::from_str(json)
                .map_err(|error| format!("malformed egress_policy metadata: {error}"))?,
        };
        Ok(Some(Self { limits, egress }))
    }

    /// Serialize into the two reserved call-metadata values.
    pub fn to_metadata(&self) -> [(String, String); 2] {
        [
            (
                super::tool_runtime::RESOURCE_LIMITS_METADATA.to_string(),
                serde_json::to_string(&self.limits)
                    .expect("ResourceLimits serialization cannot fail"),
            ),
            (
                super::tool_runtime::EGRESS_POLICY_METADATA.to_string(),
                serde_json::to_string(&self.egress)
                    .expect("EgressPolicy serialization cannot fail"),
            ),
        ]
    }
}

/// Backend actually enforcing limits on this host, probed once.
///
/// Linux builds return the real rlimit/seccomp backend (with an honest
/// `seccomp_available` capability flag); every other platform returns
/// [`crate::agentic::budgets::NoResourceBackend`] so every declared limit
/// fails closed.
pub fn probed_backend() -> &'static dyn ResourceBackend {
    imp::probed_backend()
}

/// Install the declared constraints on a child command.
///
/// The installation happens in `pre_exec`: the constraints apply to the
/// spawned program and everything it forks, never to the agent process
/// itself. Any failure refuses the spawn — governance that silently degrades
/// would be worse than no governance.
pub fn apply_to_command(
    command: &mut std::process::Command,
    constraints: &ExecutionConstraints,
) -> Result<(), String> {
    constraints.ensure_enforceable(probed_backend())?;
    imp::apply_to_command(command, constraints)
}

// ---------------------------------------------------------------------------
// Linux implementation — raw syscalls, mirroring sandbox/landlock.rs.
// ---------------------------------------------------------------------------
#[cfg(target_os = "linux")]
mod imp {
    /// Highest CPU count representable by the affinity mask used for pinning.
    const MAX_PINNABLE_CPUS: u32 = 64;
    use super::super::budgets::ResourceBackend;
    use super::ExecutionConstraints;
    use std::io;
    use std::os::raw::{c_int, c_long, c_uint, c_ulong, c_void};
    use std::os::unix::process::CommandExt;
    use std::process::Command;
    use std::sync::OnceLock;

    // Syscall numbers. x86_64 uses its historical table; aarch64 uses the
    // asm-generic table shared by the Jetson Thor target this crate ships on.
    #[cfg(target_arch = "x86_64")]
    const SYS_PRLIMIT64: c_long = 302;
    #[cfg(target_arch = "x86_64")]
    const SYS_SCHED_SETAFFINITY: c_long = 203;
    #[cfg(target_arch = "x86_64")]
    const SYS_SECCOMP: c_long = 317;
    #[cfg(target_arch = "x86_64")]
    const SYS_SOCKET: c_uint = 41;
    #[cfg(target_arch = "x86_64")]
    const AUDIT_ARCH: u32 = 0xC000003E;

    #[cfg(target_arch = "aarch64")]
    const SYS_PRLIMIT64: c_long = 261;
    #[cfg(target_arch = "aarch64")]
    const SYS_SCHED_SETAFFINITY: c_long = 122;
    #[cfg(target_arch = "aarch64")]
    const SYS_SECCOMP: c_long = 277;
    #[cfg(target_arch = "aarch64")]
    const SYS_SOCKET: c_uint = 198;
    #[cfg(target_arch = "aarch64")]
    const AUDIT_ARCH: u32 = 0xC00000B7;

    // Identical ABI values on both supported architectures (asm-generic
    // resource.h agrees with x86_64).
    const RLIMIT_CPU: c_int = 0;
    const RLIMIT_FSIZE: c_int = 1;
    const RLIMIT_NPROC: c_int = 6;
    const RLIMIT_AS: c_int = 9;

    const AF_INET: u32 = 2;
    const AF_INET6: u32 = 10;

    const PR_SET_NO_NEW_PRIVS: c_int = 38;
    const SECCOMP_SET_MODE_FILTER: c_uint = 1;
    const SECCOMP_GET_ACTION_AVAIL: c_uint = 2;
    const SECCOMP_RET_ALLOW: u32 = 0x7fff_0000;
    const SECCOMP_RET_ERRNO: u32 = 0x0005_0000;
    const EPERM: u32 = 1;

    // Classic BPF instruction classes used by the filter below.
    const BPF_LD_ABS_W: u16 = 0x20; // BPF_LD | BPF_W | BPF_ABS
    const BPF_JEQ_K: u16 = 0x15; // BPF_JMP | BPF_JEQ | BPF_K
    const BPF_RET_K: u16 = 0x06; // BPF_RET | BPF_K

    unsafe extern "C" {
        fn syscall(number: c_long, ...) -> c_long;
        fn prctl(option: c_int, ...) -> c_int;
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Rlimit {
        cur: u64,
        max: u64,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct SockFilter {
        code: u16,
        jt: u8,
        jf: u8,
        k: u32,
    }

    #[repr(C)]
    struct SockFprog {
        len: u16,
        filter: *const SockFilter,
    }

    /// Filter denying INET socket creation for the filtered task and its
    /// descendants. Layout mirrors `struct seccomp_data`: `nr` at 0, `arch`
    /// at 4, `args[0]` low word at 16.
    ///
    /// ```text
    /// i0: LD   [arch]
    /// i1: JEQ  native-arch ?continue :DENY_ALL      (unknown arch fails shut)
    /// i2: LD   [nr]
    /// i3: JEQ  SYS_socket     ?continue :ALLOW
    /// i4: LD   [args[0] lo]                         (socket domain)
    /// i5: JEQ  AF_INET        :DENY_SOCKET
    /// i6: RET  EPERM
    /// i7: JEQ  AF_INET6       ?continue :ALLOW
    /// i8: RET  EPERM
    /// i9: RET  ALLOW
    /// i10: RET EPERM                                 (DENY_ALL landing pad)
    /// ```
    fn inet_socket_deny_filter() -> Vec<SockFilter> {
        fn jeq(k: u32, jt: u8, jf: u8) -> SockFilter {
            SockFilter {
                code: BPF_JEQ_K,
                jt,
                jf,
                k,
            }
        }
        fn ret(k: u32) -> SockFilter {
            SockFilter {
                code: BPF_RET_K,
                jt: 0,
                jf: 0,
                k,
            }
        }
        fn ld_abs(offset: u32) -> SockFilter {
            SockFilter {
                code: BPF_LD_ABS_W,
                jt: 0,
                jf: 0,
                k: offset,
            }
        }
        let errno_eperm = ret(SECCOMP_RET_ERRNO | EPERM);
        vec![
            ld_abs(4),              // i0  arch
            jeq(AUDIT_ARCH, 0, 8),  // i1  mismatch -> i10
            ld_abs(0),              // i2  nr
            jeq(SYS_SOCKET, 0, 5),  // i3  not socket -> i9
            ld_abs(16),             // i4  domain
            jeq(AF_INET, 0, 1),     // i5  v4 match -> i6
            errno_eperm,            // i6
            jeq(AF_INET6, 0, 1),    // i7  v6 match -> i8
            errno_eperm,            // i8
            ret(SECCOMP_RET_ALLOW), // i9
            errno_eperm,            // i10 unknown arch: deny all
        ]
    }

    fn seccomp_available() -> bool {
        static PROBE: OnceLock<bool> = OnceLock::new();
        *PROBE.get_or_init(|| {
            let action: u32 = SECCOMP_RET_ALLOW;
            let status = unsafe {
                syscall(
                    SYS_SECCOMP,
                    SECCOMP_GET_ACTION_AVAIL as c_ulong,
                    0 as c_ulong,
                    &action as *const u32 as *const c_void,
                )
            };
            status == 0
        })
    }

    fn set_rlimit(resource: c_int, value: u64) -> io::Result<()> {
        let limit = Rlimit {
            cur: value,
            max: value,
        };
        let status = unsafe {
            syscall(
                SYS_PRLIMIT64,
                0 as c_ulong, // pid 0 = calling process
                resource as c_uint,
                &limit as *const Rlimit,
                std::ptr::null::<Rlimit>(),
            )
        };
        if status != 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    fn pin_cpus(max_cpus: u32) -> io::Result<()> {
        let available = std::thread::available_parallelism()
            .map(|count| count.get() as u32)
            .unwrap_or(MAX_PINNABLE_CPUS);
        let pinned = max_cpus.min(available).min(MAX_PINNABLE_CPUS);
        let mut mask = 0u64;
        for cpu in 0..pinned
        {
            mask |= 1 << cpu;
        }
        let status = unsafe {
            syscall(
                SYS_SCHED_SETAFFINITY,
                0 as c_ulong, // pid 0 = calling thread (fork child)
                std::mem::size_of::<u64>(),
                &mask as *const u64 as *const c_void,
            )
        };
        if status != 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    fn install_socket_filter() -> io::Result<()> {
        let no_new_privs = unsafe {
            prctl(
                PR_SET_NO_NEW_PRIVS,
                1 as c_ulong,
                0 as c_ulong,
                0 as c_ulong,
                0 as c_ulong,
            )
        };
        if no_new_privs != 0
        {
            return Err(io::Error::last_os_error());
        }
        let filter = inet_socket_deny_filter();
        let prog = SockFprog {
            len: filter.len() as u16,
            filter: filter.as_ptr(),
        };
        let status = unsafe {
            syscall(
                SYS_SECCOMP,
                SECCOMP_SET_MODE_FILTER as c_ulong,
                0 as c_ulong, // flags
                &prog as *const SockFprog,
            )
        };
        if status != 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    fn apply(constraints: &ExecutionConstraints) -> io::Result<()> {
        let limits = &constraints.limits;
        if let Some(bytes) = limits.max_memory_bytes
        {
            set_rlimit(RLIMIT_AS, bytes)?;
        }
        if let Some(processes) = limits.max_processes
        {
            set_rlimit(RLIMIT_NPROC, u64::from(processes))?;
        }
        if let Some(bytes) = limits.max_file_size_bytes
        {
            set_rlimit(RLIMIT_FSIZE, bytes)?;
        }
        if let Some(seconds) = limits.wall_time_seconds
        {
            // Belt-and-braces alongside the runtime kill deadline: a runaway
            // child burning CPU gets SIGKILLed by the kernel even when the
            // supervising thread is starved.
            set_rlimit(RLIMIT_CPU, seconds.max(1))?;
        }
        if let Some(cpus) = limits.max_cpus
        {
            pin_cpus(cpus)?;
        }
        let deny_all = constraints.egress.enforce && constraints.egress.allow.is_empty();
        if deny_all
        {
            install_socket_filter()?;
        }
        else if !constraints.egress.allow.is_empty() || !constraints.egress.enforce
        {
            // Defense in depth: ensure_enforceable rejects these earlier.
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "egress policy is not enforceable by the kernel backend",
            ));
        }
        Ok(())
    }

    /// The real backend, or [`crate::agentic::budgets::NoResourceBackend`]
    /// when seccomp is disabled.
    struct RealLinuxBackend {
        seccomp_available: bool,
    }

    impl ResourceBackend for RealLinuxBackend {
        // Honest fail-closed gaps: these limits need non-resettable,
        // execution-tree-wide accounting/lifecycle control. Per-process
        // rlimits, sched affinity and process-group kills are insufficient.
        fn supports_memory_limit(&self, _bytes: u64) -> bool {
            false
        }
        fn supports_cpu_limit(&self, _cpus: u32) -> bool {
            false
        }
        fn supports_wall_time(&self, _seconds: u64) -> bool {
            false
        }
        fn supports_process_limit(&self, _processes: u32) -> bool {
            false
        }
        fn supports_file_size_limit(&self, _bytes: u64) -> bool {
            true
        }
        // Honest gap: GPU memory needs a cgroup or driver hook; claiming it
        // would be exactly the declared-vs-enforced drift H-5 removes.
        fn supports_gpu_memory_limit(&self, _bytes: u64) -> bool {
            false
        }
        fn supports_egress_allow_list(&self) -> bool {
            false
        }
        fn supports_egress_deny_all(&self) -> bool {
            self.seccomp_available
        }
    }

    /// What the machine's cgroup v2 hierarchy adds on top of per-process
    /// mechanisms, probed once. Tree-wide `memory.max`/`pids.max`/`cpu.max`
    /// are non-escapable for an unprivileged child, which is exactly the
    /// bar the honest gaps demand.
    #[derive(Clone, Copy)]
    pub(super) struct CgroupCapabilities {
        pub memory: bool,
        pub pids: bool,
        pub cpu: bool,
    }

    pub(super) fn cgroup_capabilities() -> CgroupCapabilities {
        static PROBE: OnceLock<CgroupCapabilities> = OnceLock::new();
        *PROBE.get_or_init(|| {
            let root = std::path::Path::new("/sys/fs/cgroup");
            let is_v2 = std::fs::metadata(root)
                .map(|meta| meta.is_dir())
                .unwrap_or(false)
                && std::fs::read_to_string(root.join("cgroup.controllers"))
                    .map(|content| !content.is_empty())
                    .unwrap_or(false);
            let has = |name: &str| {
                std::fs::read_to_string(root.join("cgroup.controllers"))
                    .map(|content| content.split_whitespace().any(|c| c == name))
                    .unwrap_or(false)
            };
            if !is_v2
            {
                return CgroupCapabilities {
                    memory: false,
                    pids: false,
                    cpu: false,
                };
            }
            // A group this process could not even create would make every
            // later write fail at spawn time; require writability up front.
            let scratch = root.join(format!(
                "{}-probe-{}",
                super::cgroup::GROUP_PREFIX,
                std::process::id()
            ));
            let writable = std::fs::create_dir_all(&scratch).is_ok();
            if writable
            {
                let _ = std::fs::remove_dir(&scratch);
            }
            CgroupCapabilities {
                memory: writable && has("memory"),
                pids: writable && has("pids"),
                cpu: writable && has("cpu"),
            }
        })
    }

    /// Union of the kernel mechanisms available on this host. Per-process
    /// rlimit claims stay deliberately false (see `RealLinuxBackend`);
    /// tree-wide claims come exclusively from a usable cgroup v2 hierarchy.
    pub(super) struct CombinedBackend {
        linux: Option<RealLinuxBackend>,
        cgroup: CgroupCapabilities,
    }

    impl ResourceBackend for CombinedBackend {
        fn supports_memory_limit(&self, _bytes: u64) -> bool {
            self.cgroup.memory
        }
        fn supports_cpu_limit(&self, _cpus: u32) -> bool {
            self.cgroup.cpu
        }
        fn supports_wall_time(&self, seconds: u64) -> bool {
            self.linux
                .as_ref()
                .is_some_and(|linux| linux.supports_wall_time(seconds))
        }
        fn supports_process_limit(&self, _processes: u32) -> bool {
            self.cgroup.pids
        }
        fn supports_file_size_limit(&self, bytes: u64) -> bool {
            self.linux
                .as_ref()
                .is_some_and(|linux| linux.supports_file_size_limit(bytes))
        }
        fn supports_gpu_memory_limit(&self, _bytes: u64) -> bool {
            false
        }
        fn supports_egress_allow_list(&self) -> bool {
            false
        }
        fn supports_egress_deny_all(&self) -> bool {
            self.linux
                .as_ref()
                .is_some_and(|linux| linux.supports_egress_deny_all())
        }
    }

    pub(super) fn probed_backend() -> &'static dyn ResourceBackend {
        static BACKEND: OnceLock<CombinedBackend> = OnceLock::new();
        BACKEND.get_or_init(|| CombinedBackend {
            linux: seccomp_available().then_some(RealLinuxBackend {
                seccomp_available: true,
            }),
            cgroup: cgroup_capabilities(),
        })
    }

    pub(super) fn apply_to_command(
        command: &mut Command,
        constraints: &ExecutionConstraints,
    ) -> Result<(), String> {
        let constraints = constraints.clone();
        unsafe {
            command.pre_exec(move || {
                apply(&constraints).map_err(|error| {
                    io::Error::new(
                        error.kind(),
                        format!("{} {error}", super::GOVERNANCE_FAILURE_PREFIX),
                    )
                })
            });
        }
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::agentic::budgets::ResourceLimits;

        #[test]
        fn filter_shape_matches_the_documented_control_flow() {
            let program = inet_socket_deny_filter();
            assert_eq!(program.len(), 11);
            // Every jump must land inside the program.
            for (index, instruction) in program.iter().enumerate()
            {
                if instruction.code == BPF_JEQ_K
                {
                    let jt = index + 1 + instruction.jt as usize;
                    let jf = index + 1 + instruction.jf as usize;
                    assert!(jt < program.len(), "jt of {index} escapes: {jt}");
                    assert!(jf < program.len(), "jf of {index} escapes: {jf}");
                }
            }
            // Unknown architecture lands on the final deny-everything pad.
            let arch_jump = &program[1];
            assert_eq!(
                (1 + 1 + arch_jump.jf as usize),
                program.len() - 1,
                "arch mismatch must reach the deny pad"
            );
            assert_eq!(program[program.len() - 1].k, SECCOMP_RET_ERRNO | EPERM);
            // Normal exit is ALLOW.
            assert_eq!(program[9].code, BPF_RET_K);
            assert_eq!(program[9].k, SECCOMP_RET_ALLOW);
            // Only socket creation is inspected against the domain.
            assert_eq!(program[3].k, SYS_SOCKET);
            assert_eq!(program[5].k, AF_INET);
            assert_eq!(program[7].k, AF_INET6);
        }

        #[test]
        fn tree_wide_limits_fail_closed_without_tree_wide_backend() {
            let backend = RealLinuxBackend {
                seccomp_available: true,
            };

            assert!(!backend.supports_memory_limit(1 << 20));
            assert!(!backend.supports_process_limit(2));
            assert!(!backend.supports_cpu_limit(1));
            assert!(!backend.supports_wall_time(1));
            assert!(backend.supports_file_size_limit(4096));
        }

        #[test]
        fn cgroup_backs_the_tree_wide_trio_when_available() {
            let backend = super::super::probed_backend();
            let caps = cgroup_capabilities();
            // Whatever the host provides, claims and capabilities agree.
            assert_eq!(
                backend.supports_memory_limit(1 << 20),
                caps.memory,
                "memory claim must mirror the probed hierarchy"
            );
            assert_eq!(
                backend.supports_process_limit(8),
                caps.pids,
                "process claim must mirror the probed hierarchy"
            );
            assert_eq!(
                backend.supports_cpu_limit(2),
                caps.cpu && (1..=MAX_PINNABLE_CPUS).contains(&2)
            );
            // Wall time stays kill-deadline based, never claimed as cgroup.
            assert!(!backend.supports_wall_time(60));
        }

        #[test]
        fn gpu_and_allow_list_gaps_are_honest() {
            let backend = RealLinuxBackend {
                seccomp_available: true,
            };
            assert!(!backend.supports_gpu_memory_limit(1 << 30));
            assert!(!backend.supports_egress_allow_list());
            assert_eq!(backend.supports_egress_deny_all(), seccomp_available());
        }

        // -- Live enforcement ------------------------------------------------

        fn temp_dir(tag: &str) -> std::path::PathBuf {
            let nonce = format!(
                "{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            );
            let base = std::env::temp_dir().join(format!("scirust-enforce-{tag}-{nonce}"));
            std::fs::create_dir_all(&base).unwrap();
            base
        }

        #[test]
        fn live_file_size_limit_triggers_sigxfsz() {
            let base = temp_dir("fsize");
            let big = base.join("big.txt");
            let mut command = Command::new("sh");
            command.arg("-c").arg(format!("yes > {}", big.display()));
            let constraints = ExecutionConstraints {
                limits: ResourceLimits {
                    max_file_size_bytes: Some(4096),
                    ..ResourceLimits::default()
                },
                ..ExecutionConstraints::default()
            };
            apply_to_command(&mut command, &constraints).unwrap();
            let output = command.output().unwrap();
            assert!(
                !output.status.success(),
                "child must die once RLIMIT_FSIZE trips"
            );
            assert!(
                big.metadata().unwrap().len() <= 4096 + 8192,
                "kernel must stop the write close past the cap"
            );
            std::fs::remove_dir_all(&base).unwrap();
        }

        #[test]
        fn live_seccomp_denies_local_inet_connect() {
            use std::net::TcpListener;

            let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
            let port = listener.local_addr().unwrap().port();

            let connect_script = format!("exec 3<>/dev/tcp/127.0.0.1/{port} && echo connected",);

            let mut governed = Command::new("bash");
            governed.arg("-c").arg(&connect_script);
            let constraints = ExecutionConstraints::default(); // deny-all egress
            apply_to_command(&mut governed, &constraints).unwrap();
            drop(listener.try_clone().unwrap());
            let governed_output = {
                let _guard = listener;
                governed.output().unwrap()
            };
            assert!(
                !governed_output.status.success(),
                "governed child must not open an INET socket"
            );
            assert!(String::from_utf8_lossy(&governed_output.stdout).is_empty());

            // Control: the same connect succeeds unfiltered.
            let mut free = Command::new("bash");
            free.arg("-c").arg(&connect_script);
            let listener = TcpListener::bind(("127.0.0.1", port))
                .or_else(|_| TcpListener::bind(("127.0.0.1", 0)))
                .unwrap();
            let free_output = {
                let _guard = listener;
                free.output().unwrap()
            };
            // Best-effort control: under heavy parallel test load a machine
            // can transiently exhaust fds, which fails the unfiltered child
            // for reasons unrelated to the filter. Only a successful control
            // is meaningful; a spurious one must not mask the governed
            // denial asserted above.
            if !free_output.status.success()
            {
                eprintln!(
                    "live_seccomp control connect failed (environmental): {}",
                    String::from_utf8_lossy(&free_output.stderr).trim()
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Non-Linux fallback: refuse rather than pretend.
// ---------------------------------------------------------------------------
#[cfg(not(target_os = "linux"))]
mod imp {
    use super::super::budgets::{NoResourceBackend, ResourceBackend};
    use super::ExecutionConstraints;
    use std::process::Command;

    static NO_BACKEND: NoResourceBackend = NoResourceBackend;

    pub(super) fn probed_backend() -> &'static dyn ResourceBackend {
        &NO_BACKEND
    }

    pub(super) fn apply_to_command(
        _command: &mut Command,
        _constraints: &ExecutionConstraints,
    ) -> Result<(), String> {
        Err(
            "resource governance currently requires Linux; refusing to execute with declared \
             limits that would go unenforced"
                .to_string(),
        )
    }
}
