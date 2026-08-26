//! Real cgroup v2 enforcement for declared resource limits.
//!
//! rlimits are per-process; a cgroup bounds the WHOLE spawned tree
//! aggregate. This module creates one throwaway subgroup per governed
//! execution, writes the enforceable subset of the declared limits, and the
//! sandbox's `pre_exec` hook moves the child into it before exec.
//!
//! Enforced here: `max_memory_bytes` → `memory.max` (tree-wide),
//! `max_processes` → `pids.max`, `max_cpus` → `cpu.max` quota.
//! Deliberately NOT claimable: GPU memory (no mainline controller), wall
//! time (runtime kill deadline) and file size (rlimit) keep their existing
//! mechanisms.
//!
//! Everything fails closed: an unusable hierarchy reports "not available"
//! (per-process limits still apply through the rlimit backend), while an
//! existing hierarchy that REFUSES a declared limit is an error.

use std::path::PathBuf;

/// Root of the unified cgroup v2 hierarchy.
const CGROUP_ROOT: &str = "/sys/fs/cgroup";

/// Directory prefix for throwaway groups this crate creates.
pub(super) const GROUP_PREFIX: &str = "scirust-governed";

pub struct V2Group {
    path: PathBuf,
}

impl std::fmt::Debug for V2Group {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("V2Group").field("path", &self.path).finish()
    }
}

fn read_to_string(path: &std::path::Path) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

fn hierarchy_is_v2() -> bool {
    std::fs::metadata(CGROUP_ROOT)
        .map(|meta| meta.file_type().is_dir())
        .unwrap_or(false)
        && read_to_string(&PathBuf::from(CGROUP_ROOT).join("cgroup.controllers"))
            .map(|content| !content.is_empty())
            .unwrap_or(false)
}

fn root_controllers() -> Vec<String> {
    read_to_string(&PathBuf::from(CGROUP_ROOT).join("cgroup.controllers"))
        .map(|content| content.split_whitespace().map(str::to_string).collect())
        .unwrap_or_default()
}

impl V2Group {
    /// Create a throwaway group and write the enforceable subset of the
    /// declared limits.
    ///
    /// Returns:
    /// - `Ok(Some(group))` — hierarchy usable, requested tree-wide limits
    ///   written;
    /// - `Ok(None)` — no cgroup v2 hierarchy (or missing controllers):
    ///   per-process rlimits remain the only mechanism;
    /// - `Err` — hierarchy exists but refused a DECLARED limit (fail closed).
    pub fn prepare(limits: &super::super::budgets::ResourceLimits) -> Result<Option<Self>, String> {
        let controllers = if hierarchy_is_v2()
        {
            root_controllers()
        }
        else
        {
            return Ok(None);
        };
        let has = |name: &str| controllers.iter().any(|c| c == name);

        // Only create a group when at least one tree-wide limit was declared
        // AND its controller exists; otherwise rlimits suffice.
        let wants_memory = limits.max_memory_bytes.is_some();
        let wants_pids = limits.max_processes.is_some();
        let wants_cpu = limits.max_cpus.is_some();
        let needed: Vec<(&str, bool)> = vec![
            ("memory", wants_memory),
            ("pids", wants_pids),
            ("cpu", wants_cpu),
        ];
        if !needed.iter().any(|(_, wanted)| *wanted)
        {
            return Ok(None);
        }
        for (controller, wanted) in &needed
        {
            if *wanted && !has(controller)
            {
                return Err(format!(
                    "cgroup v2 hierarchy lacks the `{controller}` controller required by a \
                     declared limit; refusing unbounded execution"
                ));
            }
        }

        let unique = format!(
            "{GROUP_PREFIX}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        );
        let path = PathBuf::from(CGROUP_ROOT).join(unique);
        std::fs::create_dir_all(&path)
            .map_err(|error| format!("cannot create cgroup {}: {error}", path.display()))?;

        let group = Self { path };
        let write = |name: &str, value: String| -> Result<(), String> {
            std::fs::write(group.path.join(name), value).map_err(|error| {
                format!("cannot write {} in {}: {error}", name, group.path.display())
            })
        };
        if let Some(bytes) = limits.max_memory_bytes
        {
            write("memory.max", format!("{bytes}"))?;
        }
        if let Some(processes) = limits.max_processes
        {
            write("pids.max", format!("{processes}"))?;
        }
        if let Some(cpus) = limits.max_cpus
        {
            // cpu.max: "$QUOTA $PERIOD" — pin the quota to cpus*period with a
            // 100 ms period so short children still get scheduled fairly.
            const PERIOD_US: u64 = 100_000;
            let quota = u64::from(cpus).saturating_mul(PERIOD_US);
            write("cpu.max", format!("{quota} {PERIOD_US}"))?;
        }
        Ok(Some(group))
    }

    /// Move the about-to-exec child into this group. Runs inside `pre_exec`,
    /// after fork and before exec, where getpid() is exactly the future pid.
    pub fn attach_to_command(&self, command: &mut std::process::Command) -> Result<(), String> {
        use std::os::unix::process::CommandExt;
        let procs = self.path.join("cgroup.procs");
        unsafe {
            command.pre_exec(move || {
                let pid = std::process::id();
                std::fs::write(&procs, format!("{pid}")).map_err(std::io::Error::other)?;
                Ok(())
            });
        }
        Ok(())
    }

    /// Current aggregate memory consumption of the group, for tests.
    pub fn memory_current_bytes(&self) -> Result<u64, String> {
        std::fs::read_to_string(self.path.join("memory.current"))
            .map_err(|e| e.to_string())?
            .trim()
            .parse()
            .map_err(|e| format!("memory.current unparsable: {e}"))
    }

    fn remove(&self) {
        let _ = std::fs::remove_dir(&self.path);
    }
}

impl Drop for V2Group {
    fn drop(&mut self) {
        self.remove();
    }
}

#[cfg(target_os = "linux")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentic::budgets::ResourceLimits;

    fn writable_hierarchy() -> bool {
        // The live tests need to CREATE groups under the root: probe cheaply.
        if !hierarchy_is_v2()
        {
            return false;
        }
        let probe =
            PathBuf::from(CGROUP_ROOT).join(format!("{GROUP_PREFIX}-probe-{}", std::process::id()));
        match std::fs::create_dir_all(&probe)
        {
            Ok(()) =>
            {
                let _ = std::fs::remove_dir(&probe);
                true
            },
            Err(_) => false,
        }
    }

    #[test]
    fn no_declared_tree_limits_creates_no_group() {
        if !writable_hierarchy()
        {
            return;
        }
        assert!(
            V2Group::prepare(&ResourceLimits::default())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn missing_controller_for_a_declared_limit_fails_closed() {
        if !writable_hierarchy()
        {
            return;
        }
        let controllers = root_controllers();
        if controllers.iter().any(|c| c == "memory")
        {
            return; // only meaningful when memory is genuinely absent
        }
        let limits = ResourceLimits {
            max_memory_bytes: Some(1 << 20),
            ..ResourceLimits::default()
        };
        let error = V2Group::prepare(&limits).expect_err("missing controller must refuse");
        assert!(error.contains("refusing"), "{error}");
    }

    #[test]
    fn declared_limits_are_written_and_enforced() {
        if !writable_hierarchy()
        {
            return;
        }
        let controllers = root_controllers();
        let has_memory = controllers.iter().any(|c| c == "memory");
        let has_pids = controllers.iter().any(|c| c == "pids");

        let mut limits = ResourceLimits::default();
        if has_memory
        {
            limits.max_memory_bytes = Some(64 << 20);
        }
        // Two is deliberately below what the probe child needs (shell plus
        // two background sleeps): the second fork must trip the ceiling.
        if has_pids
        {
            limits.max_processes = Some(2);
        }
        if !has_memory && !has_pids
        {
            return;
        }

        let group = V2Group::prepare(&limits)
            .expect("writable hierarchy must accept declared limits")
            .expect("some limits were declared");

        if has_memory
        {
            let written = std::fs::read_to_string(group.path.join("memory.max")).unwrap();
            assert_eq!(written.trim(), (64 << 20).to_string());
        }
        if has_pids
        {
            let written = std::fs::read_to_string(group.path.join("pids.max")).unwrap();
            assert_eq!(written.trim(), "2");
        }

        // Live enforcement: a child moved into the group cannot exceed the
        // process-number ceiling.
        if has_pids
        {
            use std::os::unix::process::CommandExt;
            use std::process::{Command, Stdio};
            let procs = group.path.join("cgroup.procs");
            let procs_clone = procs.clone();
            let mut command = Command::new("sh");
            command.arg("-c").arg("sleep 0.2 & sleep 0.2 & wait");
            unsafe {
                command.pre_exec(move || {
                    let pid = std::process::id();
                    std::fs::write(&procs_clone, format!("{pid}"))
                        .map_err(std::io::Error::other)?;
                    Ok(())
                });
            }
            command.stdout(Stdio::null()).stderr(Stdio::null());
            let status = command.status().unwrap();
            assert!(!status.success(), "child exceeding pids.max must fail");
        }

        drop(group); // cleanup path exercised implicitly
    }
}
