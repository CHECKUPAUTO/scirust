use super::{SandboxEnforcement, SandboxMode};
use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Support {
    pub(super) abi: u32,
    pub(super) enforcement: SandboxEnforcement,
}

pub(super) fn probe() -> Option<Support> {
    imp::probe()
}

pub(super) fn configure_command(
    command: &mut Command,
    mode: SandboxMode,
    workspace_root: &Path,
    support: Support,
) -> Result<(), String> {
    imp::configure_command(command, mode, workspace_root, support)
}

#[cfg(target_os = "linux")]
mod imp {
    use super::{SandboxEnforcement, SandboxMode, Support};
    use std::ffi::CString;
    use std::io;
    use std::mem::size_of;
    use std::os::raw::{c_char, c_int, c_long, c_uint, c_ulong};
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::process::CommandExt;
    use std::path::Path;
    use std::process::Command;

    const LANDLOCK_CREATE_RULESET_VERSION: c_uint = 1;
    const LANDLOCK_RULE_PATH_BENEATH: c_int = 1;
    const SYS_LANDLOCK_CREATE_RULESET: c_long = 444;
    const SYS_LANDLOCK_ADD_RULE: c_long = 445;
    const SYS_LANDLOCK_RESTRICT_SELF: c_long = 446;
    const PR_SET_NO_NEW_PRIVS: c_int = 38;
    const O_CLOEXEC: c_int = 0o2_000_000;
    const O_PATH: c_int = 0o10_000_000;
    const MIN_STRICT_ABI: u32 = 3;
    const MAX_ABI: u32 = 5;

    const FS_EXECUTE: u64 = 1 << 0;
    const FS_WRITE_FILE: u64 = 1 << 1;
    const FS_READ_FILE: u64 = 1 << 2;
    const FS_READ_DIR: u64 = 1 << 3;
    const FS_REFER: u64 = 1 << 13;
    const FS_TRUNCATE: u64 = 1 << 14;
    const FS_IOCTL_DEV: u64 = 1 << 15;
    const ABI1_MASK: u64 = FS_REFER - 1;
    const READ_SIDE: u64 = FS_EXECUTE | FS_READ_FILE | FS_READ_DIR;
    const FILE_COMPATIBLE: u64 =
        FS_EXECUTE | FS_WRITE_FILE | FS_READ_FILE | FS_TRUNCATE | FS_IOCTL_DEV;

    #[repr(C)]
    struct RulesetAttr {
        handled_access_fs: u64,
    }

    #[repr(C, packed)]
    struct PathBeneathAttr {
        allowed_access: u64,
        parent_fd: i32,
    }

    struct Rule {
        path: CString,
        access: u64,
    }

    struct Plan {
        handled: u64,
        rules: Vec<Rule>,
    }

    unsafe extern "C" {
        fn syscall(number: c_long, ...) -> c_long;
        fn open(pathname: *const c_char, flags: c_int, ...) -> c_int;
        fn close(fd: c_int) -> c_int;
        fn prctl(option: c_int, ...) -> c_int;
    }

    pub(super) fn probe() -> Option<Support> {
        let abi = unsafe {
            syscall(
                SYS_LANDLOCK_CREATE_RULESET,
                std::ptr::null::<RulesetAttr>(),
                0usize,
                LANDLOCK_CREATE_RULESET_VERSION,
            )
        };
        if abi < 1
        {
            return None;
        }
        let abi = u32::try_from(abi).ok()?;
        if abi < MIN_STRICT_ABI
        {
            return None;
        }
        Some(Support {
            abi,
            enforcement: enforcement_for_abi(abi),
        })
    }

    pub(super) fn configure_command(
        command: &mut Command,
        mode: SandboxMode,
        workspace_root: &Path,
        support: Support,
    ) -> Result<(), String> {
        if !mode.is_confined()
        {
            return Err("Landlock may only be installed for confined sandbox modes".to_string());
        }
        if support.abi < MIN_STRICT_ABI
        {
            return Err(format!(
                "Landlock ABI {} is too old for strict confined semantics; ABI {MIN_STRICT_ABI}+ is required",
                support.abi
            ));
        }
        if matches!(mode, SandboxMode::WorkspaceWrite)
        {
            let private_tmp = workspace_root.join("target/.sciagent-tmp");
            std::fs::create_dir_all(&private_tmp).map_err(|error| {
                format!(
                    "cannot prepare workspace-local TMPDIR `{}` for Landlock: {error}",
                    private_tmp.display()
                )
            })?;
            command.env("TMPDIR", private_tmp);
        }
        let plan = build_plan(mode, workspace_root, support.abi)?;
        unsafe {
            command.pre_exec(move || apply_plan(&plan));
        }
        Ok(())
    }

    fn enforcement_for_abi(abi: u32) -> SandboxEnforcement {
        if abi < MAX_ABI
        {
            SandboxEnforcement::Partial
        }
        else
        {
            SandboxEnforcement::Full
        }
    }

    fn fs_mask_for_abi(abi: u32) -> u64 {
        let mut mask = ABI1_MASK;
        if abi >= 2
        {
            mask |= FS_REFER;
        }
        if abi >= 3
        {
            mask |= FS_TRUNCATE;
        }
        if abi >= 5
        {
            mask |= FS_IOCTL_DEV;
        }
        mask
    }

    fn c_path(path: &Path) -> Result<CString, String> {
        CString::new(path.as_os_str().as_bytes())
            .map_err(|_| format!("Landlock grant path contains NUL: {}", path.display()))
    }

    fn rule(path: &Path, access: u64) -> Result<Rule, String> {
        Ok(Rule {
            path: c_path(path)?,
            access,
        })
    }

    fn build_plan(mode: SandboxMode, workspace_root: &Path, abi: u32) -> Result<Plan, String> {
        if abi < MIN_STRICT_ABI
        {
            return Err(format!(
                "Landlock ABI {abi} cannot enforce strict truncate semantics; ABI {MIN_STRICT_ABI}+ is required"
            ));
        }
        let handled = fs_mask_for_abi(abi.min(MAX_ABI));
        let mut rules = vec![
            rule(Path::new("/"), READ_SIDE & handled)?,
            rule(Path::new("/dev/null"), FILE_COMPATIBLE & handled)?,
        ];
        if matches!(mode, SandboxMode::WorkspaceWrite)
        {
            rules.push(rule(workspace_root, handled)?);
        }
        Ok(Plan { handled, rules })
    }

    fn apply_plan(plan: &Plan) -> io::Result<()> {
        let attr = RulesetAttr {
            handled_access_fs: plan.handled,
        };
        let ruleset_fd = unsafe {
            syscall(
                SYS_LANDLOCK_CREATE_RULESET,
                &attr as *const RulesetAttr,
                size_of::<RulesetAttr>(),
                0u32,
            )
        };
        if ruleset_fd < 0
        {
            return Err(io::Error::last_os_error());
        }
        let ruleset_fd = ruleset_fd as c_int;

        for rule in &plan.rules
        {
            let path_fd = unsafe { open(rule.path.as_ptr(), O_PATH | O_CLOEXEC) };
            if path_fd < 0
            {
                let error = io::Error::last_os_error();
                unsafe {
                    close(ruleset_fd);
                }
                return Err(error);
            }
            let path_attr = PathBeneathAttr {
                allowed_access: rule.access,
                parent_fd: path_fd,
            };
            let added = unsafe {
                syscall(
                    SYS_LANDLOCK_ADD_RULE,
                    ruleset_fd,
                    LANDLOCK_RULE_PATH_BENEATH,
                    &path_attr as *const PathBeneathAttr,
                    0u32,
                )
            };
            let add_error = if added != 0
            {
                Some(io::Error::last_os_error())
            }
            else
            {
                None
            };
            unsafe {
                close(path_fd);
            }
            if let Some(error) = add_error
            {
                unsafe {
                    close(ruleset_fd);
                }
                return Err(error);
            }
        }

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
            let error = io::Error::last_os_error();
            unsafe {
                close(ruleset_fd);
            }
            return Err(error);
        }

        let restricted = unsafe { syscall(SYS_LANDLOCK_RESTRICT_SELF, ruleset_fd, 0u32) };
        let restrict_error = if restricted != 0
        {
            Some(io::Error::last_os_error())
        }
        else
        {
            None
        };
        unsafe {
            close(ruleset_fd);
        }
        if let Some(error) = restrict_error
        {
            return Err(error);
        }
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn abi_masks_match_the_harness_contract() {
            assert_eq!(fs_mask_for_abi(1), (1 << 13) - 1);
            assert_eq!(fs_mask_for_abi(2), (1 << 14) - 1);
            assert_eq!(fs_mask_for_abi(3), (1 << 15) - 1);
            assert_eq!(fs_mask_for_abi(4), (1 << 15) - 1);
            assert_eq!(fs_mask_for_abi(5), (1 << 16) - 1);
            assert_eq!(fs_mask_for_abi(99), (1 << 16) - 1);
        }

        #[test]
        fn strict_fallback_requires_abi_three_or_newer() {
            assert!(build_plan(SandboxMode::ReadOnly, Path::new("/workspace"), 1).is_err());
            assert!(build_plan(SandboxMode::ReadOnly, Path::new("/workspace"), 2).is_err());
            assert!(build_plan(SandboxMode::ReadOnly, Path::new("/workspace"), 3).is_ok());
        }

        #[test]
        fn enforcement_is_partial_before_abi_five() {
            assert_eq!(enforcement_for_abi(3), SandboxEnforcement::Partial);
            assert_eq!(enforcement_for_abi(4), SandboxEnforcement::Partial);
            assert_eq!(enforcement_for_abi(5), SandboxEnforcement::Full);
            assert_eq!(enforcement_for_abi(9), SandboxEnforcement::Full);
        }

        #[test]
        fn read_only_plan_grants_root_read_and_dev_null_write() {
            let plan = build_plan(SandboxMode::ReadOnly, Path::new("/workspace"), 5).unwrap();
            assert_eq!(plan.rules.len(), 2);
            assert_eq!(plan.rules[0].path.to_bytes(), b"/");
            assert_eq!(plan.rules[0].access, READ_SIDE);
            assert_eq!(plan.rules[1].path.to_bytes(), b"/dev/null");
            assert_eq!(plan.rules[1].access, FILE_COMPATIBLE);
        }

        #[test]
        fn workspace_write_grants_only_the_workspace_beyond_dev_null() {
            let plan = build_plan(SandboxMode::WorkspaceWrite, Path::new("/workspace"), 5).unwrap();
            assert_eq!(plan.rules.len(), 3);
            assert_eq!(plan.rules[2].path.to_bytes(), b"/workspace");
            assert_eq!(plan.rules[2].access, plan.handled);
        }

        #[test]
        fn live_landlock_enforces_workspace_boundary_when_supported() {
            let Some(support) = probe()
            else
            {
                return;
            };
            assert!(support.abi >= MIN_STRICT_ABI);

            let nonce = format!(
                "{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            );
            let base = std::env::temp_dir().join(format!("scirust-landlock-{nonce}"));
            let workspace = base.join("workspace");
            let outside = base.join("outside.txt");
            let inside = workspace.join("inside.txt");
            std::fs::create_dir_all(&workspace).unwrap();
            std::fs::write(&inside, b"seed").unwrap();

            let write_command = |path: &Path, mode: SandboxMode| {
                let mut command = Command::new("sh");
                command.args(["-c", r#"printf '%s' "$1" > "$2""#, "sh", "payload"]);
                command.arg(path);
                configure_command(&mut command, mode, &workspace, support).unwrap();
                command.output().unwrap()
            };

            let read_only = write_command(&inside, SandboxMode::ReadOnly);
            assert!(!read_only.status.success());
            assert_eq!(std::fs::read(&inside).unwrap(), b"seed");

            let workspace_write = write_command(&inside, SandboxMode::WorkspaceWrite);
            assert!(workspace_write.status.success());
            assert_eq!(std::fs::read(&inside).unwrap(), b"payload");

            let outside_write = write_command(&outside, SandboxMode::WorkspaceWrite);
            assert!(!outside_write.status.success());
            assert!(!outside.exists());

            std::fs::remove_dir_all(&base).unwrap();
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    use super::{SandboxMode, Support};
    use std::path::Path;
    use std::process::Command;

    pub(super) fn probe() -> Option<Support> {
        None
    }

    pub(super) fn configure_command(
        _command: &mut Command,
        _mode: SandboxMode,
        _workspace_root: &Path,
        _support: Support,
    ) -> Result<(), String> {
        Err("Landlock is available only on Linux".to_string())
    }
}
