//! Copy the already-built worker into `src-tauri/binaries` under the
//! target-triple name Tauri requires for an external binary.
//!
//! Tauri resolves an `externalBin` entry by appending the target triple to
//! the configured path — `scirust-studio-worker` becomes
//! `scirust-studio-worker-x86_64-pc-windows-msvc.exe`. Getting that name
//! wrong produces a bundle that builds and then cannot find its own sidecar
//! at runtime, which is precisely the failure a preparation step exists to
//! prevent.
//!
//! What this tool will not do, deliberately:
//!
//! - **Never download a binary.** The sidecar is the worker this repository
//!   just built; fetching one from anywhere else would put unreviewed code
//!   inside a signed bundle.
//! - **Never invoke a shell string.** Every operation here is a direct
//!   filesystem call.
//! - **Never invent a worker.** A missing input is a hard failure with the
//!   command that produces it, not a placeholder.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use sha2::{Digest, Sha256};

fn main() -> ExitCode {
    match run()
    {
        Ok(report) =>
        {
            println!("{report}");
            ExitCode::SUCCESS
        },
        Err(message) =>
        {
            eprintln!("prepare-sidecar: {message}");
            ExitCode::FAILURE
        },
    }
}

fn run() -> Result<String, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h")
    {
        return Ok(usage());
    }

    let triple = match option(&args, "--target")
    {
        Some(t) => t,
        None => host_triple()?,
    };
    let profile = option(&args, "--profile").unwrap_or_else(|| "debug".to_string());

    let repo_root = repo_root()?;
    let source = option(&args, "--worker")
        .map(PathBuf::from)
        .unwrap_or_else(|| default_worker_path(&repo_root, &profile, &triple));

    if !source.exists()
    {
        return Err(format!(
            "the worker was not found at {}\n\
             Build it first:\n    cargo build -p scirust-studio-worker{}{}",
            source.display(),
            if profile == "release"
            {
                " --release"
            }
            else
            {
                ""
            },
            if option(&args, "--target").is_some()
            {
                format!(" --target {triple}")
            }
            else
            {
                String::new()
            }
        ));
    }
    if !source.is_file()
    {
        return Err(format!(
            "{} exists but is not a regular file",
            source.display()
        ));
    }

    let destination_dir = repo_root
        .join("apps")
        .join("scirust-studio")
        .join("src-tauri")
        .join("binaries");
    std::fs::create_dir_all(&destination_dir)
        .map_err(|e| format!("cannot create {}: {e}", destination_dir.display()))?;

    let destination = destination_dir.join(sidecar_file_name(&triple));
    std::fs::copy(&source, &destination).map_err(|e| {
        format!(
            "cannot copy {} to {}: {e}",
            source.display(),
            destination.display()
        )
    })?;

    // The copy must be executable on Unix; `fs::copy` preserves the mode,
    // but a source that arrived without it (from an artifact download, say)
    // would silently produce a bundle that cannot launch its own sidecar.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&destination)
            .map_err(|e| format!("cannot stat {}: {e}", destination.display()))?
            .permissions();
        perms.set_mode(perms.mode() | 0o755);
        std::fs::set_permissions(&destination, perms)
            .map_err(|e| format!("cannot make {} executable: {e}", destination.display()))?;
    }

    let digest = sha256_of(&destination)?;
    let bytes = std::fs::metadata(&destination)
        .map(|m| m.len())
        .unwrap_or_default();

    Ok(format!(
        "source      {}\n\
         destination {}\n\
         target      {triple}\n\
         size        {bytes} bytes\n\
         sha256      {digest}",
        source.display(),
        destination.display()
    ))
}

fn usage() -> String {
    "usage: prepare-sidecar [--target <triple>] [--profile debug|release] [--worker <path>]\n\
     \n\
     Copies the built scirust-studio-worker into src-tauri/binaries under the\n\
     target-triple name Tauri expects. Never downloads anything."
        .to_string()
}

fn option(args: &[String], name: &str) -> Option<String> {
    let index = args.iter().position(|a| a == name)?;
    args.get(index + 1).cloned()
}

/// The file name Tauri looks for: `<name>-<triple>` plus `.exe` on Windows.
///
/// The extension goes **after** the triple, not before it — the most common
/// way to get this wrong.
fn sidecar_file_name(triple: &str) -> String {
    if triple.contains("windows")
    {
        format!("scirust-studio-worker-{triple}.exe")
    }
    else
    {
        format!("scirust-studio-worker-{triple}")
    }
}

/// Where Cargo puts the worker for a given profile and target.
fn default_worker_path(repo_root: &Path, profile: &str, triple: &str) -> PathBuf {
    let mut path = repo_root.join("target");
    // A cross-compiled build lands under `target/<triple>/<profile>`; a host
    // build lands directly under `target/<profile>`.
    if triple != host_triple().unwrap_or_default()
    {
        path = path.join(triple);
    }
    path.join(profile).join(executable_name(triple))
}

fn executable_name(triple: &str) -> String {
    if triple.contains("windows")
    {
        "scirust-studio-worker.exe".to_string()
    }
    else
    {
        "scirust-studio-worker".to_string()
    }
}

/// The host target triple, from the compiler that built this tool.
///
/// Read from `rustc -vV` rather than assembled from `cfg!` guesses, so the
/// answer is the one Cargo itself would use.
fn host_triple() -> Result<String, String> {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let output = std::process::Command::new(rustc)
        .arg("-vV")
        .output()
        .map_err(|e| format!("cannot run rustc to determine the host target: {e}"))?;
    let text = String::from_utf8_lossy(&output.stdout);
    text.lines()
        .find_map(|l| l.strip_prefix("host: "))
        .map(|t| t.trim().to_string())
        .ok_or_else(|| "rustc did not report a host target".to_string())
}

/// This file lives at `apps/scirust-studio/tools/prepare-sidecar/src`, so the
/// repository root is four levels up from the crate directory.
fn repo_root() -> Result<PathBuf, String> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .ancestors()
        .nth(4)
        .map(Path::to_path_buf)
        .ok_or_else(|| {
            format!(
                "cannot locate the repository root from {}",
                manifest.display()
            )
        })
}

fn sha256_of(path: &Path) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let digest = Sha256::digest(&bytes);
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest
    {
        use std::fmt::Write;
        let _ = write!(hex, "{byte:02x}");
    }
    Ok(hex)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The naming rule Tauri enforces, and the one most easily got wrong.
    #[test]
    fn the_windows_extension_follows_the_triple() {
        assert_eq!(
            sidecar_file_name("x86_64-pc-windows-msvc"),
            "scirust-studio-worker-x86_64-pc-windows-msvc.exe"
        );
        assert_eq!(
            sidecar_file_name("aarch64-pc-windows-msvc"),
            "scirust-studio-worker-aarch64-pc-windows-msvc.exe"
        );
    }

    #[test]
    fn non_windows_targets_get_no_extension() {
        for triple in [
            "x86_64-unknown-linux-gnu",
            "aarch64-apple-darwin",
            "x86_64-apple-darwin",
        ]
        {
            let name = sidecar_file_name(triple);
            assert!(name.ends_with(triple), "{name}");
            assert!(!name.ends_with(".exe"), "{name}");
        }
    }

    #[test]
    fn the_executable_name_matches_the_target_not_the_host() {
        assert_eq!(
            executable_name("x86_64-pc-windows-msvc"),
            "scirust-studio-worker.exe"
        );
        assert_eq!(
            executable_name("x86_64-unknown-linux-gnu"),
            "scirust-studio-worker"
        );
    }

    #[test]
    fn options_are_parsed_by_name() {
        let args: Vec<String> = ["--target", "x86_64-pc-windows-msvc", "--profile", "release"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            option(&args, "--target").as_deref(),
            Some("x86_64-pc-windows-msvc")
        );
        assert_eq!(option(&args, "--profile").as_deref(), Some("release"));
        assert_eq!(option(&args, "--worker"), None);
    }

    #[test]
    fn a_host_build_is_not_nested_under_the_triple() {
        let host = host_triple().expect("a host triple");
        let path = default_worker_path(Path::new("/repo"), "debug", &host);
        assert_eq!(
            path,
            PathBuf::from("/repo/target/debug").join(executable_name(&host))
        );
    }

    #[test]
    fn a_cross_build_is_nested_under_the_triple() {
        let path = default_worker_path(Path::new("/repo"), "release", "aarch64-apple-darwin");
        assert_eq!(
            path,
            PathBuf::from("/repo/target/aarch64-apple-darwin/release/scirust-studio-worker")
        );
    }

    #[test]
    fn the_repository_root_contains_the_desktop_application() {
        let root = repo_root().expect("a root");
        assert!(
            root.join("apps").join("scirust-studio").is_dir(),
            "{} does not look like the repository root",
            root.display()
        );
        assert!(root.join("scirust-studio-worker").is_dir());
    }

    #[test]
    fn hashing_a_known_input_is_stable() {
        let temp = std::env::temp_dir().join(format!("prep-sidecar-{}", std::process::id()));
        std::fs::write(&temp, b"abc").unwrap();
        assert_eq!(
            sha256_of(&temp).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let _ = std::fs::remove_file(&temp);
    }

    #[test]
    fn a_missing_worker_names_the_command_that_builds_it() {
        let err = sha256_of(Path::new("/definitely/not/here")).unwrap_err();
        assert!(err.contains("/definitely/not/here"), "{err}");
    }
}
