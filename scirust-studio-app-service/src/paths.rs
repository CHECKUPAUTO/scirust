//! Where the application keeps its data.
//!
//! Implemented directly from the platform conventions rather than through a
//! dependency, because the rule is three lines per platform and the
//! *decision* — which directory, and why not another one — is the part worth
//! writing down.

use std::path::PathBuf;

use crate::error::AppServiceError;

/// The vendor directory component.
pub const VENDOR: &str = "Memorithm";
/// The application directory component.
pub const APPLICATION: &str = "SciRust Studio";

/// The per-user application data directory for this platform.
///
/// - **Windows**: `%APPDATA%\Memorithm\SciRust Studio`
/// - **macOS**: `~/Library/Application Support/Memorithm/SciRust Studio`
/// - **Linux/other**: `$XDG_DATA_HOME/Memorithm/SciRust Studio`, or
///   `~/.local/share/...` when `XDG_DATA_HOME` is unset
///
/// Deliberately **not** the installation directory (which may be read-only,
/// is shared between users, and is replaced wholesale by an installer) and
/// **not** the user's Documents folder (which belongs to the user, not to
/// this application — a program that writes run history there without being
/// asked is misbehaving).
pub fn application_data_dir() -> Result<PathBuf, AppServiceError> {
    let base = platform_data_root()?;
    Ok(base.join(VENDOR).join(APPLICATION))
}

fn platform_data_root() -> Result<PathBuf, AppServiceError> {
    let from_env = |key: &str| {
        std::env::var_os(key)
            .map(PathBuf::from)
            .filter(|p| !p.as_os_str().is_empty())
    };

    if cfg!(windows)
    {
        return from_env("APPDATA")
            .or_else(|| from_env("LOCALAPPDATA"))
            .ok_or(AppServiceError::NoApplicationDirectory {
                reason: "neither APPDATA nor LOCALAPPDATA is set".to_string(),
            });
    }

    let home = from_env("HOME").ok_or(AppServiceError::NoApplicationDirectory {
        reason: "HOME is not set".to_string(),
    })?;

    if cfg!(target_os = "macos")
    {
        Ok(home.join("Library").join("Application Support"))
    }
    else
    {
        Ok(from_env("XDG_DATA_HOME").unwrap_or_else(|| home.join(".local").join("share")))
    }
}

/// The standard sub-directories the application maintains under its data
/// directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppDirectories {
    /// Root of everything below.
    pub root: PathBuf,
    /// User preferences.
    pub settings: PathBuf,
    /// The immutable run store's root.
    pub runs: PathBuf,
    /// Diagnostic logs.
    pub logs: PathBuf,
    /// Evidence left by interrupted work.
    pub recovery: PathBuf,
}

impl AppDirectories {
    /// Derive the layout under `root` without touching the filesystem.
    pub fn under(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        AppDirectories {
            settings: root.join("settings"),
            runs: root.join("runs"),
            logs: root.join("logs"),
            recovery: root.join("recovery"),
            root,
        }
    }

    /// The layout under the platform's application data directory.
    pub fn platform_default() -> Result<Self, AppServiceError> {
        Ok(AppDirectories::under(application_data_dir()?))
    }

    /// Create every directory that does not exist yet.
    pub fn create_all(&self) -> Result<(), AppServiceError> {
        for dir in [
            &self.root,
            &self.settings,
            &self.runs,
            &self.logs,
            &self.recovery,
        ]
        {
            std::fs::create_dir_all(dir).map_err(|e| AppServiceError::Io {
                path: dir.clone(),
                source: e.to_string(),
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_layout_is_derived_from_its_root() {
        let dirs = AppDirectories::under("/tmp/example");
        assert_eq!(dirs.root, PathBuf::from("/tmp/example"));
        assert!(dirs.runs.ends_with("runs"));
        assert!(dirs.settings.ends_with("settings"));
        assert!(dirs.logs.ends_with("logs"));
        assert!(dirs.recovery.ends_with("recovery"));
    }

    #[test]
    fn create_all_makes_every_directory() {
        let temp = tempfile::tempdir().unwrap();
        let dirs = AppDirectories::under(temp.path().join("studio"));
        dirs.create_all().unwrap();
        assert!(dirs.root.is_dir());
        assert!(dirs.settings.is_dir());
        assert!(dirs.runs.is_dir());
        assert!(dirs.logs.is_dir());
        assert!(dirs.recovery.is_dir());
    }

    #[test]
    fn create_all_is_idempotent() {
        let temp = tempfile::tempdir().unwrap();
        let dirs = AppDirectories::under(temp.path().join("studio"));
        dirs.create_all().unwrap();
        dirs.create_all().expect("running twice is not an error");
    }

    /// The path must be under the platform's data area and carry both the
    /// vendor and the application name, so an installation is identifiable
    /// and uninstallable.
    #[test]
    fn the_platform_directory_is_vendor_and_application_qualified() {
        // Only meaningful when the environment provides a home; a CI
        // container without one legitimately has nowhere to put this.
        let Ok(dir) = application_data_dir()
        else
        {
            return;
        };
        let text = dir.to_string_lossy().into_owned();
        assert!(text.contains(VENDOR), "{text}");
        assert!(text.contains(APPLICATION), "{text}");
    }

    #[test]
    fn the_platform_directory_is_absolute() {
        let Ok(dir) = application_data_dir()
        else
        {
            return;
        };
        assert!(dir.is_absolute(), "{}", dir.display());
    }
}
