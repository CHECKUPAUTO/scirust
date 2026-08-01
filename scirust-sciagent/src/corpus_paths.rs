//! Stable locations and safety checks for generated SCIAGENT training data.
//!
//! Training corpora are large, mutable machine-local artifacts.  They must not
//! live below the source checkout: tools such as `grep -R` and `find` do not
//! honour ignore files and would otherwise traverse the corpus and its
//! aggregate symlinks.  This module centralises the external storage policy
//! used by the corpus producer and shard collector.

use std::env;
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};

/// Environment variable that overrides the external raw-corpus directory.
pub const CORPUS_DIR_ENV: &str = "SCIRUST_SCIAGENT_CORPUS_DIR";

/// Environment variable that overrides the external packed-shards directory.
pub const SHARDS_DIR_ENV: &str = "SCIRUST_SCIAGENT_SHARDS_DIR";

/// Return the platform-appropriate default for the downloaded crate corpus.
pub fn default_corpus_dir() -> io::Result<PathBuf> {
    Ok(application_data_dir()?
        .join("scirust")
        .join("sciagent")
        .join("crates_raw"))
}

/// Return the platform-appropriate default for packed training shards.
pub fn default_shards_dir() -> io::Result<PathBuf> {
    Ok(application_data_dir()?
        .join("scirust")
        .join("sciagent")
        .join("shards"))
}

/// Resolve a raw-corpus location and reject a path inside the source checkout.
pub fn resolve_external_corpus_dir(requested: Option<PathBuf>) -> io::Result<PathBuf> {
    let path = match requested
    {
        Some(path) => path,
        None => match env::var_os(CORPUS_DIR_ENV)
        {
            Some(path) => PathBuf::from(path),
            None => default_corpus_dir()?,
        },
    };
    resolve_external_directory(&path)
}

/// Resolve a shard location and reject a path inside the source checkout.
pub fn resolve_external_shards_dir(requested: Option<PathBuf>) -> io::Result<PathBuf> {
    let path = match requested
    {
        Some(path) => path,
        None => match env::var_os(SHARDS_DIR_ENV)
        {
            Some(path) => PathBuf::from(path),
            None => default_shards_dir()?,
        },
    };
    resolve_external_directory(&path)
}

/// Canonical repository root of the current checkout. When the binary is run
/// outside a checkout, fall back to the checkout that compiled this crate.
pub fn workspace_root() -> io::Result<PathBuf> {
    fs::canonicalize(workspace_root_lexical()?)
}

/// Resolve `path` and reject both a lexical and symlink-mediated path below
/// the source checkout.  The lexical check is essential: an in-tree symlink
/// to an external corpus would still make `grep -R` traverse the corpus.
pub fn resolve_external_directory(path: &Path) -> io::Result<PathBuf> {
    validate_external_directory_against(path, &workspace_root_lexical()?)
}

/// Variant of [`resolve_external_directory`] with an explicit workspace root,
/// used by the migration utility and unit tests.
pub fn validate_external_directory_against(
    path: &Path,
    workspace_root: &Path,
) -> io::Result<PathBuf> {
    let lexical_path = absolute_lexical(path)?;
    let lexical_workspace = absolute_lexical(workspace_root)?;
    let resolved_path = canonicalize_with_missing_tail(&lexical_path)?;
    let resolved_workspace = fs::canonicalize(&lexical_workspace)?;

    if lexical_path.starts_with(&lexical_workspace)
        || resolved_path.starts_with(&resolved_workspace)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "refusing generated data inside the SciRust checkout: {} (workspace: {})",
                path.display(),
                resolved_workspace.display()
            ),
        ));
    }

    Ok(resolved_path)
}

/// Canonicalise an existing path or its closest existing ancestor.  This lets
/// callers validate a new output directory before creating it, while still
/// resolving any symlink in its existing parent chain.
pub fn canonicalize_with_missing_tail(path: &Path) -> io::Result<PathBuf> {
    let absolute = absolute_lexical(path)?;
    let mut existing = absolute.clone();
    let mut missing = Vec::new();

    loop
    {
        match fs::symlink_metadata(&existing)
        {
            Ok(_) => break,
            Err(error) if error.kind() == io::ErrorKind::NotFound =>
            {
                let name = existing.file_name().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("no existing ancestor for {}", absolute.display()),
                    )
                })?;
                missing.push(name.to_os_string());
                if !existing.pop()
                {
                    return Err(io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("no existing ancestor for {}", absolute.display()),
                    ));
                }
            },
            Err(error) => return Err(error),
        }
    }

    let mut resolved = fs::canonicalize(existing)?;
    for component in missing.into_iter().rev()
    {
        resolved.push(component);
    }
    Ok(resolved)
}

fn application_data_dir() -> io::Result<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        if let Some(path) = env::var_os("LOCALAPPDATA")
        {
            Ok(PathBuf::from(path))
        }
        else if let Some(path) = env::var_os("USERPROFILE")
        {
            Ok(PathBuf::from(path).join("AppData").join("Local"))
        }
        else
        {
            Err(io::Error::new(
                io::ErrorKind::NotFound,
                "LOCALAPPDATA or USERPROFILE is required for SCIAGENT generated data",
            ))
        }
    }

    #[cfg(target_os = "macos")]
    {
        Ok(home_dir()?.join("Library").join("Application Support"))
    }

    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        if let Some(path) = env::var_os("XDG_DATA_HOME")
        {
            let path = PathBuf::from(path);
            if path.is_absolute()
            {
                Ok(path)
            }
            else
            {
                Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "XDG_DATA_HOME must be an absolute path",
                ))
            }
        }
        else
        {
            Ok(home_dir()?.join(".local").join("share"))
        }
    }
}

#[cfg(not(target_os = "windows"))]
fn home_dir() -> io::Result<PathBuf> {
    env::var_os("HOME").map(PathBuf::from).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "HOME is required for SCIAGENT generated data",
        )
    })
}

fn workspace_root_lexical() -> io::Result<PathBuf> {
    let current_directory = env::current_dir()?;
    if let Some(workspace_root) = git_worktree_root(&current_directory)
    {
        return absolute_lexical(&workspace_root);
    }

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.parent().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "scirust-sciagent manifest has no workspace parent",
        )
    })?;
    absolute_lexical(workspace_root)
}

fn git_worktree_root(start: &Path) -> Option<PathBuf> {
    for ancestor in start.ancestors()
    {
        let marker = ancestor.join(".git");
        if marker.is_dir() || marker.is_file()
        {
            return Some(ancestor.to_path_buf());
        }
    }
    None
}

fn absolute_lexical(path: &Path) -> io::Result<PathBuf> {
    let absolute = if path.is_absolute()
    {
        path.to_path_buf()
    }
    else
    {
        env::current_dir()?.join(path)
    };
    Ok(normalize_lexical(&absolute))
}

fn normalize_lexical(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components()
    {
        match component
        {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(component.as_os_str()),
            Component::CurDir =>
            {},
            Component::ParentDir =>
            {
                normalized.pop();
            },
            Component::Normal(name) => normalized.push(name),
        }
    }
    normalized
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary_directory(test_name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before Unix epoch")
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "scirust-corpus-paths-{test_name}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&directory).expect("create temporary directory");
        directory
    }

    #[test]
    fn rejects_generated_data_below_workspace() {
        let root = temporary_directory("inside-workspace");
        let inside = root.join("data/crates_raw");

        let error = validate_external_directory_against(&inside, &root)
            .expect_err("in-tree generated data must be rejected");

        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        fs::remove_dir_all(root).expect("remove temporary directory");
    }

    #[test]
    fn allows_an_external_generated_data_directory() {
        let root = temporary_directory("external-workspace");
        let external = root
            .parent()
            .expect("temporary directory has a parent")
            .join(format!("scirust-corpus-external-{}", std::process::id()));

        let resolved = validate_external_directory_against(&external, &root)
            .expect("external generated data directory is valid");

        assert!(resolved.ends_with(external.file_name().expect("external file name")));
        fs::remove_dir_all(root).expect("remove temporary directory");
    }

    #[test]
    fn discovers_a_worktree_from_a_nested_directory() {
        let root = temporary_directory("discover-worktree");
        let nested = root.join("crate/src");
        fs::create_dir_all(&nested).expect("create nested directory");
        fs::create_dir(root.join(".git")).expect("create git marker");

        assert_eq!(git_worktree_root(&nested), Some(root.clone()));
        fs::remove_dir_all(root).expect("remove temporary directory");
    }
}
