//! `sos init` — create (or open) a reasoning repository.

use crate::error::Result;
use crate::store;

/// Run `sos init [path]`: create the store directory layout at `path` (default
/// [`store::DEFAULT_STORE_DIR`]) if it does not already exist.
///
/// # Errors
/// [`crate::error::CliError::Store`] if the directory cannot be created.
pub fn run(path: Option<&str>) -> Result<String> {
    let root = store::resolve_root(path)?;
    let s = store::open(&root)?;
    Ok(format!(
        "Initialized empty SOS repository in {} ({} object(s))",
        root.display(),
        s.len()
    ))
}
