//! Shared store-access helpers: opening the on-disk store at a path, and
//! reading every object's [`GenericHeader`] regardless of kind.

use std::path::{Path, PathBuf};

use sos_core::ObjectId;
use sos_store::{FileStore, ObjectStore};

use crate::error::{CliError, Result};
use crate::header::GenericHeader;

/// The default store directory name, relative to the current directory — the
/// `sos` analogue of `.git`.
pub const DEFAULT_STORE_DIR: &str = ".sos";

/// Resolve the store root a command should operate on: `explicit` if given,
/// else [`DEFAULT_STORE_DIR`] under the current directory.
///
/// # Errors
/// [`CliError::Io`] if the current directory cannot be read.
pub fn resolve_root(explicit: Option<&str>) -> Result<PathBuf> {
    match explicit
    {
        Some(path) => Ok(PathBuf::from(path)),
        None => Ok(std::env::current_dir()?.join(DEFAULT_STORE_DIR)),
    }
}

/// Open (creating if absent) the store at `root`.
///
/// # Errors
/// [`CliError::Store`] if `root` cannot be created or an existing store there
/// cannot be read.
pub fn open(root: &Path) -> Result<FileStore> {
    Ok(FileStore::open(root)?)
}

/// Read every stored object's [`GenericHeader`], in sorted-by-id order (the
/// same determinism [`ObjectStore::object_ids`] already guarantees).
///
/// # Errors
/// [`CliError::Serde`] if a stored record cannot be parsed as a header (a
/// corrupted or foreign record); this never happens for anything the `sos`
/// tools themselves wrote.
pub fn all_headers(store: &FileStore) -> Result<Vec<GenericHeader>> {
    store
        .object_ids()
        .into_iter()
        .map(|id| header_of(store, id))
        .collect()
}

/// Read one object's [`GenericHeader`] by id.
///
/// # Errors
/// [`CliError::NotFound`] if no object is stored at `id`; [`CliError::Serde`]
/// if the stored bytes do not parse as a header.
pub fn header_of(store: &FileStore, id: ObjectId) -> Result<GenericHeader> {
    let record = store.get_raw(id).ok_or(CliError::NotFound(id))?;
    Ok(GenericHeader::parse(&record.bytes)?)
}
