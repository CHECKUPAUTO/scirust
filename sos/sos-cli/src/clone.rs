//! `sos clone` / `sos push` — copy a reasoning repository.
//!
//! Both commands are the same operation in opposite rhetorical directions
//! (`clone` pulls a source into a fresh local copy; `push` sends the current
//! store to a destination) — exactly like `git clone`/`git push` already work
//! for a local path "remote": there is no network transport here, so both are
//! implemented as one honest local copy. A real network remote (fetching from
//! or sharing to another machine) needs a transport this workspace does not
//! have yet — that is `sos-mcp`'s domain, not this one, and is not stubbed
//! here.
//!
//! Refs are merged, not overwritten: every ref in the source is written into
//! the destination (last-writer-wins per name), so unrelated refs already in
//! the destination survive. Objects are copied via [`ObjectStore`] (present
//! ids are left alone — first-wins, the same idempotence every content-address
//! put already guarantees). Blobs are copied at the filesystem level directly
//! (there is no blob-enumeration primitive on the [`ObjectStore`] trait, since
//! bodies are opaque to the store layer; [`FileStore`]'s own on-disk blob
//! directory can be walked directly instead).

use std::fs;
use std::path::Path;

use sos_store::{FileStore, ObjectStore};

use crate::error::Result;
use crate::store;

/// Copy every object, blob, and ref from the store at `src` into the store at
/// `dest` (created if absent). Used by both `sos clone <src> <dest>` and `sos
/// push <dest>` (which passes the current store as `src`).
///
/// # Errors
/// [`crate::error::CliError::Store`] / [`crate::error::CliError::Io`] if either
/// store cannot be opened or read/written.
pub fn run(src: &str, dest: &str) -> Result<String> {
    let src_store = store::open(Path::new(src))?;
    let mut dest_store = store::open(Path::new(dest))?;

    let mut objects_copied = 0usize;
    for id in src_store.object_ids()
    {
        if !dest_store.has(id)
        {
            if let Some(record) = src_store.get_raw(id)
            {
                dest_store.put_raw(id, record);
                objects_copied += 1;
            }
        }
    }

    let blobs_copied = copy_blob_tree(&src_store, &dest_store)?;

    let mut refs_copied = 0usize;
    for named in src_store.refs()
    {
        dest_store.set_ref(&named.name, named.target);
        refs_copied += 1;
    }

    Ok(format!(
        "Copied {objects_copied} object(s), {blobs_copied} blob(s), {refs_copied} ref(s) from {src} to {dest}"
    ))
}

/// Recursively copy every blob file under `src`'s blob directory into `dest`'s,
/// skipping any that already exist (blobs are content-addressed and immutable,
/// so an existing file at the same path is already byte-identical).
fn copy_blob_tree(src: &FileStore, dest: &FileStore) -> Result<usize> {
    let src_blobs = src.root().join("blobs");
    let dest_blobs = dest.root().join("blobs");
    if !src_blobs.is_dir()
    {
        return Ok(0);
    }
    let mut copied = 0usize;
    for shard in fs::read_dir(&src_blobs)?
    {
        let shard = shard?;
        if !shard.path().is_dir()
        {
            continue;
        }
        let dest_shard = dest_blobs.join(shard.file_name());
        fs::create_dir_all(&dest_shard)?;
        for file in fs::read_dir(shard.path())?
        {
            let file = file?;
            let dest_file = dest_shard.join(file.file_name());
            if !dest_file.exists()
            {
                fs::copy(file.path(), &dest_file)?;
                copied += 1;
            }
        }
    }
    Ok(copied)
}
