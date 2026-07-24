//! `sos log` — list every object in the store, newest (by logical clock) last.

use crate::error::Result;
use crate::store;

/// Run `sos log [path]`: print every object's id, kind, determinism level, and
/// parent count, ordered by logical clock (ties broken by id, so the listing
/// is fully deterministic).
///
/// # Errors
/// [`crate::error::CliError::Store`] if the store cannot be opened.
pub fn run(path: Option<&str>) -> Result<String> {
    let root = store::resolve_root(path)?;
    let s = store::open(&root)?;
    let mut headers = store::all_headers(&s)?;
    headers.sort_by(|a, b| a.logical.cmp(&b.logical).then(a.id.cmp(&b.id)));

    if headers.is_empty()
    {
        return Ok("(empty repository)".to_owned());
    }

    let mut out = String::new();
    for h in &headers
    {
        out.push_str(&format!(
            "{}  {:<20} {}  logical={} parents={}\n",
            h.id,
            h.kind.to_string(),
            h.level.code(),
            h.logical.get(),
            h.parents.len()
        ));
    }
    out.pop(); // trailing newline
    Ok(out)
}
