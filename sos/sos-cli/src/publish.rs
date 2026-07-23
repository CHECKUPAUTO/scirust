//! `sos publish` — render and seal a publication candidate.

use sos_core::Author;
use sos_publication::{Format, Publication, render, seal_publication};
use sos_store::TypedStore;

use crate::args::Args;
use crate::error::{CliError, Result};
use crate::store;

/// Run `sos publish <publication.json> [--format md|html|json] [--author <name>]
/// [--store <path>]`.
///
/// Loads a [`Publication`] from JSON, seals it (authored by `--author`, default
/// `"sos-cli"`), and — if `--store` is given — stores the sealed object. If
/// `--format` is given, also renders and returns the document in that format
/// (Markdown/HTML/JSON only; LaTeX/PDF need a typesetting backend and are not
/// emitted here, matching `sos-publication` itself).
///
/// # Errors
/// [`CliError::Io`]/[`CliError::Serde`] if the publication file cannot be read
/// or parsed; [`CliError::Publication`] for an unsupported `--format`.
pub fn run(args: &Args) -> Result<String> {
    let path = args.positional(0, "publication.json")?;
    let bytes = std::fs::read(path)?;
    let publication: Publication = serde_json::from_slice(&bytes)?;

    let author = Author::human(args.flag("author").unwrap_or("sos-cli"));
    let sealed = seal_publication(publication.clone(), author);

    let mut out = format!(
        "Sealed publication `{}` (title: {:?})",
        sealed.id, publication.meta.title
    );

    if let Some(path) = args.flag("store")
    {
        let root = store::resolve_root(Some(path))?;
        let mut s = store::open(&root)?;
        s.put_object(&sealed)?;
        out.push_str(&format!("\nStored in {}", root.display()));
    }

    if let Some(fmt) = args.flag("format")
    {
        let format = parse_format(fmt)?;
        let artifact = render(&publication, format)?;
        out.push_str("\n\n");
        out.push_str(&artifact.content);
    }

    Ok(out)
}

/// Parse a `--format` value into a [`Format`].
fn parse_format(s: &str) -> Result<Format> {
    match s
    {
        "md" | "markdown" => Ok(Format::Markdown),
        "html" => Ok(Format::Html),
        "json" => Ok(Format::Json),
        other => Err(CliError::Usage(format!(
            "unknown --format `{other}` (expected md, html, or json)"
        ))),
    }
}
