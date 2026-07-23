//! [`Format`], [`Artifact`], and deterministic rendering of a [`Publication`].
//!
//! Rendering resolves each section block against the registries — a claim block
//! prints the claim and, crucially, the objects bound to it; a figure/table
//! block prints the caption and its sources; a citation block prints the
//! bibliography entry, marked verifiable or external. The output is a pure
//! function of the publication (same input ⇒ byte-identical output), so a
//! rendered artifact is itself content-addressable. Markdown, HTML, and
//! canonical JSON are emitted here; LaTeX/PDF need a typesetting backend
//! (Invariant VIII) and are reported as [`PublicationError::UnsupportedFormat`].

use core::fmt::Write as _;
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sos_core::Author;

use crate::claim::{Claim, ClaimBinding};
use crate::error::{PublicationError, Result};
use crate::key::{ClaimKey, FigureKey, RefKey, TableKey};
use crate::publication::Publication;
use crate::reference::Reference;
use crate::section::{Block, Section};

/// A render target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Format {
    /// Publication-ready Markdown.
    Markdown,
    /// Self-contained HTML.
    Html,
    /// Canonical JSON (the publication's interchange form).
    Json,
    /// LaTeX (needs a typesetting backend; not emitted).
    Latex,
    /// PDF (needs a typesetting backend; not emitted).
    Pdf,
}

impl Format {
    /// A short, stable code.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self
        {
            Self::Markdown => "markdown",
            Self::Html => "html",
            Self::Json => "json",
            Self::Latex => "latex",
            Self::Pdf => "pdf",
        }
    }

    /// The conventional file extension.
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self
        {
            Self::Markdown => "md",
            Self::Html => "html",
            Self::Json => "json",
            Self::Latex => "tex",
            Self::Pdf => "pdf",
        }
    }
}

/// A rendered publication: its `format` and the document `content`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Artifact {
    /// The format the content is in.
    pub format: Format,
    /// The rendered document.
    pub content: String,
}

/// Render `publication` to `format`.
///
/// Deterministic: the same publication renders byte-identically every time.
///
/// # Errors
/// [`PublicationError::UnsupportedFormat`] for [`Format::Latex`] / [`Format::Pdf`]
/// (no typesetting backend); [`PublicationError::Serde`] if JSON serialization
/// fails.
pub fn render(publication: &Publication, format: Format) -> Result<Artifact> {
    let content = match format
    {
        Format::Markdown => render_markdown(publication),
        Format::Html => render_html(publication),
        Format::Json => serde_json::to_string(publication)?,
        Format::Latex | Format::Pdf => return Err(PublicationError::UnsupportedFormat(format)),
    };
    Ok(Artifact { format, content })
}

/// Registry lookups used while rendering section blocks.
struct Resolver<'p> {
    claims: BTreeMap<&'p ClaimKey, &'p Claim>,
    figure_captions: BTreeMap<&'p FigureKey, &'p str>,
    table_captions: BTreeMap<&'p TableKey, &'p str>,
    references: BTreeMap<&'p RefKey, &'p Reference>,
}

impl<'p> Resolver<'p> {
    fn new(publication: &'p Publication) -> Self {
        Self {
            claims: publication.claims.iter().map(|c| (&c.key, c)).collect(),
            figure_captions: publication
                .figures
                .iter()
                .map(|f| (&f.key, f.caption.as_str()))
                .collect(),
            table_captions: publication
                .tables
                .iter()
                .map(|t| (&t.key, t.caption.as_str()))
                .collect(),
            references: publication
                .bibliography
                .iter()
                .map(|r| (r.key(), r))
                .collect(),
        }
    }
}

/// A human label for an author principal, e.g. `human:ada`.
fn author_label(author: &Author) -> String {
    let principal = match author
    {
        Author::Human(_) => "human",
        Author::Agent(_) => "agent",
        Author::Engine(_) => "engine",
    };
    format!("{principal}:{}", author.id())
}

/// One binding rendered as a bullet, e.g. `- directly-supports → sos1:… (note)`.
fn render_binding_line(binding: &ClaimBinding) -> String {
    match &binding.note
    {
        Some(note) => format!(
            "  - {} → `{}` ({note})",
            binding.role.code(),
            binding.object
        ),
        None => format!("  - {} → `{}`", binding.role.code(), binding.object),
    }
}

/// Render the publication as Markdown.
fn render_markdown(p: &Publication) -> String {
    let resolver = Resolver::new(p);
    let mut s = String::new();

    let _ = writeln!(s, "# {}\n", p.meta.title);
    if let Some(subtitle) = &p.meta.subtitle
    {
        let _ = writeln!(s, "*{subtitle}*\n");
    }
    if !p.meta.authors.is_empty()
    {
        let authors = p
            .meta
            .authors
            .iter()
            .map(author_label)
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(s, "**Authors:** {authors}\n");
    }
    if !p.meta.summary.is_empty()
    {
        let _ = writeln!(s, "## Abstract\n\n{}\n", p.meta.summary);
    }

    // Body sections, resolving typed blocks.
    for section in &p.sections
    {
        render_section_markdown(&mut s, section, &resolver);
    }

    // Claim registry — surfaces exactly which objects bear on each claim.
    let _ = writeln!(s, "## Claims\n");
    if p.claims.is_empty()
    {
        let _ = writeln!(s, "_No claims._\n");
    }
    for claim in &p.claims
    {
        let _ = writeln!(s, "- **{}** — {}", claim.key, claim.statement);
        let _ = writeln!(s, "  `{}`", claim.content_id());
        for binding in &claim.bindings
        {
            let _ = writeln!(s, "{}", render_binding_line(binding));
        }
    }
    let _ = writeln!(s);

    // Figures and tables.
    if !p.figures.is_empty()
    {
        let _ = writeln!(s, "## Figures\n");
        for (i, figure) in p.figures.iter().enumerate()
        {
            let _ = writeln!(
                s,
                "- **Figure {} ({})** — {}",
                i + 1,
                figure.key,
                figure.caption
            );
            for source in &figure.sources
            {
                let _ = writeln!(s, "  - from `{source}`");
            }
        }
        let _ = writeln!(s);
    }
    if !p.tables.is_empty()
    {
        let _ = writeln!(s, "## Tables\n");
        for (i, table) in p.tables.iter().enumerate()
        {
            let _ = writeln!(
                s,
                "- **Table {} ({})** — {}",
                i + 1,
                table.key,
                table.caption
            );
            for source in &table.sources
            {
                let _ = writeln!(s, "  - from `{source}`");
            }
        }
        let _ = writeln!(s);
    }

    // Bibliography, honest about what is evidence and what is not.
    if !p.bibliography.is_empty()
    {
        let _ = writeln!(s, "## References\n");
        for reference in &p.bibliography
        {
            let _ = writeln!(s, "- {}", reference_line(reference));
        }
        let _ = writeln!(s);
    }

    // Provenance footer.
    let _ = writeln!(s, "## Provenance\n");
    let _ = writeln!(s, "- Policy: `{}`", p.verification_policy);
    let _ = writeln!(s, "- Reproducibility: {}", repro_label(p));
    let _ = writeln!(s, "- Declared roots:");
    for root in &p.declared_roots
    {
        let _ = writeln!(s, "  - `{root}`");
    }
    s
}

/// Render one section as Markdown.
fn render_section_markdown(s: &mut String, section: &Section, resolver: &Resolver) {
    let _ = writeln!(s, "## {}\n", section.heading);
    for block in &section.blocks
    {
        match block
        {
            Block::Prose(text) =>
            {
                let _ = writeln!(s, "{text}\n");
            },
            Block::Claim(key) => match resolver.claims.get(key)
            {
                Some(claim) =>
                {
                    let _ = writeln!(s, "> **Claim {}:** {}\n", key, claim.statement);
                },
                None =>
                {
                    let _ = writeln!(s, "> **Claim {key}:** _[unresolved claim]_\n");
                },
            },
            Block::Figure(key) => match resolver.figure_captions.get(key)
            {
                Some(caption) =>
                {
                    let _ = writeln!(s, "_Figure {key}: {caption}_\n");
                },
                None =>
                {
                    let _ = writeln!(s, "_Figure {key}: [unresolved figure]_\n");
                },
            },
            Block::Table(key) => match resolver.table_captions.get(key)
            {
                Some(caption) =>
                {
                    let _ = writeln!(s, "_Table {key}: {caption}_\n");
                },
                None =>
                {
                    let _ = writeln!(s, "_Table {key}: [unresolved table]_\n");
                },
            },
            Block::Cite(key) => match resolver.references.get(key)
            {
                Some(reference) =>
                {
                    let _ = writeln!(s, "See {}.\n", reference_line(reference));
                },
                None =>
                {
                    let _ = writeln!(s, "See _[unresolved citation {key}]_.\n");
                },
            },
        }
    }
}

/// A one-line rendering of a bibliography entry, labelling internal vs external.
fn reference_line(reference: &Reference) -> String {
    match reference
    {
        Reference::Internal { key, object } => format!("[{key}] (in-graph) `{object}`"),
        Reference::External { key, citation } =>
        {
            let authors = citation.authors.join(", ");
            let year = citation
                .year
                .map_or_else(String::new, |y| format!(" ({y})"));
            let id = citation
                .identifier
                .as_ref()
                .map_or_else(String::new, |i| format!(" [{i}]"));
            format!(
                "[{key}] (external, unverified) {authors}{year}. {}{id}",
                citation.title
            )
        },
    }
}

/// A short description of the reproducibility requirement.
fn repro_label(p: &Publication) -> String {
    match p.reproducibility.minimum()
    {
        None => "not required".to_owned(),
        Some(level) => format!("minimum {}", level.code()),
    }
}

/// Render the publication as self-contained HTML.
fn render_html(p: &Publication) -> String {
    let resolver = Resolver::new(p);
    let mut s = String::new();
    let _ = writeln!(s, "<!DOCTYPE html>");
    let _ = writeln!(s, "<article>");
    let _ = writeln!(s, "<h1>{}</h1>", escape_html(&p.meta.title));
    if let Some(subtitle) = &p.meta.subtitle
    {
        let _ = writeln!(
            s,
            "<p class=\"subtitle\"><em>{}</em></p>",
            escape_html(subtitle)
        );
    }
    if !p.meta.authors.is_empty()
    {
        let authors = p
            .meta
            .authors
            .iter()
            .map(|a| escape_html(&author_label(a)))
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(s, "<p class=\"authors\">{authors}</p>");
    }
    if !p.meta.summary.is_empty()
    {
        let _ = writeln!(
            s,
            "<section><h2>Abstract</h2><p>{}</p></section>",
            escape_html(&p.meta.summary)
        );
    }

    for section in &p.sections
    {
        let _ = writeln!(s, "<section><h2>{}</h2>", escape_html(&section.heading));
        for block in &section.blocks
        {
            render_block_html(&mut s, block, &resolver);
        }
        let _ = writeln!(s, "</section>");
    }

    let _ = writeln!(s, "<section><h2>Claims</h2><ul>");
    for claim in &p.claims
    {
        let _ = writeln!(
            s,
            "<li><strong>{}</strong> — {} <code>{}</code><ul>",
            escape_html(claim.key.as_str()),
            escape_html(&claim.statement),
            claim.content_id()
        );
        for binding in &claim.bindings
        {
            let _ = writeln!(
                s,
                "<li>{} → <code>{}</code></li>",
                escape_html(binding.role.code()),
                binding.object
            );
        }
        let _ = writeln!(s, "</ul></li>");
    }
    let _ = writeln!(s, "</ul></section>");

    let _ = writeln!(s, "<section><h2>References</h2><ul>");
    for reference in &p.bibliography
    {
        let _ = writeln!(s, "<li>{}</li>", escape_html(&reference_line(reference)));
    }
    let _ = writeln!(s, "</ul></section>");

    let _ = writeln!(
        s,
        "<footer><p>Policy: <code>{}</code>. Reproducibility: {}.</p></footer>",
        escape_html(&p.verification_policy.to_string()),
        escape_html(&repro_label(p))
    );
    let _ = writeln!(s, "</article>");
    s
}

/// Render one block as HTML.
fn render_block_html(s: &mut String, block: &Block, resolver: &Resolver) {
    match block
    {
        Block::Prose(text) =>
        {
            let _ = writeln!(s, "<p>{}</p>", escape_html(text));
        },
        Block::Claim(key) =>
        {
            let statement = resolver
                .claims
                .get(key)
                .map_or("[unresolved claim]", |c| c.statement.as_str());
            let _ = writeln!(
                s,
                "<blockquote><strong>Claim {}:</strong> {}</blockquote>",
                escape_html(key.as_str()),
                escape_html(statement)
            );
        },
        Block::Figure(key) =>
        {
            let caption = resolver
                .figure_captions
                .get(key)
                .copied()
                .unwrap_or("[unresolved figure]");
            let _ = writeln!(
                s,
                "<figure><figcaption>Figure {}: {}</figcaption></figure>",
                escape_html(key.as_str()),
                escape_html(caption)
            );
        },
        Block::Table(key) =>
        {
            let caption = resolver
                .table_captions
                .get(key)
                .copied()
                .unwrap_or("[unresolved table]");
            let _ = writeln!(
                s,
                "<figure><figcaption>Table {}: {}</figcaption></figure>",
                escape_html(key.as_str()),
                escape_html(caption)
            );
        },
        Block::Cite(key) =>
        {
            let line = resolver.references.get(key).map_or_else(
                || format!("[unresolved citation {key}]"),
                |r| reference_line(r),
            );
            let _ = writeln!(s, "<p>See {}.</p>", escape_html(&line));
        },
    }
}

/// Escape the five characters that are special in HTML text/attribute content.
fn escape_html(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars()
    {
        match ch
        {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            other => out.push(other),
        }
    }
    out
}
