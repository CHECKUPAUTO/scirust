//! [`Section`], [`SectionKind`], and [`Block`] — the ordered, typed body of a
//! publication.
//!
//! Prose is not free text interleaved with hope: a section is a sequence of
//! typed [`Block`]s, and a block that references a claim, figure, table, or
//! citation does so by **key**, not by restating it. Rendering resolves those
//! keys against the registries, and verification reports any that do not
//! resolve. A reader — and the verifier — can therefore see exactly which claims
//! and exhibits a section leans on.

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};

use crate::key::{ClaimKey, FigureKey, RefKey, SectionId, TableKey};

/// The rhetorical role of a section (IMRaD and friends), for rendering and for
/// completeness checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SectionKind {
    /// The abstract.
    Abstract,
    /// Introduction / background.
    Introduction,
    /// Methods / materials.
    Methods,
    /// Results.
    Results,
    /// Discussion.
    Discussion,
    /// Conclusion.
    Conclusion,
    /// An appendix.
    Appendix,
    /// A section that does not fit the standard roles, named freely.
    Other(String),
}

impl SectionKind {
    const fn discriminant(&self) -> u64 {
        match self
        {
            Self::Abstract => 0,
            Self::Introduction => 1,
            Self::Methods => 2,
            Self::Results => 3,
            Self::Discussion => 4,
            Self::Conclusion => 5,
            Self::Appendix => 6,
            Self::Other(_) => 7,
        }
    }

    /// A short, stable code.
    #[must_use]
    pub fn code(&self) -> &str {
        match self
        {
            Self::Abstract => "abstract",
            Self::Introduction => "introduction",
            Self::Methods => "methods",
            Self::Results => "results",
            Self::Discussion => "discussion",
            Self::Conclusion => "conclusion",
            Self::Appendix => "appendix",
            Self::Other(name) => name,
        }
    }
}

impl Canonical for SectionKind {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.u64(self.discriminant());
        if let Self::Other(name) = self
        {
            enc.str(name);
        }
    }
}

/// One element of a section's body.
///
/// A [`Block::Prose`] carries text; the others are **typed references** into the
/// publication's registries, resolved by key at render/verify time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Block {
    /// A paragraph of prose.
    Prose(String),
    /// A reference to a claim by key.
    Claim(ClaimKey),
    /// A reference to a figure by key.
    Figure(FigureKey),
    /// A reference to a table by key.
    Table(TableKey),
    /// A reference to a bibliography entry by key.
    Cite(RefKey),
}

impl Block {
    const fn discriminant(&self) -> u64 {
        match self
        {
            Self::Prose(_) => 0,
            Self::Claim(_) => 1,
            Self::Figure(_) => 2,
            Self::Table(_) => 3,
            Self::Cite(_) => 4,
        }
    }
}

impl Canonical for Block {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.u64(self.discriminant());
        match self
        {
            Self::Prose(text) => enc.str(text),
            Self::Claim(key) => enc.value(key),
            Self::Figure(key) => enc.value(key),
            Self::Table(key) => enc.value(key),
            Self::Cite(key) => enc.value(key),
        }
    }
}

/// A titled, typed section: an ordered sequence of [`Block`]s.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Section {
    /// The section's handle within its publication.
    pub id: SectionId,
    /// The section's rhetorical role.
    pub kind: SectionKind,
    /// The displayed heading.
    pub heading: String,
    /// The section body, in order.
    pub blocks: Vec<Block>,
}

impl Section {
    /// A section `id` of `kind` with `heading` and no blocks yet.
    #[must_use]
    pub fn new(id: impl Into<SectionId>, kind: SectionKind, heading: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            kind,
            heading: heading.into(),
            blocks: Vec::new(),
        }
    }

    /// Append a block (builder style).
    #[must_use]
    pub fn block(mut self, block: Block) -> Self {
        self.blocks.push(block);
        self
    }

    /// Append a prose paragraph.
    #[must_use]
    pub fn prose(self, text: impl Into<String>) -> Self {
        self.block(Block::Prose(text.into()))
    }
}

impl Canonical for Section {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.value(&self.id);
        enc.value(&self.kind);
        enc.str(&self.heading);
        enc.seq(&self.blocks);
    }
}
