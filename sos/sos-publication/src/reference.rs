//! [`Reference`] — a bibliography entry that is honest about what it is.
//!
//! The mandate is explicit: the engine must **never confuse a citation with
//! evidence**. So the bibliography is a two-constructor sum type. An
//! [`Reference::Internal`] cites an object *in the SOS graph* — it has a content
//! address, it can be resolved, and a claim may be supported by it. A
//! [`Reference::External`] cites outside literature — a DOI, an arXiv id, a
//! book. It is recorded faithfully and rendered, but it is **unverifiable** by
//! construction: the engine cannot and does not treat it as evidence, and the
//! verifier never resolves it against the graph.

use serde::{Deserialize, Serialize};
use sos_core::ObjectId;
use sos_core::canonical::{Canonical, CanonicalEncoder};

use crate::key::RefKey;

/// A citation to outside literature. Recorded, rendered — never verified.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalCitation {
    /// The work's title.
    pub title: String,
    /// The authors, in listed order.
    pub authors: Vec<String>,
    /// The year of publication, if known.
    pub year: Option<u32>,
    /// A stable external identifier (DOI, arXiv id, ISBN), if any. This is not a
    /// content address and is not resolved by the engine.
    pub identifier: Option<String>,
}

impl ExternalCitation {
    /// A citation with a title and authors; year and identifier unset.
    #[must_use]
    pub fn new(title: impl Into<String>, authors: Vec<String>) -> Self {
        Self {
            title: title.into(),
            authors,
            year: None,
            identifier: None,
        }
    }

    /// Set the publication year.
    #[must_use]
    pub fn year(mut self, year: u32) -> Self {
        self.year = Some(year);
        self
    }

    /// Set the external identifier (DOI/arXiv/ISBN).
    #[must_use]
    pub fn identifier(mut self, identifier: impl Into<String>) -> Self {
        self.identifier = Some(identifier.into());
        self
    }
}

impl Canonical for ExternalCitation {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.str(&self.title);
        enc.seq(&self.authors);
        enc.option(&self.year);
        enc.option(&self.identifier);
    }
}

/// A bibliography entry: an in-graph object (verifiable) or outside literature
/// (unverifiable). Every entry carries a [`RefKey`] so prose can cite it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Reference {
    /// A citation to an SOS object with a content address. Resolvable, and
    /// eligible to be a claim's evidence.
    Internal {
        /// The entry's handle within its publication.
        key: RefKey,
        /// The cited object's content address.
        object: ObjectId,
    },
    /// A citation to outside literature. Recorded faithfully; **not** evidence
    /// and never resolved against the graph.
    External {
        /// The entry's handle within its publication.
        key: RefKey,
        /// The external bibliographic details.
        citation: ExternalCitation,
    },
}

impl Reference {
    const fn discriminant(&self) -> u64 {
        match self
        {
            Self::Internal { .. } => 0,
            Self::External { .. } => 1,
        }
    }

    /// An internal (in-graph) reference.
    #[must_use]
    pub fn internal(key: impl Into<RefKey>, object: ObjectId) -> Self {
        Self::Internal {
            key: key.into(),
            object,
        }
    }

    /// An external (outside-literature) reference.
    #[must_use]
    pub fn external(key: impl Into<RefKey>, citation: ExternalCitation) -> Self {
        Self::External {
            key: key.into(),
            citation,
        }
    }

    /// This entry's key.
    #[must_use]
    pub fn key(&self) -> &RefKey {
        match self
        {
            Self::Internal { key, .. } | Self::External { key, .. } => key,
        }
    }

    /// The in-graph object this entry cites, if it is internal. `None` for an
    /// external citation — the caller must not treat external literature as
    /// resolvable evidence.
    #[must_use]
    pub fn object(&self) -> Option<ObjectId> {
        match self
        {
            Self::Internal { object, .. } => Some(*object),
            Self::External { .. } => None,
        }
    }

    /// Whether this entry is verifiable (an in-graph object).
    #[must_use]
    pub fn is_verifiable(&self) -> bool {
        matches!(self, Self::Internal { .. })
    }
}

impl Canonical for Reference {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.u64(self.discriminant());
        match self
        {
            Self::Internal { key, object } =>
            {
                enc.value(key);
                enc.value(object);
            },
            Self::External { key, citation } =>
            {
                enc.value(key);
                enc.value(citation);
            },
        }
    }
}
