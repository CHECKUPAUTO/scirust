//! [`Publication`] — a verifiable projection of the SOS object graph — with its
//! [`PublicationMeta`], the [`ReproRequirement`] it declares, and its builder.
//!
//! A publication is a content-addressed [`Body`]: sealing it yields an
//! `Object<Publication>` whose id covers every claim, exhibit, section, and
//! declared root, so a third party checks the whole document's integrity from
//! one 32-byte address. But integrity is not truth. A well-formed, correctly
//! hashed publication may still be unsupported, contradicted, or irreproducible
//! — that is what [`verify`](crate::verify::verify) is for. This type only fixes
//! *what is being claimed and from what*; the verifier decides *how it stands*.

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{Author, Body, DeterminismLevel, Object, ObjectId};

use crate::claim::Claim;
use crate::exhibit::{FigureSpec, TableSpec};
use crate::policy::PolicyId;
use crate::reference::Reference;
use crate::section::Section;

/// The reproducibility bar a publication declares for its supporting evidence.
///
/// This is a *policy input*, not a measurement: it says how reproducible the
/// evidence behind a claim must be for the claim to stand. The verifier reads
/// each supporting object's realized [`DeterminismLevel`] from the graph (it
/// never re-runs anything) and compares against this bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ReproRequirement {
    /// No reproducibility bar — evidence is accepted at whatever level it
    /// realized. Honest for a review or a position paper; weak for a result.
    None,
    /// Every supporting object behind a claim must realize **at least** this
    /// level (`meet` over the claim's supporting evidence `>=` this).
    MinimumLevel(DeterminismLevel),
}

impl ReproRequirement {
    const fn discriminant(&self) -> u64 {
        match self
        {
            Self::None => 0,
            Self::MinimumLevel(_) => 1,
        }
    }

    /// The required minimum level, if any.
    #[must_use]
    pub const fn minimum(&self) -> Option<DeterminismLevel> {
        match self
        {
            Self::None => None,
            Self::MinimumLevel(level) => Some(*level),
        }
    }
}

impl Canonical for ReproRequirement {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.u64(self.discriminant());
        if let Self::MinimumLevel(level) = self
        {
            enc.value(level);
        }
    }
}

/// The front-matter of a publication.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicationMeta {
    /// The title.
    pub title: String,
    /// The subtitle, if any.
    pub subtitle: Option<String>,
    /// The authors, in listed order. These are the *paper's* authors; the
    /// principal who *sealed* the object is recorded separately in the sealed
    /// [`Object`]'s provenance.
    pub authors: Vec<Author>,
    /// The abstract.
    pub summary: String,
}

impl Canonical for PublicationMeta {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.str(&self.title);
        enc.option(&self.subtitle);
        enc.seq(&self.authors);
        enc.str(&self.summary);
    }
}

/// A verifiable projection of the SOS scientific object graph.
///
/// See the [module docs](self) for the integrity-is-not-truth distinction. The
/// registries (`claims`, `figures`, `tables`, `bibliography`) are the entries
/// prose refers to by key; `declared_roots` fixes the sub-graph the publication
/// claims to draw from, and is what the verifier takes the dependency closure
/// of; `verification_policy` names the versioned support policy that governs how
/// claims are judged; `reproducibility` is the bar that policy applies.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Publication {
    /// Front-matter (title, authors, abstract).
    pub meta: PublicationMeta,
    /// The ordered, typed body.
    pub sections: Vec<Section>,
    /// The claim registry.
    pub claims: Vec<Claim>,
    /// The figure registry.
    pub figures: Vec<FigureSpec>,
    /// The table registry.
    pub tables: Vec<TableSpec>,
    /// The bibliography (internal, verifiable entries and external, unverifiable
    /// ones).
    pub bibliography: Vec<Reference>,
    /// The object roots this publication declares it draws from — the scope the
    /// verifier takes the dependency closure of (held sorted + deduplicated).
    pub declared_roots: Vec<ObjectId>,
    /// The versioned support policy governing this publication.
    pub verification_policy: PolicyId,
    /// The reproducibility bar the policy applies to supporting evidence.
    pub reproducibility: ReproRequirement,
}

impl Publication {
    /// Start building a publication titled `title`.
    #[must_use]
    pub fn builder(title: impl Into<String>) -> PublicationBuilder {
        PublicationBuilder {
            inner: Publication {
                meta: PublicationMeta {
                    title: title.into(),
                    subtitle: None,
                    authors: Vec::new(),
                    summary: String::new(),
                },
                sections: Vec::new(),
                claims: Vec::new(),
                figures: Vec::new(),
                tables: Vec::new(),
                bibliography: Vec::new(),
                declared_roots: Vec::new(),
                verification_policy: PolicyId::standard(),
                reproducibility: ReproRequirement::None,
            },
        }
    }

    /// Look up a claim by key.
    #[must_use]
    pub fn claim(&self, key: &crate::key::ClaimKey) -> Option<&Claim> {
        self.claims.iter().find(|c| &c.key == key)
    }
}

impl Canonical for Publication {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.value(&self.meta);
        enc.seq(&self.sections);
        enc.seq(&self.claims);
        enc.seq(&self.figures);
        enc.seq(&self.tables);
        enc.seq(&self.bibliography);
        enc.seq(&self.declared_roots);
        enc.value(&self.verification_policy);
        enc.value(&self.reproducibility);
    }
}

impl Body for Publication {
    const KIND: &'static str = "Publication";
    const SCHEMA_VERSION: u32 = 1;
}

/// A builder for [`Publication`].
///
/// Registries and sections keep insertion order (it is editorially meaningful);
/// `declared_roots` is normalized to a sorted, deduplicated set on
/// [`build`](Self::build) so the publication's content address does not depend
/// on the order roots were declared.
#[derive(Debug, Clone)]
pub struct PublicationBuilder {
    inner: Publication,
}

impl PublicationBuilder {
    /// Set the subtitle.
    #[must_use]
    pub fn subtitle(mut self, subtitle: impl Into<String>) -> Self {
        self.inner.meta.subtitle = Some(subtitle.into());
        self
    }

    /// Append an author.
    #[must_use]
    pub fn author(mut self, author: Author) -> Self {
        self.inner.meta.authors.push(author);
        self
    }

    /// Set the abstract.
    #[must_use]
    pub fn summary(mut self, summary: impl Into<String>) -> Self {
        self.inner.meta.summary = summary.into();
        self
    }

    /// Append a section.
    #[must_use]
    pub fn section(mut self, section: Section) -> Self {
        self.inner.sections.push(section);
        self
    }

    /// Append a claim to the registry.
    #[must_use]
    pub fn claim(mut self, claim: Claim) -> Self {
        self.inner.claims.push(claim);
        self
    }

    /// Append a figure to the registry.
    #[must_use]
    pub fn figure(mut self, figure: FigureSpec) -> Self {
        self.inner.figures.push(figure);
        self
    }

    /// Append a table to the registry.
    #[must_use]
    pub fn table(mut self, table: TableSpec) -> Self {
        self.inner.tables.push(table);
        self
    }

    /// Append a bibliography entry.
    #[must_use]
    pub fn reference(mut self, reference: Reference) -> Self {
        self.inner.bibliography.push(reference);
        self
    }

    /// Declare a root object the publication draws from.
    #[must_use]
    pub fn declared_root(mut self, root: ObjectId) -> Self {
        self.inner.declared_roots.push(root);
        self
    }

    /// Set the versioned support policy.
    #[must_use]
    pub fn verification_policy(mut self, policy: PolicyId) -> Self {
        self.inner.verification_policy = policy;
        self
    }

    /// Set the reproducibility bar.
    #[must_use]
    pub fn reproducibility(mut self, requirement: ReproRequirement) -> Self {
        self.inner.reproducibility = requirement;
        self
    }

    /// Finish, normalizing `declared_roots` to a sorted, deduplicated set.
    #[must_use]
    pub fn build(mut self) -> Publication {
        self.inner.declared_roots.sort_unstable();
        self.inner.declared_roots.dedup();
        self.inner
    }
}

/// Seal a [`Publication`] as a storable, content-addressed
/// `Object<Publication>`, authored by the sealing `principal`.
///
/// The object's id covers the entire document; the principal is the curator who
/// released it (distinct from the paper's authors in [`PublicationMeta`]).
/// Cryptographic *signing* of the sealed root is `sos-provenance`'s job (this
/// crate does not fake a signature); the content address is intrinsic regardless.
#[must_use]
pub fn seal_publication(publication: Publication, principal: Author) -> Object<Publication> {
    Object::builder(publication).author(principal).seal()
}
