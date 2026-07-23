//! [`Claim`] — a first-class, content-addressed scientific statement, and the
//! typed [`ClaimBinding`]s that wire it to the objects that bear on it.
//!
//! A claim is the unit the Publication Engine actually verifies. It is not free
//! prose: it is a statement plus an explicit, typed set of edges into the object
//! graph. Every edge declares *how* an object bears on the claim — does it
//! directly support it, merely supply the method, or **contradict** it? The
//! verifier reads exactly these edges; nothing about a claim's standing is
//! inferred from wording. This is what stops the engine confusing a
//! well-written sentence with a supported result.

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{HashAlgo, ObjectId};

use crate::key::ClaimKey;

/// The hash domain for a claim's content address (see [`Claim::content_id`]).
const CLAIM_DOMAIN: &[u8] = b"sos-claim-body:v1";

/// How a bound object bears on a claim.
///
/// The role is the whole semantics of a binding: the verifier's policy decides a
/// claim's [`ClaimStatus`](crate::verify::ClaimStatus) from the roles of its
/// **resolved** bindings, never from the claim's text. Roles fall into three
/// polarities — [`Polarity::Support`], [`Polarity::Contradict`], and
/// [`Polarity::Context`] (provenance/relation that is neither) — reported by
/// [`BindingRole::polarity`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum BindingRole {
    /// The object **directly** supports the claim (e.g. a proof-grade derivation
    /// whose conclusion *is* the claim).
    DirectlySupports,
    /// The object supports the claim **indirectly** (corroborating evidence, a
    /// consistent secondary result).
    IndirectlySupports,
    /// The object **contradicts** the claim. A resolved contradiction is never
    /// hidden and dominates support.
    Contradicts,
    /// The object **qualifies** the claim — bounds its scope, notes a caveat —
    /// without supporting or refuting it.
    Qualifies,
    /// The claim **depends on** the object (an assumption, a prior result it is
    /// conditional on) without that object being evidence for it.
    DependsOn,
    /// The object is a **reproduction** of the claimed result (a `sos-repro`
    /// verify transcript) — direct support specific to reproducibility.
    Reproduces,
    /// The object **supersedes** what the claim asserts (a later revision). Pure
    /// lineage, neither support nor contradiction.
    Supersedes,
    /// The object **supplies the method** behind the claim (a workflow plan /
    /// derivation of the procedure) — indirect support.
    SuppliesMethod,
    /// The object **supplies the data** behind the claim (a dataset, an
    /// experiment/simulation record) — indirect support.
    SuppliesData,
    /// The object **supplies the parameters** the claim was computed with —
    /// indirect support.
    SuppliesParameters,
    /// The object **supplies the uncertainty** quantification for the claim (an
    /// estimate with a standard error) — indirect support.
    SuppliesUncertainty,
}

/// The three ways an object can bear on a claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Polarity {
    /// Evidence for the claim (direct or indirect).
    Support,
    /// Evidence against the claim.
    Contradict,
    /// Context: provenance or relation that is neither for nor against.
    Context,
}

impl BindingRole {
    /// A stable discriminant used in the canonical encoding and for deterministic
    /// ordering. Never reused across variants.
    #[must_use]
    pub const fn discriminant(self) -> u64 {
        match self
        {
            Self::DirectlySupports => 0,
            Self::IndirectlySupports => 1,
            Self::Contradicts => 2,
            Self::Qualifies => 3,
            Self::DependsOn => 4,
            Self::Reproduces => 5,
            Self::Supersedes => 6,
            Self::SuppliesMethod => 7,
            Self::SuppliesData => 8,
            Self::SuppliesParameters => 9,
            Self::SuppliesUncertainty => 10,
        }
    }

    /// A short, stable code.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self
        {
            Self::DirectlySupports => "directly-supports",
            Self::IndirectlySupports => "indirectly-supports",
            Self::Contradicts => "contradicts",
            Self::Qualifies => "qualifies",
            Self::DependsOn => "depends-on",
            Self::Reproduces => "reproduces",
            Self::Supersedes => "supersedes",
            Self::SuppliesMethod => "supplies-method",
            Self::SuppliesData => "supplies-data",
            Self::SuppliesParameters => "supplies-parameters",
            Self::SuppliesUncertainty => "supplies-uncertainty",
        }
    }

    /// Whether this role is **direct** support (the strongest positive edge).
    #[must_use]
    pub const fn is_direct_support(self) -> bool {
        matches!(self, Self::DirectlySupports | Self::Reproduces)
    }

    /// Whether this role is **indirect** support (corroboration / supplies-*).
    #[must_use]
    pub const fn is_indirect_support(self) -> bool {
        matches!(
            self,
            Self::IndirectlySupports
                | Self::SuppliesMethod
                | Self::SuppliesData
                | Self::SuppliesParameters
                | Self::SuppliesUncertainty
        )
    }

    /// Whether this role supports the claim at all (direct or indirect).
    #[must_use]
    pub const fn is_support(self) -> bool {
        self.is_direct_support() || self.is_indirect_support()
    }

    /// Whether this role contradicts the claim.
    #[must_use]
    pub const fn is_contradiction(self) -> bool {
        matches!(self, Self::Contradicts)
    }

    /// This role's [`Polarity`].
    #[must_use]
    pub const fn polarity(self) -> Polarity {
        if self.is_contradiction()
        {
            Polarity::Contradict
        }
        else if self.is_support()
        {
            Polarity::Support
        }
        else
        {
            Polarity::Context
        }
    }
}

impl Canonical for BindingRole {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.u64(self.discriminant());
    }
}

/// One typed edge from a [`Claim`] to an object in the graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimBinding {
    /// How `object` bears on the claim.
    pub role: BindingRole,
    /// The content address of the object this edge points at.
    pub object: ObjectId,
    /// An optional human note explaining the edge (does not affect verification).
    pub note: Option<String>,
}

impl ClaimBinding {
    /// A binding of `role` to `object`, with no note.
    #[must_use]
    pub fn new(role: BindingRole, object: ObjectId) -> Self {
        Self {
            role,
            object,
            note: None,
        }
    }

    /// Attach an explanatory note.
    #[must_use]
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }

    /// A deterministic sort key: `(role discriminant, object, note)`. Used to
    /// normalize a claim's bindings so equal claims encode identically.
    fn sort_key(&self) -> (u64, ObjectId, Option<&str>) {
        (self.role.discriminant(), self.object, self.note.as_deref())
    }
}

impl Canonical for ClaimBinding {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.value(&self.role);
        enc.value(&self.object);
        enc.option(&self.note);
    }
}

/// A first-class, content-addressed scientific statement.
///
/// A claim carries a stable [`ClaimKey`] (its handle within a publication), the
/// `statement` itself, and the typed [`ClaimBinding`]s that connect it to the
/// object graph. Its bindings are held **sorted and deduplicated**, so two
/// claims with the same statement and the same set of edges have the same
/// [`content_id`](Self::content_id) regardless of the order they were added.
///
/// A claim has no provenance of its own; it is asserted *by* the publication
/// that embeds it, and that publication is the sealed, authored
/// [`Object`](sos_core::Object). The claim's [`content_id`](Self::content_id) is
/// its scientific-content address — stable, forge-proof, and independent of the
/// publication it appears in, so the same claim can be recognized across papers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Claim {
    /// The claim's handle within its publication.
    pub key: ClaimKey,
    /// The statement being asserted.
    pub statement: String,
    /// The typed edges into the object graph (sorted, deduplicated).
    pub bindings: Vec<ClaimBinding>,
}

impl Claim {
    /// A claim `key`: `statement`, with `bindings` normalized (sorted +
    /// deduplicated) so its content address is order-independent.
    #[must_use]
    pub fn new(
        key: impl Into<ClaimKey>,
        statement: impl Into<String>,
        bindings: Vec<ClaimBinding>,
    ) -> Self {
        let mut bindings = bindings;
        bindings.sort_by(|a, b| a.sort_key().cmp(&b.sort_key()));
        bindings.dedup();
        Self {
            key: key.into(),
            statement: statement.into(),
            bindings,
        }
    }

    /// The claim's content address: the hash of its canonical form under the
    /// claim domain. Independent of the publication that embeds it.
    #[must_use]
    pub fn content_id(&self) -> ObjectId {
        ObjectId::of(HashAlgo::default(), CLAIM_DOMAIN, self)
    }

    /// The objects bound with a supporting role (direct or indirect), in binding
    /// order.
    #[must_use]
    pub fn supporting_objects(&self) -> Vec<ObjectId> {
        self.bindings
            .iter()
            .filter(|b| b.role.is_support())
            .map(|b| b.object)
            .collect()
    }

    /// The objects bound with the contradicting role, in binding order.
    #[must_use]
    pub fn contradicting_objects(&self) -> Vec<ObjectId> {
        self.bindings
            .iter()
            .filter(|b| b.role.is_contradiction())
            .map(|b| b.object)
            .collect()
    }
}

impl Canonical for Claim {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.value(&self.key);
        enc.str(&self.statement);
        enc.seq(&self.bindings);
    }
}
