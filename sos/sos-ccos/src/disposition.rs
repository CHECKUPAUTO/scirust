//! The trust boundary: [`dispose`] an untrusted [`Proposal`] into an
//! [`Admission`] — the mechanism that enforces **Invariant IX** (a cognitive
//! proposal becomes trusted only by surviving deterministic verification).
//!
//! Cognition proposes; determinism disposes. A caller runs a deterministic
//! engine (reasoning, theory, reproduction) over a proposal and forms a
//! [`Ruling`]. [`dispose`] turns that ruling into an [`Admission`] — but it is
//! not a rubber stamp: even an `Admit` ruling is **downgraded to a rejection**
//! if the proposal's content address does not verify or it is ungrounded. The
//! only way to obtain a [`Trusted`] reference is from an admitted [`Admission`],
//! so a proposal can never be used as trusted without a recorded verdict.

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{Author, Body, Object, ObjectId};

use crate::proposal::Proposal;

/// A caller's deterministic decision about a proposal, formed after running a
/// deterministic engine over it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Ruling {
    /// Admit the proposal — a deterministic engine verified it. Carries the
    /// content address of the **verdict object** (e.g. a `Derivation`) that did
    /// so, so the admission is auditable back to real reasoning.
    Admit {
        /// The verdict object that verified the proposal.
        verdict: ObjectId,
    },
    /// Reject the proposal, with a reason.
    Reject {
        /// Why the proposal was rejected.
        reason: String,
    },
}

/// The recorded outcome of disposing of a proposal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Disposition {
    /// The proposal was admitted, verified by `verdict`.
    Admitted {
        /// The verdict object that admitted it.
        verdict: ObjectId,
    },
    /// The proposal was rejected, with a reason.
    Rejected {
        /// Why it was rejected.
        reason: String,
    },
}

impl Disposition {
    const fn discriminant(&self) -> u64 {
        match self
        {
            Self::Admitted { .. } => 0,
            Self::Rejected { .. } => 1,
        }
    }

    /// Whether the proposal was admitted.
    #[must_use]
    pub const fn is_admitted(&self) -> bool {
        matches!(self, Self::Admitted { .. })
    }

    /// The verdict object, if admitted.
    #[must_use]
    pub const fn verdict(&self) -> Option<ObjectId> {
        match self
        {
            Self::Admitted { verdict } => Some(*verdict),
            Self::Rejected { .. } => None,
        }
    }
}

impl Canonical for Disposition {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.u64(self.discriminant());
        match self
        {
            Self::Admitted { verdict } => enc.value(verdict),
            Self::Rejected { reason } => enc.str(reason),
        }
    }
}

/// Dispose of a proposal given a caller's [`Ruling`].
///
/// The determinism check is not bypassable: a proposal whose content address
/// does not verify, or that grounds in nothing, is **always** rejected — even if
/// the caller passed [`Ruling::Admit`]. This is the safety property that keeps a
/// tampered or ungrounded cognitive artifact out of the trusted graph regardless
/// of an over-eager ruling.
#[must_use]
pub fn dispose(proposal: &Object<Proposal>, ruling: Ruling) -> Disposition {
    if !proposal.verify_id()
    {
        return Disposition::Rejected {
            reason: "proposal content address does not verify (tampered)".to_owned(),
        };
    }
    if !proposal.body.is_grounded()
    {
        return Disposition::Rejected {
            reason: "ungrounded proposal".to_owned(),
        };
    }
    match ruling
    {
        Ruling::Admit { verdict } => Disposition::Admitted { verdict },
        Ruling::Reject { reason } => Disposition::Rejected { reason },
    }
}

/// A content-addressed record that a proposal was disposed of by determinism.
///
/// The admission is the **trusted anchor**: it names the proposal and the
/// [`Disposition`] a deterministic engine reached. It is authored by the engine,
/// not the proposing agent, and never hides a rejection — a rejected admission
/// is a first-class, recorded object too.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Admission {
    /// The proposal that was disposed of.
    pub proposal: ObjectId,
    /// The deterministic outcome.
    pub disposition: Disposition,
}

impl Admission {
    /// Construct an admission record.
    #[must_use]
    pub fn new(proposal: ObjectId, disposition: Disposition) -> Self {
        Self {
            proposal,
            disposition,
        }
    }

    /// Whether the proposal was admitted into the trusted graph.
    #[must_use]
    pub const fn is_admitted(&self) -> bool {
        self.disposition.is_admitted()
    }

    /// A [`Trusted`] reference to the proposal — `Some` **only** if it was
    /// admitted. This is the sole constructor of [`Trusted`], so trusted use of a
    /// cognitive proposal always carries the verdict that earned it.
    #[must_use]
    pub fn into_trusted(&self) -> Option<Trusted> {
        self.disposition.verdict().map(|verdict| Trusted {
            proposal: self.proposal,
            verdict,
        })
    }
}

impl Canonical for Admission {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.value(&self.proposal);
        enc.value(&self.disposition);
    }
}

impl Body for Admission {
    const KIND: &'static str = "Admission";
    const SCHEMA_VERSION: u32 = 1;
}

/// A proposal that has been admitted into the trusted graph, carrying the
/// verdict object that admitted it.
///
/// There is no public constructor: a `Trusted` can be obtained only from
/// [`Admission::into_trusted`] on an admitted admission. That is the type-level
/// expression of Invariant IX — you cannot hold a trusted cognitive reference
/// without a deterministic verdict behind it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Trusted {
    proposal: ObjectId,
    verdict: ObjectId,
}

impl Trusted {
    /// The admitted proposal's content address.
    #[must_use]
    pub const fn proposal(&self) -> ObjectId {
        self.proposal
    }

    /// The verdict object that admitted it.
    #[must_use]
    pub const fn verdict(&self) -> ObjectId {
        self.verdict
    }
}

/// Seal an [`Admission`] as an `Object<Admission>` authored by the deterministic
/// `engine`, with the proposal (and, if admitted, the verdict) as provenance
/// parents.
///
/// The engine — not the proposing agent — authors the admission: the trusted
/// disposition is a determinism act.
#[must_use]
pub fn seal_admission(admission: Admission, engine: Author) -> Object<Admission> {
    let mut parents = vec![admission.proposal];
    if let Some(verdict) = admission.disposition.verdict()
    {
        parents.push(verdict);
    }
    Object::builder(admission)
        .parents(parents)
        .author(engine)
        .seal()
}
