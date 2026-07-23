//! [`Proposal`] — a first-class **untrusted** cognitive suggestion, and its
//! [`ProposalKind`].
//!
//! A proposal is what the cognitive backend (an LLM, an agent) contributes: a
//! question, hypothesis, analogy, or conjecture. It is content-addressed and
//! enters the graph like any other object — but it is **untrusted by
//! construction**. Nothing about a proposal being well-formed, well-written, or
//! confidently phrased makes it true; it becomes part of the trusted graph only
//! by surviving a deterministic [`disposition`](crate::disposition) (Invariant
//! IX). A proposal must **ground** in at least one real object: cognition
//! supplies leads *about the graph*, never free-floating assertions.

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{Author, Body, Object, ObjectId};

use crate::error::{CcosError, Result};

/// What kind of cognitive suggestion a [`Proposal`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ProposalKind {
    /// A question worth investigating.
    Question,
    /// A hypothesis to be tested.
    Hypothesis,
    /// An analogy between two areas of the graph.
    Analogy,
    /// A conjecture — a claim advanced without proof.
    Conjecture,
}

impl ProposalKind {
    /// A stable discriminant used in the canonical encoding.
    #[must_use]
    pub const fn discriminant(self) -> u64 {
        match self
        {
            Self::Question => 0,
            Self::Hypothesis => 1,
            Self::Analogy => 2,
            Self::Conjecture => 3,
        }
    }

    /// A short, stable code.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self
        {
            Self::Question => "question",
            Self::Hypothesis => "hypothesis",
            Self::Analogy => "analogy",
            Self::Conjecture => "conjecture",
        }
    }
}

impl Canonical for ProposalKind {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.u64(self.discriminant());
    }
}

/// A content-addressed, **untrusted** cognitive suggestion grounded in the graph.
///
/// The `concerns` are the objects the proposal is about — its grounding. A
/// proposal with no concerns cannot be built ([`Proposal::new`] returns
/// [`CcosError::Ungrounded`]); concerns are held sorted and deduplicated so the
/// proposal's content address does not depend on their order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Proposal {
    /// What kind of suggestion this is.
    pub kind: ProposalKind,
    /// The suggestion itself.
    pub statement: String,
    /// The objects this proposal is about — its grounding (non-empty, sorted,
    /// deduplicated).
    pub concerns: Vec<ObjectId>,
    /// The cognitive backend's stated rationale (a *lead*, not evidence).
    pub rationale: String,
}

impl Proposal {
    /// A proposal of `kind` stating `statement`, grounded in `concerns`, with
    /// `rationale`.
    ///
    /// # Errors
    /// [`CcosError::Ungrounded`] if `concerns` is empty.
    pub fn new(
        kind: ProposalKind,
        statement: impl Into<String>,
        concerns: Vec<ObjectId>,
        rationale: impl Into<String>,
    ) -> Result<Self> {
        if concerns.is_empty()
        {
            return Err(CcosError::Ungrounded);
        }
        let mut concerns = concerns;
        concerns.sort_unstable();
        concerns.dedup();
        Ok(Self {
            kind,
            statement: statement.into(),
            concerns,
            rationale: rationale.into(),
        })
    }

    /// Whether this proposal grounds in at least one object.
    #[must_use]
    pub fn is_grounded(&self) -> bool {
        !self.concerns.is_empty()
    }
}

impl Canonical for Proposal {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.value(&self.kind);
        enc.str(&self.statement);
        enc.seq(&self.concerns);
        enc.str(&self.rationale);
    }
}

impl Body for Proposal {
    const KIND: &'static str = "Proposal";
    const SCHEMA_VERSION: u32 = 1;
}

/// Seal a [`Proposal`] as an `Object<Proposal>` authored by the proposing
/// `agent`, with its grounding objects as provenance parents.
///
/// The sealed object enters the graph as an **untrusted** cognitive artifact.
/// Marking it as authored by an [`Author::Agent`] records that it came from the
/// cognitive side; its parents are exactly the objects it concerns, so the
/// grounding is part of its Merkle identity.
#[must_use]
pub fn seal_proposal(proposal: Proposal, agent: Author) -> Object<Proposal> {
    let parents = proposal.concerns.clone();
    Object::builder(proposal)
        .parents(parents)
        .author(agent)
        .seal()
}
