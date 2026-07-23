//! [`Cognition`] — the capability-scoped cognitive session.
//!
//! This is the orchestrator that puts the pieces together the way the RFC frames
//! the cognitive backend: a **capability-scoped** front to the untrusted
//! proposer. Proposing and recalling are cognitive acts that require an explicit
//! capability grant (refused by default — least privilege over the untrusted
//! side); every act is **attested** into the session's chain; and disposing of a
//! proposal is the deterministic, always-available trusted operation. Cognition
//! proposes (gated, attested); determinism disposes (open, recorded).

use sos_core::canonical::Canonical;
use sos_core::{Author, Object};
use sos_registry::{Capability, Grant};

use crate::attest::CcosChainRef;
use crate::disposition::{Admission, Ruling, dispose, seal_admission};
use crate::error::{CcosError, Result};
use crate::memory::{Recall, Remember};
use crate::proposal::{Proposal, seal_proposal};

/// The capability required to generate a proposal (the untrusted cognitive act).
#[must_use]
pub fn propose_capability() -> Capability {
    Capability::Custom("ccos:propose".to_owned())
}

/// The capability required to recall from scientific memory.
#[must_use]
pub fn recall_capability() -> Capability {
    Capability::Custom("ccos:recall".to_owned())
}

/// A capability-scoped session over a [`Remember`] backend.
///
/// The `grant` bounds what cognitive acts are permitted; a session created with
/// [`Grant::new`] (the empty grant) can dispose but can neither propose nor
/// recall. Every act performed is attested into the backend's chain.
#[derive(Debug, Clone)]
pub struct Cognition<M: Remember> {
    memory: M,
    grant: Grant,
}

impl<M: Remember> Cognition<M> {
    /// A session over `memory`, permitted the cognitive acts in `grant`.
    #[must_use]
    pub fn new(memory: M, grant: Grant) -> Self {
        Self { memory, grant }
    }

    /// Borrow the underlying memory (for ungated reads such as context paging).
    #[must_use]
    pub fn memory(&self) -> &M {
        &self.memory
    }

    /// Consume the session, returning the underlying memory.
    #[must_use]
    pub fn into_memory(self) -> M {
        self.memory
    }

    /// Generate an untrusted proposal: seal it (authored by `agent`), persist it
    /// to memory, and attest the act. The returned object is **untrusted** until
    /// [`dispose`](Self::dispose)d.
    ///
    /// # Errors
    /// [`CcosError::Denied`] if the session was not granted
    /// [`propose_capability`].
    pub fn propose(&mut self, proposal: Proposal, agent: Author) -> Result<Object<Proposal>> {
        self.require(&propose_capability())?;
        let object = seal_proposal(proposal, agent);
        self.memory.store(&object);
        self.memory.attest(
            object.body.statement.as_bytes(),
            object.id.digest().as_bytes(),
        );
        Ok(object)
    }

    /// Recall up to `k` prior proposals matching `query`.
    ///
    /// # Errors
    /// [`CcosError::Denied`] if the session was not granted [`recall_capability`].
    pub fn recall(&self, query: &Recall, k: usize) -> Result<Vec<sos_core::ObjectId>> {
        self.require(&recall_capability())?;
        Ok(self.memory.recall(query, k))
    }

    /// Dispose of a proposal under a deterministic `ruling`, sealing the
    /// [`Admission`] (authored by the deterministic `engine`) and attesting the
    /// act. Always available — deterministic disposition is the trusted side and
    /// needs no capability.
    pub fn dispose(
        &mut self,
        proposal: &Object<Proposal>,
        ruling: Ruling,
        engine: Author,
    ) -> Object<Admission> {
        let disposition = dispose(proposal, ruling);
        let admission = Admission::new(proposal.id, disposition);
        self.memory.attest(
            proposal.id.digest().as_bytes(),
            &admission.canonical_bytes(),
        );
        seal_admission(admission, engine)
    }

    /// Enforce that `capability` was granted to this session (refuse by default).
    fn require(&self, capability: &Capability) -> Result<()> {
        if self.grant.capabilities().contains(capability)
        {
            Ok(())
        }
        else
        {
            Err(CcosError::Denied {
                capability: capability.to_string(),
            })
        }
    }

    /// Attest an arbitrary cognitive act's `input`/`output` directly (e.g. a
    /// recall served, or an external agent action), returning its chain
    /// reference. Attestation is an integrity mechanism, always available.
    pub fn attest(&mut self, input: &[u8], output: &[u8]) -> CcosChainRef {
        self.memory.attest(input, output)
    }
}
