//! The [`Remember`] cognitive syscall and [`LocalMemory`] — its deterministic,
//! no-LLM fallback.
//!
//! `Remember` is the cognitive backend's interface: persist a proposal to
//! scientific memory, recall prior proposals, attest an act, and page a bounded
//! context window. The cognitive backend is **optional** (RFC-0002 §10.3):
//! without it, [`LocalMemory`] provides a fully deterministic implementation —
//! recall degrades from generative similarity to **exact structural overlap**
//! (prior proposals that concern the same objects), so "what has been proposed
//! about this?" stays replay-exact and auditable. It is a real memory, not a
//! mock; a persistent, embedding-backed memory is a follow-on that implements the
//! same trait.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sos_core::{Object, ObjectId};

use crate::attest::{CcosChain, CcosChainRef};
use crate::proposal::{Proposal, ProposalKind};

/// A bounded token budget for a [`ContextPage`] — the number of objects the page
/// may include.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TokenBudget(pub u64);

impl TokenBudget {
    /// Construct a budget.
    #[must_use]
    pub const fn new(objects: u64) -> Self {
        Self(objects)
    }
}

/// A recall query: prior proposals concerning any of `seeds`, optionally
/// filtered to one `kind`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Recall {
    /// The seed objects to recall proposals about.
    pub seeds: Vec<ObjectId>,
    /// If set, only recall proposals of this kind.
    pub kind: Option<ProposalKind>,
}

impl Recall {
    /// Recall proposals concerning any of `seeds` (no kind filter).
    #[must_use]
    pub fn about(seeds: Vec<ObjectId>) -> Self {
        Self { seeds, kind: None }
    }

    /// Restrict the recall to a single proposal kind.
    #[must_use]
    pub fn of_kind(mut self, kind: ProposalKind) -> Self {
        self.kind = Some(kind);
        self
    }
}

/// A bounded, deterministic window of scientific memory around a focus object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextPage {
    /// The objects in the page, sorted and within budget.
    pub objects: Vec<ObjectId>,
    /// Whether the page was truncated to fit the budget (more was available).
    pub truncated: bool,
}

/// The cognitive syscall: persist, recall, attest, and page scientific memory.
///
/// Storing and recalling are *memory* operations (trusted for persistence, not
/// for judgement — a recalled proposal is still untrusted); attestation is the
/// integrity mechanism; context paging keeps cognition within a bounded,
/// reproducible window.
pub trait Remember {
    /// Persist a proposal to scientific memory.
    fn store(&mut self, proposal: &Object<Proposal>);

    /// Recall up to `k` prior proposal ids matching `query`, best match first.
    fn recall(&self, query: &Recall, k: usize) -> Vec<ObjectId>;

    /// Attest a cognitive act with `input`/`output` bytes, returning a reference
    /// into the attestation chain.
    fn attest(&mut self, input: &[u8], output: &[u8]) -> CcosChainRef;

    /// A bounded, deterministic context window around `focus`.
    fn context(&self, focus: ObjectId, budget: TokenBudget) -> ContextPage;
}

/// What [`LocalMemory`] retains about a stored proposal.
#[derive(Debug, Clone, PartialEq, Eq)]
struct StoredProposal {
    kind: ProposalKind,
    concerns: Vec<ObjectId>,
}

/// A deterministic, in-memory [`Remember`] implementation — the fallback used
/// when no cognitive backend is attached.
///
/// Recall is exact structural overlap: proposals are indexed by the objects they
/// concern, and [`recall`](Remember::recall) ranks candidates by how many of the
/// query seeds they share (ties broken by object id). Nothing is generative;
/// everything is replay-exact.
#[derive(Debug, Clone, Default)]
pub struct LocalMemory {
    proposals: BTreeMap<ObjectId, StoredProposal>,
    by_concern: BTreeMap<ObjectId, BTreeSet<ObjectId>>,
    chain: CcosChain,
}

impl LocalMemory {
    /// An empty memory.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// How many proposals are stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.proposals.len()
    }

    /// Whether the memory holds no proposals.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.proposals.is_empty()
    }

    /// The attestation chain of cognitive acts recorded through this memory.
    #[must_use]
    pub fn chain(&self) -> &CcosChain {
        &self.chain
    }
}

impl Remember for LocalMemory {
    fn store(&mut self, proposal: &Object<Proposal>) {
        let id = proposal.id;
        for concern in &proposal.body.concerns
        {
            self.by_concern.entry(*concern).or_default().insert(id);
        }
        self.proposals.insert(
            id,
            StoredProposal {
                kind: proposal.body.kind,
                concerns: proposal.body.concerns.clone(),
            },
        );
    }

    fn recall(&self, query: &Recall, k: usize) -> Vec<ObjectId> {
        // Count, for each candidate proposal, how many query seeds it shares.
        let mut overlap: BTreeMap<ObjectId, usize> = BTreeMap::new();
        for seed in &query.seeds
        {
            if let Some(candidates) = self.by_concern.get(seed)
            {
                for &candidate in candidates
                {
                    *overlap.entry(candidate).or_insert(0) += 1;
                }
            }
        }
        let mut ranked: Vec<(usize, ObjectId)> = overlap
            .into_iter()
            .filter(|(id, _)| match query.kind
            {
                Some(kind) => self.proposals.get(id).is_some_and(|p| p.kind == kind),
                None => true,
            })
            .map(|(id, count)| (count, id))
            .collect();
        // Best overlap first; ties broken by object id ascending (deterministic).
        ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        ranked.into_iter().take(k).map(|(_, id)| id).collect()
    }

    fn attest(&mut self, input: &[u8], output: &[u8]) -> CcosChainRef {
        self.chain.append(input, output)
    }

    fn context(&self, focus: ObjectId, budget: TokenBudget) -> ContextPage {
        // A one-hop, deterministic neighborhood: the focus, the proposals about
        // it, and the other objects those proposals concern.
        let mut objects: BTreeSet<ObjectId> = BTreeSet::new();
        objects.insert(focus);
        if let Some(proposals) = self.by_concern.get(&focus)
        {
            for proposal in proposals
            {
                objects.insert(*proposal);
                if let Some(stored) = self.proposals.get(proposal)
                {
                    for concern in &stored.concerns
                    {
                        objects.insert(*concern);
                    }
                }
            }
        }
        let budget = budget.0 as usize;
        let truncated = objects.len() > budget;
        let objects = objects.into_iter().take(budget).collect();
        ContextPage { objects, truncated }
    }
}
