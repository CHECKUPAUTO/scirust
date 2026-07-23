//! # `sos-ccos` — the SOS Cognitive Backend Adapter (deterministic core)
//!
//! The cognitive backend (an LLM, an agent, a "Causal-Context OS") is where the
//! Scientific Operating System touches generative intelligence — and it is the
//! one place the system must **never trust**. RFC-0002's Invariant IX draws the
//! line: **cognition is a proposer only.** It supplies leads and memory;
//! determinism supplies verdicts. This crate is the deterministic boundary that
//! makes that safe, with the LLM itself deferred to an out-of-process backend
//! (Invariant VIII) — no FFI, no network, no LLM in the core.
//!
//! ## The trust boundary
//!
//! * [`Proposal`] — a first-class, content-addressed **untrusted** suggestion
//!   (question / hypothesis / analogy / conjecture). It must **ground** in at
//!   least one real object; a proposal about nothing is refused at construction.
//! * [`dispose`] / [`Admission`] — the gate. A caller runs a deterministic engine
//!   over a proposal and forms a [`Ruling`]; `dispose` records an [`Admission`].
//!   The check is not bypassable: a tampered or ungrounded proposal is rejected
//!   even under an `Admit` ruling. A [`Trusted`] reference exists **only** via an
//!   admitted admission — the type system enforces "no verdict, no trust."
//! * [`CcosChain`] — a tamper-evident, hash-chained attestation of every
//!   cognitive act (`input → output → chain`), storing only hashes; any
//!   alteration is detected and localized by [`CcosChain::verify`].
//! * [`Remember`] / [`LocalMemory`] — the cognitive syscall and its deterministic
//!   fallback: with no backend attached, recall degrades from generative
//!   similarity to **exact structural overlap**, staying replay-exact.
//! * [`Cognition`] — a capability-scoped session: proposing and recalling require
//!   an explicit [`Grant`](sos_registry::Grant) (refused by default); disposing
//!   is the always-available trusted side.
//!
//! ## What is deliberately deferred (Invariant VIII) — no stub
//!
//! Generation itself (the LLM/CCOS proposer) and embedding-backed semantic recall
//! live in the out-of-process backend and its `scirust-retrieval` bridge; this
//! crate is the untrusted-proposal envelope, the deterministic disposition gate,
//! the attestation chain, and the deterministic memory fallback. Persistent,
//! store-backed cognitive memory is a follow-on that implements the same
//! [`Remember`] trait.
//!
//! ## Example
//!
//! ```
//! use sos_core::{Author, HashAlgo, ObjectId};
//! use sos_registry::{Capability, Grant};
//! use sos_ccos::{
//!     propose_capability, Cognition, LocalMemory, Proposal, ProposalKind, Ruling,
//! };
//!
//! // Two real objects the cognitive backend will reason about.
//! let node = |t: &[u8]| ObjectId::compute(HashAlgo::default(), b"demo", t);
//! let (oscillator, ou_process) = (node(b"damped-oscillator"), node(b"ornstein-uhlenbeck"));
//!
//! // A session granted the right to propose (least privilege: nothing else).
//! let grant = Grant::new().allow(propose_capability());
//! let mut ccos = Cognition::new(LocalMemory::new(), grant);
//!
//! // The backend proposes an analogy — grounded in both objects, and untrusted.
//! let proposal = Proposal::new(
//!     ProposalKind::Analogy,
//!     "A damped oscillator is analogous to an Ornstein–Uhlenbeck process.",
//!     vec![oscillator, ou_process],
//!     "Both relax to equilibrium at a rate set by a single time constant.",
//! )
//! .unwrap();
//! let untrusted = ccos.propose(proposal, Author::agent("ccos")).unwrap();
//! assert_eq!(untrusted.kind.name, "Proposal");
//!
//! // Determinism disposes: a reasoning engine verified it, yielding a verdict
//! // object; the admission admits the proposal into the trusted graph.
//! let verdict = node(b"derivation-that-verifies-the-analogy");
//! let admission = ccos.dispose(&untrusted, Ruling::Admit { verdict }, Author::engine("sos-reasoning"));
//! assert!(admission.verify_id());
//! assert!(admission.body.is_admitted());
//! let trusted = admission.body.into_trusted().unwrap();
//! assert_eq!(trusted.proposal(), untrusted.id);
//! assert_eq!(trusted.verdict(), verdict);
//!
//! // A rejected proposal yields no trusted reference — cognition does not decide.
//! let rejected = ccos.dispose(
//!     &untrusted,
//!     Ruling::Reject { reason: "no supporting derivation".into() },
//!     Author::engine("sos-reasoning"),
//! );
//! assert!(!rejected.body.is_admitted());
//! assert!(rejected.body.into_trusted().is_none());
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

pub mod attest;
pub mod disposition;
pub mod error;
pub mod memory;
pub mod proposal;
pub mod session;

pub use attest::{Attestation, CcosChain, CcosChainRef};
pub use disposition::{Admission, Disposition, Ruling, Trusted, dispose, seal_admission};
pub use error::{CcosError, Result};
pub use memory::{ContextPage, LocalMemory, Recall, Remember, TokenBudget};
pub use proposal::{Proposal, ProposalKind, seal_proposal};
pub use session::{Cognition, propose_capability, recall_capability};
