//! # `sos-publication` — the SOS Publication & Claim-Verification Engine
//!
//! **A publication is a verifiable projection of the SOS scientific object
//! graph** (RFC-0002 §10.5). Not a document that *describes* results, but one
//! whose every statement is an explicit, typed edge into the immutable object
//! DAG — so a third party can ask, and the engine can answer *mechanically*:
//!
//! * which object supports this claim, and by what kind of evidence?
//! * is anything **contradicted**? (never hidden)
//! * does every dependency resolve, or does the claim lean on something absent?
//! * is the declared scope complete, or is support drawn from outside it?
//! * is the reproducibility bar met — at the determinism level the evidence
//!   actually realized?
//! * can the figures and tables still be regenerated?
//! * has the document changed since it was released?
//!
//! ## Integrity is not truth
//!
//! Sealing a [`Publication`] yields a content-addressed [`Object`](sos_core::Object)
//! whose id covers the whole document — tamper-evidence for free. But a
//! well-formed, correctly-hashed publication may still be unsupported,
//! contradicted, or irreproducible. [`verify`] is what separates the two: it
//! produces a [`VerificationReport`] of per-claim [`ClaimStatus`], structural
//! findings, unresolved dependencies, and a reproducibility summary. A
//! contradiction is *reported*, never raised as an error and never hidden.
//!
//! ## Consume, never recompute
//!
//! This engine reads decisions the other engines already made — it never
//! re-runs reasoning, simulation, Bayesian design, or theory evaluation. It
//! looks at the graph through one read-only [`PublicationObjectSource`], learning
//! only an object's kind, parents, and realized [`DeterminismLevel`](sos_core::DeterminismLevel).
//! Producing a figure's pixels or re-executing a workflow is the Workflow
//! Engine's job; [`verify_exhibits`] only checks a supplied re-render against
//! what was pinned, and localizes drift.
//!
//! ## What is here
//!
//! * [`Claim`] / [`ClaimBinding`] / [`BindingRole`] — first-class,
//!   content-addressed statements wired to the graph by typed edges.
//! * [`Publication`] / [`Section`] / [`Block`] — the ordered, typed document with
//!   claim, figure, table, and bibliography registries and a declared scope.
//! * [`FigureSpec`] / [`TableSpec`] — regenerable exhibit *recipes*, checkable
//!   for drift.
//! * [`Reference`] — a bibliography that keeps in-graph evidence and external,
//!   unverifiable literature structurally distinct (a citation is not evidence).
//! * [`StandardPolicy`] — the versioned, fully explicit support-decision engine
//!   (no opaque scoring).
//! * [`verify`], [`verify_exhibits`], [`check_release`] — the verification
//!   pipeline; [`render`] — deterministic Markdown/HTML/JSON; [`diff`] — a
//!   semantic publication diff.
//!
//! Signing is not done here: this crate never fabricates a signature and defers
//! real Merkle/Lamport attestation to `sos-provenance`; the content address is
//! intrinsic regardless. LaTeX/PDF rendering needs a typesetting backend and is
//! deferred per Invariant VIII. No stub, no placeholder.
//!
//! ## Example
//!
//! ```
//! use sos_core::{Author, DeterminismLevel, HashAlgo, Kind, ObjectId};
//! use sos_publication::{
//!     render, seal_publication, verify, BindingRole, Claim, ClaimBinding, ClaimStatus,
//!     Format, MapSource, ObjectFacts, Publication, StandardPolicy,
//! };
//!
//! // The graph the publication is checked against: one bit-reproducible derivation.
//! let evidence = ObjectId::compute(HashAlgo::default(), b"demo", b"derivation-1");
//! let mut graph = MapSource::new();
//! graph.insert(ObjectFacts::new(
//!     evidence,
//!     Kind::new("Derivation", 1),
//!     Vec::new(),
//!     DeterminismLevel::L3,
//! ));
//!
//! // A claim, directly supported by that derivation.
//! let claim = Claim::new(
//!     "C1",
//!     "Period squared scales with semi-major axis cubed.",
//!     vec![ClaimBinding::new(BindingRole::DirectlySupports, evidence)],
//! );
//! let paper = Publication::builder("Kepler's Third Law, Rederived")
//!     .author(Author::human("ada"))
//!     .declared_root(evidence)
//!     .claim(claim)
//!     .build();
//!
//! // It seals to a content-addressed object.
//! let obj = seal_publication(paper.clone(), Author::human("curator"));
//! assert!(obj.verify_id());
//! assert_eq!(obj.kind.name, "Publication");
//!
//! // Verification: the claim is supported by a resolved, in-scope derivation,
//! // and the document is fit to publish.
//! let report = verify(&paper, &graph, &StandardPolicy::new()).unwrap();
//! assert_eq!(report.claims[0].status, ClaimStatus::Supported);
//! assert!(report.is_publishable());
//!
//! // Point the same claim at a missing object and it is caught, not hidden.
//! let ghost = ObjectId::compute(HashAlgo::default(), b"demo", b"absent");
//! let broken_claim = Claim::new(
//!     "C1",
//!     "Rests on an object not in the graph.",
//!     vec![ClaimBinding::new(BindingRole::DirectlySupports, ghost)],
//! );
//! let broken = Publication::builder("Broken")
//!     .declared_root(evidence)
//!     .claim(broken_claim)
//!     .build();
//! let report = verify(&broken, &graph, &StandardPolicy::new()).unwrap();
//! assert_eq!(report.claims[0].status, ClaimStatus::DependencyMissing);
//! assert!(!report.is_publishable());
//!
//! // And it renders deterministically.
//! let md = render(&paper, Format::Markdown).unwrap();
//! assert!(md.content.contains("Kepler's Third Law, Rederived"));
//! assert!(md.content.contains(&evidence.to_string()));
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

pub mod claim;
pub mod diff;
pub mod error;
pub mod exhibit;
pub mod key;
pub mod policy;
pub mod publication;
pub mod publisher;
pub mod reference;
pub mod release;
pub mod render;
pub mod section;
pub mod source;
pub mod verify;

pub use claim::{BindingRole, Claim, ClaimBinding, Polarity};
pub use diff::{PublicationDiff, diff};
pub use error::{PublicationError, Result, SourceError};
pub use exhibit::{
    ColumnDef, FigureSpec, MediaType, Ordering, Param, ParamValue, RegenPolicy, RendererId,
    TableSpec,
};
pub use key::{ClaimKey, FigureKey, RefKey, SectionId, TableKey};
pub use policy::{BindingContext, ClaimContext, PolicyId, StandardPolicy, SupportPolicy};
pub use publication::{
    Publication, PublicationBuilder, PublicationMeta, ReproRequirement, seal_publication,
};
pub use publisher::{Publish, Publisher};
pub use reference::{ExternalCitation, Reference};
pub use release::{ReleaseManifest, seal_release};
pub use render::{Artifact, Format, render};
pub use section::{Block, Section, SectionKind};
pub use source::{
    MapSource, ObjectFacts, PublicationObjectSource, StoreSource, dependency_closure,
};
pub use verify::{
    ClaimAssessment, ClaimStatus, ExhibitAssessment, ExhibitClass, ExhibitReport, ExhibitVerdict,
    ReleaseConsistency, ReproVerdict, StructuralIssue, VerificationReport, check_release,
    structural_issues, verify, verify_exhibits,
};
