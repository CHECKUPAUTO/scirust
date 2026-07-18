//! # `scirust-hypermemory` — deterministic hypercomplex associative memory
//!
//! A **research-grade** deterministic associative-memory subsystem built on
//! SciRust's 16-dimensional sedenion algebra
//! ([`scirust_simd::hypercomplex::SedenionSimd`]) and *explicitly parenthesized*
//! sedenion compositions, evaluated against ordinary real-vector baselines.
//!
//! The authoritative design contract is
//! `docs/research/SCIRUST_HYPERMEMORY_PHASE1.md`.
//!
//! ## Positioning (read this before believing anything)
//!
//! This is **not** a database, vector database, RAG system, neural network, or
//! model of biological memory, and it makes **no** claim to be superior to
//! storing 16 plain real numbers. In fact, for exact similarity retrieval it is
//! *provably equivalent* to a real-vector index over the same components (see
//! [`Real16Index`] and the `index_matches_real16_baseline` test): the sedenion
//! algebra does nothing a real vector does not — **for retrieval**. The only
//! place the non-associative algebra is exercised is explicit relation
//! composition ([`S16Expr`]), and Phase 1 only provides the exact, auditable
//! machinery there, not a claim that it is useful.
//!
//! Every scientific claim in this crate is backed by a test, a benchmark, a
//! derivation, or is labelled a hypothesis. See the falsification criteria in
//! the research document.
//!
//! ## Phase 1 scope
//!
//! Phase 1 is the deterministic *exact-memory oracle*: generation-safe ids, a
//! slot/generation store, an exhaustive exact index, explicit relation trees,
//! and zero-divisor instrumentation. Approximate search, learning, persistence,
//! concurrency, and GPU are all out of scope (deferred to later phases).
//!
//! ## Example
//!
//! ```
//! use scirust_hypermemory::{ConceptSpec, S16Store, S16ExactIndex, SimilarityMetric};
//! use scirust_simd::hypercomplex::SedenionSimd;
//!
//! let mut store = S16Store::new();
//! let a = store
//!     .insert(ConceptSpec::new(b"apple".to_vec(), SedenionSimd::unit(1), 1.0, 0))
//!     .unwrap();
//! let b = store
//!     .insert(ConceptSpec::new(b"banana".to_vec(), SedenionSimd::unit(2), 1.0, 0))
//!     .unwrap();
//!
//! let mut index = S16ExactIndex::new(SimilarityMetric::Cosine);
//! index.insert_concept(store.get(a).unwrap());
//! index.insert_concept(store.get(b).unwrap());
//!
//! // Query nearest to unit(1): concept `a` wins.
//! let hits = index.search(&SedenionSimd::unit(1), 1).unwrap();
//! assert_eq!(hits[0].id, a);
//! ```

#![forbid(unsafe_code)]

mod baseline;
mod binding;
mod bounded;
mod diagnostics;
mod digest;
mod error;
mod experiments;
mod expr;
mod id;
mod index;
mod metadata;
mod record;
mod representation;
mod store;

pub use baseline::Real16Index;
pub use binding::{
    Encoding, RealBinding, RetrievalAccuracy, TripleShape, circular_convolution, cosine16,
    grouping_sensitivity, order_sensitivity, relative_distance16, structure_retrieval,
};
pub use bounded::{Insertion, S16BoundedMemory};
pub use diagnostics::{DEFAULT_NEAR_ZERO_THRESHOLD, ProductDiagnostics};
pub use digest::{DOMAIN_CONCEPT, DOMAIN_EXPRESSION, Digest32, concept_digest};
pub use error::{HypermemoryError, Result};
pub use experiments::{
    Lcg, OperandDistribution, StructureDiscriminationSurvey, ZeroDivisorSurvey,
    relative_associator, survey_structure_discrimination, survey_zero_divisors,
};
pub use expr::{
    ExprLimits, MAX_SUPPORTED_DEPTH, MAX_SUPPORTED_SIZE, RelationId, S16Expr, S16Relation,
};
pub use id::ConceptId;
pub use index::{S16ExactIndex, SearchHit, SimilarityMetric};
pub use metadata::{ConceptMetadata, LinearDecay, NoForgetting, RetentionPolicy};
pub use record::ConceptRecord;
pub use representation::{effective_representation, is_finite, norm_sqr_ordered};
pub use store::{ConceptSpec, DEFAULT_RESIDUAL_BOUND, LearnOutcome, S16Store};

#[cfg(test)]
mod layout_tests {
    use core::mem::{align_of, size_of};
    use scirust_simd::hypercomplex::SedenionSimd;

    /// The whole crate depends on this layout invariant of the reused
    /// representation. Re-assert it so an upstream change is caught here.
    #[test]
    fn sedenion_layout_is_reused_unchanged() {
        assert_eq!(size_of::<SedenionSimd>(), 64);
        assert_eq!(align_of::<SedenionSimd>(), 64);
    }
}
