#![forbid(unsafe_code)]
//! Exact, deterministic experimentation on locally generated toy elliptic curves.
//!
//! This crate deliberately accepts only small prime fields and locally specified
//! curve parameters. It has no key, address, SEC 1, network, or blockchain API.
//! All arithmetic is exact and delegates modular primitives to scirust-modalg.

pub mod canonical;
pub mod catalog;
pub mod classify;
pub mod controls;
pub mod curve;
pub mod enumerate;
pub mod experiment;
pub mod field;
pub mod falsify;
pub mod grammar;
pub mod invariant;
pub mod orders;
pub mod report;
pub mod search;
pub mod scope;

pub use curve::{CurveError, ToyCurve, ToyPoint};
pub use experiment::{Corpus, CorpusCurve, ExperimentManifest};
pub use field::{FieldError, Fp, PrimeError, ToyPrime};
pub use falsify::{Counterexample, first_point_counterexample};
pub use grammar::{PointExpression, Relation, generate_relations};
pub use invariant::CurveInvariants;
pub use search::{
    CandidateEvaluation, GateReport, GateState, ResearchCorpora, SearchError, SearchPlan,
    evaluate_candidate, run_search,
};
pub use report::ExperimentReport;
pub use scope::{CorpusKind, LocalResearchCase, ScopeError};
pub use catalog::{CatalogEntry, CatalogFamily, RelationSignature, catalog_entry};
pub use classify::{Classification, ClassificationStatus, classify};
pub use controls::{ControlId, ControlResult, run_control};
