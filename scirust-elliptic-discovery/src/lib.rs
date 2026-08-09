#![forbid(unsafe_code)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_is_multiple_of)]
//! Exact, deterministic experimentation on locally generated toy elliptic curves.
//!
//! This crate deliberately accepts only small prime fields and locally specified
//! curve parameters. It has no key, address, SEC 1, network, or blockchain API.
//! All arithmetic is exact and delegates modular primitives to scirust-modalg.

pub mod adjacent;
pub mod campaign;
pub mod canonical;
pub mod catalog;
pub mod classify;
pub mod controls;
pub mod curve;
pub mod edwards;
pub mod enumerate;
pub mod execution;
pub mod experiment;
pub mod falsify;
pub mod field;
pub mod grammar;
pub mod invariant;
pub mod montgomery;
pub mod orders;
pub mod proof;
pub mod report;
pub mod review;
pub mod scope;
pub mod search;

pub use adjacent::{
    CcosAuditBlock, CcosAuditChain, IbeCiphertext, IbeParams, Oct8Fp, PairingCommitment,
    QuantumVulnerabilityReport, Sedenion16Fp, apply_velu_isogeny, evaluate_line, evaluate_vertical,
    find_isogeny_path, ibe_decrypt, ibe_encrypt, ibe_hash_to_point, miller_loop,
    reduced_tate_pairing, simulate_quantum_attack_resistance, velu_isogeny_curve, weil_pairing,
};
pub use campaign::MANDATORY_CONTROLS as CAMPAIGN_MANDATORY_CONTROLS; // avoid duplicate
pub use campaign::{
    CampaignReplayReport, CampaignReport, CampaignRun, MANDATORY_CONTROLS, execute_campaign,
    replay_campaign,
};
pub use catalog::{CatalogEntry, CatalogFamily, RelationSignature, catalog_entry};
pub use classify::{Classification, ClassificationStatus, classify};
pub use controls::{ControlId, ControlResult, run_control};
pub use curve::{CurveError, ToyCurve, ToyPoint};
pub use edwards::{TwistedEdwardsCurve, TwistedEdwardsError, TwistedEdwardsPoint};
pub use execution::{
    ExecutionReceipt, ExecutionSummary, ReplayReport, execute_local, replay_local,
};
pub use experiment::{Corpus, CorpusCurve, ExperimentManifest};
pub use falsify::{
    Counterexample, FalsificationResult, first_point_counterexample,
    first_point_counterexample_bounded,
};
pub use field::{FieldError, Fp, PrimeError, ToyPrime};
pub use grammar::{PointExpression, Relation, generate_relations};
pub use invariant::CurveInvariants;
pub use montgomery::{MontgomeryCurve, MontgomeryError, MontgomeryPoint};
pub use proof::{
    Justification, PolynomialIdentityCertificate, ProofCertificate, attempt_justification,
    prove_j_zero_identity,
};
pub use report::ExperimentReport;
pub use review::{
    LiteratureDecision, LiteratureReview, ReviewError, ReviewReport, ReviewedCandidate,
    review_candidate,
};
pub use scope::{CorpusKind, LocalResearchCase, ScopeError};
pub use search::{
    CandidateEvaluation, GateReport, GateState, ResearchCorpora, SearchError, SearchPlan,
    evaluate_candidate, run_search,
};

#[cfg(feature = "portable-simd")]
pub mod hypercomplex_curve;
#[cfg(feature = "portable-simd")]
pub use hypercomplex_curve::{OctonionCurve, OctonionPoint, SedenionCurve, SedenionPoint};

pub mod quantum_eval;
pub use quantum_eval::{IsogenyAssessment, QuantumIsogenyEvaluator, ShorAssessment};
