#![forbid(unsafe_code)]

//! # `scirust-tdi` — prospective dynamic-information analysis
//!
//! A deterministic, exact toolkit ported from the **TDI** ("Dynamic
//! Information Theory") research project. It investigates whether the
//! *structure of accessible futures* carries predictive information that scalar
//! summaries such as Shannon entropy do not preserve — using exact finite-state
//! dynamics, arbitrary-precision **rational** probabilities (no floating-point
//! rounding), and honest baselines.
//!
//! Building blocks (all exact and deterministic):
//!
//! * **Exact finite-state dynamics** — [`TableSystem`] / [`TransitionSystem`],
//!   [`State`], [`Action`], and reachability [`explore`].
//! * **Future-structure descriptors** — [`uniform_future_block_distribution`]
//!   and its [`uniform_future_block_entropy_bits`]; the branching-path and
//!   branching-state distributions ([`uniform_branching_path_distribution`],
//!   [`uniform_branching_state_distribution`]); and the flagship
//!   [`distribution_overlap`], the intervention-conditioned distribution
//!   overlap that adds predictive information beyond entropy.
//! * **Honest baselines** — Shannon block entropy, the orbital baseline
//!   ([`analyze_orbit`]), and perturbation-recovery analysis
//!   ([`analyze_recovery`], [`analyze_branching_recovery`]).
//! * **Exact arithmetic** — [`ExactRatio`] and the [`TdiSignature`].
//!
//! ## Provenance & scope
//!
//! This is a faithful re-homing of the TDI `tdi-core` crate, kept exact and
//! test-for-test identical: the twelve modules differ from the frozen upstream
//! only by SciRust's `rustfmt.toml` and this documentation, so the upstream
//! confirmatory results describe this crate's behaviour directly.
//!
//! Within small synthetic finite-state families, and under preregistered
//! one-shot evaluation, the overlaps carry predictive information beyond exact
//! contraction descriptors (TDI-5.5), beyond exact spectral moments (TDI-5.6),
//! across four generator families (TDI-5.7), beyond the **literal** spectral
//! gap and mixing time (TDI-6.1), and beyond a fourth spectral moment — which
//! also shows the exact descriptor ladder saturating while the signal does not
//! (TDI-5.9).
//!
//! It does **not** establish a universal law, invariance across system sizes,
//! or superiority over every dynamical baseline: TDI-1's signal was fully
//! subsumed by the orbital baseline, and cross-width invariance, causal effect,
//! nonlinear sufficiency and information decomposition all remain unrun
//! upstream. See the crate README for the full table and open questions.

mod action;
mod baseline;
mod branching_baseline;
mod branching_distribution;
mod branching_recovery;
mod dynamics;
mod explorer;
mod recovery;
mod signature;
mod state;
mod system;

pub use action::Action;
pub use baseline::{
    BaselineError, uniform_future_block_distribution, uniform_future_block_entropy_bits,
};
pub use branching_baseline::{
    BranchingBaselineError, uniform_branching_path_distribution,
    uniform_branching_path_entropy_bits,
};
pub use branching_distribution::{
    BranchingDistributionError, DistributionMathError, distribution_overlap,
    uniform_branching_state_distribution,
};
pub use branching_recovery::{
    BranchingRecoveryAnalysis, BranchingRecoveryError, analyze_branching_recovery,
};
pub use dynamics::{OrbitAnalysis, OrbitError, analyze_orbit};
pub use explorer::{ExploreError, ReachabilityReport, explore};
pub use recovery::{RecoveryAnalysis, RecoveryError, analyze_recovery};
pub use signature::{ExactRatio, SignatureError, TdiSignature};
pub use state::{State, StateError};
pub use system::{TableSystem, TableSystemError, TransitionSystem};
