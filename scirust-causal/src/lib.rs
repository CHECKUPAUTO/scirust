//! Deterministic invertible cubic flows and constrained causal-structure
//! optimization.
//!
//! # Scope
//!
//! Two capabilities:
//!
//! 1. an exactly invertible **strictly lower-triangular cubic map**
//!    ([`TriangularCubicFlow`]); and
//! 2. a continuous **causal-structure optimizer** ([`optimize_causal`]) — a
//!    NOTEARS-style smooth score ([`CubicCausalScore`]) with a polynomial
//!    acyclicity penalty ([`PolynomialAcyclicity`]) assembled into an
//!    augmented-Lagrangian objective ([`CausalObjective`]) and minimized by a
//!    deterministic BFGS loop, with optional thresholding into a
//!    [`scirust_graph::dag::CausalDag`] ([`extract_causal_dag`]).
//!
//! # Causal interpretation — read before using the discovery API
//!
//! **A fitted interaction matrix, or a `CausalDag` extracted from it, is a model
//! selected by optimization. It is not, and must not be reported as, the true
//! causal graph.** This crate performs *structure optimization*, which is a
//! source of hypotheses, not a causal oracle. Specifically:
//!
//! - Observational structure learning can identify a causal DAG only up to its
//!   **Markov-equivalence class** (a CPDAG); returning a single directed graph is
//!   at best *one representative* of that class. This crate does **not** compute
//!   the equivalence class and does **not** mark which edges are compelled versus
//!   reversible.
//! - Even that representative is meaningful only **under strong, unverified
//!   assumptions**: acyclicity; **causal sufficiency** (no latent common causes);
//!   faithfulness; the correct functional/noise form (here an additive
//!   cubic-interaction model with the assumed noise); and an adequate sample size.
//!   None of these are checked here.
//! - There is **no guarantee of recovering the true causal graph** from
//!   observational data, and low training/optimization loss is not evidence of a
//!   causal effect. Predictive or optimization success must never be reported as
//!   causal identification.
//! - The optimizer can stop at a stationary point that is not a minimum. Always
//!   consult [`TerminationReason`]: [`TerminationReason::StationaryAtInitialPoint`]
//!   (notably for an all-zeros start, the empty-graph saddle) and every
//!   non-`Converged` reason mean the matrix carries **no** optimality guarantee.
//!
//! Effect identification, adjustment sets, conditional-independence testing,
//! equivalence-class (CPDAG/PAG) discovery, latent-confounding handling, and
//! sensitivity analysis are **out of scope for this crate as it stands** and are
//! the subject of later work.
//!
//! # Cubic-flow mathematical properties
//!
//! - The Jacobian `J = I + 3·diag((A x)²)·A` is **unit lower triangular**.
//! - Its determinant is **exactly 1** in exact arithmetic, so `log|det J|` is
//!   identically zero — a structural result, not a numerical coincidence.
//! - The **inverse** `x_i = y_i - (Σ_{j < i} A[i,j] · x_j)³` is an exact algebraic
//!   forward substitution, not an iterative approximation.
//! - The map is a **proper subclass** of the Drużkowski maps; no claim is made
//!   about general Drużkowski or Nilsson maps, and this does **not** solve or
//!   prove the Jacobian conjecture.
//!
//! # Numerical caveats
//!
//! - All operations use binary64 (`f64`). The cubic operation can **overflow**
//!   for very large arguments; such overflows are detected and reported.
//! - Non-finite weights, inputs, or computation results are always rejected.
//! - **Reproducibility** holds only for a fixed implementation, input, build, and
//!   environment. Bit-identical results across architectures, compilers, or Rust
//!   versions are not guaranteed.

#![forbid(unsafe_code)]

mod acyclicity;
mod cubic_score;
mod error;
mod graph;
mod objective;
mod optimize;
mod permutation;
mod synthetic;
mod triangular_cubic;

pub use acyclicity::PolynomialAcyclicity;
pub use cubic_score::CubicCausalScore;
pub use error::CausalError;
pub use graph::{GraphExtractionConfig, extract_causal_dag};
pub use objective::{AugmentedLagrangianConfig, CausalObjective, ObjectiveEvaluation};
pub use optimize::{CausalOptimizationResult, OptimizerConfig, TerminationReason, optimize_causal};
pub use permutation::{VariablePermutation, triangularize_from_dag};
pub use synthetic::{SyntheticDataConfig, generate_causal_samples, generate_noise_matrix};
pub use triangular_cubic::TriangularCubicFlow;
