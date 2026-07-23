//! Deterministic causal models and exactly invertible polynomial flows.
//!
//! # Scope
//!
//! The only implemented family is the **strictly lower-triangular cubic map**
//!
//! ```text
//! y_i = x_i + (Σ_{j < i} A[i,j] · x_j)³
//! ```
//!
//! where `A` is strictly lower triangular (all diagonal and upper-triangular
//! entries are zero). This is a proper subclass of the Drużkowski maps; no
//! claim is made about general Drużkowski or Nilsson maps.
//!
//! # Mathematical properties
//!
//! - The Jacobian `J = I + 3·diag((A x)²)·A` is **unit lower triangular**.
//! - Its determinant is **exactly 1** in exact arithmetic.
//! - `log|det J|` is identically zero — a structural result, not a numerical
//!   coincidence.
//! - The **inverse** is defined by deterministic forward substitution:
//!
//!   ```text
//!   x_i = y_i - (Σ_{j < i} A[i,j] · x_j)³
//!   ```
//!
//!   This is an exact algebraic formula evaluated in floating-point; it is
//!   *not* an iterative approximation.
//!
//! # Numerical caveats
//!
//! - All operations use binary64 (`f64`) arithmetic. The mathematical
//!   determinant of 1 is exact in real arithmetic, but floating-point
//!   evaluation of the LU-based determinant may accumulate round-off.
//! - The cubic operation can **overflow** when `|z_i| > 5.6 × 10¹⁰²`.
//!   Such overflows are detected and reported as errors.
//! - Non-finite weights, inputs, or computation results are always rejected.
//! - **Reproducibility** is guaranteed only for a fixed implementation,
//!   input, build, and execution environment. Bit-identical results across
//!   architectures, compilers, or Rust versions are not guaranteed.
//!
//! # Unimplemented (future phases)
//!
//! The following belong to later phases and are **not** included here:
//! NOTEARS, continuous DAG optimization, augmented Lagrangian, BFGS
//! integration, graph thresholding, `CausalDag` conversion, batch APIs,
//! GPU/SIMD support, generic scalars, symbolic algebra, finite fields,
//! arbitrary Drużkowski maps, normalizing-flow training, serialization,
//! or benchmarks.

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
