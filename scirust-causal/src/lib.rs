//! Deterministic invertible cubic flows, constrained causal-structure
//! optimization, and a typed data model for honest causal claims.
//!
//! # Scope
//!
//! Five capabilities:
//!
//! 1. an exactly invertible **strictly lower-triangular cubic map**
//!    ([`TriangularCubicFlow`]);
//! 2. a continuous **causal-structure optimizer** ([`optimize_causal`]) — a
//!    NOTEARS-style smooth score ([`CubicCausalScore`]) with a polynomial
//!    acyclicity penalty ([`PolynomialAcyclicity`]) assembled into an
//!    augmented-Lagrangian objective ([`CausalObjective`]) and minimized by a
//!    deterministic BFGS loop, with optional thresholding into a
//!    [`scirust_graph::dag::CausalDag`] ([`extract_causal_dag`]); and
//! 3. a **typed causal-contract data model** — [`CausalVariable`]s with
//!    [`VariableRole`]s, [`Intervention`]s grouped into [`Environment`]s,
//!    provenance-carrying [`CausalDataset`]s, an [`AssumptionRegistry`] that
//!    tracks *why* each assumption is believed to hold, [`GraphConstraints`]
//!    (background knowledge a candidate DAG must satisfy), and
//!    [`CausalCertificate`] — the mandated shape for reporting any causal
//!    claim as a conditional statement rather than an assertion. This layer
//!    defines *contracts*, not algorithms: it contains no discovery,
//!    identification, or estimation procedure. Its [`CausalCertificateBuilder`]
//!    structurally forbids attaching a numeric estimate to any status other
//!    than [`IdentifiabilityStatus::Identifiable`] — see its docs; and
//! 4. deterministic, robust **conditional-independence (CI) testing**
//!    ([`PartialCorrelationTest`]) — the statistical oracle a
//!    causal-discovery algorithm consumes. Classical (QR-residualized,
//!    optionally Fisher-z calibrated), robust (OGK-residualized), and
//!    deterministic-permutation-calibrated variants are provided; none of
//!    them perform causal discovery on their own — see the private
//!    `conditional_independence` module's own docs (readable via the source
//!    or `cargo doc --document-private-items`) for the exact scientific
//!    scope and honesty caveats this capability stays within; and
//! 5. constraint-based **equivalence-class discovery**
//!    ([`PcStable`]/[`EquivalenceClassDiscovery`]) — repeated CI testing
//!    (capability 4) assembled into skeleton search, unshielded-collider
//!    (v-structure) detection, and Meek's orientation rules, returning a
//!    [`Cpdag`] that honestly marks which edges the evidence *compels* versus
//!    leaves *reversible* — the equivalence-class gap capability 2's own docs
//!    (below) name explicitly. This is a **separate, additive** discovery
//!    paradigm from capability 2 (constraint-based vs. score-based); neither
//!    calls the other. See the private `equivalence_class` module's own docs
//!    for the exact scientific scope, in particular what a `Cpdag` from this
//!    procedure does and does not claim about latent confounding and
//!    faithfulness.
//!
//! # Causal interpretation — read before using the discovery API
//!
//! **A fitted interaction matrix, or a `CausalDag` extracted from it, is a model
//! selected by optimization. It is not, and must not be reported as, the true
//! causal graph.** This crate's continuous optimizer ([`optimize_causal`],
//! capability 2 above) performs *structure optimization*, which is a source
//! of hypotheses, not a causal oracle. Specifically:
//!
//! - Observational structure learning can identify a causal DAG only up to its
//!   **Markov-equivalence class** (a CPDAG); a single directed graph is at best
//!   *one representative* of that class. [`optimize_causal`]/[`extract_causal_dag`]
//!   do **not** compute the equivalence class and do **not** mark which edges
//!   are compelled versus reversible — for that, use the separate
//!   constraint-based [`PcStable`] (capability 5 above), which returns exactly
//!   that marking (subject to *its own* stated assumptions and limitations,
//!   documented in the private `equivalence_class` module — it is not a
//!   universal fix for what this optimizer cannot claim).
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
//! Conditional-independence testing (capability 4) and constraint-based
//! Markov-equivalence-class discovery (capability 5, CPDAG only — **not**
//! PAG, which needs latent-confounding-robust discovery this crate does not
//! implement) exist; see their own docs for exact scope. Effect
//! identification, adjustment sets, latent-confounding-robust discovery
//! (e.g. FCI), and sensitivity analysis remain **out of scope for this crate
//! as it stands** and are the subject of later work.
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
mod assumptions;
mod certificate;
mod conditional_independence;
mod cpdag;
mod cubic_score;
mod dataset;
mod environment;
mod equivalence_class;
mod error;
mod fingerprint;
mod graph;
mod graph_constraints;
mod intervention;
mod objective;
mod optimize;
mod orientation;
mod partial_correlation;
mod permutation;
mod permutation_calibration;
mod robust_partial_correlation;
mod skeleton_discovery;
mod synthetic;
mod triangular_cubic;
mod variable;

pub use acyclicity::PolynomialAcyclicity;
pub use assumptions::{AssumptionBasis, AssumptionRecord, AssumptionRegistry, CausalAssumption};
pub use certificate::{CausalCertificate, CausalCertificateBuilder, IdentifiabilityStatus};
pub use conditional_independence::{
    CalibrationMethod, ConditionalIndependenceConfig, ConditionalIndependenceMethod,
    ConditionalIndependenceResult, ConditionalIndependenceTest, IndependenceDecision,
    MissingValuePolicy, PartialCorrelationTest, RegimeSelection, ResidualizationMethod,
};
pub use cpdag::Cpdag;
pub use cubic_score::CubicCausalScore;
pub use dataset::{CausalDataset, SampleBlock};
pub use environment::Environment;
pub use equivalence_class::{
    EquivalenceClassConfig, EquivalenceClassDiscovery, EquivalenceClassResult, PcStable,
};
pub use error::CausalError;
pub use fingerprint::sha256_hex;
pub use graph::{GraphExtractionConfig, extract_causal_dag};
pub use graph_constraints::{ConstraintViolation, GraphConstraints};
pub use intervention::{Intervention, InterventionKind};
pub use objective::{AugmentedLagrangianConfig, CausalObjective, ObjectiveEvaluation};
pub use optimize::{CausalOptimizationResult, OptimizerConfig, TerminationReason, optimize_causal};
pub use permutation::{VariablePermutation, triangularize_from_dag};
pub use robust_partial_correlation::RobustCalibration;
pub use synthetic::{SyntheticDataConfig, generate_causal_samples, generate_noise_matrix};
pub use triangular_cubic::TriangularCubicFlow;
pub use variable::{CausalVariable, VariableKind, VariableRole, validate_variable_set};
