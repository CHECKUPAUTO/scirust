//! Deterministic invertible cubic flows, constrained causal-structure
//! optimization, and a typed data model for honest causal claims.
//!
//! # Scope
//!
//! Eleven capabilities:
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
//!    faithfulness; and
//! 6. **effect identification and estimation**
//!    ([`estimate_effect_from_dag`]/[`estimate_effect_from_cpdag`]) —
//!    Pearl's backdoor criterion ([`check_backdoor_criterion`], decided by
//!    d-separation) to establish *whether* an effect is identifiable from a
//!    given graph, then linear backdoor adjustment to estimate it. This is
//!    the one capability that produces a [`CausalCertificate`] carrying a
//!    real numeric estimate, and it is the reason capability 3's
//!    "only `Identifiable` may carry an estimate" rule exists: an
//!    unidentifiable query, an equivalence class that does not determine one
//!    effect, and an exhausted-degrees-of-freedom sample all return a
//!    certificate with **no** estimate rather than a plausible number. See
//!    the private `effect_estimation` module's own docs for the full
//!    assumption list, and the crate's adversarial tests for a latent
//!    confounder producing a confidently wrong estimate through an
//!    adjustment set that is perfectly valid for the graph it was checked
//!    against; and
//! 7. quantitative **sensitivity analysis** ([`analyze_sensitivity`],
//!    [`bound_effect_under_confounder`], [`benchmark_covariate`]) — the
//!    omitted-variable-bias framework of Cinelli & Hazlett (2020), answering
//!    *how strong would an unmeasured confounder have to be to overturn this
//!    estimate?* This exists precisely because capability 6 assumes causal
//!    sufficiency rather than establishing it: it converts that assumption
//!    from a prose caveat into a number the reader can compare against
//!    domain knowledge. It quantifies a stated assumption; it does **not**
//!    test whether a confounder exists, which no data can do. See the
//!    private `sensitivity` module's own docs — in particular the recorded
//!    finding that a high robustness value is not by itself reassurance; and
//! 8. **Invariant Causal Prediction** ([`invariant_causal_prediction`]) —
//!    Peters, Bühlmann & Meinshausen (2016). Given data from two or more
//!    labelled [`Environment`]s and **no graph at all**, finds the predictor
//!    subsets whose relationship to the target is invariant across
//!    environments; their intersection is, with the stated confidence, a
//!    *subset* of the target's direct causes. Its distinguishing property is
//!    one capability 6 structurally cannot have: when its own core assumption
//!    fails, it can **say so** — no surviving subset is positive evidence of
//!    misspecification, not an absence of findings. The crate's tests exhibit
//!    a hidden confounder that capability 6 certifies as `Identifiable` and
//!    this one flags. The trade is directness: it answers *which* variables
//!    are causes, never *how large* the effect is; and
//! 9. **structural simulation and counterfactuals** ([`LinearScm`]) — the
//!    third rung of Pearl's ladder. Given a *fully specified* linear
//!    structural causal model, [`LinearScm::simulate`] evaluates
//!    interventional worlds (rung 2) and [`LinearScm::counterfactual`]
//!    answers unit-level "what would `Y` have been for *this* unit, had `X`
//!    been `x`?" (rung 3) by abduction–action–prediction. Its epistemic
//!    position is the exact inverse of capabilities 5 and 8: it assumes the
//!    whole model — direction, coefficients, functional form — and asks
//!    questions no amount of data can answer without one. The crate's tests
//!    make the cost explicit with two Markov-equivalent models whose joint
//!    distributions are provably identical, on which capability 5 correctly
//!    returns an *unoriented* edge, and which then answer the same
//!    counterfactual query `3` versus `1`. Its certificates therefore carry
//!    zero *computational* uncertainty and an assumption load that no
//!    observational procedure in this crate can discharge; and
//! 10. **experimental design** ([`plan_next_experiment`],
//!     [`greedy_experiment_sequence`]) — the reply to capability 9's bill.
//!     Given a [`Cpdag`] (from capability 5, or hand-built via
//!     [`Cpdag::from_edges`]), it ranks the feasible single-variable
//!     interventions by how much of the equivalence class each would settle,
//!     and reports a **worst-case guarantee** rather than a best case: an
//!     experiment that resolves an edge only on some outcomes is not reported
//!     as resolving it. This is the one capability that needs no data at all —
//!     the answer is a function of the graph — and the one whose output is not
//!     a causal claim but a statement about what a causal claim would cost.
//!     It has no cost model and no opinion on whether an experiment is worth
//!     running; see the private `experiment_design` module's docs for the
//!     perfect-intervention idealisation it assumes and states; and
//! 11. **theory revision** ([`revise_assumptions`], [`audit_certificate`]) —
//!     the mechanism by which everything above can be *withdrawn*. Every
//!     certificate in this crate is a conditional statement resting on named
//!     assumptions; until this capability existed, nothing happened when one
//!     of those assumptions later turned out to be wrong. Evidence revises
//!     the [`AssumptionRegistry`], and the revision re-audits certificates:
//!     a claim whose ground has moved reports
//!     [`CertificateStanding::Retracted`] or
//!     [`CertificateStanding::InDoubt`], and in both cases its estimate stops
//!     being usable. The distinction between those two is the capability's
//!     whole substance — evidence naming several assumptions falsified their
//!     *conjunction* and cannot say which member broke, so it puts a claim in
//!     doubt without attributing blame. [`evidence_from_invariance`] applies
//!     that rule to capability 8, whose own docs say it detects that
//!     something is wrong and cannot say what. The crate's tests run the
//!     whole loop: capability 6's confidently wrong estimate, capability 8
//!     detecting the misspecification, and the certificate ceasing to stand.
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
//! Conditional-independence testing (capability 4), constraint-based
//! Markov-equivalence-class discovery (capability 5, CPDAG only — **not**
//! PAG, which needs latent-confounding-robust discovery this crate does not
//! implement), and backdoor identification plus linear effect estimation
//! (capability 6) exist; see their own docs for exact scope. Note that
//! capability 6 identifies effects **relative to a graph the caller
//! supplies** — it assumes causal sufficiency rather than establishing it,
//! so a latent confounder yields a confidently wrong estimate. Front-door
//! and instrumental-variable identification and latent-confounding-robust
//! discovery (e.g. FCI) remain **out of scope for this crate as it stands**
//! and are the subject of later work.
//! Capability 7 quantifies how badly a latent confounder *could* distort an
//! estimate; it does not detect or remove one. Capability 8 can detect that
//! *something* is wrong with the model when multiple environments are
//! available, but does not say what, and cannot repair it. Capability 9
//! simulates interventions and counterfactuals, but only **relative to a
//! structural model the caller supplies**: it neither discovers that model
//! nor validates it, and no procedure in this crate can single one out from
//! observational data alone — which is why it is the one capability whose
//! answers change when a Markov-equivalent alternative is substituted.
//! Capability 10 says which experiment would settle that substitution, but
//! plans it only; nothing here runs an experiment, and a plan is not
//! evidence. Capability 11 withdraws claims whose assumptions evidence has
//! undermined, but it re-derives nothing and establishes nothing as true:
//! corroboration there means a check that could have failed did not, which is
//! weaker than every other guarantee in this list.
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
mod adjustment;
mod assumptions;
mod certificate;
mod conditional_independence;
mod cpdag;
mod cubic_score;
mod d_separation;
mod dataset;
mod effect_estimation;
mod environment;
mod equivalence_class;
mod error;
mod experiment_design;
mod fingerprint;
mod graph;
mod graph_constraints;
mod intervention;
mod invariance;
mod objective;
mod optimize;
mod orientation;
mod partial_correlation;
mod permutation;
mod permutation_calibration;
mod robust_partial_correlation;
mod scm;
mod sensitivity;
mod skeleton_discovery;
mod synthetic;
mod theory_revision;
mod triangular_cubic;
mod variable;

pub use acyclicity::PolynomialAcyclicity;
pub use adjustment::{
    BackdoorVerdict, BackdoorViolation, canonical_adjustment_set, check_backdoor_criterion,
    find_minimal_adjustment_sets,
};
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
pub use effect_estimation::{
    AdjustmentStrategy, EffectEstimate, EffectEstimationConfig, estimate_effect_from_cpdag,
    estimate_effect_from_dag,
};
pub use environment::Environment;
pub use equivalence_class::{
    EquivalenceClassConfig, EquivalenceClassDiscovery, EquivalenceClassResult, PcStable,
};
pub use error::CausalError;
pub use experiment_design::{
    ExperimentCandidate, ExperimentDesignConfig, ExperimentPlan, ExperimentPlanOutcome,
    ExperimentSequence, greedy_experiment_sequence, plan_next_experiment,
};
pub use fingerprint::sha256_hex;
pub use graph::{GraphExtractionConfig, extract_causal_dag};
pub use graph_constraints::{ConstraintViolation, GraphConstraints};
pub use intervention::{Intervention, InterventionKind};
pub use invariance::{
    AcceptedSet, InvarianceConfig, InvariantPredictionOutcome, InvariantPredictionResult,
    invariant_causal_prediction,
};
pub use objective::{AugmentedLagrangianConfig, CausalObjective, ObjectiveEvaluation};
pub use optimize::{CausalOptimizationResult, OptimizerConfig, TerminationReason, optimize_causal};
pub use permutation::{VariablePermutation, triangularize_from_dag};
pub use robust_partial_correlation::RobustCalibration;
pub use scm::{CounterfactualOutcome, CounterfactualQuery, InterventionAssignment, LinearScm};
pub use sensitivity::{
    AdjustedEffect, ConfounderScenario, SensitivityAnalysis, analyze_sensitivity,
    benchmark_covariate, bound_effect_under_confounder,
};
pub use synthetic::{SyntheticDataConfig, generate_causal_samples, generate_noise_matrix};
pub use theory_revision::{
    AssumptionEvidence, AssumptionOutcome, AssumptionRevision, CertificateAudit,
    CertificateStanding, EvidenceDirection, RevisionVerdict, audit_certificate,
    evidence_from_invariance, revise_assumptions,
};
pub use triangular_cubic::TriangularCubicFlow;
pub use variable::{CausalVariable, VariableKind, VariableRole, validate_variable_set};
