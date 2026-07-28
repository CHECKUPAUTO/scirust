//! Backdoor-adjusted **causal effect estimation** — the first and only place
//! in this crate that produces a [`CausalCertificate`] carrying a real
//! numeric estimate.
//!
//! # What this computes
//!
//! Given a causal graph, a treatment `X`, an outcome `Y`, and data, this:
//!
//! 1. finds or validates a set `Z` satisfying the **backdoor criterion**
//!    (`crate::adjustment`) — the identification step, which is pure graph
//!    reasoning and touches no data; then
//! 2. estimates the coefficient of `X` in the linear model
//!    `Y ~ 1 + X + Z` — the estimation step, which assumes **linearity** on
//!    top of everything identification already assumed.
//!
//! Step 2 is performed via the Frisch–Waugh–Lovell decomposition: residualize
//! `X` and `Y` on `[1, Z]` (reusing `crate::partial_correlation`'s
//! QR-with-rank-check residualizer, so a rank-deficient adjustment set is a
//! typed error rather than a silent pseudo-inverse), then take
//! `β = Σ x̃ỹ / Σ x̃²`. FWL guarantees this is *identical* to the coefficient
//! from the full multiple regression, and lets the standard error be computed
//! from the same residuals: `se(β) = sqrt(σ̂² / Σ x̃²)` with
//! `σ̂² = RSS / (n − |Z| − 2)`.
//!
//! # The two-layer honesty rule
//!
//! Identification and estimation are separate claims, and this module never
//! lets the second outrun the first:
//!
//! - No valid backdoor set ⇒ [`IdentifiabilityStatus::NotIdentifiable`],
//!   **no estimate**, even though a regression would have run perfectly well
//!   and produced a confident-looking number.
//! - A CPDAG whose treatment still has an undirected edge ⇒
//!   [`IdentifiabilityStatus::EquivalenceClassOnly`], **no estimate** (see
//!   [`estimate_effect_from_cpdag`]).
//! - Residual degrees of freedom exhausted (`n ≤ |Z| + 2`) ⇒
//!   [`IdentifiabilityStatus::Inconclusive`], **no estimate** — the point
//!   estimate is arithmetically available, but an effect reported with no
//!   quantifiable uncertainty is exactly what this program exists not to do.
//!
//! [`CausalCertificateBuilder::finalize`] enforces the "only `Identifiable`
//! may carry an estimate" rule structurally, so these are belt *and* braces:
//! the certificate layer would reject a violation even if this module tried.
//!
//! # What a returned estimate does not mean
//!
//! The estimate is the causal effect **if** the supplied graph is the true
//! causal graph, **if** there is no latent confounding (a confounder missing
//! from the graph is invisible to a criterion evaluated *on* that graph),
//! **if** positivity holds, and **if** the true `Y`-on-`(X, Z)` relationship
//! is linear. None of those are checked here, and none are checkable from
//! the data alone. The crate's adversarial tests include a latent confounder
//! producing a large, confidently-reported, *wrong* estimate through an
//! adjustment set that is perfectly valid for the graph it was checked
//! against — the failure mode this caveat describes, demonstrated rather
//! than asserted.

use crate::adjustment::{BackdoorVerdict, canonical_adjustment_set, check_backdoor_criterion};
use crate::assumptions::CausalAssumption;
use crate::certificate::{CausalCertificate, IdentifiabilityStatus};
use crate::conditional_independence::{RegimeSelection, extract_column, select_rows};
use crate::cpdag::Cpdag;
use crate::dataset::CausalDataset;
use crate::error::CausalError;
use crate::partial_correlation::residualize;
use crate::variable::VariableKind;
use scirust_graph::dag::CausalDag;

const ZERO_VARIANCE_TOLERANCE: f64 = 1e-14;

/// Which adjustment set to use.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AdjustmentStrategy {
    /// The treatment's parents — always backdoor-valid under causal
    /// sufficiency (see [`canonical_adjustment_set`]).
    CanonicalParents,
    /// Exactly this set, which is still checked against the backdoor
    /// criterion; supplying an invalid set yields a `NotIdentifiable`
    /// certificate, never a silently-substituted different set.
    Explicit(Vec<usize>),
}

/// Configuration for [`estimate_effect_from_dag`] / [`estimate_effect_from_cpdag`].
#[derive(Debug, Clone, PartialEq)]
pub struct EffectEstimationConfig {
    /// Relative singular-value tolerance for detecting a rank-deficient
    /// adjustment design (see `crate::partial_correlation`).
    pub rank_tolerance: f64,
    /// Which regime's rows to draw the sample from. Defaults to
    /// [`RegimeSelection::ObservationalOnly`]: backdoor adjustment is a
    /// statement about the *observational* distribution, so silently mixing
    /// in interventional rows would answer a different question.
    pub regime: RegimeSelection,
}

impl Default for EffectEstimationConfig {
    fn default() -> Self {
        Self {
            rank_tolerance: 1e-9,
            regime: RegimeSelection::ObservationalOnly,
        }
    }
}

impl EffectEstimationConfig {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// # Errors
    ///
    /// [`CausalError::InvalidConfiguration`] if `rank_tolerance` is not
    /// finite and positive.
    pub fn with_rank_tolerance(mut self, rank_tolerance: f64) -> Result<Self, CausalError> {
        if !rank_tolerance.is_finite() || rank_tolerance <= 0.0
        {
            return Err(CausalError::InvalidConfiguration {
                name: "rank_tolerance",
                value: rank_tolerance,
            });
        }
        self.rank_tolerance = rank_tolerance;
        Ok(self)
    }

    #[must_use]
    pub fn with_regime(mut self, regime: RegimeSelection) -> Self {
        self.regime = regime;
        self
    }
}

/// One effect-estimation outcome.
///
/// `estimate`/`standard_error`/`adjustment_set` are `Some` exactly when
/// `certificate.status()` is [`IdentifiabilityStatus::Identifiable`]; the
/// certificate is the authoritative record and these fields are a structured
/// view of the same values (an invariant the crate's tests assert directly).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EffectEstimate {
    pub treatment: usize,
    pub outcome: usize,
    /// The backdoor-valid set actually adjusted for.
    pub adjustment_set: Option<Vec<usize>>,
    /// The coefficient of the treatment in `Y ~ 1 + X + Z`, under linearity.
    pub estimate: Option<f64>,
    pub standard_error: Option<f64>,
    /// Rows actually used, after regime selection.
    pub sample_count: usize,
    pub certificate: CausalCertificate,
    pub warnings: Vec<String>,
}

/// The assumptions every estimate from this module rests on, in the crate's
/// existing named vocabulary. `CorrectFunctionalForm` covers the linearity
/// this module's estimator adds on top of identification's own requirements.
fn effect_assumptions() -> Vec<CausalAssumption> {
    vec![
        CausalAssumption::Acyclicity,
        CausalAssumption::CausalSufficiency,
        CausalAssumption::CorrectFunctionalForm,
        CausalAssumption::Positivity,
        CausalAssumption::AdequateSampleSize,
    ]
}

fn query_label(dataset: &CausalDataset, treatment: usize, outcome: usize) -> String {
    format!(
        "causal effect of {} on {}",
        dataset.variables[treatment].name, dataset.variables[outcome].name
    )
}

/// The query context an abstaining result needs to describe itself: which
/// effect was asked about, and how many rows the (still-performed) regime
/// selection yielded. Bundled into a struct so [`unidentified`] stays within
/// a reasonable argument count.
struct AbstentionContext<'a> {
    dataset: &'a CausalDataset,
    treatment: usize,
    outcome: usize,
    sample_count: usize,
}

/// Builds a non-identifiable / inconclusive result: a certificate with no
/// estimate attached, plus the reason.
fn unidentified(
    context: &AbstentionContext<'_>,
    status: IdentifiabilityStatus,
    evidence: String,
    warnings: Vec<String>,
    unresolved: &[String],
) -> Result<EffectEstimate, CausalError> {
    let AbstentionContext {
        dataset,
        treatment,
        outcome,
        sample_count,
    } = *context;
    let mut builder = CausalCertificate::builder(
        query_label(dataset, treatment, outcome),
        status,
        effect_assumptions(),
        evidence,
    );
    for alternative in unresolved
    {
        builder = builder.with_unresolved_alternative(alternative.clone());
    }
    let certificate = builder
        .with_sensitivity(
            "no estimate is attached: identification was not established, so any number \
             a regression would have produced is not a causal effect",
        )
        .finalize()?;
    Ok(EffectEstimate {
        treatment,
        outcome,
        adjustment_set: None,
        estimate: None,
        standard_error: None,
        sample_count,
        certificate,
        warnings,
    })
}

/// Validates that the queried variables exist and are continuous.
fn validate_variables(
    dataset: &CausalDataset,
    treatment: usize,
    outcome: usize,
    adjustment: &[usize],
) -> Result<(), CausalError> {
    let n_vars = dataset.variables.len();
    for &v in std::iter::once(&treatment)
        .chain(std::iter::once(&outcome))
        .chain(adjustment.iter())
    {
        if v >= n_vars
        {
            return Err(CausalError::UnknownVariableIndex { index: v });
        }
        if dataset.variables[v].kind != VariableKind::Continuous
        {
            return Err(CausalError::UnsupportedVariableKind { variable: v });
        }
    }
    if treatment == outcome
    {
        return Err(CausalError::SameVariable {
            variable: treatment,
        });
    }
    Ok(())
}

/// The Frisch–Waugh–Lovell estimate of the treatment coefficient, with its
/// standard error. Returns `Ok(None)` when residual degrees of freedom are
/// exhausted — an honest non-answer, not an error.
struct FwlOutcome {
    estimate: f64,
    standard_error: f64,
}

fn fit_adjusted_effect(
    treatment_column: &[f64],
    outcome_column: &[f64],
    adjustment_columns: &[&[f64]],
    rank_tolerance: f64,
    treatment: usize,
) -> Result<Option<FwlOutcome>, CausalError> {
    let n = treatment_column.len();
    let n_adjust = adjustment_columns.len();

    // `residualize` always includes an intercept, so an empty adjustment set
    // correctly reduces to mean-centering rather than to no-op raw columns.
    let x_residual = residualize(treatment_column, adjustment_columns, rank_tolerance)?;
    let y_residual = residualize(outcome_column, adjustment_columns, rank_tolerance)?;

    let sum_xx: f64 = x_residual.iter().map(|v| v * v).sum();
    if sum_xx <= ZERO_VARIANCE_TOLERANCE
    {
        // The treatment is (numerically) a deterministic function of the
        // adjustment set: no residual variation is left to identify an effect.
        return Err(CausalError::ZeroVariance {
            variable: treatment,
        });
    }
    let sum_xy: f64 = x_residual.iter().zip(&y_residual).map(|(a, b)| a * b).sum();
    let estimate = sum_xy / sum_xx;

    // Residual df of the *full* model `Y ~ 1 + X + Z`: n − (2 + |Z|).
    let parameters = 2 + n_adjust;
    if n <= parameters
    {
        return Ok(None);
    }
    let residual_sum_of_squares: f64 = x_residual
        .iter()
        .zip(&y_residual)
        .map(|(x, y)| {
            let r = y - estimate * x;
            r * r
        })
        .sum();
    let sigma_squared = residual_sum_of_squares / (n - parameters) as f64;
    let standard_error = (sigma_squared / sum_xx).sqrt();

    if !estimate.is_finite() || !standard_error.is_finite()
    {
        return Err(CausalError::NonFiniteComputation {
            operation: "fit_adjusted_effect",
            index: treatment,
            value: if estimate.is_finite()
            {
                standard_error
            }
            else
            {
                estimate
            },
        });
    }
    Ok(Some(FwlOutcome {
        estimate,
        standard_error,
    }))
}

/// Estimates the causal effect of `treatment` on `outcome` from `dataset`,
/// identified via the backdoor criterion on `dag`.
///
/// See the module docs for the full assumption list and the cases that
/// deliberately return **no** estimate.
///
/// # Errors
///
/// [`CausalError::UnknownVariableIndex`] / [`CausalError::SameVariable`] /
/// [`CausalError::UnsupportedVariableKind`] for a malformed query;
/// [`CausalError::DimensionMismatch`] if `dag` and `dataset` disagree on the
/// variable count; [`CausalError::InsufficientSamples`] when the selected
/// regime yields too few rows to fit at all;
/// [`CausalError::RankDeficientConditioningSet`] for a collinear adjustment
/// design; [`CausalError::ZeroVariance`] when the treatment has no residual
/// variation left after adjustment.
pub fn estimate_effect_from_dag(
    dataset: &CausalDataset,
    dag: &CausalDag,
    treatment: usize,
    outcome: usize,
    strategy: &AdjustmentStrategy,
    config: &EffectEstimationConfig,
) -> Result<EffectEstimate, CausalError> {
    if dag.n_nodes() != dataset.variables.len()
    {
        return Err(CausalError::DimensionMismatch {
            expected: dataset.variables.len(),
            got: dag.n_nodes(),
        });
    }
    validate_variables(dataset, treatment, outcome, &[])?;

    // ── Identification: pure graph reasoning, no data touched ──────────
    let (adjustment, identification_note) = match strategy
    {
        AdjustmentStrategy::CanonicalParents =>
        {
            let z = canonical_adjustment_set(dag, treatment, outcome)?;
            (
                z,
                "backdoor adjustment by the treatment's parents".to_string(),
            )
        },
        AdjustmentStrategy::Explicit(z) =>
        {
            let mut z = z.clone();
            z.sort_unstable();
            (
                z,
                "backdoor adjustment by a caller-supplied set".to_string(),
            )
        },
    };
    validate_variables(dataset, treatment, outcome, &adjustment)?;

    let rows = select_rows(dataset, &config.regime)?;
    let sample_count = rows.len();

    match check_backdoor_criterion(dag, treatment, outcome, &adjustment)?
    {
        BackdoorVerdict::Satisfied =>
        {},
        BackdoorVerdict::Violated(violation) =>
        {
            return unidentified(
                &AbstentionContext {
                    dataset,
                    treatment,
                    outcome,
                    sample_count,
                },
                IdentifiabilityStatus::NotIdentifiable,
                format!(
                    "the supplied adjustment set {adjustment:?} does not satisfy the backdoor \
                     criterion for this graph ({violation:?})"
                ),
                vec![format!(
                    "no estimate produced: backdoor criterion violated ({violation:?})"
                )],
                &["a different adjustment set may still identify this effect".to_string()],
            );
        },
    }

    // ── Estimation: only now is data touched ───────────────────────────
    let minimum_rows = adjustment.len() + 2;
    if sample_count < minimum_rows
    {
        return Err(CausalError::InsufficientSamples {
            required: minimum_rows,
            actual: sample_count,
        });
    }

    let treatment_column = extract_column(dataset, treatment, &rows);
    let outcome_column = extract_column(dataset, outcome, &rows);
    let adjustment_columns: Vec<Vec<f64>> = adjustment
        .iter()
        .map(|&z| extract_column(dataset, z, &rows))
        .collect();
    let adjustment_refs: Vec<&[f64]> = adjustment_columns.iter().map(Vec::as_slice).collect();

    let fitted = fit_adjusted_effect(
        &treatment_column,
        &outcome_column,
        &adjustment_refs,
        config.rank_tolerance,
        treatment,
    )?;

    let Some(fitted) = fitted
    else
    {
        return unidentified(
            &AbstentionContext {
                dataset,
                treatment,
                outcome,
                sample_count,
            },
            IdentifiabilityStatus::Inconclusive,
            format!(
                "the effect is identified by backdoor adjustment on {adjustment:?}, but the \
                 sample ({sample_count} rows) leaves no residual degrees of freedom for the \
                 model Y ~ 1 + X + Z ({} parameters)",
                adjustment.len() + 2
            ),
            vec![
                "no estimate produced: residual degrees of freedom exhausted, so no uncertainty \
                 could be quantified"
                    .to_string(),
            ],
            &[],
        );
    };

    let certificate = CausalCertificate::builder(
        query_label(dataset, treatment, outcome),
        IdentifiabilityStatus::Identifiable,
        effect_assumptions(),
        format!(
            "{sample_count} observational rows; {identification_note} on {adjustment:?}, \
             verified against the supplied DAG by the backdoor criterion"
        ),
    )
    .with_estimate(
        "linear backdoor adjustment (Frisch-Waugh-Lovell, QR least squares)",
        fitted.estimate,
        fitted.standard_error,
    )
    .with_sensitivity(
        "valid only if the supplied graph is correct, there is no latent confounding, \
         positivity holds, and the outcome is linear in the treatment and adjustment set; \
         none of these are checked here",
    )
    .finalize()?;

    Ok(EffectEstimate {
        treatment,
        outcome,
        adjustment_set: Some(adjustment),
        estimate: Some(fitted.estimate),
        standard_error: Some(fitted.standard_error),
        sample_count,
        certificate,
        warnings: Vec::new(),
    })
}

/// Estimates the causal effect of `treatment` on `outcome` where the
/// structure is known only up to a Markov equivalence class (a [`Cpdag`], e.g.
/// from `crate::PcStable`).
///
/// # The identification gate
///
/// An effect is only reported when **every edge incident to the treatment is
/// directed** in the CPDAG. That condition is *sufficient*, and this module
/// does not claim it is necessary: when it holds, the treatment's parent set
/// is identical in every DAG in the equivalence class, so backdoor adjustment
/// by those parents yields the same estimate for every member — the effect is
/// identified without ever choosing a representative DAG. When it fails, at
/// least two members of the class disagree about what causes the treatment,
/// so the class as a whole does not determine one effect, and this returns
/// [`IdentifiabilityStatus::EquivalenceClassOnly`] with **no estimate**
/// rather than silently orienting the ambiguous edge some arbitrary way.
///
/// (Enumerating the full multiset of effects across the class — the IDA
/// approach — would report a *range* here instead of abstaining. That is
/// deliberately out of scope for this phase and named as such in the tracker.)
///
/// # Errors
///
/// As [`estimate_effect_from_dag`].
pub fn estimate_effect_from_cpdag(
    dataset: &CausalDataset,
    cpdag: &Cpdag,
    treatment: usize,
    outcome: usize,
    config: &EffectEstimationConfig,
) -> Result<EffectEstimate, CausalError> {
    if cpdag.n_nodes() != dataset.variables.len()
    {
        return Err(CausalError::DimensionMismatch {
            expected: dataset.variables.len(),
            got: cpdag.n_nodes(),
        });
    }
    validate_variables(dataset, treatment, outcome, &[])?;

    let rows = select_rows(dataset, &config.regime)?;
    let sample_count = rows.len();

    let undirected_at_treatment: Vec<usize> = cpdag
        .neighbors(treatment)
        .into_iter()
        .filter(|&v| cpdag.is_undirected(treatment, v))
        .collect();
    if !undirected_at_treatment.is_empty()
    {
        return unidentified(
            &AbstentionContext {
                dataset,
                treatment,
                outcome,
                sample_count,
            },
            IdentifiabilityStatus::EquivalenceClassOnly,
            format!(
                "the treatment still has undirected edge(s) to {undirected_at_treatment:?} in \
                 the CPDAG, so members of the equivalence class disagree about its parent set \
                 and the class does not determine a single effect"
            ),
            vec![format!(
                "no estimate produced: {} edge(s) at the treatment are unoriented",
                undirected_at_treatment.len()
            )],
            &undirected_at_treatment
                .iter()
                .map(|v| {
                    format!(
                        "the edge between the treatment and variable {v} could be oriented \
                         either way by the evidence"
                    )
                })
                .collect::<Vec<_>>(),
        );
    }

    // Every edge at the treatment is oriented, so its parent set is the same
    // in every member DAG; build a DAG carrying exactly the CPDAG's directed
    // edges and adjust by those parents. Any still-undirected edge elsewhere
    // in the graph cannot change pa(treatment), and backdoor validity of a
    // treatment's own parent set does not depend on the rest of the
    // orientation.
    let mut oriented = CausalDag::new(cpdag.n_nodes());
    for (from, to) in cpdag.directed_edges()
    {
        oriented
            .add_directed_edge(from, to)
            .map_err(|_| CausalError::CyclicGraph)?;
    }

    estimate_effect_from_dag(
        dataset,
        &oriented,
        treatment,
        outcome,
        &AdjustmentStrategy::CanonicalParents,
        config,
    )
}
