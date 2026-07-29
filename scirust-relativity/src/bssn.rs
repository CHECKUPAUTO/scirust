//! BSSN formulation core (Layer 3.3) — local variables, constraints,
//! reconstruction, and evolution right-hand sides.
//!
//! The ADM system of [`crate::adm_evolution`] is only *weakly hyperbolic* under
//! common gauge choices, which is why the BSSN (Baumgarte-Shapiro-Shibata-
//! Nakamura) conformal-traceless reformulation exists. This module implements
//! the BSSN variable transformation, its inverse, the algebraic constraints,
//! explicit projections, the conformal Ricci decomposition, and the local
//! evolution right-hand sides.
//!
//! **Implementing BSSN does not, by itself, demonstrate numerical stability.**
//! Strong hyperbolicity is a property of the full evolution system including
//! its gauge conditions and principal part on a discretized domain — none of
//! which exists here. What this module establishes is that the BSSN machinery
//! is correct and agrees with the already-validated ADM system. No stability
//! claim is made. See `docs/LAYER_3_BSSN.md`.
//!
//! **Category.** Established general relativity. The numbers are a numerical
//! approximation (central finite differences of the supplied fields), with
//! truncation error inherent to the method, never hidden.
//!
//! ## Conventions (inherited unchanged from [`crate::adm_evolution`])
//!
//! Signature `(-,+,+,+)`, `G = c = 1`,
//! `K_ij = -1/(2N)(d_t gamma_ij - D_i N_j - D_j N_i)` — an *expanding* slice has
//! *negative* `K`. Matter enters through the existing [`AdmSources`]; no second
//! source convention is introduced. Lapse and shift are **prescribed**, not
//! evolved.
//!
//! ## Variables
//!
//! ```text
//! phi           = (1/12) ln det gamma            (canonical; chi = e^{-4 phi} derived)
//! gammatilde_ij = e^{-4 phi} gamma_ij            (det gammatilde = 1)
//! K             = gamma^{ij} K_ij
//! Atilde_ij     = e^{-4 phi} ( K_ij - (1/3) gamma_ij K )
//! Gammatilde^i  = gammatilde^{jk} Gammatilde^i_jk
//! ```
//!
//! ## The constraint-substituted `d_t K`
//!
//! The standard BSSN trace equation is obtained from the ADM one by
//! substituting the Hamiltonian constraint to eliminate `R`. The two therefore
//! differ by **exactly** the Hamiltonian residual:
//!
//! ```text
//! ( d_t K )_ADM - ( d_t K )_BSSN = alpha * H ,
//! H = R + K^2 - K_ij K^{ij} - 16 pi rho .
//! ```
//!
//! This was established numerically before the implementation was written (the
//! measured difference and `alpha * H` agreed to exactly `0.000e0`). ADM/BSSN
//! right-hand-side equivalence is therefore exact **on the constraint surface**;
//! off it the discrepancy is not an error but precisely `alpha * H`, and
//! [`AdmBssnEquivalence`] reports both the raw and the constraint-corrected
//! difference. `d_t phi`, `d_t gammatilde_ij`, and `d_t Atilde_ij` are
//! equivalent to ADM unconditionally.

use std::f64::consts::PI;
use std::fmt;

use crate::adm_evolution::{
    AdmEvolutionError, AdmEvolutionSettings, AdmSources, SpatialTensorField,
    curvature_evolution_rhs, hamiltonian_constraint, metric_evolution_rhs,
};
use crate::{
    Metric, RelativityError, christoffel_from_derivatives, determinant, invert_metric,
    numerical_christoffel, ricci_tensor_from_metric,
};

/// A typed failure of a BSSN operation. It never panics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BssnError {
    /// The ADM spatial metric is singular or non-invertible.
    SingularAdmMetric,
    /// The spatial-metric determinant is not finite and strictly positive.
    InvalidDeterminant(f64),
    /// The conformal variable (`phi` or `chi`) is not finite, or `chi <= 0`.
    InvalidConformalVariable(f64),
    /// The conformal metric is singular or non-invertible.
    SingularConformalMetric,
    /// A named component of the state is not finite.
    NonFiniteState {
        /// Which quantity was non-finite.
        quantity: &'static str,
    },
    /// Raising or lowering an index failed (the required inverse was unavailable).
    FailedIndexOperation {
        /// Which quantity the operation was applied to.
        quantity: &'static str,
    },
    /// A required derivative could not be evaluated.
    UnavailableDerivative {
        /// Which quantity the derivative was requested for.
        quantity: &'static str,
    },
    /// Reconstructing the ADM state from BSSN variables failed.
    FailedAdmReconstruction {
        /// Which quantity failed to reconstruct.
        quantity: &'static str,
    },
    /// An explicit projection could not be applied.
    FailedProjection {
        /// Which constraint the projection targeted.
        constraint: &'static str,
    },
    /// A propagated geometry-core or ADM-evaluator failure.
    Evaluation(AdmEvolutionError),
}

impl fmt::Display for BssnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::SingularAdmMetric => write!(f, "the ADM spatial metric is singular"),
            Self::InvalidDeterminant(value) =>
            {
                write!(f, "metric determinant {value} must be finite and positive")
            },
            Self::InvalidConformalVariable(value) =>
            {
                write!(f, "conformal variable {value} is invalid")
            },
            Self::SingularConformalMetric => write!(f, "the conformal metric is singular"),
            Self::NonFiniteState { quantity } =>
            {
                write!(f, "state quantity '{quantity}' is not finite")
            },
            Self::FailedIndexOperation { quantity } =>
            {
                write!(f, "index operation failed for '{quantity}'")
            },
            Self::UnavailableDerivative { quantity } =>
            {
                write!(f, "derivative unavailable for '{quantity}'")
            },
            Self::FailedAdmReconstruction { quantity } =>
            {
                write!(f, "ADM reconstruction failed for '{quantity}'")
            },
            Self::FailedProjection { constraint } =>
            {
                write!(f, "projection failed for the '{constraint}' constraint")
            },
            Self::Evaluation(error) => write!(f, "evaluation failed: {error}"),
        }
    }
}

impl std::error::Error for BssnError {}

impl From<AdmEvolutionError> for BssnError {
    fn from(error: AdmEvolutionError) -> Self {
        Self::Evaluation(error)
    }
}

impl From<RelativityError> for BssnError {
    fn from(error: RelativityError) -> Self {
        Self::Evaluation(AdmEvolutionError::Curvature(error))
    }
}

/// Prescribed gauge: the lapse `alpha` and shift `beta^i`.
///
/// Gauge is **not evolved** in this increment; live conditions (1+log lapse,
/// Gamma-driver shift) are deferred to a later gauge increment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BssnGauge {
    /// The lapse `alpha`.
    pub lapse: f64,
    /// The shift `beta^i`.
    pub shift: [f64; 3],
}

impl BssnGauge {
    /// Unit lapse, zero shift (the synchronous/comoving gauge).
    pub const SYNCHRONOUS: Self = Self {
        lapse: 1.0,
        shift: [0.0; 3],
    };
}

/// The BSSN state at one point.
///
/// `phi` is canonical; [`BssnState::chi`] derives `chi = e^{-4 phi}` so the two
/// can never disagree.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BssnState {
    /// The conformal factor `phi = (1/12) ln det gamma`.
    pub conformal_factor: f64,
    /// The conformal metric `gammatilde_ij` (unit determinant).
    pub conformal_metric: [[f64; 3]; 3],
    /// The trace `K = gamma^{ij} K_ij`.
    pub mean_curvature: f64,
    /// The conformal trace-free curvature `Atilde_ij`.
    pub conformal_curvature: [[f64; 3]; 3],
    /// The conformal connection functions `Gammatilde^i`.
    pub conformal_connection: [f64; 3],
}

impl BssnState {
    /// The derived conformal variable `chi = e^{-4 phi} = (det gamma)^{-1/3}`.
    #[must_use]
    pub fn chi(&self) -> f64 {
        (-4.0 * self.conformal_factor).exp()
    }

    /// `e^{4 phi} = (det gamma)^{1/3}`.
    #[must_use]
    pub fn conformal_scale(&self) -> f64 {
        (4.0 * self.conformal_factor).exp()
    }

    /// The inverse conformal metric `gammatilde^{ij}`.
    pub fn inverse_conformal_metric(&self) -> Result<[[f64; 3]; 3], BssnError> {
        invert_metric::<3>(&self.conformal_metric).map_err(|_| BssnError::SingularConformalMetric)
    }
}

/// The decomposed algebraic constraints of a BSSN state.
///
/// Reported separately, never blended into a single scalar.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BssnAlgebraicConstraints {
    /// `det gammatilde - 1`.
    pub determinant_residual: f64,
    /// `|det gammatilde - 1|`.
    pub determinant_absolute: f64,
    /// `gammatilde^{ij} Atilde_ij`.
    pub trace_residual: f64,
    /// `|gammatilde^{ij} Atilde_ij|`.
    pub trace_absolute: f64,
    /// The scale used to normalize the trace residual
    /// (`max |Atilde_ij|`), or `0` when the curvature vanishes.
    pub trace_scale: f64,
    /// `trace_residual / trace_scale`, or `None` when the scale is too small to
    /// be meaningful.
    pub trace_normalized: Option<f64>,
    /// `Gammatilde^i_stored - Gammatilde^i_reconstructed`.
    pub connection_residual: [f64; 3],
    /// The largest absolute connection-component residual.
    pub connection_absolute: f64,
    /// Whether every stored component is finite.
    pub finite: bool,
}

/// The conformal Ricci decomposition `R_ij = Rtilde_ij + Rphi_ij`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConformalRicci {
    /// `Rtilde_ij`, the Ricci tensor of `gammatilde_ij`.
    pub conformal: [[f64; 3]; 3],
    /// `Rphi_ij`, the conformal-factor contribution.
    pub conformal_factor_part: [[f64; 3]; 3],
    /// Their sum, the physical spatial Ricci tensor.
    pub total: [[f64; 3]; 3],
}

/// The BSSN evolution right-hand sides, with each additive contribution exposed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BssnEvolutionRhs {
    /// `d_t phi`.
    pub conformal_factor: f64,
    /// The `-(1/6) alpha K` contribution to `d_t phi`.
    pub conformal_factor_lapse_term: f64,
    /// The shift contribution to `d_t phi`.
    pub conformal_factor_shift_term: f64,
    /// `d_t gammatilde_ij`.
    pub conformal_metric: [[f64; 3]; 3],
    /// The `-2 alpha Atilde_ij` contribution.
    pub conformal_metric_lapse_term: [[f64; 3]; 3],
    /// The shift advection and shift-gradient contribution.
    pub conformal_metric_shift_term: [[f64; 3]; 3],
    /// `d_t K`.
    pub mean_curvature: f64,
    /// The `-D^i D_i alpha` contribution to `d_t K`.
    pub mean_curvature_lapse_term: f64,
    /// The `alpha (Atilde_ij Atilde^{ij} + K^2/3)` contribution.
    pub mean_curvature_quadratic_term: f64,
    /// The `4 pi alpha (rho + S)` contribution.
    pub mean_curvature_matter_term: f64,
    /// The shift-advection contribution to `d_t K`.
    pub mean_curvature_shift_term: f64,
    /// `d_t Atilde_ij`.
    pub conformal_curvature: [[f64; 3]; 3],
    /// The trace-free `e^{-4 phi}[-D_iD_j alpha]^TF` contribution.
    pub conformal_curvature_lapse_term: [[f64; 3]; 3],
    /// The trace-free `e^{-4 phi}[alpha R_ij]^TF` contribution.
    pub conformal_curvature_ricci_term: [[f64; 3]; 3],
    /// The `alpha (K Atilde_ij - 2 Atilde_ik Atilde^k_j)` contribution.
    pub conformal_curvature_quadratic_term: [[f64; 3]; 3],
    /// The trace-free `e^{-4 phi}[-8 pi alpha S_ij]^TF` contribution.
    pub conformal_curvature_matter_term: [[f64; 3]; 3],
    /// The shift contribution to `d_t Atilde_ij`.
    pub conformal_curvature_shift_term: [[f64; 3]; 3],
    /// `d_t Gammatilde^i`.
    pub conformal_connection: [f64; 3],
}

/// The result of comparing a reconstructed ADM right-hand side with the
/// Layer 3.1 ADM right-hand side.
///
/// See the [module documentation](self): `d_t K` differs by exactly `alpha * H`
/// off the constraint surface, so both the raw and the constraint-corrected
/// difference are reported.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdmBssnEquivalence {
    /// The largest `|d_t gamma_ij|` difference (unconditionally zero in exact
    /// arithmetic).
    pub metric_difference: f64,
    /// The largest raw `|d_t K_ij|` difference.
    pub curvature_difference: f64,
    /// The largest `|d_t K_ij|` difference after subtracting the exactly-known
    /// `alpha * H` contribution.
    pub curvature_difference_constraint_corrected: f64,
    /// The Hamiltonian constraint residual `H` at this point.
    pub hamiltonian_residual: f64,
    /// The lapse used.
    pub lapse: f64,
}

/// The outcome of an explicit projection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BssnProjection {
    /// The residual before projection.
    pub residual_before: f64,
    /// The residual after projection.
    pub residual_after: f64,
    /// The largest absolute change applied to any component.
    pub correction_magnitude: f64,
}

// ---------------------------------------------------------------------------
// Conversion
// ---------------------------------------------------------------------------

fn require_finite_tensor(quantity: &'static str, tensor: &[[f64; 3]; 3]) -> Result<(), BssnError> {
    if tensor.iter().flatten().any(|value| !value.is_finite())
    {
        return Err(BssnError::NonFiniteState { quantity });
    }
    Ok(())
}

/// Contract a symmetric tensor with an inverse metric: `m^{ij} t_ij`.
fn trace_with(inverse: &[[f64; 3]; 3], tensor: &[[f64; 3]; 3]) -> f64 {
    let mut value = 0.0;
    for i in 0..3
    {
        for j in 0..3
        {
            value += inverse[i][j] * tensor[i][j];
        }
    }
    value
}

/// Raise both indices: `t^{ij} = m^{ia} m^{jb} t_ab`.
fn raise_both(inverse: &[[f64; 3]; 3], tensor: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0_f64; 3]; 3];
    for i in 0..3
    {
        for j in 0..3
        {
            let mut value = 0.0;
            for a in 0..3
            {
                for b in 0..3
                {
                    value += inverse[i][a] * inverse[j][b] * tensor[a][b];
                }
            }
            out[i][j] = value;
        }
    }
    out
}

/// Raise the first index: `t^i_j = m^{ia} t_aj`.
fn raise_first(inverse: &[[f64; 3]; 3], tensor: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0_f64; 3]; 3];
    for i in 0..3
    {
        for j in 0..3
        {
            let mut value = 0.0;
            for a in 0..3
            {
                value += inverse[i][a] * tensor[a][j];
            }
            out[i][j] = value;
        }
    }
    out
}

/// The conformal metric `gammatilde_ij = (det gamma)^{-1/3} gamma_ij` of a
/// spatial metric, as a [`Metric<3>`] so the geometry core can differentiate it.
struct ConformalMetricField<'a, G> {
    physical: &'a G,
}

impl<G: Metric<3>> Metric<3> for ConformalMetricField<'_, G> {
    fn components(&self, coordinates: &[f64; 3]) -> [[f64; 3]; 3] {
        let physical = self.physical.components(coordinates);
        let scale = match determinant::<3>(&physical)
        {
            Ok(det) if det.is_finite() && det > 0.0 => det.powf(-1.0 / 3.0),
            // A non-invertible or non-positive determinant cannot produce a
            // conformal metric; emit a non-finite value so every downstream
            // consumer fails loudly rather than silently using garbage.
            _ => f64::NAN,
        };
        let mut out = [[0.0_f64; 3]; 3];
        for i in 0..3
        {
            for j in 0..3
            {
                out[i][j] = scale * physical[i][j];
            }
        }
        out
    }
}

/// `phi = (1/12) ln det gamma` at a point.
fn conformal_factor_at<G: Metric<3>>(metric: &G, coordinates: &[f64; 3]) -> Result<f64, BssnError> {
    let physical = metric.components(coordinates);
    let det = determinant::<3>(&physical).map_err(|_| BssnError::SingularAdmMetric)?;
    if !det.is_finite() || det <= 0.0
    {
        return Err(BssnError::InvalidDeterminant(det));
    }
    Ok(det.ln() / 12.0)
}

/// The contracted conformal connection `Gammatilde^i = gammatilde^{jk} Gammatilde^i_jk`,
/// reconstructed from the conformal metric (the authoritative definition).
fn conformal_connection_from_metric<G: Metric<3>>(
    metric: &G,
    coordinates: &[f64; 3],
    settings: &AdmEvolutionSettings,
) -> Result<[f64; 3], BssnError> {
    let conformal = ConformalMetricField { physical: metric };
    let components = conformal.components(coordinates);
    let inverse =
        invert_metric::<3>(&components).map_err(|_| BssnError::SingularConformalMetric)?;
    let christoffel = numerical_christoffel::<_, 3>(&conformal, coordinates, settings.metric_step)
        .map_err(|_| BssnError::UnavailableDerivative {
            quantity: "conformal_christoffel",
        })?;
    let mut out = [0.0_f64; 3];
    for i in 0..3
    {
        let mut value = 0.0;
        for j in 0..3
        {
            for k in 0..3
            {
                value += inverse[j][k] * christoffel[i][j][k];
            }
        }
        if !value.is_finite()
        {
            return Err(BssnError::NonFiniteState {
                quantity: "conformal_connection",
            });
        }
        out[i] = value;
    }
    Ok(out)
}

/// Convert an ADM state (a spatial metric field plus an extrinsic-curvature
/// field) into the BSSN variables at `coordinates`.
///
/// This performs **no projection**: the returned state carries whatever
/// algebraic-constraint violation the input implies, which
/// [`bssn_algebraic_constraints`] then reports. Projection is available only as
/// the explicit [`project_unit_determinant`] / [`project_trace_free`]
/// operations.
///
/// # Example
///
/// Minkowski converts to the trivial BSSN state.
///
/// ```
/// use scirust_relativity::Metric;
/// use scirust_relativity::adm_evolution::AdmEvolutionSettings;
/// use scirust_relativity::bssn::adm_to_bssn;
///
/// struct Flat;
/// impl Metric<3> for Flat {
///     fn components(&self, _x: &[f64; 3]) -> [[f64; 3]; 3] {
///         [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
///     }
/// }
/// let zero = |_: &[f64; 3]| [[0.0; 3]; 3];
/// let settings = AdmEvolutionSettings { spatial_step: 1.0e-3, metric_step: 1.0e-3 };
/// let state = adm_to_bssn(&Flat, &zero, &[0.0, 0.0, 0.0], &settings).expect("converts");
///
/// assert!(state.conformal_factor.abs() < 1.0e-12);
/// assert!((state.chi() - 1.0).abs() < 1.0e-12);
/// assert!(state.mean_curvature.abs() < 1.0e-12);
/// ```
pub fn adm_to_bssn<G: Metric<3>>(
    spatial_metric: &G,
    extrinsic_curvature: &impl SpatialTensorField,
    coordinates: &[f64; 3],
    settings: &AdmEvolutionSettings,
) -> Result<BssnState, BssnError> {
    let physical = spatial_metric.components(coordinates);
    require_finite_tensor("spatial_metric", &physical)?;
    let inverse = invert_metric::<3>(&physical).map_err(|_| BssnError::SingularAdmMetric)?;

    let conformal_factor = conformal_factor_at(spatial_metric, coordinates)?;
    let scale = (-4.0 * conformal_factor).exp();
    if !scale.is_finite() || scale <= 0.0
    {
        return Err(BssnError::InvalidConformalVariable(scale));
    }

    let curvature = extrinsic_curvature.components(coordinates);
    require_finite_tensor("extrinsic_curvature", &curvature)?;
    let mean_curvature = trace_with(&inverse, &curvature);

    let mut conformal_metric = [[0.0_f64; 3]; 3];
    let mut conformal_curvature = [[0.0_f64; 3]; 3];
    for i in 0..3
    {
        for j in 0..3
        {
            conformal_metric[i][j] = scale * physical[i][j];
            conformal_curvature[i][j] =
                scale * (curvature[i][j] - physical[i][j] * mean_curvature / 3.0);
        }
    }
    require_finite_tensor("conformal_metric", &conformal_metric)?;
    require_finite_tensor("conformal_curvature", &conformal_curvature)?;

    let conformal_connection =
        conformal_connection_from_metric(spatial_metric, coordinates, settings)?;

    Ok(BssnState {
        conformal_factor,
        conformal_metric,
        mean_curvature,
        conformal_curvature,
        conformal_connection,
    })
}

/// An ADM state reconstructed from BSSN variables.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReconstructedAdm {
    /// The spatial metric `gamma_ij = e^{4 phi} gammatilde_ij`.
    pub spatial_metric: [[f64; 3]; 3],
    /// The extrinsic curvature
    /// `K_ij = e^{4 phi} ( Atilde_ij + (1/3) gammatilde_ij K )`.
    pub extrinsic_curvature: [[f64; 3]; 3],
}

/// The ADM variables reconstructed from a BSSN state:
/// `gamma_ij = e^{4 phi} gammatilde_ij` and
/// `K_ij = e^{4 phi} ( Atilde_ij + (1/3) gammatilde_ij K )`.
pub fn bssn_to_adm(state: &BssnState) -> Result<ReconstructedAdm, BssnError> {
    let scale = state.conformal_scale();
    if !scale.is_finite() || scale <= 0.0
    {
        return Err(BssnError::InvalidConformalVariable(scale));
    }
    let mut spatial_metric = [[0.0_f64; 3]; 3];
    let mut extrinsic_curvature = [[0.0_f64; 3]; 3];
    for i in 0..3
    {
        for j in 0..3
        {
            spatial_metric[i][j] = scale * state.conformal_metric[i][j];
            extrinsic_curvature[i][j] = scale
                * (state.conformal_curvature[i][j]
                    + state.conformal_metric[i][j] * state.mean_curvature / 3.0);
        }
    }
    if spatial_metric.iter().flatten().any(|v| !v.is_finite())
    {
        return Err(BssnError::FailedAdmReconstruction {
            quantity: "spatial_metric",
        });
    }
    if extrinsic_curvature.iter().flatten().any(|v| !v.is_finite())
    {
        return Err(BssnError::FailedAdmReconstruction {
            quantity: "extrinsic_curvature",
        });
    }
    // The reconstructed metric must still be invertible.
    invert_metric::<3>(&spatial_metric).map_err(|_| BssnError::FailedAdmReconstruction {
        quantity: "spatial_metric_inverse",
    })?;
    Ok(ReconstructedAdm {
        spatial_metric,
        extrinsic_curvature,
    })
}

/// Evaluate the decomposed algebraic constraints of a BSSN state.
///
/// `reconstructed_connection` is the `Gammatilde^i` implied by the state's own
/// conformal metric; passing it explicitly keeps this function free of any
/// derivative machinery, so a caller with an analytic connection can supply it
/// directly. [`bssn_constraints_from_adm`] wires it up from a metric field.
#[must_use]
pub fn bssn_algebraic_constraints(
    state: &BssnState,
    reconstructed_connection: &[f64; 3],
) -> BssnAlgebraicConstraints {
    let finite = state.conformal_factor.is_finite()
        && state.mean_curvature.is_finite()
        && state
            .conformal_metric
            .iter()
            .flatten()
            .all(|v| v.is_finite())
        && state
            .conformal_curvature
            .iter()
            .flatten()
            .all(|v| v.is_finite())
        && state.conformal_connection.iter().all(|v| v.is_finite());

    let determinant_residual = match determinant::<3>(&state.conformal_metric)
    {
        Ok(det) => det - 1.0,
        Err(_) => f64::NAN,
    };

    let trace_residual = match invert_metric::<3>(&state.conformal_metric)
    {
        Ok(inverse) => trace_with(&inverse, &state.conformal_curvature),
        Err(_) => f64::NAN,
    };
    let trace_scale = state
        .conformal_curvature
        .iter()
        .flatten()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let trace_normalized = (trace_scale > 1.0e-12).then(|| trace_residual / trace_scale);

    let mut connection_residual = [0.0_f64; 3];
    let mut connection_absolute = 0.0_f64;
    for i in 0..3
    {
        connection_residual[i] = state.conformal_connection[i] - reconstructed_connection[i];
        connection_absolute = connection_absolute.max(connection_residual[i].abs());
    }

    BssnAlgebraicConstraints {
        determinant_residual,
        determinant_absolute: determinant_residual.abs(),
        trace_residual,
        trace_absolute: trace_residual.abs(),
        trace_scale,
        trace_normalized,
        connection_residual,
        connection_absolute,
        finite,
    }
}

/// Convert an ADM state and evaluate its BSSN algebraic constraints in one step,
/// reconstructing `Gammatilde^i` from the supplied metric field.
pub fn bssn_constraints_from_adm<G: Metric<3>>(
    spatial_metric: &G,
    extrinsic_curvature: &impl SpatialTensorField,
    coordinates: &[f64; 3],
    settings: &AdmEvolutionSettings,
) -> Result<(BssnState, BssnAlgebraicConstraints), BssnError> {
    let state = adm_to_bssn(spatial_metric, extrinsic_curvature, coordinates, settings)?;
    let reconstructed = conformal_connection_from_metric(spatial_metric, coordinates, settings)?;
    let constraints = bssn_algebraic_constraints(&state, &reconstructed);
    Ok((state, constraints))
}

// ---------------------------------------------------------------------------
// Explicit projections (never applied silently)
// ---------------------------------------------------------------------------

/// Rescale `gammatilde_ij` so that `det gammatilde = 1`, returning the pre- and
/// post-projection residuals and the correction magnitude.
///
/// Rejects a singular or non-finite conformal metric. Never called implicitly.
pub fn project_unit_determinant(state: &mut BssnState) -> Result<BssnProjection, BssnError> {
    let det =
        determinant::<3>(&state.conformal_metric).map_err(|_| BssnError::FailedProjection {
            constraint: "unit_determinant",
        })?;
    if !det.is_finite() || det <= 0.0
    {
        return Err(BssnError::FailedProjection {
            constraint: "unit_determinant",
        });
    }
    let residual_before = det - 1.0;
    let scale = det.powf(-1.0 / 3.0);
    let mut correction_magnitude = 0.0_f64;
    for i in 0..3
    {
        for j in 0..3
        {
            let updated = scale * state.conformal_metric[i][j];
            correction_magnitude =
                correction_magnitude.max((updated - state.conformal_metric[i][j]).abs());
            state.conformal_metric[i][j] = updated;
        }
    }
    let after =
        determinant::<3>(&state.conformal_metric).map_err(|_| BssnError::FailedProjection {
            constraint: "unit_determinant",
        })?;
    Ok(BssnProjection {
        residual_before,
        residual_after: after - 1.0,
        correction_magnitude,
    })
}

/// Remove the conformal trace from `Atilde_ij` so that
/// `gammatilde^{ij} Atilde_ij = 0`, returning the pre- and post-projection
/// residuals and the correction magnitude.
///
/// Rejects a singular or non-finite conformal metric. Never called implicitly.
pub fn project_trace_free(state: &mut BssnState) -> Result<BssnProjection, BssnError> {
    let inverse =
        invert_metric::<3>(&state.conformal_metric).map_err(|_| BssnError::FailedProjection {
            constraint: "trace_free",
        })?;
    let residual_before = trace_with(&inverse, &state.conformal_curvature);
    if !residual_before.is_finite()
    {
        return Err(BssnError::FailedProjection {
            constraint: "trace_free",
        });
    }
    let mut correction_magnitude = 0.0_f64;
    for i in 0..3
    {
        for j in 0..3
        {
            let updated = state.conformal_curvature[i][j]
                - state.conformal_metric[i][j] * residual_before / 3.0;
            correction_magnitude =
                correction_magnitude.max((updated - state.conformal_curvature[i][j]).abs());
            state.conformal_curvature[i][j] = updated;
        }
    }
    let after = trace_with(&inverse, &state.conformal_curvature);
    Ok(BssnProjection {
        residual_before,
        residual_after: after,
        correction_magnitude,
    })
}

// ---------------------------------------------------------------------------
// Conformal Ricci decomposition
// ---------------------------------------------------------------------------

/// Central first derivatives of `phi`.
fn conformal_factor_gradient<G: Metric<3>>(
    metric: &G,
    coordinates: &[f64; 3],
    step: f64,
) -> Result<[f64; 3], BssnError> {
    let mut gradient = [0.0_f64; 3];
    for i in 0..3
    {
        let mut plus = *coordinates;
        let mut minus = *coordinates;
        plus[i] += step;
        minus[i] -= step;
        gradient[i] = (conformal_factor_at(metric, &plus)? - conformal_factor_at(metric, &minus)?)
            / (2.0 * step);
    }
    Ok(gradient)
}

/// Central second derivatives (coordinate Hessian) of `phi`.
fn conformal_factor_hessian<G: Metric<3>>(
    metric: &G,
    coordinates: &[f64; 3],
    step: f64,
) -> Result<[[f64; 3]; 3], BssnError> {
    let center = conformal_factor_at(metric, coordinates)?;
    let mut hessian = [[0.0_f64; 3]; 3];
    for i in 0..3
    {
        let mut plus = *coordinates;
        let mut minus = *coordinates;
        plus[i] += step;
        minus[i] -= step;
        hessian[i][i] = (conformal_factor_at(metric, &plus)? - 2.0 * center
            + conformal_factor_at(metric, &minus)?)
            / (step * step);
    }
    for i in 0..3
    {
        for j in (i + 1)..3
        {
            let mut pp = *coordinates;
            let mut pm = *coordinates;
            let mut mp = *coordinates;
            let mut mm = *coordinates;
            pp[i] += step;
            pp[j] += step;
            pm[i] += step;
            pm[j] -= step;
            mp[i] -= step;
            mp[j] += step;
            mm[i] -= step;
            mm[j] -= step;
            let value = (conformal_factor_at(metric, &pp)?
                - conformal_factor_at(metric, &pm)?
                - conformal_factor_at(metric, &mp)?
                + conformal_factor_at(metric, &mm)?)
                / (4.0 * step * step);
            hessian[i][j] = value;
            hessian[j][i] = value;
        }
    }
    Ok(hessian)
}

/// The conformal Ricci decomposition `R_ij = Rtilde_ij + Rphi_ij` at a point.
///
/// `Rtilde_ij` is the Ricci tensor of `gammatilde_ij`, obtained from the Layer 1
/// [`ricci_tensor_from_metric`] (**not** re-implemented); only `Rphi_ij` is new:
///
/// ```text
/// Rphi_ij = -2 Dt_i Dt_j phi - 2 gammatilde_ij gammatilde^{kl} Dt_k Dt_l phi
///           + 4 (d_i phi)(d_j phi) - 4 gammatilde_ij gammatilde^{kl} (d_k phi)(d_l phi)
/// ```
///
/// The sum is validated against the physical Ricci tensor computed
/// independently by the same Layer 1 routine applied to `gamma_ij` — a genuine
/// cross-check, since the two paths share no code beyond the Ricci engine.
pub fn conformal_ricci<G: Metric<3>>(
    spatial_metric: &G,
    coordinates: &[f64; 3],
    settings: &AdmEvolutionSettings,
) -> Result<ConformalRicci, BssnError> {
    let conformal = ConformalMetricField {
        physical: spatial_metric,
    };
    let conformal_components = conformal.components(coordinates);
    require_finite_tensor("conformal_metric", &conformal_components)?;
    let inverse = invert_metric::<3>(&conformal_components)
        .map_err(|_| BssnError::SingularConformalMetric)?;

    let conformal_ricci_tensor = ricci_tensor_from_metric::<_, 3>(
        &conformal,
        coordinates,
        settings.spatial_step,
        settings.metric_step,
    )?;

    let christoffel = numerical_christoffel::<_, 3>(&conformal, coordinates, settings.metric_step)
        .map_err(|_| BssnError::UnavailableDerivative {
            quantity: "conformal_christoffel",
        })?;
    let gradient = conformal_factor_gradient(spatial_metric, coordinates, settings.spatial_step)?;
    let hessian = conformal_factor_hessian(spatial_metric, coordinates, settings.spatial_step)?;

    // Covariant Hessian with respect to gammatilde.
    let mut covariant = [[0.0_f64; 3]; 3];
    for i in 0..3
    {
        for j in 0..3
        {
            let mut value = hessian[i][j];
            for k in 0..3
            {
                value -= christoffel[k][i][j] * gradient[k];
            }
            covariant[i][j] = value;
        }
    }
    let laplacian = trace_with(&inverse, &covariant);
    let mut gradient_squared = 0.0;
    for i in 0..3
    {
        for j in 0..3
        {
            gradient_squared += inverse[i][j] * gradient[i] * gradient[j];
        }
    }

    let mut conformal_factor_part = [[0.0_f64; 3]; 3];
    let mut total = [[0.0_f64; 3]; 3];
    for i in 0..3
    {
        for j in 0..3
        {
            conformal_factor_part[i][j] = -2.0 * covariant[i][j]
                - 2.0 * conformal_components[i][j] * laplacian
                + 4.0 * gradient[i] * gradient[j]
                - 4.0 * conformal_components[i][j] * gradient_squared;
            total[i][j] = conformal_ricci_tensor[i][j] + conformal_factor_part[i][j];
        }
    }
    require_finite_tensor("conformal_ricci", &total)?;

    Ok(ConformalRicci {
        conformal: conformal_ricci_tensor,
        conformal_factor_part,
        total,
    })
}

// ---------------------------------------------------------------------------
// Evolution right-hand sides
// ---------------------------------------------------------------------------

/// Evaluate the local BSSN evolution right-hand sides at `coordinates`.
///
/// The lapse and shift are **prescribed** through `gauge` (constant in space for
/// this increment's oracles, so the shift-advection and shift-gradient
/// contributions vanish identically and are reported as zero rather than
/// silently omitted). See the [module documentation](self) for the equations
/// and for the exact `alpha * H` relation between this `d_t K` and the ADM one.
// Explicit tensor-index loops read most clearly here (matching the crate's
// curvature, action, adm, and adm_evolution modules).
/// The BSSN evolution right-hand sides at a point, computing the conformal
/// Ricci tensor internally from the supplied metric field.
///
/// Behaviour is unchanged from Layer 3.3. See
/// [`bssn_evolution_rhs_with_ricci`] for the variant a grid uses, which supplies
/// `Rtilde_ij` in genuine BSSN form instead.
pub fn bssn_evolution_rhs<G: Metric<3>>(
    spatial_metric: &G,
    extrinsic_curvature: &impl SpatialTensorField,
    coordinates: &[f64; 3],
    gauge: &BssnGauge,
    sources: &AdmSources,
    settings: &AdmEvolutionSettings,
) -> Result<(BssnState, BssnEvolutionRhs), BssnError> {
    evolution_rhs_inner(
        spatial_metric,
        extrinsic_curvature,
        coordinates,
        gauge,
        sources,
        settings,
        None,
    )
}

// Tensor-component loops index several distinct rank-2/3 arrays together;
// iterating one of them would obscure that pairing.
#[allow(clippy::needless_range_loop)]
fn evolution_rhs_inner<G: Metric<3>>(
    spatial_metric: &G,
    extrinsic_curvature: &impl SpatialTensorField,
    coordinates: &[f64; 3],
    gauge: &BssnGauge,
    sources: &AdmSources,
    settings: &AdmEvolutionSettings,
    supplied: Option<&BssnSuppliedTerms<'_>>,
) -> Result<(BssnState, BssnEvolutionRhs), BssnError> {
    if !gauge.lapse.is_finite() || gauge.shift.iter().any(|v| !v.is_finite())
    {
        return Err(BssnError::NonFiniteState { quantity: "gauge" });
    }
    let alpha = gauge.lapse;
    let state = adm_to_bssn(spatial_metric, extrinsic_curvature, coordinates, settings)?;
    let physical = spatial_metric.components(coordinates);
    let physical_inverse =
        invert_metric::<3>(&physical).map_err(|_| BssnError::SingularAdmMetric)?;
    let conformal_inverse = state.inverse_conformal_metric()?;

    let trace_k = state.mean_curvature;
    let curvature_up = raise_both(&conformal_inverse, &state.conformal_curvature);
    let curvature_mixed = raise_first(&conformal_inverse, &state.conformal_curvature);
    let mut curvature_squared = 0.0;
    for i in 0..3
    {
        for j in 0..3
        {
            curvature_squared += state.conformal_curvature[i][j] * curvature_up[i][j];
        }
    }

    let stress_trace = sources.trace(&physical_inverse);
    let scale = state.chi();

    // d_t phi. With a spatially constant gauge the shift terms vanish exactly.
    let conformal_factor_lapse_term = -alpha * trace_k / 6.0;
    // beta^j d_j phi + (1/6) d_j beta^j
    let (shift_gradient, shift_hessian, shift_first) = match supplied
    {
        Some(terms) => (
            terms.shift.gradient,
            terms.shift.hessian,
            Some(terms.derivatives),
        ),
        None => ([[0.0_f64; 3]; 3], [[[0.0_f64; 3]; 3]; 3], None),
    };
    // The shift Hessian enters only the `Gammatilde^i` equation, which
    // `bssn_connection_rhs` owns.
    let _ = shift_hessian;
    let mut shift_divergence = 0.0;
    for j in 0..3
    {
        shift_divergence += shift_gradient[j][j];
    }
    let conformal_factor_shift_term = match shift_first
    {
        Some(first) =>
        {
            let mut advection = 0.0;
            for j in 0..3
            {
                advection += gauge.shift[j] * first.conformal_factor_gradient[j];
            }
            advection + shift_divergence / 6.0
        },
        None => 0.0,
    };
    let conformal_factor = conformal_factor_lapse_term + conformal_factor_shift_term;

    // d_t gammatilde_ij.
    let mut conformal_metric_lapse_term = [[0.0_f64; 3]; 3];
    // beta^k d_k gammatilde_ij + gammatilde_ik d_j beta^k + gammatilde_jk d_i beta^k
    //   - (2/3) gammatilde_ij d_k beta^k
    let mut conformal_metric_shift_term = [[0.0_f64; 3]; 3];
    if let Some(first) = shift_first
    {
        for i in 0..3
        {
            for j in 0..3
            {
                let mut value = -2.0 / 3.0 * state.conformal_metric[i][j] * shift_divergence;
                for k in 0..3
                {
                    value += gauge.shift[k] * first.conformal_metric_gradient[k][i][j]
                        + state.conformal_metric[i][k] * shift_gradient[k][j]
                        + state.conformal_metric[j][k] * shift_gradient[k][i];
                }
                conformal_metric_shift_term[i][j] = value;
            }
        }
    }
    let mut conformal_metric_rhs = [[0.0_f64; 3]; 3];
    for i in 0..3
    {
        for j in 0..3
        {
            conformal_metric_lapse_term[i][j] = -2.0 * alpha * state.conformal_curvature[i][j];
            conformal_metric_rhs[i][j] =
                conformal_metric_lapse_term[i][j] + conformal_metric_shift_term[i][j];
        }
    }

    // The covariant lapse Hessian `D_i D_j alpha`. Exactly zero for a prescribed
    // constant lapse; supplied by the caller once the lapse is live.
    let lapse_hessian = match supplied
    {
        Some(terms) => covariant_lapse_hessian(&state, terms.derivatives, terms.lapse)?,
        None => [[0.0_f64; 3]; 3],
    };
    let lapse_laplacian = trace_with(&physical_inverse, &lapse_hessian);

    // d_t K (the constraint-substituted form; see the module documentation).
    let mean_curvature_lapse_term = -lapse_laplacian;
    let mean_curvature_quadratic_term = alpha * (curvature_squared + trace_k * trace_k / 3.0);
    let mean_curvature_matter_term = 4.0 * PI * alpha * (sources.energy_density + stress_trace);
    // beta^j d_j K
    let mean_curvature_shift_term = match shift_first
    {
        Some(first) =>
        {
            let mut value = 0.0;
            for j in 0..3
            {
                value += gauge.shift[j] * first.mean_curvature_gradient[j];
            }
            value
        },
        None => 0.0,
    };
    let mean_curvature = mean_curvature_lapse_term
        + mean_curvature_quadratic_term
        + mean_curvature_matter_term
        + mean_curvature_shift_term;

    // d_t Atilde_ij.
    let ricci = match supplied
    {
        Some(terms) => *terms.ricci,
        None => conformal_ricci(spatial_metric, coordinates, settings)?,
    };
    let mut ricci_source = [[0.0_f64; 3]; 3];
    let mut matter_source = [[0.0_f64; 3]; 3];
    for i in 0..3
    {
        for j in 0..3
        {
            ricci_source[i][j] = alpha * ricci.total[i][j];
            matter_source[i][j] = -8.0 * PI * alpha * sources.stress[i][j];
        }
    }
    let ricci_trace = trace_with(&physical_inverse, &ricci_source);
    let matter_trace = trace_with(&physical_inverse, &matter_source);

    let mut conformal_curvature_lapse_term = [[0.0_f64; 3]; 3];
    let mut conformal_curvature_ricci_term = [[0.0_f64; 3]; 3];
    let mut conformal_curvature_matter_term = [[0.0_f64; 3]; 3];
    let mut conformal_curvature_quadratic_term = [[0.0_f64; 3]; 3];
    // beta^k d_k Atilde_ij + Atilde_ik d_j beta^k + Atilde_jk d_i beta^k
    //   - (2/3) Atilde_ij d_k beta^k
    let mut conformal_curvature_shift_term = [[0.0_f64; 3]; 3];
    if let Some(first) = shift_first
    {
        for i in 0..3
        {
            for j in 0..3
            {
                let mut value = -2.0 / 3.0 * state.conformal_curvature[i][j] * shift_divergence;
                for k in 0..3
                {
                    value += gauge.shift[k] * first.conformal_curvature_gradient[k][i][j]
                        + state.conformal_curvature[i][k] * shift_gradient[k][j]
                        + state.conformal_curvature[j][k] * shift_gradient[k][i];
                }
                conformal_curvature_shift_term[i][j] = value;
            }
        }
    }
    let mut conformal_curvature_rhs = [[0.0_f64; 3]; 3];
    for i in 0..3
    {
        for j in 0..3
        {
            // Trace-free parts with respect to gamma (equivalently gammatilde).
            conformal_curvature_ricci_term[i][j] =
                scale * (ricci_source[i][j] - physical[i][j] * ricci_trace / 3.0);
            conformal_curvature_matter_term[i][j] =
                scale * (matter_source[i][j] - physical[i][j] * matter_trace / 3.0);
            // -e^{-4 phi} ( D_i D_j alpha )^TF
            conformal_curvature_lapse_term[i][j] =
                scale * (-lapse_hessian[i][j] + physical[i][j] * lapse_laplacian / 3.0);
            let mut quadratic = 0.0;
            for k in 0..3
            {
                quadratic += state.conformal_curvature[i][k] * curvature_mixed[k][j];
            }
            conformal_curvature_quadratic_term[i][j] =
                alpha * (trace_k * state.conformal_curvature[i][j] - 2.0 * quadratic);
            conformal_curvature_rhs[i][j] = conformal_curvature_lapse_term[i][j]
                + conformal_curvature_ricci_term[i][j]
                + conformal_curvature_quadratic_term[i][j]
                + conformal_curvature_matter_term[i][j]
                + conformal_curvature_shift_term[i][j];
        }
    }

    // d_t Gammatilde^i. Every term of the standard equation carries either a
    // spatial derivative of a field (d_j alpha, d_j K, d_j phi, d_j beta^i) or
    // the conformal Christoffel symbols, all of which vanish identically for
    // the spatially constant fields and prescribed constant gauge this
    // increment supports. It is therefore exactly zero here -- carried so the
    // state and its rates stay complete, and exercised structurally (the
    // connection constraint) rather than dynamically.
    //
    // A grid, where those gradients are non-zero, must NOT use this value:
    // freezing `Gammatilde^i` while `gammatilde_ij` evolves reduces BSSN back to
    // a weakly hyperbolic ADM-like system. `bssn_connection_rhs` supplies the
    // full equation from caller-provided gradients; `crate::bssn_grid` uses it.
    let conformal_connection = [0.0_f64; 3];

    require_finite_tensor("conformal_metric_rhs", &conformal_metric_rhs)?;
    require_finite_tensor("conformal_curvature_rhs", &conformal_curvature_rhs)?;
    if !conformal_factor.is_finite() || !mean_curvature.is_finite()
    {
        return Err(BssnError::NonFiniteState {
            quantity: "evolution_rhs",
        });
    }

    Ok((
        state,
        BssnEvolutionRhs {
            conformal_factor,
            conformal_factor_lapse_term,
            conformal_factor_shift_term,
            conformal_metric: conformal_metric_rhs,
            conformal_metric_lapse_term,
            conformal_metric_shift_term,
            mean_curvature,
            mean_curvature_lapse_term,
            mean_curvature_quadratic_term,
            mean_curvature_matter_term,
            mean_curvature_shift_term,
            conformal_curvature: conformal_curvature_rhs,
            conformal_curvature_lapse_term,
            conformal_curvature_ricci_term,
            conformal_curvature_quadratic_term,
            conformal_curvature_matter_term,
            conformal_curvature_shift_term,
            conformal_connection,
        },
    ))
}

/// ADM right-hand sides reconstructed from BSSN right-hand sides.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReconstructedAdmRhs {
    /// `d_t gamma_ij`.
    pub spatial_metric: [[f64; 3]; 3],
    /// `d_t K_ij`.
    pub extrinsic_curvature: [[f64; 3]; 3],
}

/// Reconstruct the ADM right-hand sides `(d_t gamma_ij, d_t K_ij)` from a BSSN
/// state and its right-hand sides, by the product rule on
/// `gamma_ij = e^{4 phi} gammatilde_ij` and
/// `K_ij = e^{4 phi} ( Atilde_ij + (1/3) gammatilde_ij K )`.
#[must_use]
pub fn adm_rhs_from_bssn(state: &BssnState, rhs: &BssnEvolutionRhs) -> ReconstructedAdmRhs {
    let scale = state.conformal_scale();
    let mut metric_rhs = [[0.0_f64; 3]; 3];
    let mut curvature_rhs = [[0.0_f64; 3]; 3];
    for i in 0..3
    {
        for j in 0..3
        {
            metric_rhs[i][j] = 4.0 * scale * rhs.conformal_factor * state.conformal_metric[i][j]
                + scale * rhs.conformal_metric[i][j];

            let inner = state.conformal_curvature[i][j]
                + state.conformal_metric[i][j] * state.mean_curvature / 3.0;
            let inner_rhs = rhs.conformal_curvature[i][j]
                + (rhs.conformal_metric[i][j] * state.mean_curvature
                    + state.conformal_metric[i][j] * rhs.mean_curvature)
                    / 3.0;
            curvature_rhs[i][j] = 4.0 * scale * rhs.conformal_factor * inner + scale * inner_rhs;
        }
    }
    ReconstructedAdmRhs {
        spatial_metric: metric_rhs,
        extrinsic_curvature: curvature_rhs,
    }
}

/// Compare the ADM right-hand sides reconstructed from BSSN against the Layer
/// 3.1 ADM right-hand sides at the same point.
///
/// `d_t gamma_ij` agrees unconditionally. `d_t K_ij` agrees **only on the
/// constraint surface**: the standard BSSN `d_t K` is the constraint-substituted
/// form, so the two differ by exactly `alpha * H`. Both the raw and the
/// constraint-corrected differences are returned, so the residual is visibly
/// accounted for rather than tuned away.
pub fn adm_bssn_equivalence<G: Metric<3>>(
    spatial_metric: &G,
    extrinsic_curvature: &impl SpatialTensorField,
    coordinates: &[f64; 3],
    gauge: &BssnGauge,
    sources: &AdmSources,
    settings: &AdmEvolutionSettings,
) -> Result<AdmBssnEquivalence, BssnError> {
    let lapse = gauge.lapse;
    let lapse_field = move |_: &[f64; 3]| lapse;
    let shift = gauge.shift;
    let shift_field = move |_: &[f64; 3]| shift;

    let adm_metric_rhs = metric_evolution_rhs(
        spatial_metric,
        extrinsic_curvature,
        &lapse_field,
        &shift_field,
        coordinates,
        settings,
    )?;
    let adm_curvature_rhs = curvature_evolution_rhs(
        spatial_metric,
        extrinsic_curvature,
        &lapse_field,
        &shift_field,
        coordinates,
        sources,
        settings,
    )?;
    let hamiltonian = hamiltonian_constraint(
        spatial_metric,
        extrinsic_curvature,
        coordinates,
        sources,
        settings,
    )?;

    let (state, rhs) = bssn_evolution_rhs(
        spatial_metric,
        extrinsic_curvature,
        coordinates,
        gauge,
        sources,
        settings,
    )?;
    let reconstructed = adm_rhs_from_bssn(&state, &rhs);
    let metric_from_bssn = reconstructed.spatial_metric;
    let curvature_from_bssn = reconstructed.extrinsic_curvature;

    // The constraint-substitution enters d_t K (the trace) as alpha * H, which
    // propagates into d_t K_ij through the (1/3) gammatilde_ij K term:
    //   d_t K_ij |_ADM - d_t K_ij |_BSSN = e^{4 phi} (1/3) gammatilde_ij * alpha * H
    //                                    = (1/3) gamma_ij * alpha * H.
    let physical = spatial_metric.components(coordinates);
    let correction = gauge.lapse * hamiltonian.signed_residual / 3.0;

    let mut metric_difference = 0.0_f64;
    let mut curvature_difference = 0.0_f64;
    let mut corrected = 0.0_f64;
    for i in 0..3
    {
        for j in 0..3
        {
            metric_difference =
                metric_difference.max((metric_from_bssn[i][j] - adm_metric_rhs.total[i][j]).abs());
            let raw = adm_curvature_rhs.total[i][j] - curvature_from_bssn[i][j];
            curvature_difference = curvature_difference.max(raw.abs());
            corrected = corrected.max((raw - physical[i][j] * correction).abs());
        }
    }

    Ok(AdmBssnEquivalence {
        metric_difference,
        curvature_difference,
        curvature_difference_constraint_corrected: corrected,
        hamiltonian_residual: hamiltonian.signed_residual,
        lapse: gauge.lapse,
    })
}

// ---------------------------------------------------------------------------
// The conformal connection evolution equation
// ---------------------------------------------------------------------------

/// The spatial gradients a grid must supply to evolve `Gammatilde^i`.
///
/// Layer 3.3 evaluates every derivative internally from coordinate-sampled
/// fields, which is why it could only report `d_t Gammatilde^i = 0` for the
/// spatially constant data it supported. A grid has these gradients already, so
/// it supplies them directly rather than being differenced a second time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BssnSpatialDerivatives {
    /// `d_i phi`.
    pub conformal_factor_gradient: [f64; 3],
    /// `d_i K`.
    pub mean_curvature_gradient: [f64; 3],
    /// `d_k gammatilde_ij`, indexed `[k][i][j]`.
    pub conformal_metric_gradient: [[[f64; 3]; 3]; 3],
    /// `d_k Atilde_ij`, indexed `[k][i][j]`.
    pub conformal_curvature_gradient: [[[f64; 3]; 3]; 3],
}

impl BssnSpatialDerivatives {
    /// All gradients zero — the spatially homogeneous case Layer 3.3 supported.
    pub const ZERO: Self = Self {
        conformal_factor_gradient: [0.0; 3],
        mean_curvature_gradient: [0.0; 3],
        conformal_metric_gradient: [[[0.0; 3]; 3]; 3],
        conformal_curvature_gradient: [[[0.0; 3]; 3]; 3],
    };
}

/// The decomposed right-hand side of the `Gammatilde^i` evolution equation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BssnConnectionRhs {
    /// The total right-hand side.
    pub total: [f64; 3],
    /// The `2 alpha Gammatilde^i_jk Atilde^jk` contribution.
    pub christoffel_term: [f64; 3],
    /// The `-(4/3) alpha gammatilde^{ij} d_j K` contribution.
    pub mean_curvature_term: [f64; 3],
    /// The `12 alpha Atilde^{ij} d_j phi` contribution.
    pub conformal_factor_term: [f64; 3],
    /// The `-2 Atilde^{ij} d_j alpha` contribution (zero for a constant lapse).
    pub lapse_term: [f64; 3],
    /// The shift advection and shift-gradient contribution (zero for zero shift).
    pub shift_term: [f64; 3],
    /// The conformal Christoffel symbols used, exposed for cross-checking.
    pub conformal_christoffel: [[[f64; 3]; 3]; 3],
}

/// The evolution right-hand side of the conformal connection functions
/// `Gammatilde^i`.
///
/// # Derivation, not transcription
///
/// `Gammatilde^i` is *defined* by `Gammatilde^i = gammatilde^{jk}
/// Gammatilde^i_{jk} = -d_j gammatilde^{ij}`, so differentiating the definition
/// in time gives, for a constant lapse and zero shift,
///
/// ```text
/// d_t gammatilde_ij  = -2 alpha Atilde_ij
/// d_t gammatilde^{ij} = +2 alpha Atilde^{ij}
/// d_t Gammatilde^i    = -d_j ( 2 alpha Atilde^{ij} ) = -2 alpha d_j Atilde^{ij}
/// ```
///
/// Substituting the momentum constraint for `d_j Atilde^{ij}` gives the form
/// implemented here:
///
/// ```text
/// d_t Gammatilde^i = -2 Atilde^{ij} d_j alpha
///     + 2 alpha ( Gammatilde^i_{jk} Atilde^{jk}
///                 + 6 Atilde^{ij} d_j phi
///                 - (2/3) gammatilde^{ij} d_j K )
/// ```
///
/// **This is constraint-substituted**, exactly as the `d_t K` equation is (see
/// the module documentation). The two forms agree *on the momentum-constraint
/// surface* and differ off it by the momentum constraint itself. The
/// substituted form is the one that makes BSSN strongly hyperbolic, which is
/// the entire reason BSSN exists, so it is the form used — and the fact that it
/// is substituted is stated rather than glossed.
///
/// # Matter
///
/// The matter contribution is **not** implemented. Deriving it requires fixing a
/// convention for the momentum density that this increment has no vacuum-free
/// oracle to validate against, and shipping an unvalidated formula would be
/// worse than refusing. A non-zero momentum density is therefore rejected with a
/// typed error rather than silently ignored or silently guessed.
// Tensor-component loops index several distinct arrays together by index.
#[allow(clippy::needless_range_loop)]
pub fn bssn_connection_rhs(
    state: &BssnState,
    derivatives: &BssnSpatialDerivatives,
    second: &BssnSecondDerivatives,
    lapse: &BssnLapseDerivatives,
    shift: &BssnShiftDerivatives,
    gauge: &BssnGauge,
    sources: &AdmSources,
) -> Result<BssnConnectionRhs, BssnError> {
    if !gauge.lapse.is_finite() || gauge.shift.iter().any(|value| !value.is_finite())
    {
        return Err(BssnError::NonFiniteState { quantity: "gauge" });
    }
    if sources.momentum_density.iter().any(|value| *value != 0.0)
    {
        return Err(BssnError::UnavailableDerivative {
            quantity: "connection_matter_term",
        });
    }

    let alpha = gauge.lapse;
    let inverse = state.inverse_conformal_metric()?;
    let curvature_up = raise_both(&inverse, &state.conformal_curvature);

    // One implementation of the Christoffel formula in the crate: this reuses
    // the algebraic core that `numerical_christoffel` also calls.
    let christoffel =
        christoffel_from_derivatives::<3>(&inverse, &derivatives.conformal_metric_gradient);

    let mut christoffel_term = [0.0_f64; 3];
    let mut mean_curvature_term = [0.0_f64; 3];
    let mut conformal_factor_term = [0.0_f64; 3];
    let mut lapse_term = [0.0_f64; 3];
    let mut shift_term = [0.0_f64; 3];
    let mut total = [0.0_f64; 3];

    for i in 0..3
    {
        let mut contracted = 0.0;
        for j in 0..3
        {
            for k in 0..3
            {
                contracted += christoffel[i][j][k] * curvature_up[j][k];
            }
        }
        christoffel_term[i] = 2.0 * alpha * contracted;

        let mut trace_gradient = 0.0;
        let mut factor_gradient = 0.0;
        for j in 0..3
        {
            trace_gradient += inverse[i][j] * derivatives.mean_curvature_gradient[j];
            factor_gradient += curvature_up[i][j] * derivatives.conformal_factor_gradient[j];
        }
        mean_curvature_term[i] = -(4.0 / 3.0) * alpha * trace_gradient;
        conformal_factor_term[i] = 12.0 * alpha * factor_gradient;

        // -2 Atilde^{ij} d_j alpha. Exactly zero for a prescribed constant
        // lapse, non-zero once the lapse is live.
        let mut lapse_gradient_contraction = 0.0;
        for j in 0..3
        {
            lapse_gradient_contraction += curvature_up[i][j] * lapse.gradient[j];
        }
        lapse_term[i] = -2.0 * lapse_gradient_contraction;

        // Shift terms:
        //   beta^j d_j Gammatilde^i - Gammatilde^j d_j beta^i
        //   + (2/3) Gammatilde^i d_j beta^j
        //   + (1/3) gammatilde^{li} d_l d_j beta^j + gammatilde^{lj} d_j d_l beta^i
        let mut divergence = 0.0;
        for j in 0..3
        {
            divergence += shift.gradient[j][j];
        }
        let mut shift_contribution = 2.0 / 3.0 * state.conformal_connection[i] * divergence;
        for j in 0..3
        {
            shift_contribution += gauge.shift[j] * second.conformal_connection_gradient[i][j]
                - state.conformal_connection[j] * shift.gradient[i][j];
        }
        for l in 0..3
        {
            let mut divergence_gradient = 0.0;
            for j in 0..3
            {
                divergence_gradient += shift.hessian[j][l][j];
            }
            shift_contribution += inverse[l][i] * divergence_gradient / 3.0;
            for j in 0..3
            {
                shift_contribution += inverse[l][j] * shift.hessian[i][j][l];
            }
        }
        shift_term[i] = shift_contribution;

        total[i] = christoffel_term[i]
            + mean_curvature_term[i]
            + conformal_factor_term[i]
            + lapse_term[i]
            + shift_term[i];

        if !total[i].is_finite()
        {
            return Err(BssnError::NonFiniteState {
                quantity: "connection_rhs",
            });
        }
    }

    Ok(BssnConnectionRhs {
        total,
        christoffel_term,
        mean_curvature_term,
        conformal_factor_term,
        lapse_term,
        shift_term,
        conformal_christoffel: christoffel,
    })
}

// ---------------------------------------------------------------------------
// The conformal Ricci tensor in genuine BSSN form
// ---------------------------------------------------------------------------

/// The second spatial derivatives the BSSN-form conformal Ricci tensor needs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BssnSecondDerivatives {
    /// `d_i d_j phi`.
    pub conformal_factor_hessian: [[f64; 3]; 3],
    /// `d_k d_l gammatilde_ij`, indexed `[k][l][i][j]`.
    pub conformal_metric_hessian: [[[[f64; 3]; 3]; 3]; 3],
    /// `d_j Gammatilde^i`, indexed `[i][j]`.
    pub conformal_connection_gradient: [[f64; 3]; 3],
}

/// The conformal Ricci tensor written in **BSSN form**, using the
/// independently evolved `Gammatilde^k`.
///
/// # Why this is not the same as differentiating the metric
///
/// [`conformal_ricci`] obtains `Rtilde_ij` from [`ricci_tensor_from_metric`] —
/// the generic Ricci tensor of `gammatilde_ij`. That is *correct* as a tensor
/// identity, and it is why the two agree on the constraint surface. But it is
/// **not the BSSN system**.
///
/// The BSSN form is
///
/// ```text
/// Rtilde_ij = -1/2 gammatilde^{lm} d_l d_m gammatilde_ij
///           + gammatilde_{k(i} d_{j)} Gammatilde^k
///           + Gammatilde^k Gammatilde_{(ij)k}
///           + gammatilde^{lm} ( 2 Gammatilde^k_{l(i} Gammatilde_{j)km}
///                             + Gammatilde^k_{im} Gammatilde_{klj} )
/// ```
///
/// where `Gammatilde_{kij} = gammatilde_{kl} Gammatilde^l_{ij}` and
/// `X_{(ij)} = 1/2 (X_ij + X_ji)`.
///
/// The second term carries the **evolved** `Gammatilde^k`, and it is exactly
/// what removes the mixed second derivatives `d_i d_k gammatilde_jl` from the
/// principal part, leaving only the manifestly elliptic
/// `-1/2 gammatilde^{lm} d_l d_m`. That substitution is the entire reason BSSN
/// exists: it is what turns the weakly hyperbolic ADM system into a strongly
/// hyperbolic one.
///
/// Computing `Rtilde_ij` generically from the metric instead puts those mixed
/// derivatives back and leaves a system with ADM's principal part wearing BSSN
/// variables. Evolving `Gammatilde^i` does not help if the Ricci tensor never
/// reads it.
///
/// Being an evolved variable rather than a derived one, `Gammatilde^k` can drift
/// from `-d_j gammatilde^{kj}`; that drift is the connection constraint, which
/// is monitored and never enforced.
// Tensor-component loops index several distinct arrays together by index.
#[allow(clippy::needless_range_loop)]
pub fn conformal_ricci_from_derivatives(
    state: &BssnState,
    first: &BssnSpatialDerivatives,
    second: &BssnSecondDerivatives,
) -> Result<ConformalRicci, BssnError> {
    let inverse = state.inverse_conformal_metric()?;
    let christoffel = christoffel_from_derivatives::<3>(&inverse, &first.conformal_metric_gradient);

    // Fully covariant connection Gammatilde_{kij} = gammatilde_{kl} Gammatilde^l_{ij}.
    let mut lowered = [[[0.0_f64; 3]; 3]; 3];
    for k in 0..3
    {
        for i in 0..3
        {
            for j in 0..3
            {
                let mut value = 0.0;
                for l in 0..3
                {
                    value += state.conformal_metric[k][l] * christoffel[l][i][j];
                }
                lowered[k][i][j] = value;
            }
        }
    }

    let mut conformal = [[0.0_f64; 3]; 3];
    for i in 0..3
    {
        for j in 0..3
        {
            // -1/2 gammatilde^{lm} d_l d_m gammatilde_ij : the principal part,
            // and the only second-derivative term that survives.
            let mut laplacian = 0.0;
            for l in 0..3
            {
                for m in 0..3
                {
                    laplacian += inverse[l][m] * second.conformal_metric_hessian[l][m][i][j];
                }
            }

            // gammatilde_{k(i} d_{j)} Gammatilde^k, using the EVOLVED connection.
            let mut connection_term = 0.0;
            for k in 0..3
            {
                connection_term += state.conformal_metric[k][i]
                    * second.conformal_connection_gradient[k][j]
                    + state.conformal_metric[k][j] * second.conformal_connection_gradient[k][i];
            }
            connection_term *= 0.5;

            // Gammatilde^k Gammatilde_{(ij)k}, with `Gammatilde_{abc} = lowered[a][b][c]`.
            //
            // `Gammatilde^k` contracts the LAST index, not the first. Writing
            // `lowered[k][i][j]` here contracts the first instead, which is a
            // different tensor: it produced a systematic error that saturated
            // near 1% of the Ricci scale and refused to converge. One dimension
            // could not see it, because `Gammatilde^k` is nearly zero there.
            let mut product_term = 0.0;
            for k in 0..3
            {
                product_term +=
                    state.conformal_connection[k] * (lowered[i][j][k] + lowered[j][i][k]);
            }
            product_term *= 0.5;

            // gammatilde^{lm} ( 2 Gammatilde^k_{l(i} Gammatilde_{j)km}
            //                 + Gammatilde^k_{im} Gammatilde_{klj} )
            let mut quadratic = 0.0;
            for l in 0..3
            {
                for m in 0..3
                {
                    let mut bracket = 0.0;
                    for k in 0..3
                    {
                        bracket += christoffel[k][l][i] * lowered[j][k][m]
                            + christoffel[k][l][j] * lowered[i][k][m]
                            + christoffel[k][i][m] * lowered[k][l][j];
                    }
                    quadratic += inverse[l][m] * bracket;
                }
            }

            conformal[i][j] = -0.5 * laplacian + connection_term + product_term + quadratic;
        }
    }
    require_finite_tensor("bssn_conformal_ricci", &conformal)?;

    // The conformal-factor part, from the supplied gradients rather than a
    // second differencing of the metric.
    let mut covariant_hessian = [[0.0_f64; 3]; 3];
    for i in 0..3
    {
        for j in 0..3
        {
            let mut value = second.conformal_factor_hessian[i][j];
            for k in 0..3
            {
                value -= christoffel[k][i][j] * first.conformal_factor_gradient[k];
            }
            covariant_hessian[i][j] = value;
        }
    }
    let hessian_trace = trace_with(&inverse, &covariant_hessian);
    let mut gradient_norm = 0.0;
    for k in 0..3
    {
        for l in 0..3
        {
            gradient_norm += inverse[k][l]
                * first.conformal_factor_gradient[k]
                * first.conformal_factor_gradient[l];
        }
    }

    let mut conformal_factor_part = [[0.0_f64; 3]; 3];
    let mut total = [[0.0_f64; 3]; 3];
    for i in 0..3
    {
        for j in 0..3
        {
            conformal_factor_part[i][j] = -2.0 * covariant_hessian[i][j]
                - 2.0 * state.conformal_metric[i][j] * hessian_trace
                + 4.0 * first.conformal_factor_gradient[i] * first.conformal_factor_gradient[j]
                - 4.0 * state.conformal_metric[i][j] * gradient_norm;
            total[i][j] = conformal[i][j] + conformal_factor_part[i][j];
        }
    }
    require_finite_tensor("bssn_conformal_factor_ricci", &conformal_factor_part)?;

    Ok(ConformalRicci {
        conformal,
        conformal_factor_part,
        total,
    })
}

/// The BSSN evolution right-hand sides at a point, with the conformal Ricci
/// tensor **supplied by the caller**.
///
/// A grid computes `Rtilde_ij` in genuine BSSN form from the evolved
/// `Gammatilde^k` (see [`conformal_ricci_from_derivatives`]), which is a
/// different discrete operator from the generic metric Ricci that
/// [`bssn_evolution_rhs`] would compute internally — and the difference is
/// precisely what makes the system strongly rather than weakly hyperbolic.
///
/// Everything downstream of the Ricci tensor is shared with
/// [`bssn_evolution_rhs`]: there is one implementation of the BSSN evolution
/// equations, differing only in where `Rtilde_ij` comes from.
pub fn bssn_evolution_rhs_with_ricci<G: Metric<3>>(
    spatial_metric: &G,
    extrinsic_curvature: &impl SpatialTensorField,
    coordinates: &[f64; 3],
    gauge: &BssnGauge,
    sources: &AdmSources,
    settings: &AdmEvolutionSettings,
    supplied: &BssnSuppliedTerms<'_>,
) -> Result<(BssnState, BssnEvolutionRhs), BssnError> {
    evolution_rhs_inner(
        spatial_metric,
        extrinsic_curvature,
        coordinates,
        gauge,
        sources,
        settings,
        Some(supplied),
    )
}

/// The shift gradients a live or prescribed shift must supply.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BssnShiftDerivatives {
    /// `d_j beta^i`, indexed `[i][j]`.
    pub gradient: [[f64; 3]; 3],
    /// `d_j d_k beta^i`, indexed `[i][j][k]`.
    pub hessian: [[[f64; 3]; 3]; 3],
}

impl BssnShiftDerivatives {
    /// A zero or spatially constant shift: no gradient, no curvature.
    pub const ZERO: Self = Self {
        gradient: [[0.0; 3]; 3],
        hessian: [[[0.0; 3]; 3]; 3],
    };
}

/// The precomputed terms a grid supplies to the BSSN evolution equations.
///
/// Bundled rather than passed separately because they always travel together:
/// each is a quantity the grid can compute with compact stencils and the
/// coordinate-sampled path cannot.
#[derive(Debug, Clone, Copy)]
pub struct BssnSuppliedTerms<'a> {
    /// The conformal Ricci tensor in genuine BSSN form.
    pub ricci: &'a ConformalRicci,
    /// The first spatial derivatives of the evolved fields.
    pub derivatives: &'a BssnSpatialDerivatives,
    /// The lapse gradient and coordinate Hessian.
    pub lapse: &'a BssnLapseDerivatives,
    /// The shift gradient and coordinate Hessian.
    pub shift: &'a BssnShiftDerivatives,
}

// ---------------------------------------------------------------------------
// Live slicing: 1+log
// ---------------------------------------------------------------------------

/// The lapse gradient and coordinate Hessian a live slicing must supply.
///
/// `hessian` is the **coordinate** second derivative `d_i d_j alpha`; the
/// covariant `D_i D_j alpha` is formed internally, because doing so needs the
/// physical Christoffel symbols and a caller should not have to reconstruct
/// them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BssnLapseDerivatives {
    /// `d_i alpha`.
    pub gradient: [f64; 3],
    /// `d_i d_j alpha`.
    pub hessian: [[f64; 3]; 3],
}

impl BssnLapseDerivatives {
    /// A prescribed constant lapse: no gradient, no curvature.
    pub const ZERO: Self = Self {
        gradient: [0.0; 3],
        hessian: [[0.0; 3]; 3],
    };
}

/// The covariant lapse Hessian
/// `D_i D_j alpha = d_i d_j alpha - Gamma^k_ij d_k alpha`, using the **physical**
/// Christoffel symbols.
///
/// The physical connection follows from the conformal one by the standard
/// relation
///
/// ```text
/// Gamma^k_ij = Gammatilde^k_ij
///            + 2 ( delta^k_i d_j phi + delta^k_j d_i phi
///                  - gammatilde_ij gammatilde^{kl} d_l phi )
/// ```
///
/// so the physical metric is never differenced a second time and the grid's
/// compact stencils carry through unchanged.
#[allow(clippy::needless_range_loop)]
fn covariant_lapse_hessian(
    state: &BssnState,
    first: &BssnSpatialDerivatives,
    lapse: &BssnLapseDerivatives,
) -> Result<[[f64; 3]; 3], BssnError> {
    let inverse = state.inverse_conformal_metric()?;
    let conformal = christoffel_from_derivatives::<3>(&inverse, &first.conformal_metric_gradient);

    // Raise the conformal-factor gradient with the conformal metric.
    let mut raised_factor_gradient = [0.0_f64; 3];
    for k in 0..3
    {
        let mut value = 0.0;
        for l in 0..3
        {
            value += inverse[k][l] * first.conformal_factor_gradient[l];
        }
        raised_factor_gradient[k] = value;
    }

    let mut hessian = [[0.0_f64; 3]; 3];
    for i in 0..3
    {
        for j in 0..3
        {
            let mut correction = 0.0;
            for k in 0..3
            {
                let delta_ki = if k == i { 1.0 } else { 0.0 };
                let delta_kj = if k == j { 1.0 } else { 0.0 };
                let physical_christoffel = conformal[k][i][j]
                    + 2.0
                        * (delta_ki * first.conformal_factor_gradient[j]
                            + delta_kj * first.conformal_factor_gradient[i]
                            - state.conformal_metric[i][j] * raised_factor_gradient[k]);
                correction += physical_christoffel * lapse.gradient[k];
            }
            hessian[i][j] = lapse.hessian[i][j] - correction;
        }
    }
    require_finite_tensor("covariant_lapse_hessian", &hessian)?;
    Ok(hessian)
}

/// The 1+log slicing condition with zero shift:
///
/// ```text
/// d_t alpha = -2 alpha K
/// ```
///
/// This is the standard singularity-avoiding slicing. Linearised about flat
/// space it makes the lapse obey a wave equation with characteristic speed
/// `sqrt(2)` — *faster* than light, which is legitimate because the lapse is
/// gauge and carries no physical signal. That speed is a sharp, closed-form
/// oracle and is what this implementation is validated against.
///
/// The shift is **not** evolved: no Gamma-driver, no shift advection. With
/// `beta^i = 0` the advective term `beta^j d_j alpha` vanishes identically, so
/// this is the complete equation for the configuration supported here.
#[must_use]
pub fn one_plus_log_lapse_rhs(state: &BssnState, gauge: &BssnGauge) -> f64 {
    -2.0 * gauge.lapse * state.mean_curvature
}

// ---------------------------------------------------------------------------
// The Gamma-driver shift
// ---------------------------------------------------------------------------

/// The hyperbolic Gamma-driver shift condition, in its standard first-order form:
///
/// ```text
/// d_t beta^i = (3/4) B^i
/// d_t B^i    = d_t Gammatilde^i - eta B^i
/// ```
///
/// Together with 1+log slicing this is the "moving puncture" gauge. The `3/4`
/// sets the shift's characteristic speed to `1` — exactly the speed of light —
/// which is why that coefficient is not free.
///
/// `eta` is a damping rate with dimensions of inverse length; it suppresses the
/// long-wavelength drift the undamped driver would otherwise develop. A negative
/// `eta` would *amplify* that drift, so it is rejected.
///
/// The `d_t Gammatilde^i` passed in must be the **full** connection right-hand
/// side, shift terms included: the driver is defined to chase whatever
/// `Gammatilde^i` actually does, and feeding it a partial rate would make the
/// shift chase a quantity nothing else evolves.
pub fn gamma_driver_rhs(
    connection_rate: &[f64; 3],
    driver: &[f64; 3],
    eta: f64,
) -> Result<([f64; 3], [f64; 3]), BssnError> {
    if !eta.is_finite() || eta < 0.0
    {
        return Err(BssnError::InvalidConformalVariable(eta));
    }
    let mut shift_rate = [0.0_f64; 3];
    let mut driver_rate = [0.0_f64; 3];
    for i in 0..3
    {
        shift_rate[i] = 0.75 * driver[i];
        driver_rate[i] = connection_rate[i] - eta * driver[i];
        if !shift_rate[i].is_finite() || !driver_rate[i].is_finite()
        {
            return Err(BssnError::NonFiniteState {
                quantity: "gamma_driver",
            });
        }
    }
    Ok((shift_rate, driver_rate))
}
