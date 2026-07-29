//! ADM constraint and evolution core (Layer 3.1 — Numerical Relativity opens).
//!
//! Given the spatial metric, extrinsic curvature, lapse, and shift as
//! **independently specified data** on one spacelike slice (not extracted from
//! an already-known 4-metric — that is Layer 2's [`crate::adm`]), this module
//! evaluates the Gauss-Codazzi Hamiltonian and momentum constraints and the
//! right-hand sides of the ADM evolution equations, at a single point. It is a
//! right-hand-side *evaluator*, not a time integrator: nothing here advances
//! data in time, and no spatial grid or mesh is introduced. See
//! `docs/LAYER_3_ADM_EVOLUTION.md` for the full design, the oracle hierarchy,
//! and — critically — the independent numerical derivation of the
//! extrinsic-curvature evolution equation's matter-term sign.
//!
//! **Category.** The ADM equations themselves are established general
//! relativity; no new physics and no modified gravity. The numbers are a
//! numerical approximation (central finite differences of the supplied
//! fields), with truncation error inherent to the method, never hidden.
//!
//! ## Conventions (signature `(-,+,+,+)`, matching [`crate::adm`] exactly)
//!
//! ```text
//! K_ij = -1/(2N) ( partial_t gamma_ij - D_i N_j - D_j N_i )
//!
//! Hamiltonian:  R^(3) + K^2 - K_ij K^{ij} - 16 pi rho = 0
//! Momentum:     D_j ( K^{ij} - gamma^{ij} K ) - 8 pi S^i = 0
//!
//! partial_t gamma_ij = -2 alpha K_ij + (Lie_beta gamma)_ij
//! partial_t K_ij      = -D_i D_j alpha
//!                      + alpha ( R_ij + K K_ij - 2 K_ik K^k_j )
//!                      + (Lie_beta K)_ij
//!                      - 8 pi alpha ( S_ij - 1/2 gamma_ij (S - rho) )
//! ```
//!
//! The matter-term sign in the last equation was **corrected** from the
//! naively-copied textbook form after independent numerical verification
//! against an exact FLRW solution (see the design note, §4) — it is not the
//! sign a literal transcription would give for this repository's `K_ij`
//! convention.

use std::f64::consts::PI;
use std::fmt;

use crate::adm::{mean_curvature, raise};
use crate::{
    Metric, RelativityError, invert_metric, numerical_christoffel, ricci_tensor_from_metric,
};

/// A typed failure of the ADM constraint/evolution evaluators. It never panics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdmEvolutionError {
    /// A coordinate is not finite (carries its index).
    NonFiniteCoordinate(usize),
    /// A finite-difference step is not finite and strictly positive.
    InvalidStep(f64),
    /// The spatial metric is singular (cannot be inverted); this also covers a
    /// failed index raising, which needs the same inverse.
    SingularSpatialMetric,
    /// A supplied field (lapse, shift, or extrinsic curvature) evaluated to a
    /// non-finite value; carries the field's name.
    NonFiniteField {
        /// Which field produced the non-finite value.
        field: &'static str,
    },
    /// An assembled output quantity is not finite; carries its name.
    NonFiniteResult {
        /// Short name of the offending quantity.
        quantity: &'static str,
    },
    /// A curvature or connection evaluation failed.
    Curvature(RelativityError),
}

impl fmt::Display for AdmEvolutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::NonFiniteCoordinate(index) => write!(f, "coordinate {index} is not finite"),
            Self::InvalidStep(step) =>
            {
                write!(
                    f,
                    "finite-difference step {step} must be finite and positive"
                )
            },
            Self::SingularSpatialMetric => write!(f, "the spatial metric is singular"),
            Self::NonFiniteField { field } => write!(f, "field '{field}' is not finite"),
            Self::NonFiniteResult { quantity } => write!(f, "quantity '{quantity}' is not finite"),
            Self::Curvature(error) => write!(f, "curvature evaluation failed: {error}"),
        }
    }
}

impl std::error::Error for AdmEvolutionError {}

impl From<RelativityError> for AdmEvolutionError {
    fn from(error: RelativityError) -> Self {
        Self::Curvature(error)
    }
}

/// A scalar field on the spatial slice (for example the lapse), sampled at a
/// point.
pub trait SpatialScalarField {
    /// Evaluate the field at `coordinates`.
    fn value(&self, coordinates: &[f64; 3]) -> f64;
}

impl<F: Fn(&[f64; 3]) -> f64> SpatialScalarField for F {
    fn value(&self, coordinates: &[f64; 3]) -> f64 {
        self(coordinates)
    }
}

/// A vector field on the spatial slice (for example the shift `N^i`), sampled
/// at a point.
pub trait SpatialVectorField {
    /// Evaluate the field's contravariant components at `coordinates`.
    fn components(&self, coordinates: &[f64; 3]) -> [f64; 3];
}

impl<F: Fn(&[f64; 3]) -> [f64; 3]> SpatialVectorField for F {
    fn components(&self, coordinates: &[f64; 3]) -> [f64; 3] {
        self(coordinates)
    }
}

/// A symmetric rank-2 tensor field on the spatial slice (for example the
/// extrinsic curvature `K_ij`), sampled at a point.
///
/// Deliberately distinct from [`Metric<3>`]: it need not be positive-definite
/// or invertible — it is a curvature, not a distance.
pub trait SpatialTensorField {
    /// Evaluate the field's covariant components at `coordinates`.
    fn components(&self, coordinates: &[f64; 3]) -> [[f64; 3]; 3];
}

impl<F: Fn(&[f64; 3]) -> [[f64; 3]; 3]> SpatialTensorField for F {
    fn components(&self, coordinates: &[f64; 3]) -> [[f64; 3]; 3] {
        self(coordinates)
    }
}

/// The 3+1 (ADM) projections of the stress-energy tensor onto the spatial
/// slice, measured by the normal observer:
///
/// ```text
/// rho  = T_{mu nu} n^mu n^nu           (energy density)
/// S_i  = -gamma_{i mu} n_nu T^{mu nu}    (momentum density)
/// S_ij = gamma_{i mu} gamma_{j nu} T^{mu nu}   (spatial stress)
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdmSources {
    /// The energy density `rho`.
    pub energy_density: f64,
    /// The momentum density `S_i` (lower spatial index).
    pub momentum_density: [f64; 3],
    /// The spatial stress `S_ij` (both indices lower).
    pub stress: [[f64; 3]; 3],
}

impl AdmSources {
    /// The vacuum source: every projection is zero.
    pub const VACUUM: Self = Self {
        energy_density: 0.0,
        momentum_density: [0.0; 3],
        stress: [[0.0; 3]; 3],
    };

    /// A perfect fluid at rest relative to the normal observer: isotropic
    /// stress `S_ij = pressure * gamma_ij`, zero momentum density.
    #[must_use]
    pub fn perfect_fluid(
        energy_density: f64,
        pressure: f64,
        spatial_metric: &[[f64; 3]; 3],
    ) -> Self {
        let mut stress = [[0.0_f64; 3]; 3];
        for i in 0..3
        {
            for j in 0..3
            {
                stress[i][j] = pressure * spatial_metric[i][j];
            }
        }
        Self {
            energy_density,
            momentum_density: [0.0; 3],
            stress,
        }
    }

    /// The spatial trace `S = gamma^{ij} S_ij`.
    // Explicit tensor-index loops read most clearly here (matching the crate's
    // curvature, action, and adm modules).
    #[allow(clippy::needless_range_loop)]
    #[must_use]
    pub fn trace(&self, inverse_spatial_metric: &[[f64; 3]; 3]) -> f64 {
        let mut value = 0.0;
        for i in 0..3
        {
            for j in 0..3
            {
                value += inverse_spatial_metric[i][j] * self.stress[i][j];
            }
        }
        value
    }
}

/// The finite-difference settings shared by every evaluator in this module.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdmEvolutionSettings {
    /// The step used to differentiate the supplied fields (lapse, shift,
    /// extrinsic curvature, spatial metric) and for `R^(3)` / the spatial
    /// Christoffel symbols' outer difference.
    pub spatial_step: f64,
    /// The inner step used inside the nested Christoffel-of-metric
    /// construction (see [`crate::ricci_scalar_from_metric`]).
    pub metric_step: f64,
}

/// The decomposed Hamiltonian constraint residual
/// `R^(3) + K^2 - K_ij K^{ij} - 16 pi rho`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HamiltonianConstraint {
    /// The signed residual.
    pub signed_residual: f64,
    /// `|signed_residual|`.
    pub absolute_residual: f64,
    /// `signed_residual / scale`, when `scale` exceeds a small floor; `None`
    /// when the normalization would be numerically meaningless (all
    /// contributing terms are near zero).
    pub normalized_residual: Option<f64>,
    /// The spatial-curvature contribution `R^(3)`.
    pub spatial_ricci_scalar: f64,
    /// The trace-squared contribution `K^2`.
    pub mean_curvature_squared: f64,
    /// The extrinsic-curvature-norm contribution `K_ij K^{ij}` (subtracted).
    pub extrinsic_curvature_norm: f64,
    /// The matter contribution `-16 pi rho`.
    pub matter_term: f64,
    /// The normalization scale
    /// `|R^(3)| + K^2 + K_ij K^{ij} + |matter_term|` used for
    /// `normalized_residual`.
    pub scale: f64,
}

/// The decomposed momentum constraint residual
/// `D_j ( K^{ij} - gamma^{ij} K ) - 8 pi S^i`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MomentumConstraint {
    /// The signed residual vector `M^i` (also the per-component view).
    pub residual: [f64; 3],
    /// The metric norm `sqrt(gamma_ij M^i M^j)` of the residual.
    pub residual_norm: f64,
    /// The geometric contribution `D_j ( K^{ij} - gamma^{ij} K )`.
    pub geometric_term: [f64; 3],
    /// The matter contribution `-8 pi S^i`.
    pub matter_term: [f64; 3],
}

/// The right-hand side of `partial_t gamma_ij = -2 alpha K_ij + (Lie_beta gamma)_ij`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetricEvolutionRhs {
    /// The total right-hand side.
    pub total: [[f64; 3]; 3],
    /// The `-2 alpha K_ij` contribution.
    pub extrinsic_curvature_term: [[f64; 3]; 3],
    /// The `(Lie_beta gamma)_ij` contribution.
    pub lie_derivative: [[f64; 3]; 3],
}

/// The right-hand side of the extrinsic-curvature evolution equation (see the
/// [module documentation](self) for the full equation and its corrected sign).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurvatureEvolutionRhs {
    /// The total right-hand side:
    /// `total = -lapse_hessian + ricci_term + quadratic_extrinsic_term + lie_derivative + matter_term`.
    pub total: [[f64; 3]; 3],
    /// The raw covariant Hessian `D_i D_j alpha` (not pre-negated).
    pub lapse_hessian: [[f64; 3]; 3],
    /// The `alpha * R_ij` contribution.
    pub ricci_term: [[f64; 3]; 3],
    /// The `alpha * ( K K_ij - 2 K_ik K^k_j )` contribution.
    pub quadratic_extrinsic_term: [[f64; 3]; 3],
    /// The `(Lie_beta K)_ij` contribution.
    pub lie_derivative: [[f64; 3]; 3],
    /// The `-8 pi alpha ( S_ij - 1/2 gamma_ij (S - rho) )` contribution.
    pub matter_term: [[f64; 3]; 3],
}

fn validate_request(
    coordinates: &[f64; 3],
    settings: &AdmEvolutionSettings,
) -> Result<(), AdmEvolutionError> {
    if let Some((index, _)) = coordinates
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(AdmEvolutionError::NonFiniteCoordinate(index));
    }
    if !settings.spatial_step.is_finite() || settings.spatial_step <= 0.0
    {
        return Err(AdmEvolutionError::InvalidStep(settings.spatial_step));
    }
    if !settings.metric_step.is_finite() || settings.metric_step <= 0.0
    {
        return Err(AdmEvolutionError::InvalidStep(settings.metric_step));
    }
    Ok(())
}

fn require_finite_tensor(
    quantity: &'static str,
    tensor: &[[f64; 3]; 3],
) -> Result<(), AdmEvolutionError> {
    if tensor.iter().flatten().any(|value| !value.is_finite())
    {
        return Err(AdmEvolutionError::NonFiniteResult { quantity });
    }
    Ok(())
}

/// Central first derivatives of a scalar field: `partials[i] = d_i f(x)`.
fn scalar_first_derivatives(
    field: impl Fn(&[f64; 3]) -> f64,
    coordinates: &[f64; 3],
    step: f64,
) -> [f64; 3] {
    let mut partials = [0.0_f64; 3];
    for i in 0..3
    {
        let mut plus = *coordinates;
        let mut minus = *coordinates;
        plus[i] += step;
        minus[i] -= step;
        partials[i] = (field(&plus) - field(&minus)) / (2.0 * step);
    }
    partials
}

/// Central second derivatives (Hessian) of a scalar field:
/// `hessian[i][j] = d_i d_j f(x)`.
fn scalar_second_derivatives(
    field: impl Fn(&[f64; 3]) -> f64,
    coordinates: &[f64; 3],
    step: f64,
) -> [[f64; 3]; 3] {
    let center = field(coordinates);
    let mut hessian = [[0.0_f64; 3]; 3];
    for i in 0..3
    {
        let mut plus = *coordinates;
        let mut minus = *coordinates;
        plus[i] += step;
        minus[i] -= step;
        hessian[i][i] = (field(&plus) - 2.0 * center + field(&minus)) / (step * step);
    }
    for i in 0..3
    {
        for j in (i + 1)..3
        {
            let mut plus_plus = *coordinates;
            let mut plus_minus = *coordinates;
            let mut minus_plus = *coordinates;
            let mut minus_minus = *coordinates;
            plus_plus[i] += step;
            plus_plus[j] += step;
            plus_minus[i] += step;
            plus_minus[j] -= step;
            minus_plus[i] -= step;
            minus_plus[j] += step;
            minus_minus[i] -= step;
            minus_minus[j] -= step;
            let value = (field(&plus_plus) - field(&plus_minus) - field(&minus_plus)
                + field(&minus_minus))
                / (4.0 * step * step);
            hessian[i][j] = value;
            hessian[j][i] = value;
        }
    }
    hessian
}

/// Central first derivatives of a vector field: `partials[i][j] = d_i V^j(x)`.
fn vector_first_derivatives(
    field: impl Fn(&[f64; 3]) -> [f64; 3],
    coordinates: &[f64; 3],
    step: f64,
) -> [[f64; 3]; 3] {
    let mut partials = [[0.0_f64; 3]; 3];
    for i in 0..3
    {
        let mut plus = *coordinates;
        let mut minus = *coordinates;
        plus[i] += step;
        minus[i] -= step;
        let vector_plus = field(&plus);
        let vector_minus = field(&minus);
        for j in 0..3
        {
            partials[i][j] = (vector_plus[j] - vector_minus[j]) / (2.0 * step);
        }
    }
    partials
}

/// Central first derivatives of a rank-2 tensor field:
/// `partials[k][i][j] = d_k T_ij(x)`.
fn tensor_first_derivatives(
    field: impl Fn(&[f64; 3]) -> [[f64; 3]; 3],
    coordinates: &[f64; 3],
    step: f64,
) -> [[[f64; 3]; 3]; 3] {
    let mut partials = [[[0.0_f64; 3]; 3]; 3];
    for k in 0..3
    {
        let mut plus = *coordinates;
        let mut minus = *coordinates;
        plus[k] += step;
        minus[k] -= step;
        let tensor_plus = field(&plus);
        let tensor_minus = field(&minus);
        for i in 0..3
        {
            for j in 0..3
            {
                partials[k][i][j] = (tensor_plus[i][j] - tensor_minus[i][j]) / (2.0 * step);
            }
        }
    }
    partials
}

/// Lie derivative of a symmetric spatial 2-tensor along the shift:
/// `(Lie_beta T)_ij = beta^k d_k T_ij + T_kj d_i beta^k + T_ik d_j beta^k`.
fn lie_derivative_tensor(
    tensor: &[[f64; 3]; 3],
    tensor_partials: &[[[f64; 3]; 3]; 3],
    shift: &[f64; 3],
    shift_partials: &[[f64; 3]; 3],
) -> [[f64; 3]; 3] {
    let mut result = [[0.0_f64; 3]; 3];
    for i in 0..3
    {
        for j in 0..3
        {
            let mut value = 0.0;
            for k in 0..3
            {
                value += shift[k] * tensor_partials[k][i][j];
                value += tensor[k][j] * shift_partials[i][k];
                value += tensor[i][k] * shift_partials[j][k];
            }
            result[i][j] = value;
        }
    }
    result
}

/// The mixed extrinsic curvature `K_i^k = gamma^{kl} K_il`.
fn mixed_extrinsic_curvature(inverse: &[[f64; 3]; 3], extrinsic: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut mixed = [[0.0_f64; 3]; 3];
    for i in 0..3
    {
        for k in 0..3
        {
            let mut value = 0.0;
            for l in 0..3
            {
                value += inverse[k][l] * extrinsic[i][l];
            }
            mixed[i][k] = value;
        }
    }
    mixed
}

/// The momentum tensor `P^{ij} = K^{ij} - gamma^{ij} K` at `coordinates`, from
/// spatial data (not from a 4-metric — compare `crate::adm`'s
/// backward-extraction analogue of the same quantity).
fn momentum_tensor_upper<G: Metric<3>>(
    spatial_metric: &G,
    extrinsic_curvature: &impl SpatialTensorField,
    coordinates: &[f64; 3],
) -> Result<[[f64; 3]; 3], AdmEvolutionError> {
    let metric = spatial_metric.components(coordinates);
    let inverse =
        invert_metric::<3>(&metric).map_err(|_| AdmEvolutionError::SingularSpatialMetric)?;
    let extrinsic = extrinsic_curvature.components(coordinates);
    let trace = mean_curvature(&inverse, &extrinsic);
    let raised = raise(&inverse, &extrinsic);
    let mut momentum = [[0.0_f64; 3]; 3];
    for i in 0..3
    {
        for j in 0..3
        {
            momentum[i][j] = raised[i][j] - inverse[i][j] * trace;
        }
    }
    Ok(momentum)
}

/// Evaluate the Hamiltonian constraint `R^(3) + K^2 - K_ij K^{ij} - 16 pi rho`
/// at `coordinates`, from independently supplied spatial data.
///
/// Returns a typed [`AdmEvolutionError`] for an invalid request, a singular
/// spatial metric, a non-finite field, or a non-finite result; it never panics.
///
/// # Example
///
/// Flat space with zero extrinsic curvature and vacuum sources is Hamiltonian-
/// constraint-satisfying (Minkowski).
///
/// ```
/// use scirust_relativity::Metric;
/// use scirust_relativity::adm_evolution::{
///     AdmEvolutionSettings, AdmSources, hamiltonian_constraint,
/// };
///
/// struct FlatSpace;
/// impl Metric<3> for FlatSpace {
///     fn components(&self, _coordinates: &[f64; 3]) -> [[f64; 3]; 3] {
///         [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
///     }
/// }
///
/// let zero_k = |_: &[f64; 3]| [[0.0; 3]; 3];
/// let settings = AdmEvolutionSettings { spatial_step: 1.0e-3, metric_step: 1.0e-3 };
/// let constraint =
///     hamiltonian_constraint(&FlatSpace, &zero_k, &[1.0, 2.0, 3.0], &AdmSources::VACUUM, &settings)
///         .expect("valid constraint");
/// assert!(constraint.absolute_residual < 1.0e-6);
/// ```
pub fn hamiltonian_constraint<G: Metric<3>>(
    spatial_metric: &G,
    extrinsic_curvature: &impl SpatialTensorField,
    coordinates: &[f64; 3],
    sources: &AdmSources,
    settings: &AdmEvolutionSettings,
) -> Result<HamiltonianConstraint, AdmEvolutionError> {
    validate_request(coordinates, settings)?;

    let metric = spatial_metric.components(coordinates);
    let inverse =
        invert_metric::<3>(&metric).map_err(|_| AdmEvolutionError::SingularSpatialMetric)?;
    let extrinsic = extrinsic_curvature.components(coordinates);
    if extrinsic.iter().flatten().any(|value| !value.is_finite())
    {
        return Err(AdmEvolutionError::NonFiniteField {
            field: "extrinsic_curvature",
        });
    }

    let spatial_ricci_scalar = crate::ricci_scalar_from_metric(
        spatial_metric,
        coordinates,
        settings.spatial_step,
        settings.metric_step,
    )?;

    let trace = mean_curvature(&inverse, &extrinsic);
    let raised = raise(&inverse, &extrinsic);
    let mut extrinsic_curvature_norm = 0.0;
    for i in 0..3
    {
        for j in 0..3
        {
            extrinsic_curvature_norm += extrinsic[i][j] * raised[i][j];
        }
    }
    let mean_curvature_squared = trace * trace;
    let matter_term = -16.0 * PI * sources.energy_density;

    let signed_residual =
        spatial_ricci_scalar + mean_curvature_squared - extrinsic_curvature_norm + matter_term;
    if !signed_residual.is_finite()
    {
        return Err(AdmEvolutionError::NonFiniteResult {
            quantity: "hamiltonian_residual",
        });
    }

    let scale = spatial_ricci_scalar.abs()
        + mean_curvature_squared
        + extrinsic_curvature_norm.abs()
        + matter_term.abs();
    let normalized_residual = (scale > 1.0e-12).then_some(signed_residual / scale);

    Ok(HamiltonianConstraint {
        signed_residual,
        absolute_residual: signed_residual.abs(),
        normalized_residual,
        spatial_ricci_scalar,
        mean_curvature_squared,
        extrinsic_curvature_norm,
        matter_term,
        scale,
    })
}

/// Evaluate the momentum constraint `D_j ( K^{ij} - gamma^{ij} K ) - 8 pi S^i`
/// at `coordinates`, from independently supplied spatial data.
///
/// Returns a typed [`AdmEvolutionError`] for an invalid request, a singular
/// spatial metric, or a non-finite result; it never panics.
// Explicit tensor-index loops read most clearly here (matching the crate's
// curvature, action, and adm modules).
#[allow(clippy::needless_range_loop)]
pub fn momentum_constraint<G: Metric<3>>(
    spatial_metric: &G,
    extrinsic_curvature: &impl SpatialTensorField,
    coordinates: &[f64; 3],
    sources: &AdmSources,
    settings: &AdmEvolutionSettings,
) -> Result<MomentumConstraint, AdmEvolutionError> {
    validate_request(coordinates, settings)?;

    let metric = spatial_metric.components(coordinates);
    let inverse =
        invert_metric::<3>(&metric).map_err(|_| AdmEvolutionError::SingularSpatialMetric)?;
    let christoffel =
        numerical_christoffel::<_, 3>(spatial_metric, coordinates, settings.metric_step)?;
    let center = momentum_tensor_upper(spatial_metric, extrinsic_curvature, coordinates)?;

    let mut geometric_term = [0.0_f64; 3];
    for i in 0..3
    {
        let mut divergence = 0.0;
        for d in 0..3
        {
            let mut plus = *coordinates;
            let mut minus = *coordinates;
            plus[d] += settings.spatial_step;
            minus[d] -= settings.spatial_step;
            let plus_tensor = momentum_tensor_upper(spatial_metric, extrinsic_curvature, &plus)?;
            let minus_tensor = momentum_tensor_upper(spatial_metric, extrinsic_curvature, &minus)?;
            divergence += (plus_tensor[i][d] - minus_tensor[i][d]) / (2.0 * settings.spatial_step);
        }
        let mut connection = 0.0;
        for j in 0..3
        {
            for k in 0..3
            {
                connection +=
                    christoffel[i][j][k] * center[k][j] + christoffel[j][j][k] * center[i][k];
            }
        }
        geometric_term[i] = divergence + connection;
    }

    let mut matter_term = [0.0_f64; 3];
    for i in 0..3
    {
        let mut raised_momentum_density = 0.0;
        for j in 0..3
        {
            raised_momentum_density += inverse[i][j] * sources.momentum_density[j];
        }
        matter_term[i] = -8.0 * PI * raised_momentum_density;
    }

    let mut residual = [0.0_f64; 3];
    for i in 0..3
    {
        residual[i] = geometric_term[i] + matter_term[i];
        if !residual[i].is_finite()
        {
            return Err(AdmEvolutionError::NonFiniteResult {
                quantity: "momentum_residual",
            });
        }
    }

    let mut residual_norm_squared = 0.0;
    for i in 0..3
    {
        for j in 0..3
        {
            residual_norm_squared += metric[i][j] * residual[i] * residual[j];
        }
    }

    Ok(MomentumConstraint {
        residual,
        residual_norm: residual_norm_squared.max(0.0).sqrt(),
        geometric_term,
        matter_term,
    })
}

/// Evaluate the right-hand side of
/// `partial_t gamma_ij = -2 alpha K_ij + (Lie_beta gamma)_ij` at `coordinates`.
///
/// Returns a typed [`AdmEvolutionError`] for an invalid request or a non-finite
/// field/result; it never panics.
pub fn metric_evolution_rhs<G: Metric<3>>(
    spatial_metric: &G,
    extrinsic_curvature: &impl SpatialTensorField,
    lapse: &impl SpatialScalarField,
    shift: &impl SpatialVectorField,
    coordinates: &[f64; 3],
    settings: &AdmEvolutionSettings,
) -> Result<MetricEvolutionRhs, AdmEvolutionError> {
    validate_request(coordinates, settings)?;

    let alpha = lapse.value(coordinates);
    if !alpha.is_finite()
    {
        return Err(AdmEvolutionError::NonFiniteField { field: "lapse" });
    }
    let extrinsic = extrinsic_curvature.components(coordinates);
    if extrinsic.iter().flatten().any(|value| !value.is_finite())
    {
        return Err(AdmEvolutionError::NonFiniteField {
            field: "extrinsic_curvature",
        });
    }
    let beta = shift.components(coordinates);
    if beta.iter().any(|value| !value.is_finite())
    {
        return Err(AdmEvolutionError::NonFiniteField { field: "shift" });
    }

    let gamma = spatial_metric.components(coordinates);
    let gamma_partials = tensor_first_derivatives(
        |x| spatial_metric.components(x),
        coordinates,
        settings.spatial_step,
    );
    let beta_partials =
        vector_first_derivatives(|x| shift.components(x), coordinates, settings.spatial_step);
    let lie_derivative = lie_derivative_tensor(&gamma, &gamma_partials, &beta, &beta_partials);

    let mut extrinsic_curvature_term = [[0.0_f64; 3]; 3];
    let mut total = [[0.0_f64; 3]; 3];
    for i in 0..3
    {
        for j in 0..3
        {
            extrinsic_curvature_term[i][j] = -2.0 * alpha * extrinsic[i][j];
            total[i][j] = extrinsic_curvature_term[i][j] + lie_derivative[i][j];
        }
    }
    require_finite_tensor("metric_evolution_rhs", &total)?;

    Ok(MetricEvolutionRhs {
        total,
        extrinsic_curvature_term,
        lie_derivative,
    })
}

/// Evaluate the right-hand side of the extrinsic-curvature evolution equation
/// (see the [module documentation](self)) at `coordinates`.
///
/// Returns a typed [`AdmEvolutionError`] for an invalid request, a singular
/// spatial metric, a non-finite field, or a non-finite result; it never panics.
pub fn curvature_evolution_rhs<G: Metric<3>>(
    spatial_metric: &G,
    extrinsic_curvature: &impl SpatialTensorField,
    lapse: &impl SpatialScalarField,
    shift: &impl SpatialVectorField,
    coordinates: &[f64; 3],
    sources: &AdmSources,
    settings: &AdmEvolutionSettings,
) -> Result<CurvatureEvolutionRhs, AdmEvolutionError> {
    validate_request(coordinates, settings)?;

    let alpha = lapse.value(coordinates);
    if !alpha.is_finite()
    {
        return Err(AdmEvolutionError::NonFiniteField { field: "lapse" });
    }
    let gamma = spatial_metric.components(coordinates);
    let inverse =
        invert_metric::<3>(&gamma).map_err(|_| AdmEvolutionError::SingularSpatialMetric)?;
    let extrinsic = extrinsic_curvature.components(coordinates);
    if extrinsic.iter().flatten().any(|value| !value.is_finite())
    {
        return Err(AdmEvolutionError::NonFiniteField {
            field: "extrinsic_curvature",
        });
    }
    let beta = shift.components(coordinates);
    if beta.iter().any(|value| !value.is_finite())
    {
        return Err(AdmEvolutionError::NonFiniteField { field: "shift" });
    }

    // Lapse Hessian: D_i D_j alpha = d_i d_j alpha - Gamma^{(3)k}_ij d_k alpha.
    let alpha_first =
        scalar_first_derivatives(|x| lapse.value(x), coordinates, settings.spatial_step);
    let alpha_second =
        scalar_second_derivatives(|x| lapse.value(x), coordinates, settings.spatial_step);
    let christoffel =
        numerical_christoffel::<_, 3>(spatial_metric, coordinates, settings.metric_step)?;
    let mut lapse_hessian = [[0.0_f64; 3]; 3];
    for i in 0..3
    {
        for j in 0..3
        {
            let mut connection = 0.0;
            for k in 0..3
            {
                connection += christoffel[k][i][j] * alpha_first[k];
            }
            lapse_hessian[i][j] = alpha_second[i][j] - connection;
        }
    }

    // Spatial Ricci tensor.
    let ricci = ricci_tensor_from_metric::<_, 3>(
        spatial_metric,
        coordinates,
        settings.spatial_step,
        settings.metric_step,
    )?;
    let mut ricci_term = [[0.0_f64; 3]; 3];
    for i in 0..3
    {
        for j in 0..3
        {
            ricci_term[i][j] = alpha * ricci[i][j];
        }
    }

    // Quadratic extrinsic-curvature term: alpha * ( K K_ij - 2 K_ik K^k_j ).
    let trace = mean_curvature(&inverse, &extrinsic);
    let mixed = mixed_extrinsic_curvature(&inverse, &extrinsic);
    let mut quadratic_extrinsic_term = [[0.0_f64; 3]; 3];
    for i in 0..3
    {
        for j in 0..3
        {
            let mut k_ik_k_kj = 0.0;
            for k in 0..3
            {
                k_ik_k_kj += mixed[i][k] * extrinsic[k][j];
            }
            quadratic_extrinsic_term[i][j] = alpha * (trace * extrinsic[i][j] - 2.0 * k_ik_k_kj);
        }
    }

    // Lie derivative of K along the shift.
    let k_partials = tensor_first_derivatives(
        |x| extrinsic_curvature.components(x),
        coordinates,
        settings.spatial_step,
    );
    let beta_partials =
        vector_first_derivatives(|x| shift.components(x), coordinates, settings.spatial_step);
    let lie_derivative = lie_derivative_tensor(&extrinsic, &k_partials, &beta, &beta_partials);

    // Matter term: -8 pi alpha ( S_ij - 1/2 gamma_ij (S - rho) ), corrected sign
    // (see the module documentation and docs/LAYER_3_ADM_EVOLUTION.md §4).
    let trace_stress = sources.trace(&inverse);
    let mut matter_term = [[0.0_f64; 3]; 3];
    for i in 0..3
    {
        for j in 0..3
        {
            matter_term[i][j] = -8.0
                * PI
                * alpha
                * (sources.stress[i][j]
                    - 0.5 * gamma[i][j] * (trace_stress - sources.energy_density));
        }
    }

    let mut total = [[0.0_f64; 3]; 3];
    for i in 0..3
    {
        for j in 0..3
        {
            total[i][j] = -lapse_hessian[i][j]
                + ricci_term[i][j]
                + quadratic_extrinsic_term[i][j]
                + lie_derivative[i][j]
                + matter_term[i][j];
        }
    }
    require_finite_tensor("curvature_evolution_rhs", &total)?;

    Ok(CurvatureEvolutionRhs {
        total,
        lapse_hessian,
        ricci_term,
        quadratic_extrinsic_term,
        lie_derivative,
        matter_term,
    })
}
