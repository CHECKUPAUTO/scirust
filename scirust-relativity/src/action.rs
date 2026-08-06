//! The Einstein-Hilbert action and its **numerical** variation.
//!
//! For a static, axisymmetric background this module computes the directional
//! functional derivative of the gravitational action
//! `S[g] = integral (R - 2 Lambda) sqrt(-g) d^4x` against a compact test
//! perturbation, by a finite difference in the perturbation amplitude, and
//! compares it to the analytic-integrand prediction
//! `-integral sqrt(-g) E^{ab} h_{ab}` with `E_{mu nu} = G_{mu nu} + Lambda g_{mu nu}`.
//! A solution of `G_{mu nu} + Lambda g_{mu nu} = 0` makes the action stationary,
//! so the variation vanishes; this is the established-GR statement the module
//! validates. See `docs/LAYER_2_ACTION_VARIATION.md` for the full design.
//!
//! **Category.** The physics (the Einstein-Hilbert variation yields the vacuum
//! field equations) is established general relativity. The *numbers* are a
//! numerical approximation: a metric-only nested-difference Ricci scalar
//! ([`ricci_scalar_from_metric`]), a Simpson quadrature of the action, and a
//! central difference in the amplitude. Truncation error is reported, never
//! hidden, and this is never presented as a closed-form (exact) variation.
//!
//! **Scope.** Static, axisymmetric backgrounds only: the perturbation is compact
//! in `(r, theta)` and constant in `(t, phi)`, so the ignorable-coordinate
//! integrals factor out and their boundary fluxes telescope, reducing the 4D
//! variation to a 2D `(r, theta)` integral. Non-stationary or non-axisymmetric
//! backgrounds, matter sources, modified-gravity actions, and closed-form
//! variation are out of scope (see the design note).

use std::f64::consts::PI;
use std::fmt;

use crate::{
    Connection, CurvatureTensors, Metric, RelativityError, determinant, invert_metric,
    ricci_scalar_from_metric,
};

/// The minimum (and required-odd) grid resolution per axis for Simpson quadrature.
const MIN_GRID: usize = 5;

/// A typed failure of the action-variation extractor. It never panics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActionError {
    /// The grid resolution is even or below [`MIN_GRID`] (Simpson needs an odd
    /// node count `>= 5`).
    InvalidGridResolution(usize),
    /// The radial range is not `0 < lower < upper` with finite bounds.
    InvalidRadialRange {
        /// Lower radial bound.
        lower: f64,
        /// Upper radial bound.
        upper: f64,
    },
    /// The polar range is not `0 < lower < upper < pi` with finite bounds.
    InvalidPolarRange {
        /// Lower polar bound.
        lower: f64,
        /// Upper polar bound.
        upper: f64,
    },
    /// A perturbation half-width is not finite and strictly positive.
    InvalidPerturbationWidth(f64),
    /// The perturbation center is not finite.
    InvalidPerturbationCenter {
        /// Radial center.
        radius: f64,
        /// Polar center.
        polar: f64,
    },
    /// The compact support `[center +/- half_width]` is not contained in the
    /// integration range, so the bump would not vanish at the boundary.
    SupportOutsideDomain,
    /// A perturbation tensor index is out of range (`>= 4`).
    InvalidComponent {
        /// Row index.
        row: usize,
        /// Column index.
        col: usize,
    },
    /// The amplitude is not finite and strictly positive.
    InvalidAmplitude(f64),
    /// A finite-difference step is not finite and strictly positive.
    InvalidStep(f64),
    /// The cosmological constant in the action is not finite.
    InvalidCosmologicalConstant(f64),
    /// A curvature or metric evaluation failed.
    Curvature(RelativityError),
    /// A non-finite value reached the result.
    NonFiniteResult,
}

impl fmt::Display for ActionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidGridResolution(grid) => write!(
                f,
                "grid resolution {grid} must be odd and at least {MIN_GRID}"
            ),
            Self::InvalidRadialRange { lower, upper } =>
            {
                write!(f, "invalid radial range: 0 < {lower} < {upper} required")
            },
            Self::InvalidPolarRange { lower, upper } =>
            {
                write!(
                    f,
                    "invalid polar range: 0 < {lower} < {upper} < pi required"
                )
            },
            Self::InvalidPerturbationWidth(width) =>
            {
                write!(
                    f,
                    "perturbation half-width {width} must be finite and positive"
                )
            },
            Self::InvalidPerturbationCenter { radius, polar } =>
            {
                write!(f, "perturbation center ({radius}, {polar}) must be finite")
            },
            Self::SupportOutsideDomain =>
            {
                write!(
                    f,
                    "perturbation support is not contained in the integration range"
                )
            },
            Self::InvalidComponent { row, col } =>
            {
                write!(f, "perturbation component ({row}, {col}) is out of range")
            },
            Self::InvalidAmplitude(amplitude) =>
            {
                write!(f, "amplitude {amplitude} must be finite and positive")
            },
            Self::InvalidStep(step) =>
            {
                write!(
                    f,
                    "finite-difference step {step} must be finite and positive"
                )
            },
            Self::InvalidCosmologicalConstant(value) =>
            {
                write!(f, "cosmological constant {value} must be finite")
            },
            Self::Curvature(error) => write!(f, "curvature evaluation failed: {error}"),
            Self::NonFiniteResult => write!(f, "a non-finite value reached the result"),
        }
    }
}

impl std::error::Error for ActionError {}

impl From<RelativityError> for ActionError {
    fn from(error: RelativityError) -> Self {
        Self::Curvature(error)
    }
}

/// A compact test perturbation `h_{ab}(r, theta) = phi(r, theta) B_{ab}` where
/// `B_{ab}` is the symmetric indicator of one component and `phi` is a compact
/// polynomial bump (a product of `(1 - u^2)^4` factors, zero with its first
/// three derivatives at the support boundary).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActionPerturbation {
    /// The perturbed covariant metric component `(row, col)` (symmetrized).
    pub component: (usize, usize),
    /// The bump center `(radius, polar)`.
    pub center: (f64, f64),
    /// The bump half-widths `(radial, polar)`; the support is `center +/- these`.
    pub half_widths: (f64, f64),
}

/// The `(r, theta)` integration domain and grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActionDomain {
    /// The radial integration range `(lower, upper)`.
    pub radial_range: (f64, f64),
    /// The polar integration range `(lower, upper)`.
    pub polar_range: (f64, f64),
    /// The Simpson grid resolution per axis (odd, `>= 5`).
    pub grid: usize,
}

/// The numerical settings of the variation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActionSettings {
    /// The perturbation amplitude `eps` (the central-difference step in `eps`).
    pub amplitude: f64,
    /// The outer finite-difference step (Christoffel derivatives).
    pub connection_step: f64,
    /// The inner finite-difference step (Christoffel from the metric).
    pub metric_step: f64,
    /// The cosmological constant `Lambda` in the action.
    pub cosmological_constant: f64,
}

/// The result of a numerical action variation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActionVariation {
    /// The numeric directional derivative `(S[g + eps h] - S[g - eps h]) / (2 eps)`.
    pub numeric: f64,
    /// The analytic-integrand prediction `-integral sqrt(-g) E^{ab} h_{ab}`.
    pub predicted: f64,
    /// The absolute residual `|numeric - predicted|`.
    pub residual: f64,
}

/// A background with one covariant metric component perturbed by a compact bump.
struct BumpedMetric<'a, B> {
    base: &'a B,
    perturbation: &'a ActionPerturbation,
    amplitude: f64,
}

impl<B: Metric<4>> Metric<4> for BumpedMetric<'_, B> {
    fn components(&self, coordinates: &[f64; 4]) -> [[f64; 4]; 4] {
        let mut g = self.base.components(coordinates);
        let delta =
            self.amplitude * bump_profile(coordinates[1], coordinates[2], self.perturbation);
        let (row, col) = self.perturbation.component;
        g[row][col] += delta;
        if row != col
        {
            g[col][row] += delta;
        }
        g
    }
}

/// The compact bump profile `phi(r, theta)`: a product of `(1 - u^2)^4` factors,
/// zero outside the support.
fn bump_profile(radius: f64, polar: f64, perturbation: &ActionPerturbation) -> f64 {
    let (radius_center, polar_center) = perturbation.center;
    let (radial_width, polar_width) = perturbation.half_widths;
    let mut product = 1.0;
    for (value, center, width) in [
        (radius, radius_center, radial_width),
        (polar, polar_center, polar_width),
    ]
    {
        let u = (value - center) / width;
        if u * u >= 1.0
        {
            return 0.0;
        }
        let q = 1.0 - u * u;
        product *= q * q * q * q;
    }
    product
}

/// Composite Simpson weights (length `n`, `n` odd) over `[lo, hi]`, including the
/// `h/3` factor.
fn simpson_weights(n: usize, lo: f64, hi: f64) -> Vec<f64> {
    let step = (hi - lo) / (n as f64 - 1.0);
    (0..n)
        .map(|index| {
            let coefficient = if index == 0 || index == n - 1
            {
                1.0
            }
            else if index % 2 == 1
            {
                4.0
            }
            else
            {
                2.0
            };
            coefficient * step / 3.0
        })
        .collect()
}

/// Uniform nodes over `[lo, hi]` with `n` points.
fn nodes(n: usize, lo: f64, hi: f64) -> Vec<f64> {
    (0..n)
        .map(|index| lo + (hi - lo) * index as f64 / (n as f64 - 1.0))
        .collect()
}

/// `sqrt(-det g)`, erroring on a non-Lorentzian or non-finite determinant.
fn sqrt_minus_g(metric: &[[f64; 4]; 4]) -> Result<f64, ActionError> {
    let det = determinant(metric)?;
    let value = (-det).sqrt();
    if !value.is_finite()
    {
        return Err(ActionError::NonFiniteResult);
    }
    Ok(value)
}

/// The 2D-reduced discretized action `sum (R - 2 Lambda) sqrt(-g) w`, metric-only `R`.
#[allow(clippy::too_many_arguments)]
fn discretized_action<M: Metric<4>>(
    metric: &M,
    radial_nodes: &[f64],
    polar_nodes: &[f64],
    radial_weights: &[f64],
    polar_weights: &[f64],
    cosmological_constant: f64,
    connection_step: f64,
    metric_step: f64,
) -> Result<f64, ActionError> {
    let mut sum = 0.0;
    for (i, &radius) in radial_nodes.iter().enumerate()
    {
        for (j, &polar) in polar_nodes.iter().enumerate()
        {
            let point = [0.0, radius, polar, 0.0];
            let scalar = ricci_scalar_from_metric(metric, &point, connection_step, metric_step)?;
            let density = sqrt_minus_g(&metric.components(&point))?;
            sum += (scalar - 2.0 * cosmological_constant)
                * density
                * radial_weights[i]
                * polar_weights[j];
        }
    }
    Ok(sum)
}

/// Raise both indices of a symmetric covariant tensor with the inverse metric.
fn raise_symmetric(inverse: &[[f64; 4]; 4], tensor: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut raised = [[0.0_f64; 4]; 4];
    for p in 0..4
    {
        for q in 0..4
        {
            let mut value = 0.0;
            for m in 0..4
            {
                for n in 0..4
                {
                    value += inverse[p][m] * inverse[q][n] * tensor[m][n];
                }
            }
            raised[p][q] = value;
        }
    }
    raised
}

/// The 2D-reduced analytic-integrand prediction `-sum sqrt(-g) E^{ab} h_{ab} w`.
#[allow(clippy::too_many_arguments)]
fn discretized_prediction<B: Metric<4> + Connection<4>>(
    background: &B,
    perturbation: &ActionPerturbation,
    radial_nodes: &[f64],
    polar_nodes: &[f64],
    radial_weights: &[f64],
    polar_weights: &[f64],
    cosmological_constant: f64,
    connection_step: f64,
) -> Result<f64, ActionError> {
    let (row, col) = perturbation.component;
    let mut sum = 0.0;
    for (i, &radius) in radial_nodes.iter().enumerate()
    {
        for (j, &polar) in polar_nodes.iter().enumerate()
        {
            let point = [0.0, radius, polar, 0.0];
            let metric = background.components(&point);
            let inverse = invert_metric(&metric)?;
            let curvature = CurvatureTensors::compute(background, &point, connection_step)?;
            let einstein = curvature.einstein();

            // E_{mu nu} = G_{mu nu} + Lambda g_{mu nu}, then raised.
            let mut euler_lagrange = [[0.0_f64; 4]; 4];
            for m in 0..4
            {
                for n in 0..4
                {
                    euler_lagrange[m][n] = einstein[m][n] + cosmological_constant * metric[m][n];
                }
            }
            let raised = raise_symmetric(&inverse, &euler_lagrange);

            // E^{ab} h_{ab}: the symmetric indicator picks (row, col) (and its
            // transpose for an off-diagonal component).
            let contraction = if row == col
            {
                raised[row][col]
            }
            else
            {
                raised[row][col] + raised[col][row]
            };
            let profile = bump_profile(radius, polar, perturbation);
            let density = sqrt_minus_g(&metric)?;
            sum += -contraction * profile * density * radial_weights[i] * polar_weights[j];
        }
    }
    Ok(sum)
}

/// Numerically vary the Einstein-Hilbert action of a static, axisymmetric
/// `background` against the compact test `perturbation`, over the `(r, theta)`
/// `domain`, with the given `settings`.
///
/// Returns the numeric directional derivative, the analytic-integrand
/// prediction, and their residual. For a vacuum-with-`Lambda` solution the
/// prediction is zero, so the residual measures how well the numerical variation
/// reproduces `G_{mu nu} + Lambda g_{mu nu} = 0`.
///
/// Returns a typed [`ActionError`] for any invalid request or non-finite
/// evaluation; it never panics.
///
/// # Example
///
/// De Sitter, with the action's `Lambda` matched to the background, is a
/// stationary point: the numerical variation is near zero.
///
/// ```
/// use scirust_relativity::DeSitter;
/// use scirust_relativity::action::{
///     ActionDomain, ActionPerturbation, ActionSettings, einstein_hilbert_action_variation,
/// };
/// use std::f64::consts::FRAC_PI_2;
///
/// let lambda = 0.03;
/// let spacetime = DeSitter::try_new(lambda).expect("valid cosmological constant");
/// let perturbation = ActionPerturbation {
///     component: (1, 1),
///     center: (3.0, FRAC_PI_2),
///     half_widths: (1.0, 1.0),
/// };
/// let domain = ActionDomain {
///     radial_range: (2.0, 4.0),
///     polar_range: (FRAC_PI_2 - 1.0, FRAC_PI_2 + 1.0),
///     grid: 41,
/// };
/// let settings = ActionSettings {
///     amplitude: 1.0e-3,
///     connection_step: 1.0e-3,
///     metric_step: 1.0e-3,
///     cosmological_constant: lambda,
/// };
/// let variation =
///     einstein_hilbert_action_variation(&spacetime, &perturbation, &domain, &settings)
///         .expect("valid variation");
/// assert!(variation.residual < 1.0e-3);
/// ```
pub fn einstein_hilbert_action_variation<B>(
    background: &B,
    perturbation: &ActionPerturbation,
    domain: &ActionDomain,
    settings: &ActionSettings,
) -> Result<ActionVariation, ActionError>
where
    B: Metric<4> + Connection<4>,
{
    let grid = domain.grid;
    if grid < MIN_GRID || grid.is_multiple_of(2)
    {
        return Err(ActionError::InvalidGridResolution(grid));
    }

    let (radial_lower, radial_upper) = domain.radial_range;
    if !radial_lower.is_finite()
        || !radial_upper.is_finite()
        || radial_lower <= 0.0
        || radial_lower >= radial_upper
    {
        return Err(ActionError::InvalidRadialRange {
            lower: radial_lower,
            upper: radial_upper,
        });
    }

    let (polar_lower, polar_upper) = domain.polar_range;
    if !polar_lower.is_finite()
        || !polar_upper.is_finite()
        || polar_lower <= 0.0
        || polar_lower >= polar_upper
        || polar_upper >= PI
    {
        return Err(ActionError::InvalidPolarRange {
            lower: polar_lower,
            upper: polar_upper,
        });
    }

    let (row, col) = perturbation.component;
    if row >= 4 || col >= 4
    {
        return Err(ActionError::InvalidComponent { row, col });
    }

    let (radial_width, polar_width) = perturbation.half_widths;
    if !radial_width.is_finite() || radial_width <= 0.0
    {
        return Err(ActionError::InvalidPerturbationWidth(radial_width));
    }
    if !polar_width.is_finite() || polar_width <= 0.0
    {
        return Err(ActionError::InvalidPerturbationWidth(polar_width));
    }

    let (radius_center, polar_center) = perturbation.center;
    if !radius_center.is_finite() || !polar_center.is_finite()
    {
        return Err(ActionError::InvalidPerturbationCenter {
            radius: radius_center,
            polar: polar_center,
        });
    }
    if radius_center - radial_width < radial_lower
        || radius_center + radial_width > radial_upper
        || polar_center - polar_width < polar_lower
        || polar_center + polar_width > polar_upper
    {
        return Err(ActionError::SupportOutsideDomain);
    }

    if !settings.amplitude.is_finite() || settings.amplitude <= 0.0
    {
        return Err(ActionError::InvalidAmplitude(settings.amplitude));
    }
    if !settings.connection_step.is_finite() || settings.connection_step <= 0.0
    {
        return Err(ActionError::InvalidStep(settings.connection_step));
    }
    if !settings.metric_step.is_finite() || settings.metric_step <= 0.0
    {
        return Err(ActionError::InvalidStep(settings.metric_step));
    }
    if !settings.cosmological_constant.is_finite()
    {
        return Err(ActionError::InvalidCosmologicalConstant(
            settings.cosmological_constant,
        ));
    }

    let radial_nodes = nodes(grid, radial_lower, radial_upper);
    let polar_nodes = nodes(grid, polar_lower, polar_upper);
    let radial_weights = simpson_weights(grid, radial_lower, radial_upper);
    let polar_weights = simpson_weights(grid, polar_lower, polar_upper);

    let plus = BumpedMetric {
        base: background,
        perturbation,
        amplitude: settings.amplitude,
    };
    let minus = BumpedMetric {
        base: background,
        perturbation,
        amplitude: -settings.amplitude,
    };

    let action_plus = discretized_action(
        &plus,
        &radial_nodes,
        &polar_nodes,
        &radial_weights,
        &polar_weights,
        settings.cosmological_constant,
        settings.connection_step,
        settings.metric_step,
    )?;
    let action_minus = discretized_action(
        &minus,
        &radial_nodes,
        &polar_nodes,
        &radial_weights,
        &polar_weights,
        settings.cosmological_constant,
        settings.connection_step,
        settings.metric_step,
    )?;
    let numeric = (action_plus - action_minus) / (2.0 * settings.amplitude);

    let predicted = discretized_prediction(
        background,
        perturbation,
        &radial_nodes,
        &polar_nodes,
        &radial_weights,
        &polar_weights,
        settings.cosmological_constant,
        settings.connection_step,
    )?;

    if !numeric.is_finite() || !predicted.is_finite()
    {
        return Err(ActionError::NonFiniteResult);
    }

    Ok(ActionVariation {
        numeric,
        predicted,
        residual: (numeric - predicted).abs(),
    })
}
