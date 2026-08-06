//! 3+1 (ADM) kinematics: the lapse, shift, spatial metric, and extrinsic
//! curvature of a foliation, with the Hamiltonian and momentum constraints.
//!
//! On the foliation of a 4-metric by constant time-coordinate slices, this
//! module extracts the ADM variables and evaluates the Gauss-Codazzi
//! constraints, which vanish for a vacuum-with-`Lambda` solution. It is the
//! kinematic bridge to Layer 3 (numerical relativity); it does **not** evolve
//! the data in time. See `docs/LAYER_2_ADM.md`.
//!
//! **Category.** The ADM decomposition and the Gauss-Codazzi constraints are
//! established general relativity. The numbers are a numerical approximation:
//! `R^(3)` reuses the metric-only nested-difference [`ricci_scalar_from_metric`]
//! at `D = 3`, the spatial connection reuses [`numerical_christoffel`] at
//! `D = 3`, `partial_0 gamma` is a time difference, and the momentum constraint
//! is a spatial difference of that. Truncation error is reported, never hidden.
//!
//! **Scope.** The slicing is by the chart's time coordinate; the constraints are
//! validated in vacuum-with-`Lambda`. Time evolution, matter sources, gauge
//! conditions, and constraint damping are out of scope.

use std::fmt;

use crate::{
    Metric, RelativityError, invert_metric, numerical_christoffel, ricci_scalar_from_metric,
};

/// A typed failure of the ADM decomposition. It never panics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdmError {
    /// A coordinate is not finite (carries its index).
    NonFiniteCoordinate(usize),
    /// A finite-difference step is not finite and strictly positive.
    InvalidStep(f64),
    /// The cosmological constant is not finite.
    InvalidCosmologicalConstant(f64),
    /// The spatial 3-metric is singular (cannot be inverted).
    SingularSpatialMetric,
    /// The lapse is imaginary (`N_i N^i - g_00 <= 0`): the slice is not spacelike.
    NonSpacelikeSlice,
    /// A curvature or connection evaluation failed.
    Curvature(RelativityError),
    /// A non-finite value reached the result.
    NonFiniteResult,
}

impl fmt::Display for AdmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::NonFiniteCoordinate(index) =>
            {
                write!(f, "coordinate {index} is not finite")
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
            Self::SingularSpatialMetric => write!(f, "the spatial 3-metric is singular"),
            Self::NonSpacelikeSlice =>
            {
                write!(f, "the slice is not spacelike (imaginary lapse)")
            },
            Self::Curvature(error) => write!(f, "curvature evaluation failed: {error}"),
            Self::NonFiniteResult => write!(f, "a non-finite value reached the result"),
        }
    }
}

impl std::error::Error for AdmError {}

impl From<RelativityError> for AdmError {
    fn from(error: RelativityError) -> Self {
        Self::Curvature(error)
    }
}

/// The finite-difference settings of the decomposition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdmSettings {
    /// The time step for `partial_0 gamma`.
    pub time_step: f64,
    /// The spatial step for the connection, `R^(3)`, and the momentum divergence.
    pub spatial_step: f64,
}

/// The ADM decomposition at one point of a foliation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdmDecomposition {
    /// The lapse `N`.
    pub lapse: f64,
    /// The shift vector `N^i`.
    pub shift: [f64; 3],
    /// The spatial 3-metric `gamma_ij`.
    pub spatial_metric: [[f64; 3]; 3],
    /// The extrinsic curvature `K_ij`.
    pub extrinsic_curvature: [[f64; 3]; 3],
    /// The spatial Ricci scalar `R^(3)`.
    pub spatial_ricci_scalar: f64,
    /// The mean curvature `K = gamma^{ij} K_ij`.
    pub mean_curvature: f64,
    /// The squared extrinsic curvature `K_ij K^{ij}`.
    pub extrinsic_curvature_norm: f64,
}

/// The Gauss-Codazzi constraint residuals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdmConstraints {
    /// The Hamiltonian constraint `R^(3) + K^2 - K_ij K^{ij} - 2 Lambda`.
    pub hamiltonian: f64,
    /// The momentum constraint `D_j (K^{ij} - gamma^{ij} K)`.
    pub momentum: [f64; 3],
}

/// The spatial slice at fixed time as a [`Metric<3>`] (the spatial 3-block).
struct SpatialSlice<'a, B> {
    background: &'a B,
    time: f64,
}

impl<B: Metric<4>> Metric<3> for SpatialSlice<'_, B> {
    fn components(&self, coordinates: &[f64; 3]) -> [[f64; 3]; 3] {
        let full = self.background.components(&[
            self.time,
            coordinates[0],
            coordinates[1],
            coordinates[2],
        ]);
        let mut spatial = [[0.0_f64; 3]; 3];
        for i in 0..3
        {
            for j in 0..3
            {
                spatial[i][j] = full[i + 1][j + 1];
            }
        }
        spatial
    }
}

fn validate<const N: usize>(
    coordinates: &[f64; N],
    settings: &AdmSettings,
) -> Result<(), AdmError> {
    if let Some((index, _)) = coordinates
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(AdmError::NonFiniteCoordinate(index));
    }
    if !settings.time_step.is_finite() || settings.time_step <= 0.0
    {
        return Err(AdmError::InvalidStep(settings.time_step));
    }
    if !settings.spatial_step.is_finite() || settings.spatial_step <= 0.0
    {
        return Err(AdmError::InvalidStep(settings.spatial_step));
    }
    Ok(())
}

/// The spatial 3-metric block of the 4-metric at `coordinates`.
fn spatial_metric_at<B: Metric<4>>(background: &B, coordinates: &[f64; 4]) -> [[f64; 3]; 3] {
    let full = background.components(coordinates);
    let mut spatial = [[0.0_f64; 3]; 3];
    for i in 0..3
    {
        for j in 0..3
        {
            spatial[i][j] = full[i + 1][j + 1];
        }
    }
    spatial
}

/// The covariant shift `N_i = g_{0i}` at `coordinates`.
fn shift_covector_at<B: Metric<4>>(background: &B, coordinates: &[f64; 4]) -> [f64; 3] {
    let full = background.components(coordinates);
    [full[0][1], full[0][2], full[0][3]]
}

/// The local 3+1 frame at a point.
struct Frame {
    lapse: f64,
    shift: [f64; 3],
    spatial: [[f64; 3]; 3],
    inverse: [[f64; 3]; 3],
}

/// The extrinsic curvature together with the frame it was built from.
struct ExtrinsicCurvature {
    curvature: [[f64; 3]; 3],
    lapse: f64,
    shift: [f64; 3],
    spatial: [[f64; 3]; 3],
    inverse: [[f64; 3]; 3],
}

/// The local frame: lapse `N`, shift `N^i`, spatial metric `gamma_ij`, and its
/// inverse `gamma^{ij}`.
fn frame_at<B: Metric<4>>(background: &B, coordinates: &[f64; 4]) -> Result<Frame, AdmError> {
    let full = background.components(coordinates);
    let spatial = spatial_metric_at(background, coordinates);
    let inverse = invert_metric::<3>(&spatial).map_err(|_| AdmError::SingularSpatialMetric)?;
    let lower = shift_covector_at(background, coordinates);

    let mut shift = [0.0_f64; 3];
    for i in 0..3
    {
        let mut value = 0.0;
        for j in 0..3
        {
            value += inverse[i][j] * lower[j];
        }
        shift[i] = value;
    }

    let mut shift_norm = 0.0;
    for i in 0..3
    {
        shift_norm += shift[i] * lower[i];
    }
    let lapse_squared = shift_norm - full[0][0];
    if !lapse_squared.is_finite() || lapse_squared <= 0.0
    {
        return Err(AdmError::NonSpacelikeSlice);
    }
    Ok(Frame {
        lapse: lapse_squared.sqrt(),
        shift,
        spatial,
        inverse,
    })
}

/// The extrinsic curvature `K_ij` at `coordinates`, plus the frame it needs.
fn extrinsic_curvature_at<B: Metric<4>>(
    background: &B,
    coordinates: &[f64; 4],
    settings: &AdmSettings,
) -> Result<ExtrinsicCurvature, AdmError> {
    let Frame {
        lapse,
        shift,
        spatial,
        inverse,
    } = frame_at(background, coordinates)?;
    let lower = shift_covector_at(background, coordinates);

    // partial_0 gamma_ij by a central time difference.
    let mut forward = *coordinates;
    let mut backward = *coordinates;
    forward[0] += settings.time_step;
    backward[0] -= settings.time_step;
    let gamma_forward = spatial_metric_at(background, &forward);
    let gamma_backward = spatial_metric_at(background, &backward);

    // Spatial Christoffel symbols of gamma (D = 3).
    let slice = SpatialSlice {
        background,
        time: coordinates[0],
    };
    let spatial_point = [coordinates[1], coordinates[2], coordinates[3]];
    let christoffel = numerical_christoffel::<_, 3>(&slice, &spatial_point, settings.spatial_step)?;

    // D_i N_j = partial_i N_j - Gamma^k_{ij} N_k, with partial_i N_j a spatial
    // difference of the shift covector.
    let mut covariant_shift = [[0.0_f64; 3]; 3];
    for i in 0..3
    {
        let mut plus = *coordinates;
        let mut minus = *coordinates;
        plus[i + 1] += settings.spatial_step;
        minus[i + 1] -= settings.spatial_step;
        let shift_plus = shift_covector_at(background, &plus);
        let shift_minus = shift_covector_at(background, &minus);
        for j in 0..3
        {
            let partial = (shift_plus[j] - shift_minus[j]) / (2.0 * settings.spatial_step);
            let mut connection = 0.0;
            for k in 0..3
            {
                connection += christoffel[k][i][j] * lower[k];
            }
            covariant_shift[i][j] = partial - connection;
        }
    }

    // K_ij = -(1/(2N)) (partial_0 gamma_ij - D_i N_j - D_j N_i).
    let mut extrinsic = [[0.0_f64; 3]; 3];
    for i in 0..3
    {
        for j in 0..3
        {
            let time_derivative =
                (gamma_forward[i][j] - gamma_backward[i][j]) / (2.0 * settings.time_step);
            let value =
                -(time_derivative - covariant_shift[i][j] - covariant_shift[j][i]) / (2.0 * lapse);
            if !value.is_finite()
            {
                return Err(AdmError::NonFiniteResult);
            }
            extrinsic[i][j] = value;
        }
    }
    Ok(ExtrinsicCurvature {
        curvature: extrinsic,
        lapse,
        shift,
        spatial,
        inverse,
    })
}

/// Raise both indices of a symmetric spatial tensor with `gamma^{ij}`.
///
/// `pub(crate)`: reused by the Layer 3 `adm_evolution` module so it does not
/// duplicate this index algebra.
pub(crate) fn raise(inverse: &[[f64; 3]; 3], tensor: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut raised = [[0.0_f64; 3]; 3];
    for p in 0..3
    {
        for q in 0..3
        {
            let mut value = 0.0;
            for a in 0..3
            {
                for b in 0..3
                {
                    value += inverse[p][a] * inverse[q][b] * tensor[a][b];
                }
            }
            raised[p][q] = value;
        }
    }
    raised
}

/// The mean curvature `K = gamma^{ij} K_ij`.
///
/// `pub(crate)`: reused by the Layer 3 `adm_evolution` module so it does not
/// duplicate this index algebra.
pub(crate) fn mean_curvature(inverse: &[[f64; 3]; 3], extrinsic: &[[f64; 3]; 3]) -> f64 {
    let mut value = 0.0;
    for i in 0..3
    {
        for j in 0..3
        {
            value += inverse[i][j] * extrinsic[i][j];
        }
    }
    value
}

/// The momentum tensor `P^{ij} = K^{ij} - gamma^{ij} K` at `coordinates`.
fn momentum_tensor_at<B: Metric<4>>(
    background: &B,
    coordinates: &[f64; 4],
    settings: &AdmSettings,
) -> Result<[[f64; 3]; 3], AdmError> {
    let ExtrinsicCurvature {
        curvature: extrinsic,
        inverse,
        ..
    } = extrinsic_curvature_at(background, coordinates, settings)?;
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

/// Decompose the 4-metric of `background` on the constant-time slice through
/// `coordinates`: the lapse, shift, spatial metric, extrinsic curvature, and the
/// spatial-curvature / mean-curvature invariants.
///
/// Returns a typed [`AdmError`] for non-finite coordinates, an invalid step, a
/// singular spatial metric, a non-spacelike slice, or a non-finite result; it
/// never panics.
///
/// # Example
///
/// The FLRW slicing has mean curvature `K = -3H`.
///
/// ```
/// use scirust_relativity::{ExponentialScaleFactor, Flrw};
/// use scirust_relativity::adm::{AdmSettings, adm_decomposition};
///
/// let hubble = 0.5;
/// let flrw = Flrw::new(ExponentialScaleFactor::try_new(hubble).expect("valid H"));
/// let settings = AdmSettings { time_step: 1.0e-3, spatial_step: 1.0e-3 };
/// let adm = adm_decomposition(&flrw, &[0.0, 0.1, 0.2, 0.3], &settings).expect("valid slice");
/// assert!((adm.mean_curvature - (-3.0 * hubble)).abs() < 1.0e-5);
/// ```
pub fn adm_decomposition<B: Metric<4>>(
    background: &B,
    coordinates: &[f64; 4],
    settings: &AdmSettings,
) -> Result<AdmDecomposition, AdmError> {
    validate(coordinates, settings)?;

    let ExtrinsicCurvature {
        curvature: extrinsic,
        lapse,
        shift,
        spatial,
        inverse,
    } = extrinsic_curvature_at(background, coordinates, settings)?;

    let slice = SpatialSlice {
        background,
        time: coordinates[0],
    };
    let spatial_point = [coordinates[1], coordinates[2], coordinates[3]];
    let spatial_ricci_scalar = ricci_scalar_from_metric::<_, 3>(
        &slice,
        &spatial_point,
        settings.spatial_step,
        settings.spatial_step,
    )?;

    let trace = mean_curvature(&inverse, &extrinsic);
    let raised = raise(&inverse, &extrinsic);
    let mut norm = 0.0;
    for i in 0..3
    {
        for j in 0..3
        {
            norm += extrinsic[i][j] * raised[i][j];
        }
    }

    if !lapse.is_finite()
        || !spatial_ricci_scalar.is_finite()
        || !trace.is_finite()
        || !norm.is_finite()
    {
        return Err(AdmError::NonFiniteResult);
    }

    Ok(AdmDecomposition {
        lapse,
        shift,
        spatial_metric: spatial,
        extrinsic_curvature: extrinsic,
        spatial_ricci_scalar,
        mean_curvature: trace,
        extrinsic_curvature_norm: norm,
    })
}

/// Evaluate the Gauss-Codazzi constraint residuals on the constant-time slice
/// through `coordinates`, for a vacuum-with-`Lambda` background.
///
/// The Hamiltonian residual `R^(3) + K^2 - K_ij K^{ij} - 2 Lambda` and the
/// momentum residual `D_j (K^{ij} - gamma^{ij} K)` both vanish (to the disclosed
/// numerical tolerance) for an exact solution. Returns a typed [`AdmError`]; it
/// never panics.
pub fn adm_constraints<B: Metric<4>>(
    background: &B,
    coordinates: &[f64; 4],
    cosmological_constant: f64,
    settings: &AdmSettings,
) -> Result<AdmConstraints, AdmError> {
    if !cosmological_constant.is_finite()
    {
        return Err(AdmError::InvalidCosmologicalConstant(cosmological_constant));
    }
    let decomposition = adm_decomposition(background, coordinates, settings)?;
    let hamiltonian = decomposition.spatial_ricci_scalar
        + decomposition.mean_curvature * decomposition.mean_curvature
        - decomposition.extrinsic_curvature_norm
        - 2.0 * cosmological_constant;

    // Momentum: D_j P^{ij} = partial_j P^{ij} + Gamma^i_{jk} P^{kj} + Gamma^j_{jk} P^{ik}.
    let slice = SpatialSlice {
        background,
        time: coordinates[0],
    };
    let spatial_point = [coordinates[1], coordinates[2], coordinates[3]];
    let christoffel = numerical_christoffel::<_, 3>(&slice, &spatial_point, settings.spatial_step)?;
    let center = momentum_tensor_at(background, coordinates, settings)?;

    let mut momentum = [0.0_f64; 3];
    for i in 0..3
    {
        let mut divergence = 0.0;
        for d in 0..3
        {
            let mut plus = *coordinates;
            let mut minus = *coordinates;
            plus[d + 1] += settings.spatial_step;
            minus[d + 1] -= settings.spatial_step;
            let plus_tensor = momentum_tensor_at(background, &plus, settings)?;
            let minus_tensor = momentum_tensor_at(background, &minus, settings)?;
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
        let value = divergence + connection;
        if !value.is_finite()
        {
            return Err(AdmError::NonFiniteResult);
        }
        momentum[i] = value;
    }

    if !hamiltonian.is_finite()
    {
        return Err(AdmError::NonFiniteResult);
    }

    Ok(AdmConstraints {
        hamiltonian,
        momentum,
    })
}
