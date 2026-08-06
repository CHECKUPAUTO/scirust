//! Optional research diagnostic: a Lorentzian-correct decomposition of a
//! coordinate state-error vector relative to a timelike observer.
//!
//! [`crate::scaled_local_error_norm`] (the adaptive step controller's norm) is
//! componentwise and chart-dependent: it treats the state as a point in a
//! Euclidean `R^{2D}` and says nothing about the indefinite spacetime metric.
//! That is the right, cheap tool for *step-size control*, but it is not a
//! geometrically meaningful error measure, and a Lorentzian-signature metric
//! must never be presented as an ordinary positive Euclidean norm.
//!
//! This module provides the geometric alternative for a coordinate
//! displacement `delta` (a small difference of two nearby states' coordinates,
//! treated as an approximate tangent vector at the reference point), decomposed
//! relative to a timelike four-velocity `u`:
//!
//! - the **temporal** part `|g(delta, u)| / sqrt(-g(u,u))` — the size of the
//!   component of `delta` along the observer's proper-time direction;
//! - the **spatial** part `sqrt(g(P delta, P delta))`, where
//!   `P^mu_nu = delta^mu_nu - u^mu u_nu / g(u,u)` is the crate's existing
//!   projector onto the subspace orthogonal to `u`. Because `P delta` is
//!   orthogonal to the timelike `u`, it is spacelike, so `g(P delta, P delta)`
//!   is non-negative and its square root is a genuine positive length.
//!
//! This handles the indefinite metric correctly (the temporal and spatial
//! parts are separated *by the metric*, not lumped into one Euclidean sum), and
//! it is exact in closed form — no tetrad construction is needed for the scalar
//! temporal/spatial magnitudes. It remains chart-dependent in the weaker sense
//! that `delta` is a coordinate difference and `u`, the metric are evaluated at
//! one chart point; it is **not** a full invariant comparison of two distant
//! states, and it is deliberately a post-hoc diagnostic, **not** wired into the
//! adaptive step controllers.
//!
//! The scalar temporal/spatial split of [`timelike_state_error`] is
//! complemented by a full local-orthonormal-frame (tetrad) projection,
//! [`tetrad_state_error`], which expresses `delta` in an observer's own
//! orthonormal basis component by component. The tetrad
//! ([`build_orthonormal_tetrad`]) delegates to the reusable geometry-core
//! primitive [`scirust_relativity::orthonormal_tetrad`], which builds the frame
//! by metric Gram-Schmidt: the timelike leg is the normalized four-velocity,
//! and the spacelike legs are orthonormalized coordinate-basis directions. Both
//! are exactly verifiable
//! (the frame is orthonormal, `g(e_a, e_b) = eta_ab`, and the projected
//! components reconstruct `delta` exactly), and the tetrad's temporal/spatial
//! magnitudes agree with the scalar split — see the flat-spacetime tests in
//! `tests/geometric_error.rs`. The individual spatial components depend on the
//! (non-unique) spatial-frame choice; the magnitudes do not.

use crate::{
    NonlocalRelativityError, NonlocalResult, lower_index, metric_contraction, validate_scalar,
};
use scirust_relativity::{RelativityError, orthonormal_tetrad};

/// A local orthonormal frame (tetrad) at a chart point.
///
/// This is the reusable geometry-core [`scirust_relativity::OrthonormalTetrad`]:
/// the `D` legs `e_a` satisfy `g(e_a, e_b) = eta_ab` with
/// `eta = diag(-1, +1, ..., +1)`, leg `0` timelike and legs `1..D` spacelike.
/// It is re-exported here so this crate's observer-frame diagnostic
/// ([`build_orthonormal_tetrad`], [`tetrad_state_error`]) exposes the same type
/// the geometry core builds, rather than a duplicate.
pub use scirust_relativity::OrthonormalTetrad;

/// Metric-aware decomposition of a coordinate state-error vector relative to a
/// timelike observer. See the module documentation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimelikeStateError {
    /// `|g(delta, u)| / sqrt(-g(u,u))`: the magnitude of the error along the
    /// observer's proper-time direction. Non-negative.
    pub temporal: f64,
    /// `sqrt(g(P delta, P delta))`: the metric length of the error projected
    /// orthogonal to `u`. Non-negative (the projected vector is spacelike).
    pub spatial: f64,
    /// `g(P delta, u)`: the metric inner product of the projected error with
    /// `u`. Zero in exact arithmetic (the projection is orthogonal to `u`);
    /// exposed as a rounding-level self-consistency residual.
    pub orthogonality_residual: f64,
}

/// Decompose a coordinate state-error vector `delta` at a point with metric
/// `metric` and timelike four-velocity `four_velocity` into metric-aware
/// temporal and spatial magnitudes.
///
/// `timelike_floor` must be finite and strictly positive; the observer is
/// required to be timelike with `g(u,u) < -timelike_floor` under a
/// `(-,+,+,+)`-signature convention. A non-timelike observer (null, spacelike,
/// or from an incompatible signature) is rejected with
/// [`NonlocalRelativityError::NonTimelikeMetricNorm`] rather than silently
/// producing a meaningless "spatial" length from an indefinite contraction.
///
/// This is a diagnostic, not a controller input, and it is chart-dependent in
/// the sense described in the module documentation; it must never be described
/// as a covariant error between two distant states or as establishing
/// coordinate covariance.
pub fn timelike_state_error<const D: usize>(
    metric: &[[f64; D]; D],
    four_velocity: &[f64; D],
    delta: &[f64; D],
    timelike_floor: f64,
) -> NonlocalResult<TimelikeStateError> {
    let metric_norm = validated_timelike_norm(metric, four_velocity, timelike_floor)?;

    let lowered = lower_index(metric, four_velocity);
    let inner_delta_u = lowered
        .iter()
        .zip(delta)
        .fold(0.0, |sum, (lowered_component, delta_component)| {
            sum + *lowered_component * delta_component
        });
    validate_scalar("timelike_state_error_inner", inner_delta_u, 0)?;

    let temporal = inner_delta_u.abs() / (-metric_norm).sqrt();
    validate_scalar("timelike_state_error_temporal", temporal, 0)?;

    // Projection orthogonal to u: (P delta)^mu = delta^mu - u^mu * g(delta,u)/g(u,u).
    let projection_scale = inner_delta_u / metric_norm;
    let mut projected = [0.0_f64; D];
    for component in 0..D
    {
        projected[component] = delta[component] - four_velocity[component] * projection_scale;
        validate_scalar(
            "timelike_state_error_projected",
            projected[component],
            component,
        )?;
    }

    let spatial_squared = metric_contraction(metric, &projected, &projected);
    validate_scalar("timelike_state_error_spatial_squared", spatial_squared, 0)?;
    // The projected vector is spacelike, so this is non-negative up to
    // rounding; clamp a tiny negative rounding excursion to zero.
    let spatial = spatial_squared.max(0.0).sqrt();
    validate_scalar("timelike_state_error_spatial", spatial, 0)?;

    let orthogonality_residual =
        lowered
            .iter()
            .zip(projected)
            .fold(0.0, |sum, (lowered_component, projected_component)| {
                sum + *lowered_component * projected_component
            });
    validate_scalar(
        "timelike_state_error_orthogonality",
        orthogonality_residual,
        0,
    )?;

    Ok(TimelikeStateError {
        temporal,
        spatial,
        orthogonality_residual,
    })
}

/// Validate `timelike_floor` and the observer, returning `g(u,u)` when the
/// observer is timelike (`g(u,u) < -timelike_floor` under a `(-,+,+,+)`
/// convention). Used by [`timelike_state_error`]; the tetrad builder applies
/// the identical validation inside the geometry-core primitive it delegates to.
fn validated_timelike_norm<const D: usize>(
    metric: &[[f64; D]; D],
    four_velocity: &[f64; D],
    timelike_floor: f64,
) -> NonlocalResult<f64> {
    if !timelike_floor.is_finite() || timelike_floor <= 0.0
    {
        return Err(NonlocalRelativityError::InvalidMetricNormFloor(
            timelike_floor,
        ));
    }

    let metric_norm = metric_contraction(metric, four_velocity, four_velocity);
    if !metric_norm.is_finite()
    {
        return Err(NonlocalRelativityError::NonFiniteMetricNorm {
            step: 0,
            value: metric_norm,
        });
    }
    if metric_norm > -timelike_floor
    {
        // Not sufficiently timelike under the (-,+,+,+) convention.
        return Err(NonlocalRelativityError::NonTimelikeMetricNorm { metric_norm });
    }

    Ok(metric_norm)
}

/// Build a local orthonormal frame (tetrad) at a chart point for a timelike
/// observer, by metric Gram-Schmidt.
///
/// This delegates to the reusable geometry-core primitive
/// [`scirust_relativity::orthonormal_tetrad`] so the Gram-Schmidt construction
/// lives in exactly one place; it maps that primitive's typed errors back to
/// this crate's [`NonlocalRelativityError`] contract (see [`map_tetrad_error`]),
/// leaving the observable behaviour unchanged.
///
/// The timelike leg is the normalized four-velocity; the spacelike legs are
/// the chart's coordinate-basis directions orthonormalized against the frame
/// so far (skipping any that collapse to within `timelike_floor` of a
/// degenerate residual). `timelike_floor` must be finite and strictly
/// positive; a non-timelike observer is rejected exactly as in
/// [`timelike_state_error`], and a point where fewer than `D` independent
/// legs survive is a typed
/// [`NonlocalRelativityError::DegenerateObserverFrame`] (it never silently
/// returns an incomplete frame). This is a diagnostic construction and shares
/// [`timelike_state_error`]'s chart-dependence caveats.
pub fn build_orthonormal_tetrad<const D: usize>(
    metric: &[[f64; D]; D],
    four_velocity: &[f64; D],
    timelike_floor: f64,
) -> NonlocalResult<OrthonormalTetrad<D>> {
    orthonormal_tetrad(metric, four_velocity, timelike_floor).map_err(map_tetrad_error)
}

/// Map a geometry-core [`RelativityError`] from
/// [`scirust_relativity::orthonormal_tetrad`] back to this crate's
/// [`NonlocalRelativityError`] contract, so the delegating
/// [`build_orthonormal_tetrad`] reports exactly the variants its callers and
/// tests relied on before the Gram-Schmidt construction moved into the
/// geometry core.
fn map_tetrad_error(error: RelativityError) -> NonlocalRelativityError {
    match error
    {
        RelativityError::InvalidTetradFloor(floor) =>
        {
            NonlocalRelativityError::InvalidMetricNormFloor(floor)
        },
        // The geometry-core primitive folds "non-finite norm" and "finite but
        // not timelike" into one variant; split them back so this crate keeps
        // reporting the same two norms it did before delegating.
        RelativityError::NonTimelikeFrameVector { metric_norm } if metric_norm.is_finite() =>
        {
            NonlocalRelativityError::NonTimelikeMetricNorm { metric_norm }
        },
        RelativityError::NonTimelikeFrameVector { metric_norm } =>
        {
            NonlocalRelativityError::NonFiniteMetricNorm {
                step: 0,
                value: metric_norm,
            }
        },
        RelativityError::DegenerateFrame {
            legs_found,
            dimension,
        } => NonlocalRelativityError::DegenerateObserverFrame {
            legs_found,
            dimension,
        },
        // A non-finite leg (`RelativityError::NonFiniteTetradLeg`) — and any
        // other geometry-core error the tetrad primitive is not documented to
        // return — surfaces as a non-finite tetrad diagnostic, matching the
        // pre-delegation `validate_scalar` guards on the frame legs.
        _ => NonlocalRelativityError::NonFiniteDiagnostic {
            step: 0,
            quantity: "tetrad_leg",
            value: f64::NAN,
        },
    }
}

/// The tetrad-frame decomposition of a coordinate state-error vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TetradStateError<const D: usize> {
    /// The frame components `c^a` of `delta`: `c^a = eta_aa * g(delta, e_a)`,
    /// so `delta = sum_a c^a e_a`. `components[0]` is the (signed) temporal
    /// component; the rest are spatial.
    pub components: [f64; D],
    /// `|components[0]|`, the temporal magnitude. Matches
    /// [`TimelikeStateError::temporal`].
    pub temporal: f64,
    /// The Euclidean length of the spatial components, a genuine positive
    /// spatial length. Matches [`TimelikeStateError::spatial`].
    pub spatial: f64,
    /// Chart-Euclidean length of `sum_a c^a e_a - delta`: zero in exact
    /// arithmetic (the tetrad spans the tangent space), exposed as a
    /// rounding-level self-consistency residual.
    pub reconstruction_residual: f64,
}

/// Project a coordinate state-error vector `delta` onto the local orthonormal
/// frame of a timelike observer (see [`build_orthonormal_tetrad`]).
///
/// This is the full-tetrad complement of [`timelike_state_error`]: it returns
/// `delta`'s component in each frame leg, and its temporal and spatial
/// magnitudes agree with that scalar split. The individual spatial components
/// depend on the (non-unique) spatial-frame choice; the magnitudes do not. It
/// is a diagnostic, chart-dependent in the same sense, and never a covariant
/// comparison of distant states.
pub fn tetrad_state_error<const D: usize>(
    metric: &[[f64; D]; D],
    four_velocity: &[f64; D],
    delta: &[f64; D],
    timelike_floor: f64,
) -> NonlocalResult<TetradStateError<D>> {
    let tetrad = build_orthonormal_tetrad(metric, four_velocity, timelike_floor)?;

    let mut components = [0.0_f64; D];
    for (leg, leg_vector) in tetrad.legs().iter().enumerate()
    {
        let projection = metric_contraction(metric, delta, leg_vector);
        components[leg] = OrthonormalTetrad::<D>::signature(leg) * projection;
        validate_scalar("tetrad_component", components[leg], leg)?;
    }

    let temporal = components[0].abs();
    let mut spatial_squared = 0.0_f64;
    for &component in components.iter().skip(1)
    {
        spatial_squared += component * component;
    }
    let spatial = spatial_squared.sqrt();
    validate_scalar("tetrad_spatial", spatial, 0)?;

    // Reconstruct delta = sum_a c^a e_a and measure the chart-Euclidean residual.
    let mut reconstruction_squared = 0.0_f64;
    for (component_index, &delta_component) in delta.iter().enumerate()
    {
        let mut reconstructed = 0.0_f64;
        for (component, leg_vector) in components.iter().zip(tetrad.legs().iter())
        {
            reconstructed += component * leg_vector[component_index];
        }
        let difference = reconstructed - delta_component;
        reconstruction_squared += difference * difference;
    }
    let reconstruction_residual = reconstruction_squared.sqrt();
    validate_scalar("tetrad_reconstruction", reconstruction_residual, 0)?;

    Ok(TetradStateError {
        components,
        temporal,
        spatial,
        reconstruction_residual,
    })
}
