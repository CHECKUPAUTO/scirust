//! Local orthonormal frames (tetrads).
//!
//! An orthonormal frame `{e_a}` at a chart point satisfies
//! `g(e_a, e_b) = eta_ab` with `eta = diag(-1, +1, ..., +1)`: the timelike leg
//! `e_0` is the normalized observer four-velocity, and the spacelike legs are
//! coordinate-basis directions orthonormalized against the frame so far by
//! metric Gram-Schmidt.
//!
//! This is the reusable geometry-core construction. The experimental worldline
//! crate's observer-frame diagnostic delegates to [`orthonormal_tetrad`] rather
//! than carrying its own copy of the Gram-Schmidt algorithm.

use crate::RelativityError;

/// A local orthonormal frame (tetrad): `D` legs `e_a` with
/// `g(e_a, e_b) = eta_ab`. Leg `0` is timelike, legs `1..D` spacelike.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrthonormalTetrad<const D: usize> {
    legs: [[f64; D]; D],
}

impl<const D: usize> OrthonormalTetrad<D> {
    /// Borrow the tetrad legs `e_a` (contravariant components in the chart),
    /// leg `0` first.
    #[must_use]
    pub const fn legs(&self) -> &[[f64; D]; D] {
        &self.legs
    }

    /// The Minkowski signature entry `eta_aa = g(e_a, e_a)` of leg `index`:
    /// `-1.0` for the timelike leg `0`, `+1.0` for the spacelike legs.
    #[must_use]
    pub fn signature(index: usize) -> f64 {
        if index == 0 { -1.0 } else { 1.0 }
    }
}

/// Metric inner product `g(a, b) = g_(mu nu) a^mu b^nu`.
fn metric_inner_product<const D: usize>(
    metric: &[[f64; D]; D],
    left: &[f64; D],
    right: &[f64; D],
) -> f64 {
    let mut value = 0.0;
    for (row, &left_component) in metric.iter().zip(left.iter())
    {
        for (&entry, &right_component) in row.iter().zip(right.iter())
        {
            value += entry * left_component * right_component;
        }
    }
    value
}

/// Build a local orthonormal frame (tetrad) for the timelike observer
/// `timelike_vector` at a point with metric `metric`, by metric Gram-Schmidt.
///
/// The timelike leg is `timelike_vector` normalized to `g(e_0, e_0) = -1`; the
/// spacelike legs are the coordinate-basis directions orthonormalized against
/// the frame so far, skipping any whose residual metric norm is not greater
/// than `timelike_floor`.
///
/// Returns a typed [`RelativityError`]: [`RelativityError::InvalidTetradFloor`]
/// for a non-finite / non-positive floor,
/// [`RelativityError::NonTimelikeFrameVector`] when `g(u, u) > -floor` (or is
/// non-finite), [`RelativityError::NonFiniteTetradLeg`] for a non-finite leg,
/// and [`RelativityError::DegenerateFrame`] when fewer than `D` independent
/// legs survive. It never panics and never returns an incomplete frame.
pub fn orthonormal_tetrad<const D: usize>(
    metric: &[[f64; D]; D],
    timelike_vector: &[f64; D],
    timelike_floor: f64,
) -> Result<OrthonormalTetrad<D>, RelativityError> {
    if !timelike_floor.is_finite() || timelike_floor <= 0.0
    {
        return Err(RelativityError::InvalidTetradFloor(timelike_floor));
    }

    let metric_norm = metric_inner_product(metric, timelike_vector, timelike_vector);
    if !metric_norm.is_finite() || metric_norm > -timelike_floor
    {
        return Err(RelativityError::NonTimelikeFrameVector { metric_norm });
    }

    let mut legs = [[0.0_f64; D]; D];

    // Timelike leg e_0 = u / sqrt(-g(u,u)), so g(e_0, e_0) = -1.
    let timelike_scale = (-metric_norm).sqrt();
    for (component, slot) in timelike_vector.iter().zip(legs[0].iter_mut())
    {
        *slot = component / timelike_scale;
        if !slot.is_finite()
        {
            return Err(RelativityError::NonFiniteTetradLeg);
        }
    }

    // Spacelike legs by metric Gram-Schmidt over the coordinate basis.
    let mut built = 1_usize;
    let mut candidate = 0_usize;
    while built < D && candidate < D
    {
        let mut vector = [0.0_f64; D];
        vector[candidate] = 1.0;

        for leg_vector in legs.iter().take(built)
        {
            let inner = metric_inner_product(metric, &vector, leg_vector);
            let leg_norm = metric_inner_product(metric, leg_vector, leg_vector);
            let coefficient = inner / leg_norm;
            for (component, &leg_component) in vector.iter_mut().zip(leg_vector.iter())
            {
                *component -= coefficient * leg_component;
            }
        }

        let residual_norm = metric_inner_product(metric, &vector, &vector);
        if residual_norm.is_finite() && residual_norm > timelike_floor
        {
            let scale = residual_norm.sqrt();
            for (component, slot) in vector.iter().zip(legs[built].iter_mut())
            {
                *slot = component / scale;
                if !slot.is_finite()
                {
                    return Err(RelativityError::NonFiniteTetradLeg);
                }
            }
            built += 1;
        }

        candidate += 1;
    }

    if built < D
    {
        return Err(RelativityError::DegenerateFrame {
            legs_found: built,
            dimension: D,
        });
    }

    Ok(OrthonormalTetrad { legs })
}
