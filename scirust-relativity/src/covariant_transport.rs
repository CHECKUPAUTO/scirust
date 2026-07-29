//! Parallel transport of covectors and rank-2 covariant tensors.
//!
//! This extends [`crate::transport_along_segment`] (which transports a
//! contravariant vector) to lower-index objects, along the same straight
//! coordinate segments and with the same deterministic `scirust-sim` RK4
//! engine. The transport equations differ only in the sign and index placement
//! of the connection term:
//!
//! ```text
//! dV^a/ds  = - Gamma^a_(c b) delta^c V^b        (vector, see parallel_transport)
//! dW_a/ds  = + Gamma^b_(c a) delta^c W_b        (covector)
//! dT_(ab)/ds = + Gamma^c_(e a) delta^e T_(cb)
//!            + Gamma^c_(e b) delta^e T_(ac)     (rank-2 covariant tensor)
//! ```
//!
//! Because the Levi-Civita connection is metric compatible (`nabla g = 0`),
//! two identities hold and are checked in this crate's tests:
//!
//! - **Index lowering commutes with transport.** Transporting `V^a` and then
//!   lowering with the endpoint metric gives the same covector as lowering
//!   `V^a` at the start and transporting the resulting covector.
//! - **The metric transports to itself.** Parallel-transporting `g_(ab)` from a
//!   point along any path yields the metric at the endpoint.

use crate::parallel_transport::{validate_finite_coordinates, validate_finite_vector};
use crate::{Connection, RelativityError};
use scirust_sim::{System, simulate};

/// Covector parallel transport along one straight segment `x(s) = start + s * delta`.
struct CovectorSegmentTransport<'a, C, const D: usize> {
    connection: &'a C,
    start: [f64; D],
    delta: [f64; D],
}

impl<C, const D: usize> System for CovectorSegmentTransport<'_, C, D>
where
    C: Connection<D>,
{
    fn dim(&self) -> usize {
        D
    }

    fn derivatives(&self, parameter: f64, state: &[f64], output: &mut [f64]) {
        let mut coordinates = self.start;
        for (axis, coordinate) in coordinates.iter_mut().enumerate()
        {
            *coordinate += parameter * self.delta[axis];
        }
        let christoffel = self.connection.christoffel(&coordinates);

        // dW_a/ds = + Gamma^b_(c a) delta^c W_b.
        for (a, rate) in output.iter_mut().enumerate()
        {
            let mut value = 0.0;
            for (b, christoffel_b) in christoffel.iter().enumerate()
            {
                let covector_b = state[b];
                for (c, christoffel_bc) in christoffel_b.iter().enumerate()
                {
                    value += christoffel_bc[a] * self.delta[c] * covector_b;
                }
            }
            *rate = value;
        }
    }
}

/// Rank-2 covariant tensor parallel transport along one straight segment; state
/// `state[a * D + b] = T_(ab)`.
struct CovariantTensorSegmentTransport<'a, C, const D: usize> {
    connection: &'a C,
    start: [f64; D],
    delta: [f64; D],
}

impl<C, const D: usize> System for CovariantTensorSegmentTransport<'_, C, D>
where
    C: Connection<D>,
{
    fn dim(&self) -> usize {
        D * D
    }

    fn derivatives(&self, parameter: f64, state: &[f64], output: &mut [f64]) {
        let mut coordinates = self.start;
        for (axis, coordinate) in coordinates.iter_mut().enumerate()
        {
            *coordinate += parameter * self.delta[axis];
        }
        let christoffel = self.connection.christoffel(&coordinates);

        // dT_(ab)/ds = + Gamma^c_(e a) delta^e T_(cb) + Gamma^c_(e b) delta^e T_(ac).
        for a in 0..D
        {
            for b in 0..D
            {
                let mut value = 0.0;
                for (c, christoffel_c) in christoffel.iter().enumerate()
                {
                    for (e, christoffel_ce) in christoffel_c.iter().enumerate()
                    {
                        let weight = self.delta[e];
                        value += christoffel_ce[a] * weight * state[c * D + b];
                        value += christoffel_ce[b] * weight * state[a * D + c];
                    }
                }
                output[a * D + b] = value;
            }
        }
    }
}

fn validate_finite_tensor<const D: usize>(tensor: &[[f64; D]; D]) -> Result<(), RelativityError> {
    for row in tensor
    {
        validate_finite_vector(row)?;
    }
    Ok(())
}

fn segment_delta<const D: usize>(start: &[f64; D], end: &[f64; D]) -> [f64; D] {
    let mut delta = [0.0_f64; D];
    for (axis, component) in delta.iter_mut().enumerate()
    {
        *component = end[axis] - start[axis];
    }
    delta
}

/// Parallel-transport the covector `covector` from `start` to `end` along the
/// straight coordinate segment between them, using `substeps` RK4 steps.
///
/// # Example
///
/// In flat Minkowski spacetime covector transport is the identity.
///
/// ```
/// use scirust_relativity::{transport_covector_along_segment, Minkowski};
///
/// let covector = [0.3, -0.5, 0.2, 0.1];
/// let transported = transport_covector_along_segment(
///     &Minkowski,
///     &[0.0, 0.0, 0.0, 0.0],
///     &[1.0, 2.0, -1.0, 0.5],
///     &covector,
///     4,
/// )
/// .expect("flat covector transport");
/// assert_eq!(transported, covector);
/// ```
pub fn transport_covector_along_segment<C, const D: usize>(
    connection: &C,
    start: &[f64; D],
    end: &[f64; D],
    covector: &[f64; D],
    substeps: usize,
) -> Result<[f64; D], RelativityError>
where
    C: Connection<D>,
{
    validate_finite_coordinates(start)?;
    validate_finite_coordinates(end)?;
    validate_finite_vector(covector)?;
    if substeps == 0
    {
        return Err(RelativityError::InvalidTransportResolution);
    }

    let system = CovectorSegmentTransport {
        connection,
        start: *start,
        delta: segment_delta(start, end),
    };
    let step = 1.0 / substeps as f64;
    let trajectory = simulate(&system, covector, 0.0, 1.0, step)
        .map_err(|_| RelativityError::NonFiniteTransportedVector)?;
    let final_state = trajectory
        .last_state()
        .ok_or(RelativityError::NonFiniteTransportedVector)?;

    let mut transported = [0.0_f64; D];
    for (axis, component) in transported.iter_mut().enumerate()
    {
        let value = final_state[axis];
        if !value.is_finite()
        {
            return Err(RelativityError::NonFiniteTransportedVector);
        }
        *component = value;
    }
    Ok(transported)
}

/// Parallel-transport `covector` along the polyline through `path`, transporting
/// across each consecutive segment with `substeps` RK4 steps per segment.
pub fn transport_covector_along_polyline<C, const D: usize>(
    connection: &C,
    path: &[[f64; D]],
    covector: &[f64; D],
    substeps: usize,
) -> Result<[f64; D], RelativityError>
where
    C: Connection<D>,
{
    validate_finite_vector(covector)?;
    if substeps == 0
    {
        return Err(RelativityError::InvalidTransportResolution);
    }

    let mut transported = *covector;
    for segment in path.windows(2)
    {
        transported = transport_covector_along_segment(
            connection,
            &segment[0],
            &segment[1],
            &transported,
            substeps,
        )?;
    }
    Ok(transported)
}

/// Parallel-transport the rank-2 covariant tensor `tensor` (indexed
/// `tensor[a][b] = T_(ab)`) from `start` to `end` along the straight coordinate
/// segment, using `substeps` RK4 steps.
pub fn transport_covariant_tensor_along_segment<C, const D: usize>(
    connection: &C,
    start: &[f64; D],
    end: &[f64; D],
    tensor: &[[f64; D]; D],
    substeps: usize,
) -> Result<[[f64; D]; D], RelativityError>
where
    C: Connection<D>,
{
    validate_finite_coordinates(start)?;
    validate_finite_coordinates(end)?;
    validate_finite_tensor(tensor)?;
    if substeps == 0
    {
        return Err(RelativityError::InvalidTransportResolution);
    }

    let mut initial = vec![0.0_f64; D * D];
    for (a, row) in tensor.iter().enumerate()
    {
        initial[a * D..a * D + D].copy_from_slice(row);
    }

    let system = CovariantTensorSegmentTransport {
        connection,
        start: *start,
        delta: segment_delta(start, end),
    };
    let step = 1.0 / substeps as f64;
    let trajectory = simulate(&system, &initial, 0.0, 1.0, step)
        .map_err(|_| RelativityError::NonFiniteTransportedVector)?;
    let final_state = trajectory
        .last_state()
        .ok_or(RelativityError::NonFiniteTransportedVector)?;

    let mut transported = [[0.0_f64; D]; D];
    for (a, row) in transported.iter_mut().enumerate()
    {
        for (b, component) in row.iter_mut().enumerate()
        {
            let value = final_state[a * D + b];
            if !value.is_finite()
            {
                return Err(RelativityError::NonFiniteTransportedVector);
            }
            *component = value;
        }
    }
    Ok(transported)
}

/// Parallel-transport the rank-2 covariant tensor `tensor` along the polyline
/// through `path`, with `substeps` RK4 steps per segment.
pub fn transport_covariant_tensor_along_polyline<C, const D: usize>(
    connection: &C,
    path: &[[f64; D]],
    tensor: &[[f64; D]; D],
    substeps: usize,
) -> Result<[[f64; D]; D], RelativityError>
where
    C: Connection<D>,
{
    validate_finite_tensor(tensor)?;
    if substeps == 0
    {
        return Err(RelativityError::InvalidTransportResolution);
    }

    let mut transported = *tensor;
    for segment in path.windows(2)
    {
        transported = transport_covariant_tensor_along_segment(
            connection,
            &segment[0],
            &segment[1],
            &transported,
            substeps,
        )?;
    }
    Ok(transported)
}
