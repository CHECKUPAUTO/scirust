//! Typed history entries and discrete parallel-transport approximation.
//!
//! The Phase 2 [`crate::HistoryTransport`] contract only received a retained
//! sample's velocity components and the current worldline state: enough for
//! coordinate-identity transport, but not enough to carry a vector between
//! two distinct tangent spaces, since that requires the vector's *source*
//! point. [`HistoryEntry`] is the typed accepted sample (coordinates,
//! velocity, parameter) that supplies that source point, and
//! [`DiscreteConnectionTransport`] is a deterministic, explicitly
//! discretized approximation of parallel transport built on top of it.

use crate::{
    Connection, HistoryTransport, NonlocalRelativityError, NonlocalResult, WorldlineState,
    validate_history_velocity,
};

/// One accepted worldline sample retained for history-dependent evaluation.
///
/// This is the typed replacement for a bare velocity component array: it
/// keeps the coordinates and parameter value where the sample was accepted,
/// which a geometric [`HistoryTransport`] needs in order to carry the
/// sample's vector from the tangent space where it was recorded into the
/// tangent space at a later worldline state. Nothing in this crate transports
/// a vector between distinct tangent spaces using components alone; when only
/// components are available (for example through the legacy
/// [`HistoryBackend::push_velocity`] path), transport is limited to the
/// coordinate-identity contract.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HistoryEntry<const D: usize> {
    /// Accepted coordinates `x^rho` at this sample.
    pub coordinates: [f64; D],
    /// Accepted contravariant velocity `u^rho` at this sample.
    pub velocity: [f64; D],
    /// Accepted affine or proper-time parameter value at this sample.
    pub parameter: f64,
}

impl<const D: usize> HistoryEntry<D> {
    /// Construct a history entry from coordinates, velocity, and parameter.
    #[must_use]
    pub const fn new(coordinates: [f64; D], velocity: [f64; D], parameter: f64) -> Self {
        Self {
            coordinates,
            velocity,
            parameter,
        }
    }

    /// Construct a velocity-only entry for callers that do not track source
    /// coordinates or a parameter value.
    ///
    /// Coordinates are set to the origin and the parameter to zero. These
    /// placeholders are never read by [`crate::IdentityHistoryTransport`] or
    /// by the coordinate-memory pipeline; a geometric transport must not be
    /// combined with entries constructed this way, since it would silently
    /// transport from the wrong point.
    #[must_use]
    pub const fn from_velocity_only(velocity: [f64; D]) -> Self {
        Self {
            coordinates: [0.0; D],
            velocity,
            parameter: 0.0,
        }
    }
}

/// Transport all entries in place from their current frame (the last
/// retained entry, when one exists) to `destination`, then leave `entries`
/// ready for the caller to push `destination` itself.
///
/// This is called once per accepted (or provisional) segment by
/// [`HistoryBackend::push_entry`] implementations. It performs `O(len)`
/// transport evaluations per call; combined with `N` accepted steps this
/// gives `O(N^2)` transport evaluations over a trajectory, each costing
/// `O(D^3)` for a Christoffel contraction, i.e. `O(D^3 * N^2)` overall for the
/// discrete-transport pipeline. This is documented in
/// `scirust-nonlocal-relativity/README.md`.
pub(crate) fn transport_retained_entries<const D: usize, B, T>(
    entries: &mut [HistoryEntry<D>],
    background: &B,
    transport: &T,
    destination: &HistoryEntry<D>,
) -> NonlocalResult<()>
where
    B: Connection<D>,
    T: HistoryTransport<D>,
{
    let Some(last) = entries.last().copied()
    else
    {
        return Ok(());
    };

    let from_state = WorldlineState::new(last.coordinates, last.velocity);
    let to_state = WorldlineState::new(destination.coordinates, destination.velocity);
    let segment_step = destination.parameter - last.parameter;

    for (retained_index, existing) in entries.iter_mut().enumerate()
    {
        let transported = transport.transport_segment(
            retained_index,
            background,
            existing.velocity,
            &from_state,
            &to_state,
            segment_step,
        )?;
        validate_transported_vector(&transported, retained_index)?;
        existing.velocity = transported;
    }

    Ok(())
}

/// Deterministic discrete parallel-transport approximation for retained
/// history vectors.
///
/// Each accepted segment is transported with a single Heun
/// predict-evaluate-correct-evaluate step of the linear transport equation
///
/// `dV^mu / dlambda = - Gamma^mu_(alpha beta) u^alpha V^beta`:
///
/// 1. evaluate the transport derivative at the segment start;
/// 2. predict the vector at the segment end;
/// 3. evaluate the connection and velocity at the segment end;
/// 4. correct with the average of the two derivatives.
///
/// [`HistoryBackend::push_entry`] calls this once per accepted segment for
/// every currently retained vector, so transport accumulates along the
/// actual accepted worldline polyline rather than jumping directly between a
/// sample's original recorded point and the current point.
///
/// This is a discrete numerical approximation to parallel transport along a
/// polyline. It is **not** an exact analytic bitensor propagator, **not** a
/// proof of covariance, and discretization error accumulates with the
/// segment step and the number of transported segments. The legacy
/// [`HistoryTransport::transport_velocity`] method has no source point to
/// transport from, so this type implements it as an identity passthrough;
/// its real work happens in
/// [`transport_segment`](HistoryTransport::transport_segment), which is
/// called once per accepted segment rather than once per memory evaluation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DiscreteConnectionTransport;

impl<const D: usize> HistoryTransport<D> for DiscreteConnectionTransport {
    fn transport_velocity(
        &self,
        retained_index: usize,
        velocity: [f64; D],
        _current_state: &WorldlineState<D>,
    ) -> NonlocalResult<[f64; D]> {
        validate_history_velocity(&velocity, retained_index)?;
        Ok(velocity)
    }

    fn transport_segment<B>(
        &self,
        retained_index: usize,
        background: &B,
        vector: [f64; D],
        from_state: &WorldlineState<D>,
        to_state: &WorldlineState<D>,
        segment_step: f64,
    ) -> NonlocalResult<[f64; D]>
    where
        B: Connection<D>,
    {
        heun_discrete_parallel_transport(
            retained_index,
            background,
            vector,
            from_state,
            to_state,
            segment_step,
        )
    }
}

fn heun_discrete_parallel_transport<B, const D: usize>(
    retained_index: usize,
    background: &B,
    vector: [f64; D],
    from_state: &WorldlineState<D>,
    to_state: &WorldlineState<D>,
    segment_step: f64,
) -> NonlocalResult<[f64; D]>
where
    B: Connection<D>,
{
    if !segment_step.is_finite()
    {
        return Err(NonlocalRelativityError::InvalidTransportSegmentStep(
            segment_step,
        ));
    }

    let start_symbols = background.christoffel(&from_state.coordinates);
    validate_transport_christoffel(&start_symbols)?;
    let start_derivative =
        parallel_transport_derivative(&start_symbols, &from_state.velocity, &vector);
    validate_transported_vector(&start_derivative, retained_index)?;

    let mut predicted = [0.0_f64; D];
    for mu in 0..D
    {
        predicted[mu] = vector[mu] + segment_step * start_derivative[mu];
    }
    validate_transported_vector(&predicted, retained_index)?;

    let end_symbols = background.christoffel(&to_state.coordinates);
    validate_transport_christoffel(&end_symbols)?;
    let end_derivative =
        parallel_transport_derivative(&end_symbols, &to_state.velocity, &predicted);
    validate_transported_vector(&end_derivative, retained_index)?;

    let mut corrected = [0.0_f64; D];
    for mu in 0..D
    {
        corrected[mu] =
            vector[mu] + 0.5 * segment_step * (start_derivative[mu] + end_derivative[mu]);
    }
    validate_transported_vector(&corrected, retained_index)?;

    Ok(corrected)
}

/// Evaluate `-Gamma^mu_(alpha beta) u^alpha V^beta`, the linear parallel
/// transport derivative of `vector` along local direction `local_velocity`.
fn parallel_transport_derivative<const D: usize>(
    christoffel: &[[[f64; D]; D]; D],
    local_velocity: &[f64; D],
    vector: &[f64; D],
) -> [f64; D] {
    let mut derivative = [0.0_f64; D];

    for mu in 0..D
    {
        for alpha in 0..D
        {
            for beta in 0..D
            {
                derivative[mu] -=
                    christoffel[mu][alpha][beta] * local_velocity[alpha] * vector[beta];
            }
        }
    }

    derivative
}

fn validate_transport_christoffel<const D: usize>(
    symbols: &[[[f64; D]; D]; D],
) -> NonlocalResult<()> {
    for (rho, rho_values) in symbols.iter().enumerate()
    {
        for (mu, mu_values) in rho_values.iter().enumerate()
        {
            for (nu, value) in mu_values.iter().copied().enumerate()
            {
                if !value.is_finite()
                {
                    return Err(NonlocalRelativityError::NonFiniteTransportChristoffel {
                        rho,
                        mu,
                        nu,
                        value,
                    });
                }
            }
        }
    }

    Ok(())
}

fn validate_transported_vector<const D: usize>(
    vector: &[f64; D],
    retained_index: usize,
) -> NonlocalResult<()> {
    for (component, value) in vector.iter().copied().enumerate()
    {
        if !value.is_finite()
        {
            return Err(NonlocalRelativityError::NonFiniteTransportedVector {
                retained_index,
                component,
                value,
            });
        }
    }

    Ok(())
}
