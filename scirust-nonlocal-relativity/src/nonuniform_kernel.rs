//! Shared non-uniform Caputo velocity-memory kernels.
//!
//! These builders evaluate the Caputo L1 velocity-memory vector against each
//! retained sample's own recorded [`crate::HistoryEntry::parameter`] via
//! `scirust_fractional::caputo_l1_nonuniform` (rather than one uniform
//! [`crate::NonlocalConfig::step`] applied to the whole history). They are used
//! by two callers that must produce identical numerics:
//!
//! - the non-uniform [`crate::MemoryLaw`] implementations in
//!   [`crate::nonuniform_memory`] (the fixed-step and step-doubling
//!   architectures), and
//! - the embedded Heun–Euler adaptive controller in [`crate::adaptive`].
//!
//! Both previously carried byte-for-byte copies of this logic; centralising it
//! here keeps the single numerical definition in one place so the two callers
//! cannot drift. The function bodies are unchanged from those copies, so the
//! consolidation is behaviour-preserving (guarded by the crate's bit-identity
//! golden tests).
//!
//! Coupling `kappa` is deliberately *not* applied here; it is applied later by
//! [`crate::projected_memory_force`], never inside a memory builder.

use crate::{
    HistoryBackend, HistoryModulator, HistoryTransport, NonlocalRelativityError, NonlocalResult,
    WorldlineState, validate_history_velocity,
};
use scirust_fractional::{FractionalError, FractionalOrder, caputo_l1_nonuniform};

/// Caputo L1 velocity memory of pre-assembled `velocity_samples` evaluated
/// against their own `parameter_samples`, componentwise. Returns the zero
/// vector when only one sample is available (memory is undefined for a single
/// point), and a typed [`NonlocalRelativityError`] for any non-finite sample or
/// result.
pub(crate) fn nonuniform_caputo_velocity_memory<const D: usize>(
    velocity_samples: &[[f64; D]],
    parameter_samples: &[f64],
    order: FractionalOrder,
    step_index: usize,
) -> NonlocalResult<[f64; D]> {
    if velocity_samples.len() == 1
    {
        return Ok([0.0; D]);
    }

    let mut memory = [0.0_f64; D];
    let mut samples = Vec::with_capacity(velocity_samples.len());

    for (component, memory_component) in memory.iter_mut().enumerate()
    {
        samples.clear();
        for velocity in velocity_samples
        {
            let sample = velocity[component];

            if !sample.is_finite()
            {
                return Err(NonlocalRelativityError::FractionalMemory {
                    step: step_index,
                    component,
                    source: FractionalError::NonFiniteSample(samples.len()),
                });
            }

            samples.push(sample);
        }

        let value = caputo_l1_nonuniform(&samples, parameter_samples, order).map_err(|source| {
            NonlocalRelativityError::FractionalMemory {
                step: step_index,
                component,
                source,
            }
        })?;

        if !value.is_finite()
        {
            return Err(NonlocalRelativityError::NonFiniteMemory {
                step: step_index,
                component,
                value,
            });
        }

        *memory_component = value;
    }

    Ok(memory)
}

/// Non-uniform Caputo velocity memory of a [`HistoryBackend`], applying
/// `transport`'s evaluation-time transport to each retained sample's velocity
/// before the stencil. Reads each sample's own recorded parameter.
pub(crate) fn nonuniform_transported_caputo_velocity_memory<H, T, const D: usize>(
    history: &H,
    transport: &T,
    current_state: &WorldlineState<D>,
    order: FractionalOrder,
    step_index: usize,
) -> NonlocalResult<[f64; D]>
where
    H: HistoryBackend<D>,
    T: HistoryTransport<D>,
{
    let retained_samples = history.retained_samples();

    if retained_samples == 0
    {
        return Err(NonlocalRelativityError::FractionalMemory {
            step: step_index,
            component: 0,
            source: FractionalError::EmptySamples,
        });
    }

    let mut velocity_samples = Vec::with_capacity(retained_samples);
    let mut parameter_samples = Vec::with_capacity(retained_samples);

    for retained_index in 0..retained_samples
    {
        let entry = history
            .entry(retained_index)
            .ok_or(NonlocalRelativityError::HistoryEntryUnavailable { retained_index })?;
        let transported_velocity =
            transport.transport_velocity(retained_index, entry.velocity, current_state)?;
        validate_history_velocity(&transported_velocity, retained_index)?;

        velocity_samples.push(transported_velocity);
        parameter_samples.push(entry.parameter);
    }

    nonuniform_caputo_velocity_memory(&velocity_samples, &parameter_samples, order, step_index)
}

/// Non-uniform Caputo velocity memory of a [`HistoryBackend`], applying
/// `transport`'s evaluation-time transport and then `modulator`'s dimensionless
/// weight to each retained sample's velocity before the stencil. Mirrors the
/// fixed-step architecture's modulated-transported memory with
/// `caputo_l1_nonuniform` in place of `caputo_l1_uniform`.
pub(crate) fn nonuniform_transported_modulated_caputo_velocity_memory<H, T, M, const D: usize>(
    history: &H,
    transport: &T,
    modulator: &M,
    current_state: &WorldlineState<D>,
    order: FractionalOrder,
    step_index: usize,
) -> NonlocalResult<[f64; D]>
where
    H: HistoryBackend<D>,
    T: HistoryTransport<D>,
    M: HistoryModulator<D>,
{
    let retained_samples = history.retained_samples();

    if retained_samples == 0
    {
        return Err(NonlocalRelativityError::FractionalMemory {
            step: step_index,
            component: 0,
            source: FractionalError::EmptySamples,
        });
    }

    let mut velocity_samples = Vec::with_capacity(retained_samples);
    let mut parameter_samples = Vec::with_capacity(retained_samples);

    for retained_index in 0..retained_samples
    {
        let entry = history
            .entry(retained_index)
            .ok_or(NonlocalRelativityError::HistoryEntryUnavailable { retained_index })?;
        let transported_velocity =
            transport.transport_velocity(retained_index, entry.velocity, current_state)?;
        validate_history_velocity(&transported_velocity, retained_index)?;

        let weight = modulator.weight(&entry)?;

        if !weight.is_finite()
        {
            return Err(NonlocalRelativityError::NonFiniteModulationWeight(weight));
        }

        let mut modulated_velocity = [0.0_f64; D];
        for component in 0..D
        {
            modulated_velocity[component] = weight * transported_velocity[component];
        }
        validate_history_velocity(&modulated_velocity, retained_index)?;

        velocity_samples.push(modulated_velocity);
        parameter_samples.push(entry.parameter);
    }

    nonuniform_caputo_velocity_memory(&velocity_samples, &parameter_samples, order, step_index)
}
