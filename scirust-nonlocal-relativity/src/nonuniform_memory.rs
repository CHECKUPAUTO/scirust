//! [`crate::MemoryLaw`] implementations that evaluate the Caputo L1
//! velocity-memory vector against each retained sample's own recorded
//! [`crate::HistoryEntry::parameter`], via
//! `scirust_fractional::caputo_l1_nonuniform`, instead of a single
//! [`crate::NonlocalConfig::step`] applied to the whole retained history.
//!
//! [`crate::CaputoCoordinateMemory`] and
//! [`crate::ModulatedCaputoCoordinateMemory`] both call
//! `scirust_fractional::caputo_l1_uniform` with one shared `step` value:
//! correct exactly when every accepted segment used that same step (the
//! fixed-step architecture's only use case), and silently wrong the moment
//! segment sizes vary, which adaptive stepping does by construction. Each
//! [`crate::HistoryBackend`] already records every retained sample's own
//! accepted parameter in its [`crate::HistoryEntry::parameter`] field (both
//! crate backends supply it through
//! [`crate::HistoryBackend::push_entry`]/[`crate::HistoryBackend::entry`]);
//! [`NonuniformCaputoCoordinateMemory`] and
//! [`NonuniformModulatedCaputoCoordinateMemory`] read that field directly
//! instead of assuming uniform spacing, so they stay correct under a
//! non-uniform accepted-parameter sequence and compose with
//! [`crate::simulate_nonlocal_worldline_adaptive_with_stepper_policy`]. They
//! also compose with the fixed-step architecture
//! ([`crate::simulate_nonlocal_worldline_with_policy`]): under uniform
//! spacing they produce numerically close results to
//! [`crate::CaputoCoordinateMemory`]/[`crate::ModulatedCaputoCoordinateMemory`].
//! `caputo_l1_nonuniform` and `caputo_l1_uniform` are algebraically
//! equivalent under exactly uniform spacing (each nonuniform term reduces to
//! the matching uniform term under the index substitution `k' = last - 1 -
//! k`), but they reach that value by different floating-point paths — a
//! per-term `(final_time - t_k)`-based weight and a per-term division by the
//! local spacing, versus one weight in step-count units and a single
//! division by `step^alpha` at the end — so whether two runs agree exactly
//! to the bit or only closely is a property of the specific input, not
//! something either type guarantees.
//!
//! The [`crate::MemoryLaw`] trait signature requires a
//! [`crate::NonlocalConfig`] argument; both types here read only
//! [`crate::NonlocalConfig::fractional_order`] from it and ignore
//! [`crate::NonlocalConfig::step`], [`crate::NonlocalConfig::steps`],
//! [`crate::NonlocalConfig::coupling`], and
//! [`crate::NonlocalConfig::metric_norm_floor`] (coupling is applied later,
//! by [`crate::projected_memory_force`], not inside a [`crate::MemoryLaw`]).

use crate::nonuniform_kernel::{
    nonuniform_transported_caputo_velocity_memory,
    nonuniform_transported_modulated_caputo_velocity_memory,
};
use crate::{
    HistoryBackend, HistoryModulator, HistoryTransport, MemoryLaw, NonlocalConfig, NonlocalResult,
    WorldlineState,
};

/// Coordinate Caputo L1 velocity-memory law evaluated against each retained
/// sample's own recorded parameter.
///
/// See the module documentation for how this differs from
/// [`crate::CaputoCoordinateMemory`] and why that difference matters for
/// adaptive stepping.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NonuniformCaputoCoordinateMemory;

impl<const D: usize> MemoryLaw<D> for NonuniformCaputoCoordinateMemory {
    fn memory_vector<H, T>(
        &self,
        history: &H,
        transport: &T,
        current_state: &WorldlineState<D>,
        step_index: usize,
        config: NonlocalConfig,
    ) -> NonlocalResult<[f64; D]>
    where
        H: HistoryBackend<D>,
        T: HistoryTransport<D>,
    {
        nonuniform_transported_caputo_velocity_memory(
            history,
            transport,
            current_state,
            config.fractional_order(),
            step_index,
        )
    }
}

/// [`NonuniformCaputoCoordinateMemory`] with a [`crate::HistoryModulator`]
/// applied to each retained (and possibly transported) sample before the
/// non-uniform Caputo evaluation, mirroring
/// [`crate::ModulatedCaputoCoordinateMemory`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonuniformModulatedCaputoCoordinateMemory<M> {
    modulator: M,
}

impl<M> NonuniformModulatedCaputoCoordinateMemory<M> {
    /// Construct a non-uniform modulated memory law from a modulator.
    #[must_use]
    pub const fn new(modulator: M) -> Self {
        Self { modulator }
    }

    /// Borrow the wrapped modulator.
    #[must_use]
    pub const fn modulator(&self) -> &M {
        &self.modulator
    }
}

impl<const D: usize, M> MemoryLaw<D> for NonuniformModulatedCaputoCoordinateMemory<M>
where
    M: HistoryModulator<D>,
{
    fn memory_vector<H, T>(
        &self,
        history: &H,
        transport: &T,
        current_state: &WorldlineState<D>,
        step_index: usize,
        config: NonlocalConfig,
    ) -> NonlocalResult<[f64; D]>
    where
        H: HistoryBackend<D>,
        T: HistoryTransport<D>,
    {
        nonuniform_transported_modulated_caputo_velocity_memory(
            history,
            transport,
            &self.modulator,
            current_state,
            config.fractional_order(),
            step_index,
        )
    }
}
