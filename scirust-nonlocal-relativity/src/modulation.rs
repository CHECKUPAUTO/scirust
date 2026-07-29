//! Deterministic, dimensionless scalar modulation of retained history
//! vectors, applied componentwise before the Caputo evaluation.
//!
//! A [`HistoryModulator`] answers one narrow question: given one retained,
//! already-transported history sample, what finite dimensionless scalar
//! weight should multiply it before the Caputo L1 stencil runs? It composes
//! with any [`HistoryTransport`], any [`HistoryBackend`], and either
//! fixed-step integrator, because it only changes what number the Caputo
//! stencil consumes at each retained sample, not how that sample's vector
//! was transported or stored.
//!
//! [`SchwarzschildKretschmannModulator`] is an explicitly experimental,
//! phenomenological instance: it does not modify the Einstein field
//! equations, is not a quantum-gravity prediction, and is not an
//! experimentally derived law. It describes a Caputo derivative of a
//! dimensionless *modulated* velocity history, nothing more.

use crate::{
    FractionalError, FractionalOrder, HistoryBackend, HistoryEntry, HistoryTransport, MemoryLaw,
    NonlocalConfig, NonlocalRelativityError, NonlocalResult, WorldlineState,
    caputo_velocity_memory_at_step, validate_history_velocity,
};
use scirust_relativity::ReissnerNordstrom;

/// Deterministic, dimensionless scalar modulation of one retained history
/// entry.
pub trait HistoryModulator<const D: usize>: Clone {
    /// Evaluate a finite, dimensionless scalar weight for `entry`.
    fn weight(&self, entry: &HistoryEntry<D>) -> NonlocalResult<f64>;
}

/// Modulator whose weight is exactly `1.0` for every entry.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IdentityHistoryModulator;

impl<const D: usize> HistoryModulator<D> for IdentityHistoryModulator {
    fn weight(&self, entry: &HistoryEntry<D>) -> NonlocalResult<f64> {
        let _ = entry;
        Ok(1.0)
    }
}

/// Experimental phenomenological modulation of retained history vectors by
/// the Schwarzschild Kretschmann scalar `K = 48 M^2 / r^6`.
///
/// The weight is `q = 1 + beta * L^4 * K`, where `L` is a strictly positive
/// reference length that makes the modulation dimensionless. This describes
/// a Caputo derivative of a dimensionless *modulated* velocity history. It is
/// **never** to be described as:
///
/// - a unique consequence of general relativity;
/// - a quantum-gravity prediction;
/// - an experimentally derived law;
/// - a modification of the Einstein field equations.
///
/// It is a deliberately simple, explicitly phenomenological research hook.
/// When `beta == 0.0`, evaluation bypasses the Kretschmann computation
/// entirely and returns exactly `1.0`, so a pipeline using this modulator
/// reproduces the unmodulated baseline bit-for-bit whenever the rest of the
/// numerical path is identical.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SchwarzschildKretschmannModulator {
    mass: f64,
    reference_length: f64,
    beta: f64,
}

impl SchwarzschildKretschmannModulator {
    /// Validate and construct a Schwarzschild-Kretschmann modulator.
    ///
    /// `mass` and `reference_length` must be finite and strictly positive.
    /// `beta` must be finite and non-negative.
    pub fn try_new(mass: f64, reference_length: f64, beta: f64) -> NonlocalResult<Self> {
        if !mass.is_finite() || mass <= 0.0
        {
            return Err(NonlocalRelativityError::InvalidModulationMass(mass));
        }

        if !reference_length.is_finite() || reference_length <= 0.0
        {
            return Err(NonlocalRelativityError::InvalidModulationReferenceLength(
                reference_length,
            ));
        }

        if !beta.is_finite() || beta < 0.0
        {
            return Err(NonlocalRelativityError::InvalidModulationBeta(beta));
        }

        Ok(Self {
            mass,
            reference_length,
            beta,
        })
    }

    /// Return the geometric mass parameter `M`.
    #[must_use]
    pub const fn mass(&self) -> f64 {
        self.mass
    }

    /// Return the dimensionless-reference length `L`.
    #[must_use]
    pub const fn reference_length(&self) -> f64 {
        self.reference_length
    }

    /// Return the phenomenological coefficient `beta`.
    #[must_use]
    pub const fn beta(&self) -> f64 {
        self.beta
    }

    /// Return the Schwarzschild horizon radius `2 M`.
    #[must_use]
    pub const fn horizon_radius(&self) -> f64 {
        2.0 * self.mass
    }
}

impl HistoryModulator<4> for SchwarzschildKretschmannModulator {
    fn weight(&self, entry: &HistoryEntry<4>) -> NonlocalResult<f64> {
        if self.beta == 0.0
        {
            return Ok(1.0);
        }

        let radius = entry.coordinates[1];

        if !radius.is_finite() || radius <= self.horizon_radius()
        {
            return Err(NonlocalRelativityError::InvalidModulationRadius(radius));
        }

        let kretschmann = 48.0 * self.mass * self.mass / radius.powi(6);

        if !kretschmann.is_finite()
        {
            return Err(NonlocalRelativityError::NonFiniteModulationWeight(
                kretschmann,
            ));
        }

        let weight = 1.0 + self.beta * self.reference_length.powi(4) * kretschmann;

        if !weight.is_finite() || weight <= 0.0
        {
            return Err(NonlocalRelativityError::NonFiniteModulationWeight(weight));
        }

        Ok(weight)
    }
}

/// Experimental phenomenological modulation of retained history vectors by
/// the electromagnetic field invariant `|F_(mu nu) F^(mu nu)| = 2 Q^2 / r^4`
/// of a Reissner-Nordström background's radial Coulomb field.
///
/// Unlike [`SchwarzschildKretschmannModulator`], this is not a *curvature*
/// invariant: it is the invariant of the electromagnetic field that, in the
/// full Einstein-Maxwell theory this crate does not implement, would source
/// the spacetime's curvature. This crate uses the Reissner-Nordström metric
/// purely as fixed, externally specified background data, exactly as it uses
/// Schwarzschild elsewhere; nothing here solves the Einstein or Maxwell
/// equations, or computes the electromagnetic field's backreaction on the
/// metric.
///
/// The weight is `q = 1 + beta * L^2 * |F^2|`, where `L` is a strictly
/// positive reference length that makes the modulation dimensionless. Note
/// the power `L^2`, not `L^4`: in the geometric units this crate uses
/// (charge carries the same dimension as mass, i.e. length), `|F^2| = 2
/// Q^2 / r^4` has dimension `length^-2`, unlike the Kretschmann scalar's
/// `length^-4`, so a different reference-length power is required to make
/// the product dimensionless.
///
/// This describes a Caputo derivative of a dimensionless *modulated*
/// velocity history. It is **never** to be described as:
///
/// - a unique consequence of general relativity or electromagnetism;
/// - a quantum-gravity or quantum-electrodynamics prediction;
/// - an experimentally derived law;
/// - a modification of the Einstein field equations or Maxwell's equations.
///
/// It is a deliberately simple, explicitly phenomenological research hook.
/// When `beta == 0.0`, evaluation bypasses the field-invariant computation
/// entirely and returns exactly `1.0`, so a pipeline using this modulator
/// reproduces the unmodulated baseline bit-for-bit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReissnerNordstromFieldModulator {
    background: ReissnerNordstrom,
    reference_length: f64,
    beta: f64,
}

impl ReissnerNordstromFieldModulator {
    /// Validate and construct a Reissner-Nordström field modulator.
    ///
    /// `reference_length` must be finite and strictly positive. `beta` must
    /// be finite and non-negative.
    pub fn try_new(
        background: ReissnerNordstrom,
        reference_length: f64,
        beta: f64,
    ) -> NonlocalResult<Self> {
        if !reference_length.is_finite() || reference_length <= 0.0
        {
            return Err(NonlocalRelativityError::InvalidModulationReferenceLength(
                reference_length,
            ));
        }

        if !beta.is_finite() || beta < 0.0
        {
            return Err(NonlocalRelativityError::InvalidModulationBeta(beta));
        }

        Ok(Self {
            background,
            reference_length,
            beta,
        })
    }

    /// Return the wrapped Reissner-Nordström background.
    #[must_use]
    pub const fn background(&self) -> &ReissnerNordstrom {
        &self.background
    }

    /// Return the dimensionless-reference length `L`.
    #[must_use]
    pub const fn reference_length(&self) -> f64 {
        self.reference_length
    }

    /// Return the phenomenological coefficient `beta`.
    #[must_use]
    pub const fn beta(&self) -> f64 {
        self.beta
    }
}

impl HistoryModulator<4> for ReissnerNordstromFieldModulator {
    fn weight(&self, entry: &HistoryEntry<4>) -> NonlocalResult<f64> {
        if self.beta == 0.0
        {
            return Ok(1.0);
        }

        let radius = entry.coordinates[1];

        if !radius.is_finite() || radius <= self.background.outer_horizon_radius()
        {
            return Err(NonlocalRelativityError::InvalidModulationRadius(radius));
        }

        let charge = self.background.charge();
        let field_invariant = 2.0 * charge * charge / radius.powi(4);

        if !field_invariant.is_finite()
        {
            return Err(NonlocalRelativityError::NonFiniteModulationWeight(
                field_invariant,
            ));
        }

        let weight = 1.0 + self.beta * self.reference_length.powi(2) * field_invariant;

        if !weight.is_finite() || weight <= 0.0
        {
            return Err(NonlocalRelativityError::NonFiniteModulationWeight(weight));
        }

        Ok(weight)
    }
}

/// Coordinate Caputo L1 velocity-memory law with a [`HistoryModulator`]
/// applied to each retained (and possibly transported) sample before the
/// Caputo evaluation.
///
/// This composes with any [`HistoryBackend`] that supplies
/// [`HistoryBackend::entry`] (both crate backends do, through
/// [`HistoryBackend::push_entry`]), any [`HistoryTransport`], and either
/// fixed-step integrator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModulatedCaputoCoordinateMemory<M> {
    modulator: M,
}

impl<M> ModulatedCaputoCoordinateMemory<M> {
    /// Construct a modulated memory law from a modulator.
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

impl<const D: usize, M> MemoryLaw<D> for ModulatedCaputoCoordinateMemory<M>
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
        modulated_transported_caputo_velocity_memory_at_step(
            history,
            transport,
            &self.modulator,
            current_state,
            config.step(),
            config.fractional_order(),
            step_index,
        )
    }
}

fn modulated_transported_caputo_velocity_memory_at_step<H, T, M, const D: usize>(
    history: &H,
    transport: &T,
    modulator: &M,
    current_state: &WorldlineState<D>,
    step: f64,
    order: FractionalOrder,
    step_index: usize,
) -> NonlocalResult<[f64; D]>
where
    H: HistoryBackend<D>,
    T: HistoryTransport<D>,
    M: HistoryModulator<D>,
{
    if history.retained_samples() == 0
    {
        return Err(NonlocalRelativityError::FractionalMemory {
            step: step_index,
            component: 0,
            source: FractionalError::EmptySamples,
        });
    }

    let retained_samples = history.retained_samples();
    let mut modulated_history = Vec::with_capacity(retained_samples);

    for retained_index in 0..retained_samples
    {
        let velocity = history.sample(retained_index).ok_or(
            NonlocalRelativityError::InconsistentHistoryBackend {
                requested_sample: retained_index,
                retained_samples,
            },
        )?;
        let transported_velocity =
            transport.transport_velocity(retained_index, velocity, current_state)?;
        validate_history_velocity(&transported_velocity, retained_index)?;

        let entry = history
            .entry(retained_index)
            .ok_or(NonlocalRelativityError::HistoryEntryUnavailable { retained_index })?;
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

        modulated_history.push(modulated_velocity);
    }

    caputo_velocity_memory_at_step(&modulated_history, step, order, step_index)
}
