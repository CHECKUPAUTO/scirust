//! # `scirust-nonlocal-relativity` - EXPERIMENTAL worldline memory
//!
//! This crate is an explicitly **EXPERIMENTAL** research layer for
//! fractional-memory modifications of test-particle worldline dynamics on a
//! fixed general-relativistic background.
//!
//! The model is intentionally narrow:
//!
//! - the supplied spacetime is a fixed background implementing
//!   [`Metric`] and [`Connection`];
//! - only test-particle coordinates and velocities are evolved;
//! - no fractional Einstein equations are implemented;
//! - no Einstein tensor, stress-energy tensor, matter-generated curvature, or
//!   established general-relativistic field equation is modified;
//! - no empirical validation is claimed.
//!
//! The discretization is coordinate-dependent. All quantities use the
//! coordinate system and geometric units of the supplied background. The
//! default implementation keeps complete velocity history and recomputes the
//! Caputo L1 derivative directly, giving `O(D * N^2)` history cost over `N`
//! fixed steps. Phase 2 exposes memory law, history backend, history
//! transport, and ordinary state stepper components separately. The explicit
//! bounded short-memory backend is an approximation and is never selected
//! automatically. The semi-implicit Euler update is a deterministic reference
//! integrator, not a precision integrator:
//!
//! `u_(n+1) = u_n + h a_n`, then `x_(n+1) = x_n + h u_(n+1)`.
//!
//! Positive `kappa` is a finite non-negative phenomenological damping-like
//! coupling, not a new fundamental constant. At the first sample, where the
//! Caputo history is insufficient, the memory vector is defined to be zero.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use scirust_fractional::{FractionalError, FractionalOrder, caputo_l1_uniform};
use scirust_relativity::{Connection, Metric, Schwarzschild};
use std::error::Error;
use std::fmt;
use std::str::FromStr;

mod adaptive;
mod adaptive_control;
mod adaptive_stepper;
mod charts;
mod curved_transport;
mod geometric_error;
mod modulation;
mod nonuniform_kernel;
mod nonuniform_memory;
mod proper_time;
mod transport;

pub use adaptive::{
    AdaptiveNonlocalConfig, AdaptiveSimulationPolicy, simulate_nonlocal_worldline_adaptive,
    simulate_nonlocal_worldline_adaptive_with_policy,
};
pub use adaptive_control::{AdaptiveTolerance, scaled_local_error_norm};
pub use adaptive_stepper::{
    AdaptiveStepperPolicy, HistoryRetention, simulate_nonlocal_worldline_adaptive_with_stepper,
    simulate_nonlocal_worldline_adaptive_with_stepper_policy,
    simulate_nonlocal_worldline_adaptive_with_stepper_policy_retention,
};
pub use charts::{
    CylindricalMinkowski, cartesian_to_cylindrical_coordinates, cartesian_to_cylindrical_velocity,
    cylindrical_to_cartesian_coordinates, cylindrical_to_cartesian_velocity,
    exact_cylindrical_minkowski_transport,
};
pub use curved_transport::{
    exact_schwarzschild_circular_orbit_transport, schwarzschild_circular_orbit_angular_velocity,
    schwarzschild_circular_orbit_four_velocity,
};
pub use geometric_error::{
    OrthonormalTetrad, TetradStateError, TimelikeStateError, build_orthonormal_tetrad,
    tetrad_state_error, timelike_state_error,
};
pub use modulation::{
    HistoryModulator, IdentityHistoryModulator, ModulatedCaputoCoordinateMemory,
    ReissnerNordstromFieldModulator, SchwarzschildKretschmannModulator,
};
pub use nonuniform_memory::{
    NonuniformCaputoCoordinateMemory, NonuniformModulatedCaputoCoordinateMemory,
};
pub use proper_time::{
    ParameterizationMode, ProperTimeDiagnostics, affine_trajectory_proper_time,
    proper_time_caputo_velocity_memory, simulate_nonlocal_worldline_with_mode,
};
pub use transport::{DiscreteConnectionTransport, HistoryEntry, transport_vector_along_polyline};

/// Result type used by this experimental crate.
pub type NonlocalResult<T> = Result<T, NonlocalRelativityError>;

/// Fixed-step integrator used for the ordinary worldline state equation.
///
/// The fractional Caputo term is a history-dependent force on the right-hand
/// side. These variants select how the ordinary first-order state equation is
/// advanced once that force has been evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorldlineIntegrator {
    /// Compatibility path: evaluate the accepted state once and advance with
    /// semi-implicit Euler.
    SemiImplicitEuler,
    /// Predictor-evaluate-correct-evaluate Heun method for the existing
    /// ordinary state equation with fractional-history force.
    HeunPece,
}

impl WorldlineIntegrator {
    /// Return the stable lowercase identifier for this integrator.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self
        {
            Self::SemiImplicitEuler => "semi_implicit_euler",
            Self::HeunPece => "heun_pece",
        }
    }

    /// Parse an integrator identifier.
    ///
    /// Accepted identifiers are `semi_implicit_euler`, `euler`,
    /// `heun_pece`, `heun`, `pece`, and
    /// `predict_evaluate_correct_evaluate`.
    pub fn try_from_name(name: &str) -> NonlocalResult<Self> {
        match name.to_ascii_lowercase().as_str()
        {
            "semi_implicit_euler" | "euler" => Ok(Self::SemiImplicitEuler),
            "heun_pece" | "heun" | "pece" | "predict_evaluate_correct_evaluate" =>
            {
                Ok(Self::HeunPece)
            },
            _ => Err(NonlocalRelativityError::InvalidIntegratorConfiguration {
                name: name.to_string(),
            }),
        }
    }
}

impl fmt::Display for WorldlineIntegrator {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for WorldlineIntegrator {
    type Err = NonlocalRelativityError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::try_from_name(value)
    }
}

/// Whether a history backend evaluates the complete history or an explicit
/// approximation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryApproximation {
    /// Complete retained history is used; this is the numerical oracle.
    Exact,
    /// A deliberately truncated or otherwise approximate retained history is
    /// used.
    Approximate,
}

/// History accounting recorded at one sampled affine parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HistoryDiagnostics {
    /// Number of velocity samples retained by the backend at this evaluation.
    pub retained_samples: usize,
    /// Number of retained velocity samples used by the memory law.
    pub used_samples: usize,
    /// Exact-or-approximate classification of the backend.
    pub approximation: HistoryApproximation,
}

impl HistoryDiagnostics {
    /// Construct history accounting from validated counts.
    #[must_use]
    pub const fn new(
        retained_samples: usize,
        used_samples: usize,
        approximation: HistoryApproximation,
    ) -> Self {
        Self {
            retained_samples,
            used_samples,
            approximation,
        }
    }
}

/// Storage contract for deterministic velocity history.
///
/// Implementations own their invariants. Samples are returned by value so
/// callers cannot mutate retained history without going through
/// [`HistoryBackend::push_velocity`].
pub trait HistoryBackend<const D: usize>: Clone {
    /// Append one contravariant coordinate-velocity sample.
    fn push_velocity(&mut self, velocity: [f64; D]) -> NonlocalResult<()>;

    /// Number of samples currently retained by the backend.
    fn retained_samples(&self) -> usize;

    /// Number of retained samples the memory law should use.
    fn used_samples(&self) -> usize;

    /// Return one retained sample by value, indexed from oldest retained
    /// sample to newest retained sample.
    fn sample(&self, retained_index: usize) -> Option<[f64; D]>;

    /// Exact-or-approximate classification of this backend.
    fn approximation(&self) -> HistoryApproximation;

    /// Return the current history accounting.
    #[must_use]
    fn diagnostics(&self) -> HistoryDiagnostics {
        HistoryDiagnostics::new(
            self.retained_samples(),
            self.used_samples(),
            self.approximation(),
        )
    }

    /// Accept one new [`HistoryEntry`], transporting all currently retained
    /// vectors across the newly accepted segment before appending the new
    /// sample.
    ///
    /// This is the entry point a [`HistoryTransport`] with real geometric
    /// behavior relies on: it is called once per accepted segment (including
    /// once per provisional predictor evaluation), with the source
    /// coordinates and parameter the original [`HistoryBackend::push_velocity`]
    /// contract does not carry.
    ///
    /// The default implementation performs no geometric transport: it stores
    /// only `entry.velocity` via [`HistoryBackend::push_velocity`], which
    /// preserves the original coordinate-memory contract for backends that do
    /// not override this method.
    fn push_entry<B, T>(
        &mut self,
        background: &B,
        transport: &T,
        entry: HistoryEntry<D>,
    ) -> NonlocalResult<()>
    where
        B: Connection<D>,
        T: HistoryTransport<D>,
    {
        let _ = (background, transport);
        self.push_velocity(entry.velocity)
    }

    /// Retrieve one retained sample as a full entry, when the backend records
    /// source coordinates and a parameter value.
    ///
    /// The default implementation returns `None`: a backend that does not
    /// override this method cannot supply the source-point information a
    /// geometric [`HistoryTransport`] needs, so it must not claim to.
    fn entry(&self, retained_index: usize) -> Option<HistoryEntry<D>> {
        let _ = retained_index;
        None
    }
}

/// Complete uniform velocity-history backend.
///
/// This backend retains every accepted velocity sample and is the default
/// numerical memory oracle.
#[derive(Debug, Clone, PartialEq)]
pub struct CompleteUniformHistory<const D: usize> {
    entries: Vec<HistoryEntry<D>>,
}

impl<const D: usize> CompleteUniformHistory<D> {
    /// Construct an empty complete-history backend.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Construct an empty complete-history backend with reserved capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
        }
    }
}

impl<const D: usize> Default for CompleteUniformHistory<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize> HistoryBackend<D> for CompleteUniformHistory<D> {
    fn push_velocity(&mut self, velocity: [f64; D]) -> NonlocalResult<()> {
        validate_history_velocity(&velocity, self.entries.len())?;
        self.entries
            .push(HistoryEntry::from_velocity_only(velocity));
        Ok(())
    }

    fn retained_samples(&self) -> usize {
        self.entries.len()
    }

    fn used_samples(&self) -> usize {
        self.entries.len()
    }

    fn sample(&self, retained_index: usize) -> Option<[f64; D]> {
        self.entries.get(retained_index).map(|entry| entry.velocity)
    }

    fn approximation(&self) -> HistoryApproximation {
        HistoryApproximation::Exact
    }

    fn push_entry<B, T>(
        &mut self,
        background: &B,
        transport: &T,
        entry: HistoryEntry<D>,
    ) -> NonlocalResult<()>
    where
        B: Connection<D>,
        T: HistoryTransport<D>,
    {
        validate_history_velocity(&entry.coordinates, self.entries.len())?;
        validate_history_velocity(&entry.velocity, self.entries.len())?;
        transport::transport_retained_entries(&mut self.entries, background, transport, &entry)?;
        self.entries.push(entry);
        Ok(())
    }

    fn entry(&self, retained_index: usize) -> Option<HistoryEntry<D>> {
        self.entries.get(retained_index).copied()
    }
}

/// Explicit bounded short-memory velocity-history backend.
///
/// This backend retains only the most recent `window_samples` accepted
/// velocity samples. It is an approximation and is never selected by default.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundedShortMemoryHistory<const D: usize> {
    window_samples: usize,
    entries: Vec<HistoryEntry<D>>,
    total_samples_seen: usize,
}

impl<const D: usize> BoundedShortMemoryHistory<D> {
    /// Construct a bounded short-memory backend.
    ///
    /// `window_samples` is a sample count and must be at least two so the
    /// Caputo L1 stencil can form at least one difference once the window is
    /// populated.
    pub fn new(window_samples: usize) -> NonlocalResult<Self> {
        if window_samples < 2
        {
            return Err(NonlocalRelativityError::InvalidHistoryWindow { window_samples });
        }

        Ok(Self {
            window_samples,
            entries: Vec::with_capacity(window_samples),
            total_samples_seen: 0,
        })
    }

    /// Return the configured maximum retained sample count.
    #[must_use]
    pub const fn window_samples(&self) -> usize {
        self.window_samples
    }

    /// Return the number of samples accepted by this backend instance.
    #[must_use]
    pub const fn total_samples_seen(&self) -> usize {
        self.total_samples_seen
    }
}

impl<const D: usize> HistoryBackend<D> for BoundedShortMemoryHistory<D> {
    fn push_velocity(&mut self, velocity: [f64; D]) -> NonlocalResult<()> {
        validate_history_velocity(&velocity, self.total_samples_seen)?;

        if self.entries.len() == self.window_samples
        {
            self.entries.remove(0);
        }

        self.entries
            .push(HistoryEntry::from_velocity_only(velocity));
        self.total_samples_seen += 1;
        Ok(())
    }

    fn retained_samples(&self) -> usize {
        self.entries.len()
    }

    fn used_samples(&self) -> usize {
        self.entries.len()
    }

    fn sample(&self, retained_index: usize) -> Option<[f64; D]> {
        self.entries.get(retained_index).map(|entry| entry.velocity)
    }

    fn approximation(&self) -> HistoryApproximation {
        HistoryApproximation::Approximate
    }

    fn push_entry<B, T>(
        &mut self,
        background: &B,
        transport: &T,
        entry: HistoryEntry<D>,
    ) -> NonlocalResult<()>
    where
        B: Connection<D>,
        T: HistoryTransport<D>,
    {
        validate_history_velocity(&entry.coordinates, self.total_samples_seen)?;
        validate_history_velocity(&entry.velocity, self.total_samples_seen)?;

        if self.entries.len() == self.window_samples
        {
            self.entries.remove(0);
        }

        transport::transport_retained_entries(&mut self.entries, background, transport, &entry)?;
        self.entries.push(entry);
        self.total_samples_seen += 1;
        Ok(())
    }

    fn entry(&self, retained_index: usize) -> Option<HistoryEntry<D>> {
        self.entries.get(retained_index).copied()
    }
}

/// Transport contract for historical velocity samples.
///
/// Transport is separated from the memory law so future experiments can study
/// transported histories without changing the Caputo discretization or the
/// history storage backend. The current production implementation is
/// coordinate identity transport.
pub trait HistoryTransport<const D: usize>: Clone {
    /// Transport one retained velocity sample into the current state frame.
    fn transport_velocity(
        &self,
        retained_index: usize,
        velocity: [f64; D],
        current_state: &WorldlineState<D>,
    ) -> NonlocalResult<[f64; D]>;

    /// Transport one already-retained vector across one newly accepted
    /// segment, from `from_state` to `to_state`, under the fixed background
    /// connection `background`.
    ///
    /// [`HistoryBackend::push_entry`] calls this once per accepted segment,
    /// for every vector currently retained by the backend, before appending
    /// the newly accepted sample. It is a single discrete transport step
    /// between two consecutive accepted (or provisional) worldline states; it
    /// does not re-derive transport from a sample's original recorded source
    /// point, which is why repeated calls across accepted segments accumulate
    /// path-dependent transport along the worldline instead of jumping
    /// directly between distant points.
    ///
    /// The default implementation performs no geometric transport: it returns
    /// `vector` unchanged, exactly like the coordinate-memory contract used by
    /// [`transport_velocity`](HistoryTransport::transport_velocity).
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
        let _ = (
            retained_index,
            background,
            from_state,
            to_state,
            segment_step,
        );
        Ok(vector)
    }
}

/// Coordinate identity transport for retained velocity history.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IdentityHistoryTransport;

impl<const D: usize> HistoryTransport<D> for IdentityHistoryTransport {
    fn transport_velocity(
        &self,
        retained_index: usize,
        velocity: [f64; D],
        _current_state: &WorldlineState<D>,
    ) -> NonlocalResult<[f64; D]> {
        validate_history_velocity(&velocity, retained_index)?;
        Ok(velocity)
    }
}

/// Memory-law contract for converting retained history into a memory vector.
pub trait MemoryLaw<const D: usize>: Clone {
    /// Evaluate the coordinate memory vector at the current state.
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
        T: HistoryTransport<D>;
}

/// Current coordinate Caputo L1 velocity-memory law.
///
/// The state equation remains ordinary in affine parameter; this law evaluates
/// a coordinate-history vector that is used as a force term on the right-hand
/// side.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CaputoCoordinateMemory;

impl<const D: usize> MemoryLaw<D> for CaputoCoordinateMemory {
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
        transported_caputo_velocity_memory_at_step(
            history,
            transport,
            current_state,
            config.step,
            config.fractional_order,
            step_index,
        )
    }
}

/// Fixed-step state advancement contract for the ordinary worldline equation.
pub trait WorldlineStepper<const D: usize>: Clone {
    /// Advance one step from an accepted state and accepted acceleration.
    fn advance<B, H, L, T>(
        &self,
        context: StepperContext<'_, B, H, L, T, D>,
    ) -> NonlocalResult<WorldlineState<D>>
    where
        B: Metric<D> + Connection<D>,
        H: HistoryBackend<D>,
        L: MemoryLaw<D>,
        T: HistoryTransport<D>;
}

/// Inputs supplied to a [`WorldlineStepper`] for one accepted step.
pub struct StepperContext<'a, B, H, L, T, const D: usize> {
    /// Fixed background providing metric and connection data.
    pub background: &'a B,
    /// Current accepted worldline state.
    pub state: &'a WorldlineState<D>,
    /// Total acceleration evaluated at the current accepted state.
    pub accepted_acceleration: &'a [f64; D],
    /// Accepted velocity-history backend before the next state is appended.
    pub history: &'a H,
    /// Memory law used for provisional predictor evaluations.
    pub memory_law: &'a L,
    /// History transport used for provisional predictor evaluations.
    pub transport: &'a T,
    /// Metric norm at the initial sample.
    pub initial_metric_norm: f64,
    /// True accumulated affine parameter at the current accepted `state`.
    ///
    /// A stepper that needs the provisional (post-step) affine parameter must
    /// compute it as `current_parameter + config.step`, **not** by
    /// reconstructing it from `step_index`, which is only valid under uniform
    /// spacing. This lets a stepper participate soundly in an adaptive loop
    /// where accepted step sizes vary.
    pub current_parameter: f64,
    /// Current accepted step index.
    ///
    /// Retained for diagnostics and error reporting only; it must not be used
    /// to reconstruct an absolute affine parameter (see `current_parameter`).
    pub step_index: usize,
    /// Validated nonlocal simulation configuration.
    pub config: NonlocalConfig,
}

/// Semi-implicit Euler stepper for the ordinary state equation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SemiImplicitEulerStepper;

impl<const D: usize> WorldlineStepper<D> for SemiImplicitEulerStepper {
    fn advance<B, H, L, T>(
        &self,
        context: StepperContext<'_, B, H, L, T, D>,
    ) -> NonlocalResult<WorldlineState<D>>
    where
        B: Metric<D> + Connection<D>,
        H: HistoryBackend<D>,
        L: MemoryLaw<D>,
        T: HistoryTransport<D>,
    {
        semi_implicit_euler_step(
            context.state,
            context.accepted_acceleration,
            context.config.step,
            context.step_index,
        )
    }
}

/// Heun predictor-evaluate-correct-evaluate stepper for the ordinary state
/// equation with a history-dependent force.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HeunPeceStepper;

impl<const D: usize> WorldlineStepper<D> for HeunPeceStepper {
    fn advance<B, H, L, T>(
        &self,
        context: StepperContext<'_, B, H, L, T, D>,
    ) -> NonlocalResult<WorldlineState<D>>
    where
        B: Metric<D> + Connection<D>,
        H: HistoryBackend<D>,
        L: MemoryLaw<D>,
        T: HistoryTransport<D>,
    {
        let mut predicted_velocity = [0.0_f64; D];
        let mut predicted_coordinates = [0.0_f64; D];

        for rho in 0..D
        {
            predicted_velocity[rho] = context.state.velocity[rho]
                + context.config.step * context.accepted_acceleration[rho];
            validate_generated_velocity(predicted_velocity[rho], context.step_index + 1, rho)?;

            predicted_coordinates[rho] =
                context.state.coordinates[rho] + context.config.step * predicted_velocity[rho];
            validate_generated_coordinate(predicted_coordinates[rho], context.step_index + 1, rho)?;
        }

        let predicted_state = WorldlineState::new(predicted_coordinates, predicted_velocity);
        // The provisional point is one trial step past the current accepted
        // parameter. This is exact for both uniform and non-uniform accepted
        // spacing, unlike the previous `step_index * step` reconstruction,
        // which assumed every accepted step shared one size.
        let predicted_parameter = context.current_parameter + context.config.step;
        let mut provisional_history = context.history.clone();
        provisional_history.push_entry(
            context.background,
            context.transport,
            HistoryEntry::new(
                predicted_coordinates,
                predicted_velocity,
                predicted_parameter,
            ),
        )?;
        let predicted_evaluation = evaluate_step_with_policy(StepEvaluationInput {
            background: context.background,
            state: &predicted_state,
            history: &provisional_history,
            memory_law: context.memory_law,
            transport: context.transport,
            initial_metric_norm: context.initial_metric_norm,
            affine_parameter: predicted_parameter,
            step_index: context.step_index + 1,
            config: context.config,
        })?;

        let mut next_velocity = [0.0_f64; D];
        let mut next_coordinates = [0.0_f64; D];

        for rho in 0..D
        {
            next_velocity[rho] = context.state.velocity[rho]
                + 0.5
                    * context.config.step
                    * (context.accepted_acceleration[rho] + predicted_evaluation.acceleration[rho]);
            validate_generated_velocity(next_velocity[rho], context.step_index + 1, rho)?;

            next_coordinates[rho] = context.state.coordinates[rho]
                + 0.5 * context.config.step * (context.state.velocity[rho] + next_velocity[rho]);
            validate_generated_coordinate(next_coordinates[rho], context.step_index + 1, rho)?;
        }

        Ok(WorldlineState::new(next_coordinates, next_velocity))
    }
}

/// Typed advanced simulation policy.
///
/// The policy owns one history backend and immutable algorithm components. It
/// is consumed by the simulation call, making short-memory use explicit and
/// preventing hidden global history state.
#[derive(Debug, Clone, PartialEq)]
pub struct NonlocalSimulationPolicy<const D: usize, H, L, T, S> {
    history_backend: H,
    memory_law: L,
    history_transport: T,
    stepper: S,
}

impl<const D: usize, H, L, T, S> NonlocalSimulationPolicy<D, H, L, T, S> {
    /// Construct a policy from explicit architecture components.
    #[must_use]
    pub const fn new(history_backend: H, memory_law: L, history_transport: T, stepper: S) -> Self {
        Self {
            history_backend,
            memory_law,
            history_transport,
            stepper,
        }
    }

    /// Borrow the policy history backend.
    #[must_use]
    pub const fn history_backend(&self) -> &H {
        &self.history_backend
    }

    /// Borrow the policy memory law.
    #[must_use]
    pub const fn memory_law(&self) -> &L {
        &self.memory_law
    }

    /// Borrow the policy history transport.
    #[must_use]
    pub const fn history_transport(&self) -> &T {
        &self.history_transport
    }

    /// Borrow the policy stepper.
    #[must_use]
    pub const fn stepper(&self) -> &S {
        &self.stepper
    }
}

/// Default exact semi-implicit-Euler policy used by the compatibility API.
pub type DefaultNonlocalSimulationPolicy<const D: usize> = NonlocalSimulationPolicy<
    D,
    CompleteUniformHistory<D>,
    CaputoCoordinateMemory,
    IdentityHistoryTransport,
    SemiImplicitEulerStepper,
>;

impl<const D: usize> Default for DefaultNonlocalSimulationPolicy<D> {
    fn default() -> Self {
        Self::new(
            CompleteUniformHistory::new(),
            CaputoCoordinateMemory,
            IdentityHistoryTransport,
            SemiImplicitEulerStepper,
        )
    }
}

/// Configuration for the fixed-step fractional-memory worldline integrator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonlocalConfig {
    fractional_order: FractionalOrder,
    coupling: f64,
    step: f64,
    steps: usize,
    metric_norm_floor: f64,
}

impl NonlocalConfig {
    /// Validate and construct a non-local worldline configuration.
    ///
    /// The fractional order must satisfy the first-release
    /// `scirust-fractional` interval `0 < alpha < 1`. The coupling is finite
    /// and non-negative. The step and metric-norm floor are finite and
    /// strictly positive. At least one integration step is required.
    pub fn new(
        alpha: f64,
        coupling: f64,
        step: f64,
        steps: usize,
        metric_norm_floor: f64,
    ) -> NonlocalResult<Self> {
        let fractional_order = FractionalOrder::new(alpha)
            .map_err(|_| NonlocalRelativityError::InvalidFractionalOrder(alpha))?;

        Self::from_fractional_order(fractional_order, coupling, step, steps, metric_norm_floor)
    }

    /// Construct a configuration from an already validated fractional order.
    pub fn from_fractional_order(
        fractional_order: FractionalOrder,
        coupling: f64,
        step: f64,
        steps: usize,
        metric_norm_floor: f64,
    ) -> NonlocalResult<Self> {
        if !coupling.is_finite() || coupling < 0.0
        {
            return Err(NonlocalRelativityError::InvalidCoupling(coupling));
        }

        if !step.is_finite() || step <= 0.0
        {
            return Err(NonlocalRelativityError::InvalidStep(step));
        }

        if steps == 0
        {
            return Err(NonlocalRelativityError::InvalidStepCount(steps));
        }

        if !metric_norm_floor.is_finite() || metric_norm_floor <= 0.0
        {
            return Err(NonlocalRelativityError::InvalidMetricNormFloor(
                metric_norm_floor,
            ));
        }

        Ok(Self {
            fractional_order,
            coupling,
            step,
            steps,
            metric_norm_floor,
        })
    }

    /// Return the validated fractional order.
    #[must_use]
    pub const fn fractional_order(self) -> FractionalOrder {
        self.fractional_order
    }

    /// Return the phenomenological memory coupling `kappa`.
    #[must_use]
    pub const fn coupling(self) -> f64 {
        self.coupling
    }

    /// Return the uniform affine-parameter step.
    #[must_use]
    pub const fn step(self) -> f64 {
        self.step
    }

    /// Return the number of fixed integration steps.
    #[must_use]
    pub const fn steps(self) -> usize {
        self.steps
    }

    /// Return the positive lower bound for `|g_(mu nu) u^mu u^nu|`.
    #[must_use]
    pub const fn metric_norm_floor(self) -> f64 {
        self.metric_norm_floor
    }
}

/// Coordinate and velocity sample for a worldline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorldlineState<const D: usize> {
    /// Coordinates `x^rho` in the supplied background chart.
    pub coordinates: [f64; D],
    /// Contravariant coordinate velocity `u^rho = dx^rho / d lambda`.
    pub velocity: [f64; D],
}

impl<const D: usize> WorldlineState<D> {
    /// Construct a worldline state from coordinates and velocity.
    #[must_use]
    pub const fn new(coordinates: [f64; D], velocity: [f64; D]) -> Self {
        Self {
            coordinates,
            velocity,
        }
    }
}

/// Diagnostics evaluated at a sampled affine parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepDiagnostics {
    /// Affine parameter value `lambda_n`.
    pub affine_parameter: f64,
    /// Metric norm `g_(mu nu) u^mu u^nu`.
    pub metric_norm: f64,
    /// Difference between the current and initial metric norm.
    pub metric_norm_drift: f64,
    /// Coordinate L2 norm of the Caputo velocity-memory vector.
    pub memory_l2_norm: f64,
    /// Coordinate L2 norm of the projected memory force.
    pub memory_force_l2_norm: f64,
    /// Residual `u_rho F_memory^rho`; numerically near zero for non-null states.
    pub orthogonality_residual: f64,
    /// Coordinate L2 norm of the ordinary general-relativistic acceleration.
    pub gr_acceleration_l2_norm: f64,
}

/// Output trajectory from the stateful non-local worldline integrator.
#[derive(Debug, Clone, PartialEq)]
pub struct NonlocalTrajectory<const D: usize> {
    states: Vec<WorldlineState<D>>,
    diagnostics: Vec<StepDiagnostics>,
    history_diagnostics: Vec<HistoryDiagnostics>,
}

impl<const D: usize> NonlocalTrajectory<D> {
    pub(crate) fn new(
        states: Vec<WorldlineState<D>>,
        diagnostics: Vec<StepDiagnostics>,
        history_diagnostics: Vec<HistoryDiagnostics>,
    ) -> Self {
        Self {
            states,
            diagnostics,
            history_diagnostics,
        }
    }

    /// Borrow all sampled states, including the initial state.
    #[must_use]
    pub fn states(&self) -> &[WorldlineState<D>] {
        &self.states
    }

    /// Borrow all sampled diagnostics, including the initial sample.
    #[must_use]
    pub fn diagnostics(&self) -> &[StepDiagnostics] {
        &self.diagnostics
    }

    /// Borrow all history-backend diagnostics, including the initial sample.
    #[must_use]
    pub fn history_diagnostics(&self) -> &[HistoryDiagnostics] {
        &self.history_diagnostics
    }

    /// Return the number of sampled states.
    #[must_use]
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Return true when the trajectory contains no sampled states.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Borrow the final sampled state.
    #[must_use]
    pub fn final_state(&self) -> Option<&WorldlineState<D>> {
        self.states.last()
    }

    /// Borrow the final sampled diagnostics.
    #[must_use]
    pub fn final_diagnostics(&self) -> Option<&StepDiagnostics> {
        self.diagnostics.last()
    }
}

/// Endpoint summary for one refinement in a deterministic convergence study.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RefinementEndpoint<const D: usize> {
    /// Uniform affine-parameter step used by this run.
    pub step: f64,
    /// Number of accepted integration steps.
    pub steps: usize,
    /// Final affine parameter reached by this run.
    pub final_affine_parameter: f64,
    /// Endpoint coordinates in the supplied chart.
    pub coordinates: [f64; D],
    /// Endpoint contravariant coordinate velocity.
    pub velocity: [f64; D],
    /// Endpoint metric-norm drift from the initial sample.
    pub metric_norm_drift: f64,
    /// Endpoint coordinate L2 norm of the projected memory force.
    pub memory_force_l2_norm: f64,
}

/// Self-convergence result comparing equal final affine parameter at
/// `h`, `h/2`, and `h/4`.
///
/// The finest run is a refinement reference for this numerical study, not an
/// exact solution of the continuous model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConvergenceStudy<const D: usize> {
    /// Integrator used for all three refinements.
    pub integrator: WorldlineIntegrator,
    /// Coarse endpoint at step `h`.
    pub coarse: RefinementEndpoint<D>,
    /// Fine endpoint at step `h/2`.
    pub fine: RefinementEndpoint<D>,
    /// Finest endpoint at step `h/4`.
    pub finest: RefinementEndpoint<D>,
    /// Coordinate endpoint L2 difference between `h` and `h/2`.
    pub endpoint_coordinate_error_h_h2: f64,
    /// Coordinate endpoint L2 difference between `h/2` and `h/4`.
    pub endpoint_coordinate_error_h2_h4: f64,
    /// Velocity endpoint L2 difference between `h` and `h/2`.
    pub endpoint_velocity_error_h_h2: f64,
    /// Velocity endpoint L2 difference between `h/2` and `h/4`.
    pub endpoint_velocity_error_h2_h4: f64,
    /// Observed coordinate self-convergence ratio, when the denominator is
    /// non-zero and finite.
    pub coordinate_self_convergence_ratio: Option<f64>,
    /// Observed velocity self-convergence ratio, when the denominator is
    /// non-zero and finite.
    pub velocity_self_convergence_ratio: Option<f64>,
}

/// Schwarzschild exterior chart diagnostics for one worldline state.
///
/// These quantities are specific to standard Schwarzschild coordinates
/// `(t, r, theta, phi)` in a fixed background with signature `(-,+,+,+)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SchwarzschildInvariants {
    /// Specific energy `E = -u_t` associated with the stationary Killing
    /// coordinate.
    pub specific_energy: f64,
    /// Azimuthal angular momentum `L_z = u_phi`.
    pub azimuthal_angular_momentum: f64,
    /// Metric norm `g_(mu nu) u^mu u^nu`.
    pub metric_norm: f64,
}

/// Errors reported by the experimental non-local worldline integrator.
#[derive(Debug, Clone, PartialEq)]
pub enum NonlocalRelativityError {
    /// Fractional order is non-finite or outside the supported interval.
    InvalidFractionalOrder(f64),

    /// Coupling `kappa` is negative or non-finite.
    InvalidCoupling(f64),

    /// Uniform step is non-finite or non-positive.
    InvalidStep(f64),

    /// The configured number of integration steps is invalid.
    InvalidStepCount(usize),

    /// Metric-norm floor is non-finite or non-positive.
    InvalidMetricNormFloor(f64),

    /// Short-memory history window is too small.
    InvalidHistoryWindow {
        /// Requested maximum retained sample count.
        window_samples: usize,
    },

    /// Integrator selection is not one of the supported fixed-step methods.
    InvalidIntegratorConfiguration {
        /// Rejected integrator identifier.
        name: String,
    },

    /// Refinement would overflow the number of integration steps.
    RefinementStepCountOverflow {
        /// Base number of steps.
        base_steps: usize,
        /// Integer refinement factor.
        factor: usize,
    },

    /// Coordinates are outside the standard Schwarzschild exterior chart.
    InvalidSchwarzschildExteriorCoordinates {
        /// Radial coordinate.
        radius: f64,
        /// Polar angle coordinate.
        polar_angle: f64,
        /// Horizon radius `2 M`.
        horizon_radius: f64,
    },

    /// Initial coordinate component is not finite.
    NonFiniteInitialCoordinate {
        /// Coordinate component index.
        index: usize,
        /// Invalid component value.
        value: f64,
    },

    /// Initial velocity component is not finite.
    NonFiniteInitialVelocity {
        /// Velocity component index.
        index: usize,
        /// Invalid component value.
        value: f64,
    },

    /// Retained history sample component is not finite.
    NonFiniteHistorySample {
        /// Retained or accepted sample index.
        sample: usize,
        /// Coordinate component index.
        component: usize,
        /// Invalid component value.
        value: f64,
    },

    /// History backend reported a sample count that it could not supply.
    InconsistentHistoryBackend {
        /// Requested retained sample index.
        requested_sample: usize,
        /// Reported retained sample count.
        retained_samples: usize,
    },

    /// A discrete-transport segment step is not finite.
    InvalidTransportSegmentStep(f64),

    /// A transport polyline has no waypoints.
    EmptyTransportPolyline,

    /// A Christoffel symbol evaluated during discrete parallel transport is
    /// not finite.
    NonFiniteTransportChristoffel {
        /// Contravariant index.
        rho: usize,
        /// First lower index.
        mu: usize,
        /// Second lower index.
        nu: usize,
        /// Invalid symbol value.
        value: f64,
    },

    /// A transported history-vector component is not finite.
    NonFiniteTransportedVector {
        /// Retained sample index being transported.
        retained_index: usize,
        /// Vector component index.
        component: usize,
        /// Invalid component value.
        value: f64,
    },

    /// Proper-time-mode tolerance is non-finite or non-positive.
    InvalidProperTimeTolerance(f64),

    /// A sampled step's metric norm is not within the configured proper-time
    /// tolerance of `-1`.
    ProperTimeNormDrift {
        /// Sample step index (`0` is the initial sample).
        step: usize,
        /// Observed metric norm `g_(mu nu) u^mu u^nu`.
        metric_norm: f64,
        /// Configured positive tolerance for `|g(u,u) - (-1)|`.
        tolerance: f64,
    },

    /// A state's metric norm is not negative, so it cannot be used to
    /// estimate a proper-time increment.
    NonTimelikeMetricNorm {
        /// Observed metric norm `g_(mu nu) u^mu u^nu`.
        metric_norm: f64,
    },

    /// A cylindrical-chart radial coordinate is not finite or not strictly
    /// positive.
    InvalidCylindricalRadius(f64),

    /// A coordinate-chart transform produced a non-finite coordinate
    /// component.
    NonFiniteChartCoordinate {
        /// Coordinate component index.
        component: usize,
        /// Invalid component value.
        value: f64,
    },

    /// A coordinate-chart transform produced a non-finite velocity component.
    NonFiniteChartVelocity {
        /// Velocity component index.
        component: usize,
        /// Invalid component value.
        value: f64,
    },

    /// A history backend could not supply a full [`HistoryEntry`] for a
    /// retained sample, which a [`HistoryModulator`] requires.
    HistoryEntryUnavailable {
        /// Requested retained sample index.
        retained_index: usize,
    },

    /// A curvature-modulator mass is non-finite or non-positive.
    InvalidModulationMass(f64),

    /// A curvature-modulator reference length is non-finite or non-positive.
    InvalidModulationReferenceLength(f64),

    /// A curvature-modulator phenomenological coefficient is non-finite or
    /// negative.
    InvalidModulationBeta(f64),

    /// A curvature-modulator radius is non-finite or not a valid Schwarzschild
    /// exterior radius.
    InvalidModulationRadius(f64),

    /// A curvature-modulator weight is non-finite or non-positive.
    NonFiniteModulationWeight(f64),

    /// Generated coordinate component is not finite.
    NonFiniteGeneratedCoordinate {
        /// Step index that produced the invalid state.
        step: usize,
        /// Coordinate component index.
        index: usize,
        /// Invalid component value.
        value: f64,
    },

    /// Generated velocity component is not finite.
    NonFiniteGeneratedVelocity {
        /// Step index that produced the invalid state.
        step: usize,
        /// Velocity component index.
        index: usize,
        /// Invalid component value.
        value: f64,
    },

    /// Metric component is not finite.
    NonFiniteMetricComponent {
        /// Sample step index.
        step: usize,
        /// Metric row index.
        row: usize,
        /// Metric column index.
        column: usize,
        /// Invalid component value.
        value: f64,
    },

    /// Metric norm is not finite.
    NonFiniteMetricNorm {
        /// Sample step index.
        step: usize,
        /// Invalid metric norm.
        value: f64,
    },

    /// Absolute metric norm is below the configured floor.
    MetricNormBelowFloor {
        /// Sample step index.
        step: usize,
        /// Metric norm value.
        metric_norm: f64,
        /// Configured positive floor.
        floor: f64,
    },

    /// Christoffel symbol is not finite.
    NonFiniteChristoffel {
        /// Sample step index.
        step: usize,
        /// Contravariant index.
        rho: usize,
        /// First lower index.
        mu: usize,
        /// Second lower index.
        nu: usize,
        /// Invalid symbol value.
        value: f64,
    },

    /// Caputo velocity-memory component is not finite.
    NonFiniteMemory {
        /// Sample step index.
        step: usize,
        /// Coordinate component index.
        component: usize,
        /// Invalid memory value.
        value: f64,
    },

    /// Projected memory-force component is not finite.
    NonFiniteForce {
        /// Sample step index.
        step: usize,
        /// Coordinate component index.
        component: usize,
        /// Invalid force value.
        value: f64,
    },

    /// Ordinary geodesic acceleration component is not finite.
    NonFiniteAcceleration {
        /// Sample step index.
        step: usize,
        /// Coordinate component index.
        component: usize,
        /// Invalid acceleration value.
        value: f64,
    },

    /// Diagnostic scalar is not finite.
    NonFiniteDiagnostic {
        /// Sample step index.
        step: usize,
        /// Diagnostic name.
        quantity: &'static str,
        /// Invalid diagnostic value.
        value: f64,
    },

    /// Fractional operator rejected a velocity-history component.
    FractionalMemory {
        /// Sample step index.
        step: usize,
        /// Coordinate component index.
        component: usize,
        /// Fractional-calculus error.
        source: FractionalError,
    },

    /// A circular-orbit radius is non-finite or does not exceed the
    /// existence bound `3 M` for a circular equatorial geodesic.
    InvalidCircularOrbitRadius(f64),

    /// An adaptive-stepping configuration parameter is non-finite or outside
    /// its required range.
    InvalidAdaptiveConfiguration {
        /// Name of the offending configuration field.
        field: &'static str,
        /// Rejected value.
        value: f64,
    },

    /// The adaptive integrator accepted its configured maximum number of
    /// steps without reaching the target affine parameter.
    AdaptiveStepBudgetExhausted {
        /// Number of accepted steps taken.
        accepted_steps: usize,
        /// Affine parameter actually reached.
        reached_parameter: f64,
        /// Requested target affine parameter.
        target_affine_parameter: f64,
    },

    /// The adaptive integrator rejected a single accepted step more times than
    /// the configured `max_rejections_per_step` retry budget while shrinking
    /// toward the tolerance.
    ///
    /// Distinct from [`NonlocalRelativityError::AdaptiveMinimumStepExhausted`]:
    /// here the retry *count* was reached; there the proposed shrink would have
    /// crossed `min_step`.
    AdaptiveRejectionBudgetExhausted {
        /// Accepted step index at which the retry budget was exhausted.
        accepted_step: usize,
        /// Number of rejections that were counted against the budget.
        rejections: usize,
        /// Most recently attempted step size.
        attempted_step: f64,
        /// Most recent local error estimate (scaled RMS norm; `1.0` is at
        /// tolerance).
        error_estimate: f64,
    },

    /// The adaptive integrator could not bring a single accepted step within
    /// tolerance without proposing a step below the configured `min_step`.
    ///
    /// Distinct from
    /// [`NonlocalRelativityError::AdaptiveRejectionBudgetExhausted`]: here the
    /// step-size floor was reached before the retry count.
    AdaptiveMinimumStepExhausted {
        /// Accepted step index at which shrinking hit the floor.
        accepted_step: usize,
        /// Step size that was rejected before the proposed shrink fell below
        /// `min_step`.
        attempted_step: f64,
        /// The proposed shrunk step that fell below the floor.
        proposed_step: f64,
        /// Configured minimum step size.
        min_step: f64,
        /// Most recent local error estimate (scaled RMS norm; `1.0` is at
        /// tolerance).
        error_estimate: f64,
    },

    /// A local orthonormal frame (tetrad) could not be built at a point: after
    /// the timelike leg, fewer than `D - 1` independent spacelike legs survived
    /// metric Gram-Schmidt, so the observer frame is degenerate there.
    DegenerateObserverFrame {
        /// Number of orthonormal legs successfully constructed (including the
        /// timelike leg) before the construction ran out of independent
        /// directions.
        legs_found: usize,
        /// Total number of legs required (the spacetime dimension `D`).
        dimension: usize,
    },
}

impl fmt::Display for NonlocalRelativityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidFractionalOrder(alpha) => write!(
                formatter,
                "fractional order must be finite and satisfy 0 < alpha < 1; got {alpha}"
            ),
            Self::InvalidCoupling(coupling) => write!(
                formatter,
                "memory coupling kappa must be finite and non-negative; got {coupling}"
            ),
            Self::InvalidStep(step) => write!(
                formatter,
                "uniform affine-parameter step must be finite and positive; got {step}"
            ),
            Self::InvalidStepCount(steps) =>
            {
                write!(
                    formatter,
                    "number of integration steps must be positive; got {steps}"
                )
            },
            Self::InvalidMetricNormFloor(floor) => write!(
                formatter,
                "metric-norm floor must be finite and positive; got {floor}"
            ),
            Self::InvalidHistoryWindow { window_samples } => write!(
                formatter,
                "short-memory history window must retain at least two samples; got {window_samples}"
            ),
            Self::InvalidIntegratorConfiguration { name } => write!(
                formatter,
                "unsupported worldline integrator configuration '{name}'"
            ),
            Self::RefinementStepCountOverflow { base_steps, factor } => write!(
                formatter,
                "refinement by factor {factor} overflows step count {base_steps}"
            ),
            Self::InvalidSchwarzschildExteriorCoordinates {
                radius,
                polar_angle,
                horizon_radius,
            } => write!(
                formatter,
                "Schwarzschild diagnostics require exterior coordinates r > {horizon_radius} \
                 and 0 < theta < pi; got r={radius}, theta={polar_angle}"
            ),
            Self::NonFiniteInitialCoordinate { index, value } => write!(
                formatter,
                "initial coordinate at index {index} is not finite; got {value}"
            ),
            Self::NonFiniteInitialVelocity { index, value } => write!(
                formatter,
                "initial velocity at index {index} is not finite; got {value}"
            ),
            Self::NonFiniteHistorySample {
                sample,
                component,
                value,
            } => write!(
                formatter,
                "history sample {sample}, component {component}, is not finite; got {value}"
            ),
            Self::InconsistentHistoryBackend {
                requested_sample,
                retained_samples,
            } => write!(
                formatter,
                "history backend reported {retained_samples} retained samples but sample \
                 {requested_sample} was unavailable"
            ),
            Self::InvalidTransportSegmentStep(step) => write!(
                formatter,
                "discrete-transport segment step must be finite; got {step}"
            ),
            Self::EmptyTransportPolyline => write!(
                formatter,
                "transport polyline must contain at least one waypoint"
            ),
            Self::NonFiniteTransportChristoffel { rho, mu, nu, value } => write!(
                formatter,
                "transport Christoffel symbol Gamma^{rho}_({mu} {nu}) is not finite; got {value}"
            ),
            Self::NonFiniteTransportedVector {
                retained_index,
                component,
                value,
            } => write!(
                formatter,
                "transported vector component {component} for retained sample \
                 {retained_index} is not finite; got {value}"
            ),
            Self::InvalidProperTimeTolerance(tolerance) => write!(
                formatter,
                "proper-time tolerance must be finite and positive; got {tolerance}"
            ),
            Self::ProperTimeNormDrift {
                step,
                metric_norm,
                tolerance,
            } => write!(
                formatter,
                "metric norm at step {step} is not within tolerance {tolerance} of -1; got {metric_norm}"
            ),
            Self::NonTimelikeMetricNorm { metric_norm } => write!(
                formatter,
                "metric norm must be negative (timelike) to estimate proper time; got {metric_norm}"
            ),
            Self::InvalidCylindricalRadius(radius) => write!(
                formatter,
                "cylindrical radial coordinate must be finite and strictly positive; got {radius}"
            ),
            Self::NonFiniteChartCoordinate { component, value } => write!(
                formatter,
                "chart-transformed coordinate at index {component} is not finite; got {value}"
            ),
            Self::NonFiniteChartVelocity { component, value } => write!(
                formatter,
                "chart-transformed velocity at index {component} is not finite; got {value}"
            ),
            Self::HistoryEntryUnavailable { retained_index } => write!(
                formatter,
                "history backend did not supply a full entry for retained sample \
                 {retained_index}, required by this memory law"
            ),
            Self::InvalidModulationMass(mass) => write!(
                formatter,
                "curvature-modulator mass must be finite and strictly positive; got {mass}"
            ),
            Self::InvalidModulationReferenceLength(length) => write!(
                formatter,
                "curvature-modulator reference length must be finite and strictly positive; \
                 got {length}"
            ),
            Self::InvalidModulationBeta(beta) => write!(
                formatter,
                "curvature-modulator beta must be finite and non-negative; got {beta}"
            ),
            Self::InvalidModulationRadius(radius) => write!(
                formatter,
                "curvature-modulator radius must be finite and exceed the horizon radius; \
                 got {radius}"
            ),
            Self::NonFiniteModulationWeight(weight) => write!(
                formatter,
                "curvature-modulator weight must be finite and positive; got {weight}"
            ),
            Self::NonFiniteGeneratedCoordinate { step, index, value } => write!(
                formatter,
                "generated coordinate at step {step}, index {index}, is not finite; got {value}"
            ),
            Self::NonFiniteGeneratedVelocity { step, index, value } => write!(
                formatter,
                "generated velocity at step {step}, index {index}, is not finite; got {value}"
            ),
            Self::NonFiniteMetricComponent {
                step,
                row,
                column,
                value,
            } => write!(
                formatter,
                "metric component ({row}, {column}) at step {step} is not finite; got {value}"
            ),
            Self::NonFiniteMetricNorm { step, value } =>
            {
                write!(
                    formatter,
                    "metric norm at step {step} is not finite; got {value}"
                )
            },
            Self::MetricNormBelowFloor {
                step,
                metric_norm,
                floor,
            } => write!(
                formatter,
                "absolute metric norm at step {step} is below floor {floor}; got {metric_norm}"
            ),
            Self::NonFiniteChristoffel {
                step,
                rho,
                mu,
                nu,
                value,
            } => write!(
                formatter,
                "Christoffel symbol Gamma^{rho}_({mu} {nu}) at step {step} is not finite; got {value}"
            ),
            Self::NonFiniteMemory {
                step,
                component,
                value,
            } => write!(
                formatter,
                "Caputo memory component {component} at step {step} is not finite; got {value}"
            ),
            Self::NonFiniteForce {
                step,
                component,
                value,
            } => write!(
                formatter,
                "memory-force component {component} at step {step} is not finite; got {value}"
            ),
            Self::NonFiniteAcceleration {
                step,
                component,
                value,
            } => write!(
                formatter,
                "acceleration component {component} at step {step} is not finite; got {value}"
            ),
            Self::NonFiniteDiagnostic {
                step,
                quantity,
                value,
            } => write!(
                formatter,
                "diagnostic {quantity} at step {step} is not finite; got {value}"
            ),
            Self::FractionalMemory {
                step,
                component,
                source,
            } => write!(
                formatter,
                "Caputo memory evaluation failed at step {step}, component {component}: {source}"
            ),
            Self::InvalidCircularOrbitRadius(radius) => write!(
                formatter,
                "circular-orbit radius must be finite and exceed 3 M; got {radius}"
            ),
            Self::InvalidAdaptiveConfiguration { field, value } => write!(
                formatter,
                "adaptive configuration field '{field}' is invalid; got {value}"
            ),
            Self::AdaptiveStepBudgetExhausted {
                accepted_steps,
                reached_parameter,
                target_affine_parameter,
            } => write!(
                formatter,
                "adaptive integrator exhausted its budget of {accepted_steps} accepted steps at \
                 affine parameter {reached_parameter}, short of target {target_affine_parameter}"
            ),
            Self::AdaptiveRejectionBudgetExhausted {
                accepted_step,
                rejections,
                attempted_step,
                error_estimate,
            } => write!(
                formatter,
                "adaptive integrator exhausted its rejection budget after accepted step \
                 {accepted_step} ({rejections} rejections), last attempted step {attempted_step} \
                 with scaled error estimate {error_estimate}"
            ),
            Self::AdaptiveMinimumStepExhausted {
                accepted_step,
                attempted_step,
                proposed_step,
                min_step,
                error_estimate,
            } => write!(
                formatter,
                "adaptive integrator could not meet tolerance after accepted step \
                 {accepted_step} without shrinking step {attempted_step} to {proposed_step}, \
                 below the minimum {min_step} (scaled error estimate {error_estimate})"
            ),
            Self::DegenerateObserverFrame {
                legs_found,
                dimension,
            } => write!(
                formatter,
                "could not build a local orthonormal frame: found {legs_found} of {dimension} \
                 independent orthonormal legs"
            ),
        }
    }
}

impl Error for NonlocalRelativityError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self
        {
            Self::FractionalMemory { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Lower a contravariant vector index with a covariant metric.
#[must_use]
pub fn lower_index<const D: usize>(metric: &[[f64; D]; D], vector: &[f64; D]) -> [f64; D] {
    let mut lowered = [0.0_f64; D];

    for sigma in 0..D
    {
        for nu in 0..D
        {
            lowered[sigma] += metric[sigma][nu] * vector[nu];
        }
    }

    lowered
}

/// Contract two contravariant vectors with a covariant metric.
#[must_use]
pub fn metric_contraction<const D: usize>(
    metric: &[[f64; D]; D],
    left: &[f64; D],
    right: &[f64; D],
) -> f64 {
    let mut value = 0.0;

    for mu in 0..D
    {
        for nu in 0..D
        {
            value += metric[mu][nu] * left[mu] * right[nu];
        }
    }

    value
}

/// Compute the coordinate L2 norm of a vector.
#[must_use]
pub fn coordinate_l2_norm<const D: usize>(vector: &[f64; D]) -> f64 {
    let mut sum = 0.0;

    for value in vector
    {
        sum += *value * *value;
    }

    sum.sqrt()
}

/// Evaluate Schwarzschild exterior chart diagnostics for a worldline state.
///
/// Coordinates are interpreted as standard Schwarzschild `(t, r, theta, phi)`.
/// This helper is chart- and background-specific; it is not a generic
/// invariant extractor for arbitrary metrics.
pub fn schwarzschild_invariants(
    background: &Schwarzschild,
    state: &WorldlineState<4>,
) -> NonlocalResult<SchwarzschildInvariants> {
    validate_initial_state(state)?;

    if !background.is_in_exterior(&state.coordinates)
    {
        return Err(
            NonlocalRelativityError::InvalidSchwarzschildExteriorCoordinates {
                radius: state.coordinates[1],
                polar_angle: state.coordinates[2],
                horizon_radius: background.horizon_radius(),
            },
        );
    }

    let metric = validated_metric(background, &state.coordinates, 0)?;
    let lowered_velocity = lower_index(&metric, &state.velocity);
    let specific_energy = -lowered_velocity[0];
    let azimuthal_angular_momentum = lowered_velocity[3];
    let metric_norm = metric_contraction(&metric, &state.velocity, &state.velocity);

    validate_scalar("schwarzschild_specific_energy", specific_energy, 0)?;
    validate_scalar(
        "schwarzschild_azimuthal_angular_momentum",
        azimuthal_angular_momentum,
        0,
    )?;

    if !metric_norm.is_finite()
    {
        return Err(NonlocalRelativityError::NonFiniteMetricNorm {
            step: 0,
            value: metric_norm,
        });
    }

    Ok(SchwarzschildInvariants {
        specific_energy,
        azimuthal_angular_momentum,
        metric_norm,
    })
}

/// Return the Schwarzschild exterior specific energy `E = -u_t`.
///
/// This helper assumes standard Schwarzschild coordinates in a fixed exterior
/// chart and does not apply to arbitrary stationary metrics.
pub fn schwarzschild_specific_energy(
    background: &Schwarzschild,
    state: &WorldlineState<4>,
) -> NonlocalResult<f64> {
    Ok(schwarzschild_invariants(background, state)?.specific_energy)
}

/// Return the Schwarzschild exterior azimuthal angular momentum `L_z = u_phi`.
///
/// This helper assumes standard Schwarzschild coordinates in a fixed exterior
/// chart and does not apply to arbitrary axisymmetric metrics.
pub fn schwarzschild_azimuthal_angular_momentum(
    background: &Schwarzschild,
    state: &WorldlineState<4>,
) -> NonlocalResult<f64> {
    Ok(schwarzschild_invariants(background, state)?.azimuthal_angular_momentum)
}

/// Return `g_(mu nu) u^mu u^nu` in standard Schwarzschild exterior
/// coordinates.
pub fn schwarzschild_metric_norm(
    background: &Schwarzschild,
    state: &WorldlineState<4>,
) -> NonlocalResult<f64> {
    Ok(schwarzschild_invariants(background, state)?.metric_norm)
}

/// Compute the ordinary geodesic acceleration
/// `-Gamma^rho_(mu nu) u^mu u^nu`.
#[must_use]
pub fn gr_acceleration<const D: usize>(
    christoffel: &[[[f64; D]; D]; D],
    velocity: &[f64; D],
) -> [f64; D] {
    let mut acceleration = [0.0_f64; D];

    for rho in 0..D
    {
        for mu in 0..D
        {
            for nu in 0..D
            {
                acceleration[rho] -= christoffel[rho][mu][nu] * velocity[mu] * velocity[nu];
            }
        }
    }

    acceleration
}

/// Project the memory vector orthogonally to the current velocity and apply
/// the experimental force law `F_memory^rho = -kappa P^rho_sigma m^sigma`.
#[must_use]
pub fn projected_memory_force<const D: usize>(
    velocity: &[f64; D],
    lowered_velocity: &[f64; D],
    metric_norm: f64,
    memory: &[f64; D],
    coupling: f64,
) -> [f64; D] {
    let mut memory_along_velocity = 0.0;

    for sigma in 0..D
    {
        memory_along_velocity += lowered_velocity[sigma] * memory[sigma];
    }

    let mut force = [0.0_f64; D];

    for rho in 0..D
    {
        let projected = memory[rho] - velocity[rho] * memory_along_velocity / metric_norm;
        force[rho] = -coupling * projected;
    }

    force
}

/// Evaluate the Caputo L1 velocity-memory vector from complete uniform
/// velocity history.
///
/// The first-sample convention is explicit: when history contains only the
/// current velocity, the returned memory vector is zero because the Caputo L1
/// stencil has insufficient history.
pub fn caputo_velocity_memory<const D: usize>(
    velocity_history: &[[f64; D]],
    step: f64,
    order: FractionalOrder,
) -> NonlocalResult<[f64; D]> {
    caputo_velocity_memory_at_step(velocity_history, step, order, 0)
}

/// Simulate the experimental fractional-memory worldline model.
///
/// The background is fixed and supplies both metric components and
/// Christoffel symbols. The returned trajectory contains `steps + 1` sampled
/// states and diagnostics, including the initial sample. This compatibility
/// path uses [`WorldlineIntegrator::SemiImplicitEuler`].
pub fn simulate_nonlocal_worldline<B, const D: usize>(
    background: &B,
    initial_state: WorldlineState<D>,
    config: NonlocalConfig,
) -> NonlocalResult<NonlocalTrajectory<D>>
where
    B: Metric<D> + Connection<D>,
{
    simulate_nonlocal_worldline_with_integrator(
        background,
        initial_state,
        config,
        WorldlineIntegrator::SemiImplicitEuler,
    )
}

/// Simulate the experimental fractional-memory worldline model with an
/// explicit fixed-step integrator selection.
///
/// The state equation remains ordinary in affine parameter. The Caputo L1
/// term is evaluated as a coordinate-memory force from complete velocity
/// history.
pub fn simulate_nonlocal_worldline_with_integrator<B, const D: usize>(
    background: &B,
    initial_state: WorldlineState<D>,
    config: NonlocalConfig,
    integrator: WorldlineIntegrator,
) -> NonlocalResult<NonlocalTrajectory<D>>
where
    B: Metric<D> + Connection<D>,
{
    match integrator
    {
        WorldlineIntegrator::SemiImplicitEuler => simulate_nonlocal_worldline_with_components(
            background,
            initial_state,
            config,
            CompleteUniformHistory::with_capacity(config.steps + 1),
            CaputoCoordinateMemory,
            IdentityHistoryTransport,
            SemiImplicitEulerStepper,
        ),
        WorldlineIntegrator::HeunPece => simulate_nonlocal_worldline_with_components(
            background,
            initial_state,
            config,
            CompleteUniformHistory::with_capacity(config.steps + 1),
            CaputoCoordinateMemory,
            IdentityHistoryTransport,
            HeunPeceStepper,
        ),
    }
}

/// Simulate with a typed advanced policy.
///
/// This API keeps memory law, history storage, history transport, and the
/// ordinary state stepper separate. Passing a bounded short-memory backend is
/// an explicit approximation choice; the default policy uses complete exact
/// history.
pub fn simulate_nonlocal_worldline_with_policy<B, H, L, T, S, const D: usize>(
    background: &B,
    initial_state: WorldlineState<D>,
    config: NonlocalConfig,
    policy: NonlocalSimulationPolicy<D, H, L, T, S>,
) -> NonlocalResult<NonlocalTrajectory<D>>
where
    B: Metric<D> + Connection<D>,
    H: HistoryBackend<D>,
    L: MemoryLaw<D>,
    T: HistoryTransport<D>,
    S: WorldlineStepper<D>,
{
    simulate_nonlocal_worldline_with_components(
        background,
        initial_state,
        config,
        policy.history_backend,
        policy.memory_law,
        policy.history_transport,
        policy.stepper,
    )
}

/// Simulate with explicit architecture components.
///
/// The exact complete-history backend is the numerical oracle. Short-memory
/// backends are deliberately approximate and must be supplied explicitly.
pub fn simulate_nonlocal_worldline_with_components<B, H, L, T, S, const D: usize>(
    background: &B,
    initial_state: WorldlineState<D>,
    config: NonlocalConfig,
    mut history_backend: H,
    memory_law: L,
    history_transport: T,
    stepper: S,
) -> NonlocalResult<NonlocalTrajectory<D>>
where
    B: Metric<D> + Connection<D>,
    H: HistoryBackend<D>,
    L: MemoryLaw<D>,
    T: HistoryTransport<D>,
    S: WorldlineStepper<D>,
{
    validate_initial_state(&initial_state)?;

    let mut states = Vec::with_capacity(config.steps + 1);
    let mut diagnostics = Vec::with_capacity(config.steps + 1);
    let mut history_diagnostics = Vec::with_capacity(config.steps + 1);

    states.push(initial_state);
    history_backend.push_entry(
        background,
        &history_transport,
        HistoryEntry::new(initial_state.coordinates, initial_state.velocity, 0.0),
    )?;

    let initial_metric = validated_metric(background, &initial_state.coordinates, 0)?;
    let initial_metric_norm = validated_metric_norm(
        &initial_metric,
        &initial_state.velocity,
        config.metric_norm_floor,
        0,
    )?;

    for step_index in 0..config.steps
    {
        let state = states[step_index];
        let affine_parameter = step_index as f64 * config.step;
        let evaluation = evaluate_step_with_policy(StepEvaluationInput {
            background,
            state: &state,
            history: &history_backend,
            memory_law: &memory_law,
            transport: &history_transport,
            initial_metric_norm,
            affine_parameter,
            step_index,
            config,
        })?;

        diagnostics.push(evaluation.diagnostics);
        history_diagnostics.push(history_backend.diagnostics());

        let next_state = stepper.advance(StepperContext {
            background,
            state: &state,
            accepted_acceleration: &evaluation.acceleration,
            history: &history_backend,
            memory_law: &memory_law,
            transport: &history_transport,
            initial_metric_norm,
            current_parameter: affine_parameter,
            step_index,
            config,
        })?;
        states.push(next_state);
        let next_parameter = (step_index + 1) as f64 * config.step;
        history_backend.push_entry(
            background,
            &history_transport,
            HistoryEntry::new(next_state.coordinates, next_state.velocity, next_parameter),
        )?;
    }

    let final_step = config.steps;
    let final_state = states[final_step];
    let final_evaluation = evaluate_step_with_policy(StepEvaluationInput {
        background,
        state: &final_state,
        history: &history_backend,
        memory_law: &memory_law,
        transport: &history_transport,
        initial_metric_norm,
        affine_parameter: final_step as f64 * config.step,
        step_index: final_step,
        config,
    })?;
    diagnostics.push(final_evaluation.diagnostics);
    history_diagnostics.push(history_backend.diagnostics());

    Ok(NonlocalTrajectory::new(
        states,
        diagnostics,
        history_diagnostics,
    ))
}

/// Run a deterministic endpoint self-convergence study at `h`, `h/2`, and
/// `h/4`.
///
/// The returned errors are endpoint differences between successive
/// refinements. They are useful numerical self-consistency diagnostics, but
/// they are not an exact-error certificate for the continuous model.
pub fn run_convergence_study<B, const D: usize>(
    background: &B,
    initial_state: WorldlineState<D>,
    base_config: NonlocalConfig,
    integrator: WorldlineIntegrator,
) -> NonlocalResult<ConvergenceStudy<D>>
where
    B: Metric<D> + Connection<D>,
{
    let fine_config = refined_config(base_config, 2)?;
    let finest_config = refined_config(base_config, 4)?;

    let coarse_trajectory = simulate_nonlocal_worldline_with_integrator(
        background,
        initial_state,
        base_config,
        integrator,
    )?;
    let fine_trajectory = simulate_nonlocal_worldline_with_integrator(
        background,
        initial_state,
        fine_config,
        integrator,
    )?;
    let finest_trajectory = simulate_nonlocal_worldline_with_integrator(
        background,
        initial_state,
        finest_config,
        integrator,
    )?;

    let coarse = refinement_endpoint(&coarse_trajectory, base_config)?;
    let fine = refinement_endpoint(&fine_trajectory, fine_config)?;
    let finest = refinement_endpoint(&finest_trajectory, finest_config)?;

    let endpoint_coordinate_error_h_h2 = checked_l2_distance(
        &coarse.coordinates,
        &fine.coordinates,
        "coordinate_error_h_h2",
    )?;
    let endpoint_coordinate_error_h2_h4 = checked_l2_distance(
        &fine.coordinates,
        &finest.coordinates,
        "coordinate_error_h2_h4",
    )?;
    let endpoint_velocity_error_h_h2 =
        checked_l2_distance(&coarse.velocity, &fine.velocity, "velocity_error_h_h2")?;
    let endpoint_velocity_error_h2_h4 =
        checked_l2_distance(&fine.velocity, &finest.velocity, "velocity_error_h2_h4")?;

    Ok(ConvergenceStudy {
        integrator,
        coarse,
        fine,
        finest,
        endpoint_coordinate_error_h_h2,
        endpoint_coordinate_error_h2_h4,
        endpoint_velocity_error_h_h2,
        endpoint_velocity_error_h2_h4,
        coordinate_self_convergence_ratio: finite_ratio(
            endpoint_coordinate_error_h_h2,
            endpoint_coordinate_error_h2_h4,
        ),
        velocity_self_convergence_ratio: finite_ratio(
            endpoint_velocity_error_h_h2,
            endpoint_velocity_error_h2_h4,
        ),
    })
}

pub(crate) struct StepEvaluation<const D: usize> {
    pub(crate) acceleration: [f64; D],
    pub(crate) diagnostics: StepDiagnostics,
}

fn semi_implicit_euler_step<const D: usize>(
    state: &WorldlineState<D>,
    acceleration: &[f64; D],
    step: f64,
    step_index: usize,
) -> NonlocalResult<WorldlineState<D>> {
    let mut next_velocity = [0.0_f64; D];
    let mut next_coordinates = [0.0_f64; D];

    for rho in 0..D
    {
        next_velocity[rho] = state.velocity[rho] + step * acceleration[rho];
        validate_generated_velocity(next_velocity[rho], step_index + 1, rho)?;

        next_coordinates[rho] = state.coordinates[rho] + step * next_velocity[rho];
        validate_generated_coordinate(next_coordinates[rho], step_index + 1, rho)?;
    }

    Ok(WorldlineState::new(next_coordinates, next_velocity))
}

pub(crate) fn validate_generated_velocity(
    value: f64,
    step: usize,
    index: usize,
) -> NonlocalResult<()> {
    if !value.is_finite()
    {
        return Err(NonlocalRelativityError::NonFiniteGeneratedVelocity { step, index, value });
    }

    Ok(())
}

pub(crate) fn validate_generated_coordinate(
    value: f64,
    step: usize,
    index: usize,
) -> NonlocalResult<()> {
    if !value.is_finite()
    {
        return Err(NonlocalRelativityError::NonFiniteGeneratedCoordinate { step, index, value });
    }

    Ok(())
}

fn refined_config(config: NonlocalConfig, factor: usize) -> NonlocalResult<NonlocalConfig> {
    let steps = config.steps.checked_mul(factor).ok_or(
        NonlocalRelativityError::RefinementStepCountOverflow {
            base_steps: config.steps,
            factor,
        },
    )?;
    let step = config.step / factor as f64;

    NonlocalConfig::from_fractional_order(
        config.fractional_order,
        config.coupling,
        step,
        steps,
        config.metric_norm_floor,
    )
}

fn refinement_endpoint<const D: usize>(
    trajectory: &NonlocalTrajectory<D>,
    config: NonlocalConfig,
) -> NonlocalResult<RefinementEndpoint<D>> {
    let state = match trajectory.final_state()
    {
        Some(state) => *state,
        None => return Err(NonlocalRelativityError::InvalidStepCount(0)),
    };
    let diagnostics = match trajectory.final_diagnostics()
    {
        Some(diagnostics) => *diagnostics,
        None => return Err(NonlocalRelativityError::InvalidStepCount(0)),
    };

    validate_scalar(
        "final_affine_parameter",
        diagnostics.affine_parameter,
        config.steps,
    )?;
    validate_scalar(
        "metric_norm_drift",
        diagnostics.metric_norm_drift,
        config.steps,
    )?;
    validate_scalar(
        "memory_force_l2_norm",
        diagnostics.memory_force_l2_norm,
        config.steps,
    )?;

    Ok(RefinementEndpoint {
        step: config.step,
        steps: config.steps,
        final_affine_parameter: diagnostics.affine_parameter,
        coordinates: state.coordinates,
        velocity: state.velocity,
        metric_norm_drift: diagnostics.metric_norm_drift,
        memory_force_l2_norm: diagnostics.memory_force_l2_norm,
    })
}

pub(crate) fn checked_l2_distance<const D: usize>(
    left: &[f64; D],
    right: &[f64; D],
    quantity: &'static str,
) -> NonlocalResult<f64> {
    let mut sum = 0.0;

    for component in 0..D
    {
        let difference = left[component] - right[component];
        validate_scalar(quantity, difference, component)?;
        sum += difference * difference;
        validate_scalar(quantity, sum, component)?;
    }

    let value = sum.sqrt();
    validate_scalar(quantity, value, D)?;
    Ok(value)
}

fn finite_ratio(numerator: f64, denominator: f64) -> Option<f64> {
    if denominator <= 0.0
    {
        return None;
    }

    let ratio = numerator / denominator;

    if ratio.is_finite() { Some(ratio) } else { None }
}

pub(crate) fn validate_initial_state<const D: usize>(
    state: &WorldlineState<D>,
) -> Result<(), NonlocalRelativityError> {
    for (index, value) in state.coordinates.iter().copied().enumerate()
    {
        if !value.is_finite()
        {
            return Err(NonlocalRelativityError::NonFiniteInitialCoordinate { index, value });
        }
    }

    for (index, value) in state.velocity.iter().copied().enumerate()
    {
        if !value.is_finite()
        {
            return Err(NonlocalRelativityError::NonFiniteInitialVelocity { index, value });
        }
    }

    Ok(())
}

pub(crate) fn validated_metric<B, const D: usize>(
    background: &B,
    coordinates: &[f64; D],
    step: usize,
) -> NonlocalResult<[[f64; D]; D]>
where
    B: Metric<D>,
{
    let metric = background.components(coordinates);

    for (row, row_values) in metric.iter().enumerate()
    {
        for (column, value) in row_values.iter().copied().enumerate()
        {
            if !value.is_finite()
            {
                return Err(NonlocalRelativityError::NonFiniteMetricComponent {
                    step,
                    row,
                    column,
                    value,
                });
            }
        }
    }

    Ok(metric)
}

pub(crate) fn validated_metric_norm<const D: usize>(
    metric: &[[f64; D]; D],
    velocity: &[f64; D],
    floor: f64,
    step: usize,
) -> NonlocalResult<f64> {
    let norm = metric_contraction(metric, velocity, velocity);

    if !norm.is_finite()
    {
        return Err(NonlocalRelativityError::NonFiniteMetricNorm { step, value: norm });
    }

    if norm.abs() < floor
    {
        return Err(NonlocalRelativityError::MetricNormBelowFloor {
            step,
            metric_norm: norm,
            floor,
        });
    }

    Ok(norm)
}

pub(crate) fn validated_christoffel<B, const D: usize>(
    background: &B,
    coordinates: &[f64; D],
    step: usize,
) -> NonlocalResult<[[[f64; D]; D]; D]>
where
    B: Connection<D>,
{
    let symbols = background.christoffel(coordinates);

    for (rho, rho_values) in symbols.iter().enumerate()
    {
        for (mu, mu_values) in rho_values.iter().enumerate()
        {
            for (nu, value) in mu_values.iter().copied().enumerate()
            {
                if !value.is_finite()
                {
                    return Err(NonlocalRelativityError::NonFiniteChristoffel {
                        step,
                        rho,
                        mu,
                        nu,
                        value,
                    });
                }
            }
        }
    }

    Ok(symbols)
}

pub(crate) fn caputo_velocity_memory_at_step<const D: usize>(
    velocity_history: &[[f64; D]],
    step: f64,
    order: FractionalOrder,
    step_index: usize,
) -> NonlocalResult<[f64; D]> {
    if !step.is_finite() || step <= 0.0
    {
        return Err(NonlocalRelativityError::InvalidStep(step));
    }

    if velocity_history.is_empty()
    {
        return Err(NonlocalRelativityError::FractionalMemory {
            step: step_index,
            component: 0,
            source: FractionalError::EmptySamples,
        });
    }

    let mut memory = [0.0_f64; D];

    let mut samples = Vec::with_capacity(velocity_history.len());

    for component in 0..D
    {
        samples.clear();
        for (sample_index, velocity) in velocity_history.iter().enumerate()
        {
            let sample = velocity[component];

            if !sample.is_finite()
            {
                return Err(NonlocalRelativityError::FractionalMemory {
                    step: step_index,
                    component,
                    source: FractionalError::NonFiniteSample(sample_index),
                });
            }

            samples.push(sample);
        }

        if samples.len() == 1
        {
            continue;
        }

        let value = caputo_l1_uniform(&samples, step, order).map_err(|source| {
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

        memory[component] = value;
    }

    Ok(memory)
}

fn transported_caputo_velocity_memory_at_step<H, T, const D: usize>(
    history: &H,
    transport: &T,
    current_state: &WorldlineState<D>,
    step: f64,
    order: FractionalOrder,
    step_index: usize,
) -> NonlocalResult<[f64; D]>
where
    H: HistoryBackend<D>,
    T: HistoryTransport<D>,
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
    let mut transported_history = Vec::with_capacity(retained_samples);

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
        transported_history.push(transported_velocity);
    }

    caputo_velocity_memory_at_step(&transported_history, step, order, step_index)
}

pub(crate) fn validate_history_velocity<const D: usize>(
    velocity: &[f64; D],
    sample: usize,
) -> NonlocalResult<()> {
    for (component, value) in velocity.iter().copied().enumerate()
    {
        if !value.is_finite()
        {
            return Err(NonlocalRelativityError::NonFiniteHistorySample {
                sample,
                component,
                value,
            });
        }
    }

    Ok(())
}

pub(crate) struct StepEvaluationInput<'a, B, H, L, T, const D: usize> {
    pub(crate) background: &'a B,
    pub(crate) state: &'a WorldlineState<D>,
    pub(crate) history: &'a H,
    pub(crate) memory_law: &'a L,
    pub(crate) transport: &'a T,
    pub(crate) initial_metric_norm: f64,
    pub(crate) affine_parameter: f64,
    pub(crate) step_index: usize,
    pub(crate) config: NonlocalConfig,
}

pub(crate) fn evaluate_step_with_policy<B, H, L, T, const D: usize>(
    input: StepEvaluationInput<'_, B, H, L, T, D>,
) -> NonlocalResult<StepEvaluation<D>>
where
    B: Metric<D> + Connection<D>,
    H: HistoryBackend<D>,
    L: MemoryLaw<D>,
    T: HistoryTransport<D>,
{
    let metric = validated_metric(input.background, &input.state.coordinates, input.step_index)?;
    let metric_norm = validated_metric_norm(
        &metric,
        &input.state.velocity,
        input.config.metric_norm_floor,
        input.step_index,
    )?;
    let lowered_velocity = lower_index(&metric, &input.state.velocity);

    let symbols =
        validated_christoffel(input.background, &input.state.coordinates, input.step_index)?;
    let gr = gr_acceleration(&symbols, &input.state.velocity);
    validate_vector(&gr, input.step_index, |step, component, value| {
        NonlocalRelativityError::NonFiniteAcceleration {
            step,
            component,
            value,
        }
    })?;

    let memory = input.memory_law.memory_vector(
        input.history,
        input.transport,
        input.state,
        input.step_index,
        input.config,
    )?;
    validate_vector(&memory, input.step_index, |step, component, value| {
        NonlocalRelativityError::NonFiniteMemory {
            step,
            component,
            value,
        }
    })?;

    let force = projected_memory_force(
        &input.state.velocity,
        &lowered_velocity,
        metric_norm,
        &memory,
        input.config.coupling,
    );
    validate_vector(&force, input.step_index, |step, component, value| {
        NonlocalRelativityError::NonFiniteForce {
            step,
            component,
            value,
        }
    })?;

    let mut acceleration = [0.0_f64; D];
    for rho in 0..D
    {
        acceleration[rho] = gr[rho] + force[rho];
    }
    validate_vector(&acceleration, input.step_index, |step, component, value| {
        NonlocalRelativityError::NonFiniteAcceleration {
            step,
            component,
            value,
        }
    })?;

    let memory_l2_norm = coordinate_l2_norm(&memory);
    let memory_force_l2_norm = coordinate_l2_norm(&force);
    let gr_acceleration_l2_norm = coordinate_l2_norm(&gr);
    let orthogonality_residual = lowered_velocity
        .iter()
        .zip(force)
        .fold(0.0, |sum, (lowered, force_component)| {
            sum + *lowered * force_component
        });

    let diagnostics = StepDiagnostics {
        affine_parameter: input.affine_parameter,
        metric_norm,
        metric_norm_drift: metric_norm - input.initial_metric_norm,
        memory_l2_norm,
        memory_force_l2_norm,
        orthogonality_residual,
        gr_acceleration_l2_norm,
    };

    validate_diagnostics(&diagnostics, input.step_index)?;

    Ok(StepEvaluation {
        acceleration,
        diagnostics,
    })
}

pub(crate) fn validate_vector<const D: usize, F>(
    vector: &[f64; D],
    step: usize,
    error: F,
) -> NonlocalResult<()>
where
    F: Fn(usize, usize, f64) -> NonlocalRelativityError,
{
    for (component, value) in vector.iter().copied().enumerate()
    {
        if !value.is_finite()
        {
            return Err(error(step, component, value));
        }
    }

    Ok(())
}

pub(crate) fn validate_diagnostics(
    diagnostics: &StepDiagnostics,
    step: usize,
) -> Result<(), NonlocalRelativityError> {
    let quantities = [
        ("affine_parameter", diagnostics.affine_parameter),
        ("metric_norm", diagnostics.metric_norm),
        ("metric_norm_drift", diagnostics.metric_norm_drift),
        ("memory_l2_norm", diagnostics.memory_l2_norm),
        ("memory_force_l2_norm", diagnostics.memory_force_l2_norm),
        ("orthogonality_residual", diagnostics.orthogonality_residual),
        (
            "gr_acceleration_l2_norm",
            diagnostics.gr_acceleration_l2_norm,
        ),
    ];

    for (quantity, value) in quantities
    {
        if !value.is_finite()
        {
            return Err(NonlocalRelativityError::NonFiniteDiagnostic {
                step,
                quantity,
                value,
            });
        }
    }

    Ok(())
}

pub(crate) fn validate_scalar(
    quantity: &'static str,
    value: f64,
    step: usize,
) -> NonlocalResult<()> {
    if !value.is_finite()
    {
        return Err(NonlocalRelativityError::NonFiniteDiagnostic {
            step,
            quantity,
            value,
        });
    }

    Ok(())
}
