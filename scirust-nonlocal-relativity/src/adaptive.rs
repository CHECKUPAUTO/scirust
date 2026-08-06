//! Adaptive-step worldline integration with an embedded Heun-Euler error
//! estimate.
//!
//! This closes the "proper-time history sampled at its own adaptive
//! resolution" gap left open by [`crate::proper_time_caputo_velocity_memory`]:
//! rather than resampling an already uniformly-stepped trajectory onto a
//! non-uniform axis after the fact, [`simulate_nonlocal_worldline_adaptive`]
//! chooses its own non-uniform affine-parameter step *during* integration,
//! and evaluates the Caputo memory force directly against the resulting
//! non-uniform history with `scirust_fractional::caputo_l1_nonuniform`.
//!
//! The step-size controller is the classical embedded Heun-Euler pair: the
//! same Euler predictor and Heun corrector [`crate::HeunPeceStepper`] already
//! computes are reused as a first-order/second-order embedded pair, so the
//! local error estimate `||corrected - predicted||` costs no extra
//! acceleration evaluation beyond what one Heun step already needs. This is
//! a standard, well-established adaptive-Runge-Kutta technique, not a new
//! numerical method.
//!
//! [`simulate_nonlocal_worldline_adaptive_with_policy`] composes this
//! step-size controller with the same [`crate::HistoryTransport`] and
//! [`crate::HistoryModulator`] contracts the fixed-step architecture uses:
//! each accepted step's history is transported across the newly accepted
//! segment via [`crate::HistoryBackend::push_entry`] (exactly the mechanism
//! [`crate::CompleteUniformHistory`] and [`crate::BoundedShortMemoryHistory`]
//! already use for the fixed-step path), and each retained sample is
//! modulated by [`crate::HistoryModulator::weight`] before the Caputo
//! evaluation, exactly like [`crate::ModulatedCaputoCoordinateMemory`] does.
//! [`simulate_nonlocal_worldline_adaptive`] is the plain-coordinate-memory
//! special case, `IdentityHistoryTransport` and `IdentityHistoryModulator`
//! composed with `CompleteUniformHistory`.
//!
//! This module's controller does **not** itself reuse [`crate::MemoryLaw`] or
//! [`crate::WorldlineStepper`]: both thread a single fixed [`NonlocalConfig`]
//! step through their signatures ([`crate::StepperContext`] for the latter),
//! which this module's embedded-pair controller cannot satisfy without
//! changing those contracts. Composing adaptive stepping with a
//! curvature-modulated *and* transported pipeline together is exercised in
//! this crate's tests exactly as the fixed-step architecture exercises it.
//!
//! [`crate::adaptive_stepper`] closes the `MemoryLaw`/`WorldlineStepper` gap
//! left open here, for [`crate::SemiImplicitEulerStepper`] specifically, via
//! a different step-size-control mechanism (classical step-doubling rather
//! than this module's embedded pair); see that module's documentation for
//! the mechanism and for why [`crate::HeunPeceStepper`] is not offered
//! through that step-doubling entry point — this module's embedded
//! Heun-Euler controller *is* adaptive Heun-PECE, so a step-doubling variant
//! would be a strictly inferior duplicate rather than a new capability.

use crate::adaptive_control::{StepControl, clamp_step_to_target, control_step};
use crate::nonuniform_kernel::nonuniform_transported_modulated_caputo_velocity_memory;
use crate::{
    AdaptiveTolerance, CompleteUniformHistory, Connection, HistoryBackend, HistoryEntry,
    HistoryModulator, HistoryTransport, IdentityHistoryModulator, IdentityHistoryTransport, Metric,
    NonlocalRelativityError, NonlocalResult, NonlocalTrajectory, StepDiagnostics, WorldlineState,
    coordinate_l2_norm, gr_acceleration, lower_index, projected_memory_force,
    scaled_local_error_norm, validate_diagnostics, validate_generated_coordinate,
    validate_generated_velocity, validate_initial_state, validate_vector, validated_christoffel,
    validated_metric, validated_metric_norm,
};
use scirust_fractional::FractionalOrder;

/// Configuration for the adaptive-step fractional-memory worldline
/// integrator.
///
/// Unlike [`crate::NonlocalConfig`], there is no fixed step or step count:
/// the integrator chooses its own affine-parameter step at each accepted
/// sample, bounded by `min_step` and `max_step`, targeting the componentwise
/// scaled root-mean-square local error defined by [`AdaptiveTolerance`] and
/// [`crate::scaled_local_error_norm`], via the embedded Heun-Euler pair
/// described in this module's documentation. Both adaptive controllers in
/// this crate share that one scaled-norm implementation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveNonlocalConfig {
    fractional_order: FractionalOrder,
    coupling: f64,
    initial_step: f64,
    min_step: f64,
    max_step: f64,
    tolerance: AdaptiveTolerance,
    metric_norm_floor: f64,
    target_affine_parameter: f64,
    max_accepted_steps: usize,
    max_rejections_per_step: usize,
}

impl AdaptiveNonlocalConfig {
    /// Validate and construct an adaptive worldline configuration with a
    /// single scalar error tolerance.
    ///
    /// This is the compatibility constructor: the scalar `error_tolerance`
    /// seeds a uniform [`AdaptiveTolerance`] (the same magnitude for the
    /// relative, coordinate-absolute, and velocity-absolute fields) via
    /// [`AdaptiveTolerance::uniform`]. The step-acceptance decision then uses
    /// the componentwise scaled root-mean-square norm
    /// ([`crate::scaled_local_error_norm`]) rather than the previous unscaled
    /// sum of coordinate and velocity L2 differences, so a run built this way
    /// is *not* bit-for-bit identical to the pre-scaling controller. Use
    /// [`AdaptiveNonlocalConfig::with_tolerance`] to set the three tolerance
    /// fields independently.
    ///
    /// The fractional order must satisfy the `scirust-fractional` interval
    /// `0 < alpha < 1`. The coupling is finite and non-negative. `min_step`,
    /// `max_step`, and `initial_step` are finite and strictly positive, with
    /// `min_step <= initial_step <= max_step`. `error_tolerance`,
    /// `metric_norm_floor`, and `target_affine_parameter` are finite and
    /// strictly positive. `max_accepted_steps` and `max_rejections_per_step`
    /// must be at least one.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        alpha: f64,
        coupling: f64,
        initial_step: f64,
        min_step: f64,
        max_step: f64,
        error_tolerance: f64,
        metric_norm_floor: f64,
        target_affine_parameter: f64,
        max_accepted_steps: usize,
        max_rejections_per_step: usize,
    ) -> NonlocalResult<Self> {
        if !error_tolerance.is_finite() || error_tolerance <= 0.0
        {
            return Err(NonlocalRelativityError::InvalidAdaptiveConfiguration {
                field: "error_tolerance",
                value: error_tolerance,
            });
        }

        let tolerance = AdaptiveTolerance::uniform(error_tolerance)?;

        Self::with_tolerance(
            alpha,
            coupling,
            initial_step,
            min_step,
            max_step,
            tolerance,
            metric_norm_floor,
            target_affine_parameter,
            max_accepted_steps,
            max_rejections_per_step,
        )
    }

    /// Validate and construct an adaptive worldline configuration with an
    /// explicit componentwise [`AdaptiveTolerance`].
    ///
    /// Identical to [`AdaptiveNonlocalConfig::new`] except that the caller
    /// supplies the already-validated relative, coordinate-absolute, and
    /// velocity-absolute tolerances directly, so coordinate and velocity
    /// components can be held to different absolute scales.
    #[allow(clippy::too_many_arguments)]
    pub fn with_tolerance(
        alpha: f64,
        coupling: f64,
        initial_step: f64,
        min_step: f64,
        max_step: f64,
        tolerance: AdaptiveTolerance,
        metric_norm_floor: f64,
        target_affine_parameter: f64,
        max_accepted_steps: usize,
        max_rejections_per_step: usize,
    ) -> NonlocalResult<Self> {
        let fractional_order = FractionalOrder::new(alpha)
            .map_err(|_| NonlocalRelativityError::InvalidFractionalOrder(alpha))?;

        if !coupling.is_finite() || coupling < 0.0
        {
            return Err(NonlocalRelativityError::InvalidCoupling(coupling));
        }

        if !min_step.is_finite() || min_step <= 0.0
        {
            return Err(NonlocalRelativityError::InvalidAdaptiveConfiguration {
                field: "min_step",
                value: min_step,
            });
        }

        if !max_step.is_finite() || max_step < min_step
        {
            return Err(NonlocalRelativityError::InvalidAdaptiveConfiguration {
                field: "max_step",
                value: max_step,
            });
        }

        if !initial_step.is_finite() || initial_step < min_step || initial_step > max_step
        {
            return Err(NonlocalRelativityError::InvalidAdaptiveConfiguration {
                field: "initial_step",
                value: initial_step,
            });
        }

        if !metric_norm_floor.is_finite() || metric_norm_floor <= 0.0
        {
            return Err(NonlocalRelativityError::InvalidMetricNormFloor(
                metric_norm_floor,
            ));
        }

        if !target_affine_parameter.is_finite() || target_affine_parameter <= 0.0
        {
            return Err(NonlocalRelativityError::InvalidAdaptiveConfiguration {
                field: "target_affine_parameter",
                value: target_affine_parameter,
            });
        }

        if max_accepted_steps == 0
        {
            return Err(NonlocalRelativityError::InvalidStepCount(
                max_accepted_steps,
            ));
        }

        if max_rejections_per_step == 0
        {
            return Err(NonlocalRelativityError::InvalidAdaptiveConfiguration {
                field: "max_rejections_per_step",
                value: 0.0,
            });
        }

        Ok(Self {
            fractional_order,
            coupling,
            initial_step,
            min_step,
            max_step,
            tolerance,
            metric_norm_floor,
            target_affine_parameter,
            max_accepted_steps,
            max_rejections_per_step,
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

    /// Return the initial affine-parameter step proposal.
    #[must_use]
    pub const fn initial_step(self) -> f64 {
        self.initial_step
    }

    /// Return the minimum permitted affine-parameter step.
    #[must_use]
    pub const fn min_step(self) -> f64 {
        self.min_step
    }

    /// Return the maximum permitted affine-parameter step.
    #[must_use]
    pub const fn max_step(self) -> f64 {
        self.max_step
    }

    /// Return the componentwise scaled-error tolerance applied per accepted
    /// step.
    #[must_use]
    pub const fn tolerance(self) -> AdaptiveTolerance {
        self.tolerance
    }

    /// Return the positive lower bound for `|g_(mu nu) u^mu u^nu|`.
    #[must_use]
    pub const fn metric_norm_floor(self) -> f64 {
        self.metric_norm_floor
    }

    /// Return the target affine parameter at which integration stops.
    #[must_use]
    pub const fn target_affine_parameter(self) -> f64 {
        self.target_affine_parameter
    }

    /// Return the maximum number of accepted steps before integration must
    /// give up.
    #[must_use]
    pub const fn max_accepted_steps(self) -> usize {
        self.max_accepted_steps
    }

    /// Return the maximum number of consecutive rejections permitted while
    /// shrinking toward one accepted step.
    #[must_use]
    pub const fn max_rejections_per_step(self) -> usize {
        self.max_rejections_per_step
    }
}

/// Typed architecture policy for the adaptive-step integrator: which history
/// backend, geometric transport, and curvature/field modulator compose the
/// non-uniform memory evaluation.
///
/// This mirrors [`crate::NonlocalSimulationPolicy`]'s role for the
/// fixed-step architecture, narrowed to the three components adaptive
/// stepping actually varies (there is no [`crate::WorldlineStepper`] or
/// [`crate::MemoryLaw`] here; see this module's documentation for why).
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveSimulationPolicy<H, T, M> {
    history_backend: H,
    transport: T,
    modulator: M,
}

impl<H, T, M> AdaptiveSimulationPolicy<H, T, M> {
    /// Construct an adaptive policy from explicit architecture components.
    #[must_use]
    pub const fn new(history_backend: H, transport: T, modulator: M) -> Self {
        Self {
            history_backend,
            transport,
            modulator,
        }
    }

    /// Borrow the policy history backend.
    #[must_use]
    pub const fn history_backend(&self) -> &H {
        &self.history_backend
    }

    /// Borrow the policy history transport.
    #[must_use]
    pub const fn transport(&self) -> &T {
        &self.transport
    }

    /// Borrow the policy history modulator.
    #[must_use]
    pub const fn modulator(&self) -> &M {
        &self.modulator
    }
}

struct AdaptiveEvaluation<const D: usize> {
    acceleration: [f64; D],
    diagnostics: StepDiagnostics,
}

#[allow(clippy::too_many_arguments)]
fn evaluate_adaptive_step<B, H, T, M, const D: usize>(
    background: &B,
    state: &WorldlineState<D>,
    history: &H,
    transport: &T,
    modulator: &M,
    initial_metric_norm: f64,
    affine_parameter: f64,
    config: AdaptiveNonlocalConfig,
    step_index: usize,
) -> NonlocalResult<AdaptiveEvaluation<D>>
where
    B: Metric<D> + Connection<D>,
    H: HistoryBackend<D>,
    T: HistoryTransport<D>,
    M: HistoryModulator<D>,
{
    let metric = validated_metric(background, &state.coordinates, step_index)?;
    let metric_norm = validated_metric_norm(
        &metric,
        &state.velocity,
        config.metric_norm_floor(),
        step_index,
    )?;
    let lowered_velocity = lower_index(&metric, &state.velocity);

    let symbols = validated_christoffel(background, &state.coordinates, step_index)?;
    let gr = gr_acceleration(&symbols, &state.velocity);
    validate_vector(&gr, step_index, |step, component, value| {
        NonlocalRelativityError::NonFiniteAcceleration {
            step,
            component,
            value,
        }
    })?;

    let memory = nonuniform_transported_modulated_caputo_velocity_memory(
        history,
        transport,
        modulator,
        state,
        config.fractional_order(),
        step_index,
    )?;
    validate_vector(&memory, step_index, |step, component, value| {
        NonlocalRelativityError::NonFiniteMemory {
            step,
            component,
            value,
        }
    })?;

    let force = projected_memory_force(
        &state.velocity,
        &lowered_velocity,
        metric_norm,
        &memory,
        config.coupling(),
    );
    validate_vector(&force, step_index, |step, component, value| {
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
    validate_vector(&acceleration, step_index, |step, component, value| {
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
        affine_parameter,
        metric_norm,
        metric_norm_drift: metric_norm - initial_metric_norm,
        memory_l2_norm,
        memory_force_l2_norm,
        orthogonality_residual,
        gr_acceleration_l2_norm,
    };

    validate_diagnostics(&diagnostics, step_index)?;

    Ok(AdaptiveEvaluation {
        acceleration,
        diagnostics,
    })
}

struct EmbeddedStepResult<const D: usize> {
    predicted_state: WorldlineState<D>,
    corrected_state: WorldlineState<D>,
}

/// Attempt one embedded Heun-Euler step of size `step` from `state`. Builds
/// its own throwaway clone of `history` (with the trial predicted point
/// pushed into it, transported across the trial segment) to evaluate memory
/// at the predicted point; the caller's persistent `history` is untouched,
/// exactly like [`crate::HeunPeceStepper::advance`]'s internal provisional
/// history is discarded once the fixed-step main loop reads its returned
/// state.
#[allow(clippy::too_many_arguments)]
fn embedded_heun_euler_step<B, H, T, M, const D: usize>(
    background: &B,
    state: &WorldlineState<D>,
    accepted_acceleration: &[f64; D],
    history: &H,
    transport: &T,
    modulator: &M,
    current_parameter: f64,
    step: f64,
    initial_metric_norm: f64,
    config: AdaptiveNonlocalConfig,
    step_index: usize,
) -> NonlocalResult<EmbeddedStepResult<D>>
where
    B: Metric<D> + Connection<D>,
    H: HistoryBackend<D>,
    T: HistoryTransport<D>,
    M: HistoryModulator<D>,
{
    let mut predicted_velocity = [0.0_f64; D];
    let mut predicted_coordinates = [0.0_f64; D];

    for rho in 0..D
    {
        predicted_velocity[rho] = state.velocity[rho] + step * accepted_acceleration[rho];
        validate_generated_velocity(predicted_velocity[rho], step_index, rho)?;

        predicted_coordinates[rho] = state.coordinates[rho] + step * predicted_velocity[rho];
        validate_generated_coordinate(predicted_coordinates[rho], step_index, rho)?;
    }

    let predicted_state = WorldlineState::new(predicted_coordinates, predicted_velocity);
    let predicted_parameter = current_parameter + step;

    let mut provisional_history = history.clone();
    provisional_history.push_entry(
        background,
        transport,
        HistoryEntry::new(
            predicted_coordinates,
            predicted_velocity,
            predicted_parameter,
        ),
    )?;

    let predicted_evaluation = evaluate_adaptive_step(
        background,
        &predicted_state,
        &provisional_history,
        transport,
        modulator,
        initial_metric_norm,
        predicted_parameter,
        config,
        step_index,
    )?;

    let mut corrected_velocity = [0.0_f64; D];
    let mut corrected_coordinates = [0.0_f64; D];

    for rho in 0..D
    {
        corrected_velocity[rho] = state.velocity[rho]
            + 0.5 * step * (accepted_acceleration[rho] + predicted_evaluation.acceleration[rho]);
        validate_generated_velocity(corrected_velocity[rho], step_index, rho)?;

        corrected_coordinates[rho] =
            state.coordinates[rho] + 0.5 * step * (state.velocity[rho] + corrected_velocity[rho]);
        validate_generated_coordinate(corrected_coordinates[rho], step_index, rho)?;
    }

    let corrected_state = WorldlineState::new(corrected_coordinates, corrected_velocity);

    Ok(EmbeddedStepResult {
        predicted_state,
        corrected_state,
    })
}

/// Simulate the experimental fractional-memory worldline model with an
/// adaptive affine-parameter step, composed with a [`AdaptiveSimulationPolicy`]
/// history backend, geometric transport, and curvature/field modulator.
///
/// The step-size controller is the embedded Heun-Euler pair described in
/// this module's documentation. History is transported across each newly
/// accepted segment via [`HistoryBackend::push_entry`] (the same mechanism
/// the fixed-step architecture uses), and each retained sample is weighted
/// by `policy.modulator()` before the non-uniform Caputo evaluation — so
/// `DiscreteConnectionTransport` and `SchwarzschildKretschmannModulator` (or
/// `ReissnerNordstromFieldModulator`) compose with adaptive stepping exactly
/// as they compose with the fixed-step integrators.
///
/// The returned [`NonlocalTrajectory`] samples a generally non-uniform
/// affine-parameter axis: **do not** pass it to
/// [`crate::affine_trajectory_proper_time`], whose `step` argument assumes
/// uniform spacing; read `diagnostics()[i].affine_parameter` directly
/// instead.
///
/// Integration stops once the accumulated affine parameter reaches
/// `config.target_affine_parameter()`, and the final accepted step is
/// clamped so the trajectory does not overshoot it. An
/// [`NonlocalRelativityError::AdaptiveStepBudgetExhausted`] is returned if
/// `config.max_accepted_steps()` accepted steps are used before reaching the
/// target, and an
/// [`NonlocalRelativityError::AdaptiveRejectionBudgetExhausted`] is returned
/// if a single step's error estimate cannot be brought within tolerance
/// without shrinking below `config.min_step()` or exceeding
/// `config.max_rejections_per_step()` retries; neither case silently returns
/// a truncated or out-of-tolerance trajectory.
pub fn simulate_nonlocal_worldline_adaptive_with_policy<B, H, T, M, const D: usize>(
    background: &B,
    initial_state: WorldlineState<D>,
    config: AdaptiveNonlocalConfig,
    policy: AdaptiveSimulationPolicy<H, T, M>,
) -> NonlocalResult<NonlocalTrajectory<D>>
where
    B: Metric<D> + Connection<D>,
    H: HistoryBackend<D>,
    T: HistoryTransport<D>,
    M: HistoryModulator<D>,
{
    validate_initial_state(&initial_state)?;

    let initial_metric = validated_metric(background, &initial_state.coordinates, 0)?;
    let initial_metric_norm = validated_metric_norm(
        &initial_metric,
        &initial_state.velocity,
        config.metric_norm_floor(),
        0,
    )?;

    let mut history = policy.history_backend;
    let transport = policy.transport;
    let modulator = policy.modulator;

    history.push_entry(
        background,
        &transport,
        HistoryEntry::new(initial_state.coordinates, initial_state.velocity, 0.0),
    )?;

    let mut states = vec![initial_state];
    let mut diagnostics_list = Vec::new();
    let mut history_diagnostics_list = Vec::new();

    let initial_evaluation = evaluate_adaptive_step(
        background,
        &initial_state,
        &history,
        &transport,
        &modulator,
        initial_metric_norm,
        0.0,
        config,
        0,
    )?;
    diagnostics_list.push(initial_evaluation.diagnostics);
    history_diagnostics_list.push(history.diagnostics());

    let mut accepted_acceleration = initial_evaluation.acceleration;
    let mut current_parameter = 0.0_f64;
    let mut proposed_step = config.initial_step();
    let mut accepted_count = 0_usize;

    while current_parameter < config.target_affine_parameter()
    {
        if accepted_count >= config.max_accepted_steps()
        {
            return Err(NonlocalRelativityError::AdaptiveStepBudgetExhausted {
                accepted_steps: accepted_count,
                reached_parameter: current_parameter,
                target_affine_parameter: config.target_affine_parameter(),
            });
        }

        let mut step = clamp_step_to_target(
            proposed_step,
            current_parameter,
            config.target_affine_parameter(),
        );
        let step_index = accepted_count + 1;
        // Explicit per-accepted-step rejection counter, reset (by being
        // re-declared) for every new accepted step. The shared `control_step`
        // enforces `max_rejections_per_step` and `min_step` identically for
        // both adaptive controllers.
        let mut rejection_count = 0_usize;

        let (accepted_state, used_step, next_proposed_step) = loop
        {
            let result = embedded_heun_euler_step(
                background,
                &states[accepted_count],
                &accepted_acceleration,
                &history,
                &transport,
                &modulator,
                current_parameter,
                step,
                initial_metric_norm,
                config,
                step_index,
            )?;

            // Componentwise scaled RMS local error (shared by both adaptive
            // controllers); the lower-order Euler predictor and higher-order
            // Heun corrector are the embedded pair.
            let normalized_error = scaled_local_error_norm(
                &result.predicted_state,
                &result.corrected_state,
                config.tolerance(),
            )?;

            match control_step(
                normalized_error,
                step,
                &mut rejection_count,
                accepted_count,
                config.min_step(),
                config.max_step(),
                config.max_rejections_per_step(),
            )?
            {
                StepControl::Accept { next_step } =>
                {
                    break (result.corrected_state, step, next_step);
                },
                StepControl::Retry { next_step } =>
                {
                    step = next_step;
                },
            }
        };

        current_parameter += used_step;
        history.push_entry(
            background,
            &transport,
            HistoryEntry::new(
                accepted_state.coordinates,
                accepted_state.velocity,
                current_parameter,
            ),
        )?;
        states.push(accepted_state);

        let final_evaluation = evaluate_adaptive_step(
            background,
            &accepted_state,
            &history,
            &transport,
            &modulator,
            initial_metric_norm,
            current_parameter,
            config,
            step_index,
        )?;
        diagnostics_list.push(final_evaluation.diagnostics);
        history_diagnostics_list.push(history.diagnostics());

        accepted_acceleration = final_evaluation.acceleration;
        proposed_step = next_proposed_step;
        accepted_count += 1;
    }

    Ok(NonlocalTrajectory::new(
        states,
        diagnostics_list,
        history_diagnostics_list,
    ))
}

/// Simulate the experimental fractional-memory worldline model with an
/// adaptive affine-parameter step, using plain coordinate memory (complete
/// history, no geometric transport, no curvature/field modulation).
///
/// This is exactly
/// [`simulate_nonlocal_worldline_adaptive_with_policy`] with
/// [`AdaptiveSimulationPolicy::new`]`(CompleteUniformHistory::new(),
/// IdentityHistoryTransport, IdentityHistoryModulator)`; use the `_with_policy`
/// entry point directly to compose adaptive stepping with
/// `DiscreteConnectionTransport`, `SchwarzschildKretschmannModulator`,
/// `ReissnerNordstromFieldModulator`, or `BoundedShortMemoryHistory`.
pub fn simulate_nonlocal_worldline_adaptive<B, const D: usize>(
    background: &B,
    initial_state: WorldlineState<D>,
    config: AdaptiveNonlocalConfig,
) -> NonlocalResult<NonlocalTrajectory<D>>
where
    B: Metric<D> + Connection<D>,
{
    simulate_nonlocal_worldline_adaptive_with_policy(
        background,
        initial_state,
        config,
        AdaptiveSimulationPolicy::new(
            CompleteUniformHistory::new(),
            IdentityHistoryTransport,
            IdentityHistoryModulator,
        ),
    )
}
