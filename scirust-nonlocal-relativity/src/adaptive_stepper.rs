//! Adaptive-step worldline integration that genuinely composes with
//! [`crate::MemoryLaw`] and [`crate::WorldlineStepper`]
//! ([`crate::SemiImplicitEulerStepper`] specifically), unlike
//! [`crate::simulate_nonlocal_worldline_adaptive_with_policy`]'s embedded
//! Heun-Euler controller, which cannot reuse either trait: both thread a
//! single fixed [`crate::NonlocalConfig`] step through their signatures
//! ([`crate::StepperContext`] for the latter), and
//! [`crate::CaputoCoordinateMemory`] specifically applies that one step to
//! the *entire* retained history via `caputo_l1_uniform`, which is only
//! correct when every accepted segment used the same step.
//!
//! This module closes that gap for [`crate::SemiImplicitEulerStepper`],
//! composed with the two non-uniform-aware memory laws in
//! [`crate::nonuniform_memory`] ([`crate::NonuniformCaputoCoordinateMemory`],
//! [`crate::NonuniformModulatedCaputoCoordinateMemory`]) that read each
//! retained sample's own recorded parameter instead of assuming uniform
//! spacing, via classical step-doubling rather than an embedded pair:
//!
//! 1. advance one full trial step of size `h` with
//!    [`crate::SemiImplicitEulerStepper`] from the accepted state (no memory
//!    evaluation needed for this branch: the accepted acceleration is
//!    already known from the previous accepted step);
//! 2. advance two steps of size `h/2`, evaluating the memory law at the
//!    midpoint against a throwaway provisional history clone (exactly like
//!    [`crate::HeunPeceStepper`]'s predictor push);
//! 3. the difference between the one-step and two-half-step results
//!    estimates the local error; the two-half-steps result (more accurate)
//!    is kept when it is accepted.
//!
//! Both [`crate::SemiImplicitEulerStepper`] and [`crate::HeunPeceStepper`] now
//! derive any provisional affine parameter from
//! [`crate::StepperContext::current_parameter`] (the true accumulated
//! parameter at the accepted state), so both are sound under the non-uniform
//! accepted spacing adaptive stepping produces. Semi-implicit Euler never
//! forms a provisional point at all; Heun-PECE forms one at
//! `current_parameter + config.step`, which is exact for uniform *and*
//! non-uniform spacing (this replaced an earlier `step_index * config.step`
//! reconstruction that was correct only when every accepted step shared one
//! size — see [`crate::StepperContext::current_parameter`]).
//!
//! **[`crate::HeunPeceStepper`] is nonetheless deliberately not offered
//! through *this* step-doubling entry point, for a numerical-analysis reason,
//! not the old parameter-formula one.** This controller's error estimate is
//! classical step-doubling specialized to a *first-order* method: it compares
//! one full step against two half steps and uses their raw difference as the
//! local-error estimate, which is the Richardson estimate only because the
//! divisor `2^p - 1` equals `1` for `p = 1`. Heun-PECE is second order
//! (`p = 2`), so its step-doubling estimate would need the difference divided
//! by `2^2 - 1 = 3` and the step-scaling exponent changed from `1/(p+1) = 0.5`
//! to `1/3`; using this first-order controller unchanged would systematically
//! misestimate a Heun step's error. More importantly, step-doubling is the
//! *wrong tool* for Heun: a second-order method already has a natural embedded
//! first-order partner (its Euler predictor), and the embedded Heun-Euler pair
//! costs one extra acceleration evaluation per step versus step-doubling's
//! three method evaluations. That embedded pair is exactly what
//! [`crate::simulate_nonlocal_worldline_adaptive`] already implements — it
//! computes the same Heun corrector [`crate::HeunPeceStepper::advance`]
//! computes, and additionally exposes the Euler predictor the error estimate
//! needs. So **adaptive Heun-PECE already exists** (the embedded controller);
//! adding Heun-PECE to this step-doubling controller would be a strictly
//! inferior duplicate, not a new capability, and is deliberately omitted per
//! the project's rule against implementing something merely to close a
//! checklist item.
//!
//! The step-size controller's growth/shrink exponent is `1 / (p_low + 1) =
//! 0.5`, exactly as in
//! [`crate::simulate_nonlocal_worldline_adaptive_with_policy`]'s embedded
//! pair, because semi-implicit Euler is *also* a first-order method
//! (`p_low = 1`, local truncation error `O(h^2)`): for a first-order method,
//! classical step-doubling's Richardson error estimate needs no rescaling
//! (the one-step/two-half-step difference divided by `2^1 - 1 = 1` is itself
//! already the estimate), so the raw difference computed below is used
//! directly, exactly as the embedded pair uses its raw predictor/corrector
//! difference directly.
//!
//! Both adaptive controllers now enforce
//! [`crate::AdaptiveNonlocalConfig::max_rejections_per_step`] with identical
//! semantics: each keeps an explicit per-accepted-step rejection counter
//! (reset on every acceptance), returns
//! [`crate::NonlocalRelativityError::AdaptiveRejectionBudgetExhausted`] when
//! the retry count is reached, and
//! [`crate::NonlocalRelativityError::AdaptiveMinimumStepExhausted`] when the
//! proposed shrink would cross
//! [`crate::AdaptiveNonlocalConfig::min_step`] first — two distinct typed
//! errors rather than the single overloaded one earlier revisions used.

use crate::adaptive_control::{StepControl, clamp_step_to_target, control_step};
use crate::{
    AdaptiveNonlocalConfig, CompleteUniformHistory, Connection, HistoryBackend, HistoryEntry,
    HistoryTransport, IdentityHistoryTransport, MemoryLaw, Metric, NonlocalConfig,
    NonlocalRelativityError, NonlocalResult, NonlocalTrajectory, NonuniformCaputoCoordinateMemory,
    SemiImplicitEulerStepper, StepEvaluationInput, StepperContext, WorldlineState,
    WorldlineStepper, evaluate_step_with_policy, scaled_local_error_norm, validate_initial_state,
    validated_metric, validated_metric_norm,
};

/// Typed architecture policy for the step-doubling adaptive integrator:
/// which history backend, [`crate::MemoryLaw`], and
/// [`crate::HistoryTransport`] compose the non-uniform memory evaluation
/// around [`crate::SemiImplicitEulerStepper`].
///
/// This mirrors [`crate::NonlocalSimulationPolicy`]'s role for the
/// fixed-step architecture, narrowed to the components this module actually
/// varies: there is no stepper type parameter, because
/// [`crate::SemiImplicitEulerStepper`] is the only
/// [`crate::WorldlineStepper`] this module can soundly drive (see the module
/// documentation for why).
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveStepperPolicy<H, L, T> {
    history_backend: H,
    memory_law: L,
    transport: T,
}

impl<H, L, T> AdaptiveStepperPolicy<H, L, T> {
    /// Construct a policy from explicit architecture components.
    #[must_use]
    pub const fn new(history_backend: H, memory_law: L, transport: T) -> Self {
        Self {
            history_backend,
            memory_law,
            transport,
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
    pub const fn transport(&self) -> &T {
        &self.transport
    }
}

/// Persistent-history retention strategy for the step-doubling adaptive
/// integrator.
///
/// Each accepted step of the step-doubling controller computes a midpoint
/// (`half_state_1`, at `current_parameter + step/2`) and an endpoint
/// (`half_state_2`, at `current_parameter + step`) along the two-half-step
/// branch. This enum selects which of them are retained in the persistent
/// history the hereditary Caputo memory reads.
///
/// This is a research control, not a tuning knob: the default is unchanged
/// behaviour. See `experiments/nonlocal-relativity-v2` and the v2 paper for
/// the measured comparison of the two strategies against a fine independent
/// reference, and the resulting decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HistoryRetention {
    /// Retain only the accepted endpoint of each accepted step (the shipped
    /// default). The two-half-step midpoint is used to evaluate the trial and
    /// is then discarded, so the persistent history samples exactly the
    /// accepted endpoints.
    #[default]
    EndpointOnly,
    /// Retain both the accepted midpoint and the accepted endpoint of each
    /// accepted step, at their true affine parameters — a denser retained
    /// history for the hereditary memory evaluation, at a higher `O(N^2)`
    /// memory cost (roughly `4x`, since the retained sample count roughly
    /// doubles). Provided for the research comparison; combine only with a
    /// coordinate-identity transport (a geometric transport across two
    /// half-segments per step is a separate, unaddressed concern).
    RefinedAcceptedHistory,
}

struct StepDoublingResult<const D: usize> {
    full_state: WorldlineState<D>,
    /// The accepted midpoint of the two-half-step branch (`half_state_1`), at
    /// affine parameter `current_parameter + step/2`. Retained persistently
    /// only under [`HistoryRetention::RefinedAcceptedHistory`].
    refined_midpoint: WorldlineState<D>,
    refined_state: WorldlineState<D>,
}

/// Attempt one step-doubling trial of size `step` from `state`: one full
/// [`crate::SemiImplicitEulerStepper`] step, compared against two half steps
/// with a memory-law evaluation at the midpoint (against a throwaway
/// provisional history clone). See the module documentation.
#[allow(clippy::too_many_arguments)]
fn step_doubling_trial<B, H, L, T, const D: usize>(
    background: &B,
    state: &WorldlineState<D>,
    accepted_acceleration: &[f64; D],
    history: &H,
    memory_law: &L,
    transport: &T,
    current_parameter: f64,
    step: f64,
    initial_metric_norm: f64,
    config_template: NonlocalConfig,
    step_index: usize,
) -> NonlocalResult<StepDoublingResult<D>>
where
    B: Metric<D> + Connection<D>,
    H: HistoryBackend<D>,
    L: MemoryLaw<D>,
    T: HistoryTransport<D>,
{
    let full_step_config = NonlocalConfig::from_fractional_order(
        config_template.fractional_order(),
        config_template.coupling(),
        step,
        1,
        config_template.metric_norm_floor(),
    )?;

    let full_state = SemiImplicitEulerStepper.advance(StepperContext {
        background,
        state,
        accepted_acceleration,
        history,
        memory_law,
        transport,
        initial_metric_norm,
        current_parameter,
        step_index,
        config: full_step_config,
    })?;

    let half_step = 0.5 * step;
    let half_step_config = NonlocalConfig::from_fractional_order(
        config_template.fractional_order(),
        config_template.coupling(),
        half_step,
        1,
        config_template.metric_norm_floor(),
    )?;

    let half_state_1 = SemiImplicitEulerStepper.advance(StepperContext {
        background,
        state,
        accepted_acceleration,
        history,
        memory_law,
        transport,
        initial_metric_norm,
        current_parameter,
        step_index,
        config: half_step_config,
    })?;

    let midpoint_parameter = current_parameter + half_step;
    let mut provisional_history = history.clone();
    provisional_history.push_entry(
        background,
        transport,
        HistoryEntry::new(
            half_state_1.coordinates,
            half_state_1.velocity,
            midpoint_parameter,
        ),
    )?;

    let midpoint_evaluation = evaluate_step_with_policy(StepEvaluationInput {
        background,
        state: &half_state_1,
        history: &provisional_history,
        memory_law,
        transport,
        initial_metric_norm,
        affine_parameter: midpoint_parameter,
        step_index,
        config: half_step_config,
    })?;

    let half_state_2 = SemiImplicitEulerStepper.advance(StepperContext {
        background,
        state: &half_state_1,
        accepted_acceleration: &midpoint_evaluation.acceleration,
        history: &provisional_history,
        memory_law,
        transport,
        initial_metric_norm,
        current_parameter: midpoint_parameter,
        step_index,
        config: half_step_config,
    })?;

    Ok(StepDoublingResult {
        full_state,
        refined_midpoint: half_state_1,
        refined_state: half_state_2,
    })
}

/// Simulate the experimental fractional-memory worldline model with an
/// adaptive affine-parameter step, genuinely composed with
/// [`crate::MemoryLaw`] and [`crate::SemiImplicitEulerStepper`] via
/// classical step-doubling. See the module documentation for the mechanism,
/// why [`crate::SemiImplicitEulerStepper`] specifically is reused, and why
/// [`crate::HeunPeceStepper`] is not offered here.
///
/// This is [`simulate_nonlocal_worldline_adaptive_with_stepper_policy_retention`]
/// with [`HistoryRetention::EndpointOnly`] (the shipped default). Use the
/// `_retention` entry point to select
/// [`HistoryRetention::RefinedAcceptedHistory`] for the research comparison.
///
/// As with [`crate::simulate_nonlocal_worldline_adaptive_with_policy`], the
/// returned [`crate::NonlocalTrajectory`] samples a generally non-uniform
/// affine-parameter axis, integration stops once the accumulated affine
/// parameter reaches `config.target_affine_parameter()` (the final accepted
/// step is clamped so the trajectory does not overshoot it), and neither an
/// exhausted step budget
/// ([`crate::NonlocalRelativityError::AdaptiveStepBudgetExhausted`]) nor an
/// exhausted rejection budget
/// ([`crate::NonlocalRelativityError::AdaptiveRejectionBudgetExhausted`])
/// is silently swallowed into a truncated or out-of-tolerance trajectory.
pub fn simulate_nonlocal_worldline_adaptive_with_stepper_policy<B, H, L, T, const D: usize>(
    background: &B,
    initial_state: WorldlineState<D>,
    config: AdaptiveNonlocalConfig,
    policy: AdaptiveStepperPolicy<H, L, T>,
) -> NonlocalResult<NonlocalTrajectory<D>>
where
    B: Metric<D> + Connection<D>,
    H: HistoryBackend<D>,
    L: MemoryLaw<D>,
    T: HistoryTransport<D>,
{
    simulate_nonlocal_worldline_adaptive_with_stepper_policy_retention(
        background,
        initial_state,
        config,
        policy,
        HistoryRetention::EndpointOnly,
    )
}

/// Simulate the step-doubling adaptive worldline model with an explicit
/// [`HistoryRetention`] strategy.
///
/// Identical to [`simulate_nonlocal_worldline_adaptive_with_stepper_policy`]
/// (which passes [`HistoryRetention::EndpointOnly`]) except that
/// [`HistoryRetention::RefinedAcceptedHistory`] additionally persists each
/// accepted step's midpoint at its true affine parameter
/// (`current_parameter + step/2`) before the endpoint, giving the hereditary
/// Caputo memory a denser retained history. The returned trajectory's
/// `states()` remain one accepted endpoint per step under either strategy;
/// only the retained history the memory reads (reflected in
/// `history_diagnostics()`) differs. See the module docs, the experiment
/// suite, and the v2 paper for the measured comparison.
pub fn simulate_nonlocal_worldline_adaptive_with_stepper_policy_retention<
    B,
    H,
    L,
    T,
    const D: usize,
>(
    background: &B,
    initial_state: WorldlineState<D>,
    config: AdaptiveNonlocalConfig,
    policy: AdaptiveStepperPolicy<H, L, T>,
    retention: HistoryRetention,
) -> NonlocalResult<NonlocalTrajectory<D>>
where
    B: Metric<D> + Connection<D>,
    H: HistoryBackend<D>,
    L: MemoryLaw<D>,
    T: HistoryTransport<D>,
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
    let memory_law = policy.memory_law;
    let transport = policy.transport;

    history.push_entry(
        background,
        &transport,
        HistoryEntry::new(initial_state.coordinates, initial_state.velocity, 0.0),
    )?;

    let mut states = vec![initial_state];
    let mut diagnostics_list = Vec::new();
    let mut history_diagnostics_list = Vec::new();

    let config_template = NonlocalConfig::from_fractional_order(
        config.fractional_order(),
        config.coupling(),
        config.initial_step(),
        1,
        config.metric_norm_floor(),
    )?;

    let initial_evaluation = evaluate_step_with_policy(StepEvaluationInput {
        background,
        state: &initial_state,
        history: &history,
        memory_law: &memory_law,
        transport: &transport,
        initial_metric_norm,
        affine_parameter: 0.0,
        step_index: 0,
        config: config_template,
    })?;
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
        let mut rejection_count = 0_usize;

        let (accepted_state, accepted_midpoint, used_step, next_proposed_step) = loop
        {
            let trial = step_doubling_trial(
                background,
                &states[accepted_count],
                &accepted_acceleration,
                &history,
                &memory_law,
                &transport,
                current_parameter,
                step,
                initial_metric_norm,
                config_template,
                step_index,
            )?;

            // Componentwise scaled RMS local error (shared by both adaptive
            // controllers); the one full step and the two half steps are the
            // step-doubling Richardson pair.
            let normalized_error = scaled_local_error_norm(
                &trial.full_state,
                &trial.refined_state,
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
                    break (trial.refined_state, trial.refined_midpoint, step, next_step);
                },
                StepControl::Retry { next_step } =>
                {
                    step = next_step;
                },
            }
        };

        // The accepted step's midpoint parameter, computed before advancing
        // `current_parameter`. `2 * (0.5 * used_step) == used_step` exactly in
        // IEEE-754, so the endpoint parameter below is unaffected.
        let midpoint_parameter = current_parameter + 0.5 * used_step;
        current_parameter += used_step;

        if retention == HistoryRetention::RefinedAcceptedHistory
        {
            // Persist the accepted midpoint (strictly between the previous
            // endpoint and this endpoint) before the endpoint, so the retained
            // history stays strictly parameter-ordered with no duplicates.
            history.push_entry(
                background,
                &transport,
                HistoryEntry::new(
                    accepted_midpoint.coordinates,
                    accepted_midpoint.velocity,
                    midpoint_parameter,
                ),
            )?;
        }

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

        let final_evaluation = evaluate_step_with_policy(StepEvaluationInput {
            background,
            state: &accepted_state,
            history: &history,
            memory_law: &memory_law,
            transport: &transport,
            initial_metric_norm,
            affine_parameter: current_parameter,
            step_index,
            config: config_template,
        })?;
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
/// adaptive, step-doubling-controlled affine-parameter step, using plain
/// non-uniform coordinate memory (complete history, no geometric transport).
///
/// This is exactly
/// [`simulate_nonlocal_worldline_adaptive_with_stepper_policy`] with
/// [`AdaptiveStepperPolicy::new`]`(CompleteUniformHistory::new(),
/// NonuniformCaputoCoordinateMemory, IdentityHistoryTransport)`; use the
/// `_with_stepper_policy` entry point directly to compose with
/// [`crate::NonuniformModulatedCaputoCoordinateMemory`],
/// [`crate::DiscreteConnectionTransport`], or
/// [`crate::BoundedShortMemoryHistory`].
pub fn simulate_nonlocal_worldline_adaptive_with_stepper<B, const D: usize>(
    background: &B,
    initial_state: WorldlineState<D>,
    config: AdaptiveNonlocalConfig,
) -> NonlocalResult<NonlocalTrajectory<D>>
where
    B: Metric<D> + Connection<D>,
{
    simulate_nonlocal_worldline_adaptive_with_stepper_policy(
        background,
        initial_state,
        config,
        AdaptiveStepperPolicy::new(
            CompleteUniformHistory::new(),
            NonuniformCaputoCoordinateMemory,
            IdentityHistoryTransport,
        ),
    )
}
