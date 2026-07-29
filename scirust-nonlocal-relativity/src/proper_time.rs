//! Affine vs. proper-time parameterization modes.
//!
//! The fixed-step integrators in this crate always advance a uniform
//! parameter `lambda_n = n * h`. [`ParameterizationMode`] does not change
//! that numerical scheme; it only says how the parameter is *interpreted*
//! and what extra timelike-normalization checks apply before and after the
//! run. `AffineParameter` keeps the original, unconstrained behavior.
//! `NormalizedTimelikeProperTime` additionally requires the initial state to
//! be timelike with `g(u,u)` close to `-1`, and rejects any sampled step
//! whose metric norm drifts away from `-1` beyond a configurable tolerance.
//! No automatic four-velocity renormalization is ever performed; drift is
//! reported as a typed error instead of silently repaired.

use crate::{
    Connection, HistoryBackend, HistoryTransport, MemoryLaw, Metric, NonlocalConfig,
    NonlocalRelativityError, NonlocalResult, NonlocalSimulationPolicy, NonlocalTrajectory,
    WorldlineState, WorldlineStepper, metric_contraction, simulate_nonlocal_worldline_with_policy,
    validate_initial_state, validated_metric,
};
use scirust_fractional::{FractionalOrder, caputo_l1_nonuniform};

/// Explicit parameterization mode for a worldline simulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParameterizationMode {
    /// The configured step is an affine-parameter step. This is the default,
    /// unconstrained mode: it makes no timelike-normalization assumption and
    /// is bit-for-bit compatible with every other simulation entry point in
    /// this crate.
    AffineParameter,
    /// The configured step is a proper-time step.
    ///
    /// The initial state must be timelike with metric norm within
    /// `tolerance` of `-1` under a `(-,+,+,+)` signature background, and the
    /// metric norm at every subsequently sampled step must stay within
    /// `tolerance` of `-1`. No automatic four-velocity renormalization is
    /// performed; a drift beyond `tolerance` is reported as a typed error.
    NormalizedTimelikeProperTime {
        /// Positive finite tolerance for `|g(u,u) - (-1)|`, checked at the
        /// initial sample and at every subsequently sampled step.
        tolerance: f64,
    },
}

/// Simulate with a typed advanced policy under an explicit parameterization
/// mode.
///
/// With [`ParameterizationMode::AffineParameter`] this is exactly
/// [`simulate_nonlocal_worldline_with_policy`] with no extra checks: it
/// reproduces that function's output bit-for-bit for identical inputs. With
/// [`ParameterizationMode::NormalizedTimelikeProperTime`], the initial state
/// is validated as timelike and normalized before the run, and every sampled
/// step's metric norm is validated against the same tolerance after the run
/// completes; both checks use the accepted-state metric norm already
/// computed by the underlying simulation; neither renormalizes the state.
pub fn simulate_nonlocal_worldline_with_mode<B, H, L, T, S, const D: usize>(
    background: &B,
    initial_state: WorldlineState<D>,
    config: NonlocalConfig,
    policy: NonlocalSimulationPolicy<D, H, L, T, S>,
    mode: ParameterizationMode,
) -> NonlocalResult<NonlocalTrajectory<D>>
where
    B: Metric<D> + Connection<D>,
    H: HistoryBackend<D>,
    L: MemoryLaw<D>,
    T: HistoryTransport<D>,
    S: WorldlineStepper<D>,
{
    if let ParameterizationMode::NormalizedTimelikeProperTime { tolerance } = mode
    {
        validate_proper_time_tolerance(tolerance)?;
        validate_initial_state(&initial_state)?;
        let metric = validated_metric(background, &initial_state.coordinates, 0)?;
        let norm = metric_contraction(&metric, &initial_state.velocity, &initial_state.velocity);
        check_proper_time_norm(0, norm, tolerance)?;
    }

    let trajectory =
        simulate_nonlocal_worldline_with_policy(background, initial_state, config, policy)?;

    if let ParameterizationMode::NormalizedTimelikeProperTime { tolerance } = mode
    {
        for (step, diagnostics) in trajectory.diagnostics().iter().enumerate()
        {
            check_proper_time_norm(step, diagnostics.metric_norm, tolerance)?;
        }
    }

    Ok(trajectory)
}

fn validate_proper_time_tolerance(tolerance: f64) -> NonlocalResult<()> {
    if !tolerance.is_finite() || tolerance <= 0.0
    {
        return Err(NonlocalRelativityError::InvalidProperTimeTolerance(
            tolerance,
        ));
    }

    Ok(())
}

fn check_proper_time_norm(step: usize, metric_norm: f64, tolerance: f64) -> NonlocalResult<()> {
    if !metric_norm.is_finite() || (metric_norm - (-1.0)).abs() > tolerance
    {
        return Err(NonlocalRelativityError::ProperTimeNormDrift {
            step,
            metric_norm,
            tolerance,
        });
    }

    Ok(())
}

/// One estimated proper-time sample derived from an affine-parameter
/// trajectory of timelike states.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProperTimeDiagnostics {
    /// Affine parameter at the start of this step.
    pub affine_parameter: f64,
    /// Estimated proper-time increment for this step.
    pub proper_time_increment: f64,
    /// Cumulative estimated proper time from the initial sample.
    pub cumulative_proper_time: f64,
}

/// Estimate proper-time diagnostics for an affine-parameter trajectory of
/// timelike states.
///
/// Uses the left-endpoint quadrature `Delta tau_n ~= h * sqrt(-g(u,u)_n)`,
/// evaluated with the metric norm accepted at the *start* of each step (the
/// same accepted-state metric norm already recorded in
/// `StepDiagnostics::metric_norm`). This is a first-order-accurate estimate
/// of physical proper time elapsed along an affine-parameter trajectory,
/// exposed purely as a diagnostic: it is not a resampling of the trajectory
/// onto a uniform proper-time grid, and the returned increments are
/// generally non-uniform. They must never be passed to
/// `scirust_fractional::caputo_l1_uniform`, which requires uniform sample
/// spacing that this estimate does not provide.
pub fn affine_trajectory_proper_time<const D: usize>(
    trajectory: &NonlocalTrajectory<D>,
    step: f64,
) -> NonlocalResult<Vec<ProperTimeDiagnostics>> {
    if !step.is_finite() || step <= 0.0
    {
        return Err(NonlocalRelativityError::InvalidStep(step));
    }

    let diagnostics = trajectory.diagnostics();
    let mut output = Vec::with_capacity(diagnostics.len().saturating_sub(1));
    let mut cumulative = 0.0_f64;

    for window in diagnostics.windows(2)
    {
        let start = window[0];

        if !start.metric_norm.is_finite() || start.metric_norm >= 0.0
        {
            return Err(NonlocalRelativityError::NonTimelikeMetricNorm {
                metric_norm: start.metric_norm,
            });
        }

        let increment = step * (-start.metric_norm).sqrt();

        if !increment.is_finite()
        {
            return Err(NonlocalRelativityError::NonFiniteDiagnostic {
                step: output.len(),
                quantity: "proper_time_increment",
                value: increment,
            });
        }

        cumulative += increment;

        if !cumulative.is_finite()
        {
            return Err(NonlocalRelativityError::NonFiniteDiagnostic {
                step: output.len(),
                quantity: "cumulative_proper_time",
                value: cumulative,
            });
        }

        output.push(ProperTimeDiagnostics {
            affine_parameter: start.affine_parameter,
            proper_time_increment: increment,
            cumulative_proper_time: cumulative,
        });
    }

    Ok(output)
}

/// Estimated proper-time value at every sampled state of an affine-parameter
/// trajectory (length equal to `trajectory.len()`, with `axis[0] == 0.0`),
/// built from [`affine_trajectory_proper_time`]'s cumulative estimates.
///
/// Because each accepted step's proper-time increment
/// `Delta tau_n ~= h * sqrt(-g(u,u)_n)` depends on that step's own metric
/// norm, this axis is generally non-uniform even though the affine parameter
/// itself is uniform; a strictly negative metric norm makes each increment
/// strictly positive, so the axis is always strictly increasing.
fn proper_time_axis<const D: usize>(
    trajectory: &NonlocalTrajectory<D>,
    step: f64,
) -> NonlocalResult<Vec<f64>> {
    let increments = affine_trajectory_proper_time(trajectory, step)?;
    let mut axis = Vec::with_capacity(trajectory.len());
    axis.push(0.0);

    for entry in &increments
    {
        axis.push(entry.cumulative_proper_time);
    }

    Ok(axis)
}

/// Evaluate the Caputo velocity-memory vector of an affine-parameter
/// trajectory with respect to its own estimated **proper time** rather than
/// the affine parameter, using [`scirust_fractional::caputo_l1_nonuniform`]
/// on the (generally non-uniform) axis from [`affine_trajectory_proper_time`].
///
/// This is a pure post-hoc diagnostic over an already-computed trajectory: it
/// does not feed back into the live integration loop, does not change any
/// accepted state, and is unrelated to [`ParameterizationMode::NormalizedTimelikeProperTime`],
/// which advances the state equation using a *uniform* proper-time step by
/// construction. This function instead asks what the Caputo memory *would
/// have been* if evaluated against the trajectory's own non-uniform elapsed
/// proper time — meaningful for any timelike affine-parameter trajectory,
/// including ones whose proper time does not advance uniformly with the
/// affine parameter (for example, a non-circular orbit).
///
/// `step` is the trajectory's uniform affine-parameter step, used only to
/// estimate the proper-time axis; every sampled state must be timelike
/// (`g(u,u) < 0`).
pub fn proper_time_caputo_velocity_memory<const D: usize>(
    trajectory: &NonlocalTrajectory<D>,
    step: f64,
    order: FractionalOrder,
) -> NonlocalResult<[f64; D]> {
    let proper_times = proper_time_axis(trajectory, step)?;
    let final_step = trajectory.len().saturating_sub(1);
    let mut memory = [0.0_f64; D];

    for (component, memory_component) in memory.iter_mut().enumerate()
    {
        let samples: Vec<f64> = trajectory
            .states()
            .iter()
            .map(|state| state.velocity[component])
            .collect();

        let value = caputo_l1_nonuniform(&samples, &proper_times, order).map_err(|source| {
            NonlocalRelativityError::FractionalMemory {
                step: final_step,
                component,
                source,
            }
        })?;

        if !value.is_finite()
        {
            return Err(NonlocalRelativityError::NonFiniteMemory {
                step: final_step,
                component,
                value,
            });
        }

        *memory_component = value;
    }

    Ok(memory)
}
