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
//! baseline implementation keeps complete velocity history and recomputes the
//! Caputo L1 derivative directly, giving `O(D * N^2)` history cost over `N`
//! fixed steps. The semi-implicit Euler update is a deterministic reference
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
use scirust_relativity::{Connection, Metric};
use std::error::Error;
use std::fmt;

/// Result type used by this experimental crate.
pub type NonlocalResult<T> = Result<T, NonlocalRelativityError>;

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
}

impl<const D: usize> NonlocalTrajectory<D> {
    fn new(states: Vec<WorldlineState<D>>, diagnostics: Vec<StepDiagnostics>) -> Self {
        Self {
            states,
            diagnostics,
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
            Self::NonFiniteInitialCoordinate { index, value } => write!(
                formatter,
                "initial coordinate at index {index} is not finite; got {value}"
            ),
            Self::NonFiniteInitialVelocity { index, value } => write!(
                formatter,
                "initial velocity at index {index} is not finite; got {value}"
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
/// states and diagnostics, including the initial sample.
pub fn simulate_nonlocal_worldline<B, const D: usize>(
    background: &B,
    initial_state: WorldlineState<D>,
    config: NonlocalConfig,
) -> NonlocalResult<NonlocalTrajectory<D>>
where
    B: Metric<D> + Connection<D>,
{
    validate_initial_state(&initial_state)?;

    let mut states = Vec::with_capacity(config.steps + 1);
    let mut diagnostics = Vec::with_capacity(config.steps + 1);
    let mut velocity_history = Vec::with_capacity(config.steps + 1);

    states.push(initial_state);
    velocity_history.push(initial_state.velocity);

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
        let evaluation = evaluate_step(
            background,
            &state,
            &velocity_history,
            initial_metric_norm,
            affine_parameter,
            step_index,
            config,
        )?;

        diagnostics.push(evaluation.diagnostics);

        let mut next_velocity = [0.0_f64; D];
        let mut next_coordinates = [0.0_f64; D];

        for rho in 0..D
        {
            next_velocity[rho] = state.velocity[rho] + config.step * evaluation.acceleration[rho];
            if !next_velocity[rho].is_finite()
            {
                return Err(NonlocalRelativityError::NonFiniteGeneratedVelocity {
                    step: step_index + 1,
                    index: rho,
                    value: next_velocity[rho],
                });
            }

            next_coordinates[rho] = state.coordinates[rho] + config.step * next_velocity[rho];
            if !next_coordinates[rho].is_finite()
            {
                return Err(NonlocalRelativityError::NonFiniteGeneratedCoordinate {
                    step: step_index + 1,
                    index: rho,
                    value: next_coordinates[rho],
                });
            }
        }

        let next_state = WorldlineState::new(next_coordinates, next_velocity);
        states.push(next_state);
        velocity_history.push(next_velocity);
    }

    let final_step = config.steps;
    let final_state = states[final_step];
    let final_evaluation = evaluate_step(
        background,
        &final_state,
        &velocity_history,
        initial_metric_norm,
        final_step as f64 * config.step,
        final_step,
        config,
    )?;
    diagnostics.push(final_evaluation.diagnostics);

    Ok(NonlocalTrajectory::new(states, diagnostics))
}

struct StepEvaluation<const D: usize> {
    acceleration: [f64; D],
    diagnostics: StepDiagnostics,
}

fn validate_initial_state<const D: usize>(
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

fn validated_metric<B, const D: usize>(
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

fn validated_metric_norm<const D: usize>(
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

fn validated_christoffel<B, const D: usize>(
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

fn caputo_velocity_memory_at_step<const D: usize>(
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

fn evaluate_step<B, const D: usize>(
    background: &B,
    state: &WorldlineState<D>,
    velocity_history: &[[f64; D]],
    initial_metric_norm: f64,
    affine_parameter: f64,
    step_index: usize,
    config: NonlocalConfig,
) -> NonlocalResult<StepEvaluation<D>>
where
    B: Metric<D> + Connection<D>,
{
    let metric = validated_metric(background, &state.coordinates, step_index)?;
    let metric_norm = validated_metric_norm(
        &metric,
        &state.velocity,
        config.metric_norm_floor,
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

    let memory = caputo_velocity_memory_at_step(
        velocity_history,
        config.step,
        config.fractional_order,
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
        config.coupling,
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

    Ok(StepEvaluation {
        acceleration,
        diagnostics,
    })
}

fn validate_vector<const D: usize, F>(
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

fn validate_diagnostics(
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
