//! Homogeneous ADM time evolution (Layer 3.2 — free evolution with constraint
//! monitoring).
//!
//! Layer 3.1 ([`crate::adm_evolution`]) evaluates the ADM constraints and
//! evolution right-hand sides at a point but evolves nothing. This module
//! integrates those right-hand sides **forward in time** in the one sector
//! where the result can be checked against exact closed-form solutions: the
//! spatially homogeneous, spatially flat (cosmological) sector, with a
//! barotropic perfect fluid `p = w rho`.
//!
//! **Why only the homogeneous sector.** The ADM system is only *weakly
//! hyperbolic*, so free evolution of generic inhomogeneous data on a grid is
//! numerically unstable — that instability is the reason BSSN and
//! generalized-harmonic formulations exist. Every field here is
//! position-independent, so all spatial derivatives vanish identically and the
//! pathology (which lives entirely in the spatial-derivative principal part)
//! cannot arise: the evolution is a well-posed system of ODEs. **Nothing here
//! says anything about ADM's stability for inhomogeneous data.** See
//! `docs/LAYER_3_HOMOGENEOUS_EVOLUTION.md`.
//!
//! **Category.** Established general relativity — the ADM evolution equations
//! and the Friedmann solutions are textbook. The numbers are a numerical
//! approximation (fixed-step RK4 through finite-difference right-hand sides),
//! with truncation error measured and reported, never hidden. This is not a
//! cosmological model: the three equations of state are validation oracles, not
//! a fit to any observation.
//!
//! ## Conventions
//!
//! Signature `(-,+,+,+)`, `G = c = 1`, inherited unchanged from
//! [`crate::adm_evolution`] and [`crate::adm`] — in particular
//! `K_ij = -1/(2N)(partial_t gamma_ij - D_i N_j - D_j N_i)`, so an *expanding*
//! universe has *negative* `K` (`K = -3H`). With unit lapse, zero shift, and
//! position-independent fields:
//!
//! ```text
//! partial_t gamma_ij = -2 K_ij
//! partial_t K_ij     = K K_ij - 2 K_ik K^k_j - 8 pi ( S_ij - 1/2 gamma_ij (S - rho) )
//! partial_t rho      = -3 H ( rho + p ) ,     H = -K/3 ,   K = gamma^{ij} K_ij
//! ```
//!
//! and the Hamiltonian constraint reduces to the first Friedmann equation
//! `H^2 = 8 pi rho / 3`. The reduction is **not** hard-coded: the right-hand
//! sides are obtained by calling the Layer 3.1 evaluators with genuinely
//! constant-in-space field adapters, so the same code path an inhomogeneous
//! evolution would take is exercised (the spatial derivatives simply evaluate
//! to exact zeros).
//!
//! Time stepping reuses [`scirust_sim::simulate`] — the platform's existing
//! deterministic fixed-step RK4 — rather than introducing a new integrator.
//!
//! **Free evolution with constraint monitoring.** The Hamiltonian constraint is
//! evaluated as a *diagnostic* at every recorded sample; it is never enforced,
//! re-projected, or damped. Constraint drift is therefore visible rather than
//! hidden.

use std::f64::consts::PI;
use std::fmt;

use scirust_sim::{SimError, System, simulate};

use crate::adm_evolution::{
    AdmEvolutionError, AdmEvolutionSettings, AdmSources, SpatialScalarField, SpatialTensorField,
    SpatialVectorField, curvature_evolution_rhs, hamiltonian_constraint, metric_evolution_rhs,
};
use crate::{Metric, determinant, invert_metric};

/// Number of flattened state components: `gamma_ij` (9), `K_ij` (9), `rho` (1).
const STATE_DIMENSION: usize = 19;

/// The single point at which the position-independent fields are sampled.
const ORIGIN: [f64; 3] = [0.0, 0.0, 0.0];

/// A typed failure of the homogeneous evolution. It never panics.
#[derive(Debug, Clone, PartialEq)]
pub enum HomogeneousEvolutionError {
    /// The equation-of-state parameter `w` is not finite.
    InvalidEquationOfState(f64),
    /// The energy density is not finite, or is negative.
    InvalidEnergyDensity(f64),
    /// The requested time span or step is malformed.
    InvalidTimeSpan {
        /// Start time.
        start: f64,
        /// End time.
        end: f64,
        /// Step size.
        step: f64,
    },
    /// The spatial metric is singular or not positive-definite.
    InvalidSpatialMetric,
    /// The integrator rejected the request or the state became non-finite.
    Integration(String),
    /// Evaluating the constraints or right-hand sides failed.
    Evaluation(AdmEvolutionError),
}

impl fmt::Display for HomogeneousEvolutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidEquationOfState(value) =>
            {
                write!(f, "equation-of-state parameter {value} must be finite")
            },
            Self::InvalidEnergyDensity(value) =>
            {
                write!(f, "energy density {value} must be finite and non-negative")
            },
            Self::InvalidTimeSpan { start, end, step } =>
            {
                write!(
                    f,
                    "invalid time span: need finite start {start} < end {end} and a finite positive step {step}"
                )
            },
            Self::InvalidSpatialMetric =>
            {
                write!(f, "the spatial metric is singular or not positive-definite")
            },
            Self::Integration(message) => write!(f, "integration failed: {message}"),
            Self::Evaluation(error) => write!(f, "evaluation failed: {error}"),
        }
    }
}

impl std::error::Error for HomogeneousEvolutionError {}

impl From<AdmEvolutionError> for HomogeneousEvolutionError {
    fn from(error: AdmEvolutionError) -> Self {
        Self::Evaluation(error)
    }
}

impl From<SimError> for HomogeneousEvolutionError {
    fn from(error: SimError) -> Self {
        Self::Integration(error.to_string())
    }
}

/// A barotropic perfect fluid `p = w rho`.
///
/// `w = -1` is vacuum energy (a cosmological constant), `w = 0` is dust, and
/// `w = 1/3` is radiation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BarotropicFluid {
    equation_of_state: f64,
}

impl BarotropicFluid {
    /// Construct from a finite equation-of-state parameter `w`.
    pub fn try_new(equation_of_state: f64) -> Result<Self, HomogeneousEvolutionError> {
        if !equation_of_state.is_finite()
        {
            return Err(HomogeneousEvolutionError::InvalidEquationOfState(
                equation_of_state,
            ));
        }
        Ok(Self { equation_of_state })
    }

    /// Vacuum energy (a cosmological constant): `w = -1`.
    #[must_use]
    pub const fn vacuum_energy() -> Self {
        Self {
            equation_of_state: -1.0,
        }
    }

    /// Pressureless dust: `w = 0`.
    #[must_use]
    pub const fn dust() -> Self {
        Self {
            equation_of_state: 0.0,
        }
    }

    /// Radiation: `w = 1/3`.
    #[must_use]
    pub fn radiation() -> Self {
        Self {
            equation_of_state: 1.0 / 3.0,
        }
    }

    /// The equation-of-state parameter `w`.
    #[must_use]
    pub const fn equation_of_state(&self) -> f64 {
        self.equation_of_state
    }

    /// The pressure `p = w rho`.
    #[must_use]
    pub fn pressure(&self, energy_density: f64) -> f64 {
        self.equation_of_state * energy_density
    }
}

/// The homogeneous ADM state on one slice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HomogeneousState {
    /// The spatial 3-metric `gamma_ij`.
    pub spatial_metric: [[f64; 3]; 3],
    /// The extrinsic curvature `K_ij`.
    pub extrinsic_curvature: [[f64; 3]; 3],
    /// The fluid energy density `rho`.
    pub energy_density: f64,
}

impl HomogeneousState {
    /// Build spatially flat, isotropic initial data
    /// `gamma_ij = a^2 delta_ij`, `K_ij = -H gamma_ij`, with the energy density
    /// **set by the Hamiltonian constraint** `rho = 3 H^2 / (8 pi)`, so the data
    /// starts exactly on the constraint surface.
    ///
    /// Returns an error for a non-finite or non-positive scale factor, or a
    /// non-finite Hubble parameter.
    pub fn friedmann(
        scale_factor: f64,
        hubble_parameter: f64,
    ) -> Result<Self, HomogeneousEvolutionError> {
        if !scale_factor.is_finite() || scale_factor <= 0.0
        {
            return Err(HomogeneousEvolutionError::InvalidSpatialMetric);
        }
        if !hubble_parameter.is_finite()
        {
            return Err(HomogeneousEvolutionError::InvalidEnergyDensity(
                hubble_parameter,
            ));
        }
        let spatial = scale_factor * scale_factor;
        let mut spatial_metric = [[0.0_f64; 3]; 3];
        let mut extrinsic_curvature = [[0.0_f64; 3]; 3];
        for i in 0..3
        {
            spatial_metric[i][i] = spatial;
            extrinsic_curvature[i][i] = -hubble_parameter * spatial;
        }
        Ok(Self {
            spatial_metric,
            extrinsic_curvature,
            energy_density: 3.0 * hubble_parameter * hubble_parameter / (8.0 * PI),
        })
    }

    /// The scale factor `a = (det gamma)^{1/6}`.
    ///
    /// Derived from the determinant rather than a single component so that it
    /// stays meaningful if rounding takes the state slightly off isotropy.
    pub fn scale_factor(&self) -> Result<f64, HomogeneousEvolutionError> {
        let det = determinant::<3>(&self.spatial_metric)
            .map_err(|_| HomogeneousEvolutionError::InvalidSpatialMetric)?;
        if !det.is_finite() || det <= 0.0
        {
            return Err(HomogeneousEvolutionError::InvalidSpatialMetric);
        }
        Ok(det.powf(1.0 / 6.0))
    }

    /// The mean curvature `K = gamma^{ij} K_ij`.
    // Explicit tensor-index loops read most clearly here (matching the crate's
    // curvature, action, adm, and adm_evolution modules).
    #[allow(clippy::needless_range_loop)]
    pub fn mean_curvature(&self) -> Result<f64, HomogeneousEvolutionError> {
        let inverse = invert_metric::<3>(&self.spatial_metric)
            .map_err(|_| HomogeneousEvolutionError::InvalidSpatialMetric)?;
        let mut value = 0.0;
        for i in 0..3
        {
            for j in 0..3
            {
                value += inverse[i][j] * self.extrinsic_curvature[i][j];
            }
        }
        Ok(value)
    }

    /// The Hubble parameter `H = -K/3`.
    pub fn hubble_parameter(&self) -> Result<f64, HomogeneousEvolutionError> {
        Ok(-self.mean_curvature()? / 3.0)
    }

    fn flatten(&self, state: &mut [f64]) {
        for i in 0..3
        {
            for j in 0..3
            {
                state[3 * i + j] = self.spatial_metric[i][j];
                state[9 + 3 * i + j] = self.extrinsic_curvature[i][j];
            }
        }
        state[18] = self.energy_density;
    }

    fn unflatten(state: &[f64]) -> Self {
        let mut spatial_metric = [[0.0_f64; 3]; 3];
        let mut extrinsic_curvature = [[0.0_f64; 3]; 3];
        for i in 0..3
        {
            for j in 0..3
            {
                spatial_metric[i][j] = state[3 * i + j];
                extrinsic_curvature[i][j] = state[9 + 3 * i + j];
            }
        }
        Self {
            spatial_metric,
            extrinsic_curvature,
            energy_density: state[18],
        }
    }
}

/// A position-independent spatial metric.
struct ConstantMetric([[f64; 3]; 3]);
impl Metric<3> for ConstantMetric {
    fn components(&self, _coordinates: &[f64; 3]) -> [[f64; 3]; 3] {
        self.0
    }
}

/// A position-independent symmetric spatial tensor.
struct ConstantTensor([[f64; 3]; 3]);
impl SpatialTensorField for ConstantTensor {
    fn components(&self, _coordinates: &[f64; 3]) -> [[f64; 3]; 3] {
        self.0
    }
}

/// Unit lapse (synchronous gauge).
struct UnitLapse;
impl SpatialScalarField for UnitLapse {
    fn value(&self, _coordinates: &[f64; 3]) -> f64 {
        1.0
    }
}

/// Zero shift (comoving gauge).
struct ZeroShift;
impl SpatialVectorField for ZeroShift {
    fn components(&self, _coordinates: &[f64; 3]) -> [f64; 3] {
        [0.0; 3]
    }
}

/// The homogeneous ADM evolution as a [`scirust_sim::System`], so it is
/// integrated by the platform's existing deterministic RK4 rather than a new
/// integrator.
///
/// The flattened state is `[gamma_00..gamma_22, K_00..K_22, rho]` in that fixed
/// order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HomogeneousEvolution {
    fluid: BarotropicFluid,
    settings: AdmEvolutionSettings,
}

impl HomogeneousEvolution {
    /// Construct from a fluid and the finite-difference settings handed to the
    /// Layer 3.1 evaluators.
    #[must_use]
    pub const fn new(fluid: BarotropicFluid, settings: AdmEvolutionSettings) -> Self {
        Self { fluid, settings }
    }

    /// The fluid.
    #[must_use]
    pub const fn fluid(&self) -> BarotropicFluid {
        self.fluid
    }

    /// Evaluate the ADM sources for a state.
    fn sources(&self, state: &HomogeneousState) -> AdmSources {
        AdmSources::perfect_fluid(
            state.energy_density,
            self.fluid.pressure(state.energy_density),
            &state.spatial_metric,
        )
    }

    /// The Hamiltonian constraint residual for a state, evaluated through the
    /// Layer 3.1 evaluator (a diagnostic; never enforced).
    pub fn hamiltonian_residual(
        &self,
        state: &HomogeneousState,
    ) -> Result<f64, HomogeneousEvolutionError> {
        let constraint = hamiltonian_constraint(
            &ConstantMetric(state.spatial_metric),
            &ConstantTensor(state.extrinsic_curvature),
            &ORIGIN,
            &self.sources(state),
            &self.settings,
        )?;
        Ok(constraint.signed_residual)
    }

    /// The right-hand sides for a state, obtained from the Layer 3.1 evaluators.
    fn right_hand_sides(
        &self,
        state: &HomogeneousState,
    ) -> Result<HomogeneousRates, HomogeneousEvolutionError> {
        let metric = ConstantMetric(state.spatial_metric);
        let curvature = ConstantTensor(state.extrinsic_curvature);
        let sources = self.sources(state);

        let metric_rhs = metric_evolution_rhs(
            &metric,
            &curvature,
            &UnitLapse,
            &ZeroShift,
            &ORIGIN,
            &self.settings,
        )?;
        let curvature_rhs = curvature_evolution_rhs(
            &metric,
            &curvature,
            &UnitLapse,
            &ZeroShift,
            &ORIGIN,
            &sources,
            &self.settings,
        )?;

        // Continuity equation: rho' = -3 H (rho + p), the normal projection of
        // nabla_mu T^{mu nu} = 0.
        let hubble = state.hubble_parameter()?;
        let pressure = self.fluid.pressure(state.energy_density);
        let energy_rhs = -3.0 * hubble * (state.energy_density + pressure);

        Ok(HomogeneousRates {
            spatial_metric: metric_rhs.total,
            extrinsic_curvature: curvature_rhs.total,
            energy_density: energy_rhs,
        })
    }
}

/// The time derivatives of the homogeneous state.
struct HomogeneousRates {
    spatial_metric: [[f64; 3]; 3],
    extrinsic_curvature: [[f64; 3]; 3],
    energy_density: f64,
}

impl System for HomogeneousEvolution {
    fn dim(&self) -> usize {
        STATE_DIMENSION
    }

    fn derivatives(&self, _time: f64, state: &[f64], derivative: &mut [f64]) {
        let current = HomogeneousState::unflatten(state);
        match self.right_hand_sides(&current)
        {
            Ok(rates) =>
            {
                for i in 0..3
                {
                    for j in 0..3
                    {
                        derivative[3 * i + j] = rates.spatial_metric[i][j];
                        derivative[9 + 3 * i + j] = rates.extrinsic_curvature[i][j];
                    }
                }
                derivative[18] = rates.energy_density;
            },
            Err(_) =>
            {
                // `System::derivatives` cannot report an error. Emitting a
                // non-finite derivative makes the integrator's own non-finite
                // guard stop the run, which `evolve_homogeneous` then surfaces
                // as a typed `Integration` error rather than a silent wrong
                // answer. Nothing is replaced by zero.
                for value in derivative.iter_mut()
                {
                    *value = f64::NAN;
                }
            },
        }
    }
}

/// One recorded sample of the evolution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HomogeneousSample {
    /// The time.
    pub time: f64,
    /// The scale factor `a = (det gamma)^{1/6}`.
    pub scale_factor: f64,
    /// The Hubble parameter `H = -K/3`.
    pub hubble_parameter: f64,
    /// The fluid energy density `rho`.
    pub energy_density: f64,
    /// The Hamiltonian constraint residual — a **diagnostic** of the free
    /// evolution, never enforced.
    pub hamiltonian_residual: f64,
}

/// Freely evolve homogeneous ADM data from `start` to `end` with fixed step
/// `step`, recording a sample (including the Hamiltonian constraint residual)
/// at every step.
///
/// Returns a typed [`HomogeneousEvolutionError`] for malformed initial data, a
/// malformed time span, an integrator failure, or a non-finite evaluation; it
/// never panics.
///
/// # Example
///
/// De Sitter: with `w = -1` the exact solution is `a(t) = exp(H t)`.
///
/// ```
/// use scirust_relativity::adm_evolution::AdmEvolutionSettings;
/// use scirust_relativity::adm_homogeneous::{
///     BarotropicFluid, HomogeneousEvolution, HomogeneousState, evolve_homogeneous,
/// };
///
/// let hubble = 0.5;
/// let evolution = HomogeneousEvolution::new(
///     BarotropicFluid::vacuum_energy(),
///     AdmEvolutionSettings { spatial_step: 1.0e-3, metric_step: 1.0e-3 },
/// );
/// let initial = HomogeneousState::friedmann(1.0, hubble).expect("valid data");
/// let history = evolve_homogeneous(&evolution, &initial, 0.0, 2.0, 0.01).expect("evolves");
///
/// let last = history.last().expect("non-empty");
/// assert!((last.scale_factor - (hubble * 2.0_f64).exp()).abs() < 1.0e-6);
/// assert!(last.hamiltonian_residual.abs() < 1.0e-6);
/// ```
pub fn evolve_homogeneous(
    evolution: &HomogeneousEvolution,
    initial: &HomogeneousState,
    start: f64,
    end: f64,
    step: f64,
) -> Result<Vec<HomogeneousSample>, HomogeneousEvolutionError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() || step <= 0.0 || start >= end
    {
        return Err(HomogeneousEvolutionError::InvalidTimeSpan { start, end, step });
    }
    if !initial.energy_density.is_finite() || initial.energy_density < 0.0
    {
        return Err(HomogeneousEvolutionError::InvalidEnergyDensity(
            initial.energy_density,
        ));
    }
    // Validates positive-definiteness and invertibility up front, so a bad
    // metric is a typed error rather than an integrator blow-up.
    initial.scale_factor()?;
    initial.mean_curvature()?;

    let mut flat = vec![0.0_f64; STATE_DIMENSION];
    initial.flatten(&mut flat);

    let trajectory = simulate(evolution, &flat, start, end, step)?;

    let mut history = Vec::with_capacity(trajectory.t.len());
    for (time, row) in trajectory.t.iter().zip(trajectory.y.iter())
    {
        let state = HomogeneousState::unflatten(row);
        history.push(HomogeneousSample {
            time: *time,
            scale_factor: state.scale_factor()?,
            hubble_parameter: state.hubble_parameter()?,
            energy_density: state.energy_density,
            hamiltonian_residual: evolution.hamiltonian_residual(&state)?,
        });
    }
    Ok(history)
}
