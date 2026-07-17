//! The simulation driver (eulerian and transport-compensated).
//!
//! [`simulate`] steps a time-dependent velocity field, computing at each step
//! the vorticity, the **curvature-weighted rotational intensity**
//! `⟨ω² · e^{L²κ}⟩` and the structural signature, then reduces those per-step
//! series into interval-integrated indices exactly as the reference engine
//! does:
//!
//! * the deformation reported at step `i` is attributed to the interval
//!   `[t_{i-1}, t_i]`;
//! * the four remaining bounded components are integrated by the trapezoidal
//!   midpoint of successive nodes;
//! * each index is the interval sum divided by the observed duration.
//!
//! [`simulate_transport_compensated`] adds the semi-Lagrangian
//! *transport-compensated* temporal-deformation mode (see [`crate::transport`]):
//! before the temporal term, the previous vorticity is advected along a
//! transport velocity to the current time, so the deformation measures genuine
//! change rather than mere advection. This mode requires the periodic boundary
//! convention.

use crate::error::{ItdError, Result};
use crate::field::Field2;
use crate::geometry::{BoundaryMode, Geometry};
use crate::operators::{bounded, spatial_mean, vorticity};
use crate::scenarios::{Config, Scenario, curvature_field};
use crate::signature::{STRUCTURAL_LENGTH, StructuralWeights, structural_metrics};
use crate::transport::{Interpolation, Trajectory, transport_previous_vorticity};

/// Names of the five structural component indices, in reported order.
pub const COMPONENT_NAMES: [&str; 5] = [
    "heterogeneity",
    "localization",
    "roughness",
    "sign_mixing",
    "temporal_deformation",
];

/// Configuration for a run: the structural length scale, the component weights
/// and the boundary convention. The curvature characteristic length is passed
/// separately (it lives in [`Config`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimConfig {
    /// Structural length `L_s` scaling the roughness component.
    pub structural_length: f64,
    /// Weights applied to the five bounded components.
    pub weights: StructuralWeights,
    /// Boundary convention for the field operators.
    pub boundary: BoundaryMode,
}

impl Default for SimConfig {
    fn default() -> Self {
        SimConfig {
            structural_length: STRUCTURAL_LENGTH,
            weights: StructuralWeights::default(),
            boundary: BoundaryMode::Finite,
        }
    }
}

/// The outcome of a run: the three headline indices, the five component
/// indices, and the per-step series (nodal; the deformation series has a
/// leading zero, since the first step has no predecessor).
#[derive(Debug, Clone, PartialEq)]
pub struct SimulationResult {
    /// The scenario name.
    pub name: String,
    /// Time-averaged curvature-weighted rotational intensity.
    pub intensity_index: f64,
    /// Time-averaged structural complexity.
    pub structure_index: f64,
    /// Time-averaged coupled diagnostic `intensity · (1 + structure)`.
    pub coupled_index: f64,
    /// The five component indices, in [`COMPONENT_NAMES`] order.
    pub component_indices: [f64; 5],
    /// Time-averaged raw (unbounded) *eulerian* temporal deformation.
    pub temporal_deformation_eulerian_index: f64,
    /// Time-averaged raw (unbounded) *compensated* temporal deformation, or
    /// `None` in eulerian mode.
    pub temporal_deformation_compensated_index: Option<f64>,
    /// Per-step intensity rate.
    pub intensity_rate: Vec<f64>,
    /// Per-step heterogeneity (raw).
    pub heterogeneity: Vec<f64>,
    /// Per-step localization (raw).
    pub localization: Vec<f64>,
    /// Per-step roughness (raw).
    pub roughness: Vec<f64>,
    /// Per-step sign-mixing.
    pub sign_mixing: Vec<f64>,
    /// Per-step active temporal deformation (compensated in transport mode;
    /// index `i` covers `[t_{i-1}, t_i]`).
    pub temporal_deformation: Vec<f64>,
}

impl SimulationResult {
    /// Looks up a component index by name (see [`COMPONENT_NAMES`]).
    pub fn component_index(&self, name: &str) -> Option<f64> {
        COMPONENT_NAMES
            .iter()
            .position(|&n| n == name)
            .map(|k| self.component_indices[k])
    }
}

/// A transport-velocity closure plus the interpolation and trajectory schemes
/// used to advect the previous vorticity in transport-compensated mode.
struct TransportCompensation<'a> {
    velocity: &'a dyn Fn(f64, f64, f64) -> (f64, f64),
    interpolation: Interpolation,
    trajectory: Trajectory,
}

/// Runs a full eulerian simulation.
///
/// `velocity` returns `(vx, vy)` and `curvature` returns the curvature field,
/// both evaluated at `(xc, yc, t)`. `times` must be strictly increasing with at
/// least two samples.
#[allow(clippy::too_many_arguments)]
pub fn simulate<VF, CF>(
    name: &str,
    velocity: VF,
    curvature: CF,
    xc: &[f64],
    yc: &[f64],
    times: &[f64],
    geometry: &Geometry,
    characteristic_length: f64,
    config: &SimConfig,
) -> Result<SimulationResult>
where
    VF: Fn(&[f64], &[f64], f64) -> (Field2, Field2),
    CF: Fn(&[f64], &[f64], f64) -> Field2,
{
    run_core(
        name,
        velocity,
        curvature,
        xc,
        yc,
        times,
        geometry,
        characteristic_length,
        config,
        None,
    )
}

/// Runs a simulation with the semi-Lagrangian **transport-compensated**
/// temporal-deformation mode. The previous vorticity is advected by
/// `transport_velocity` (a pointwise `(x, y, t) -> (vx, vy)` field) before the
/// temporal term. Requires [`BoundaryMode::Periodic`].
#[allow(clippy::too_many_arguments)]
pub fn simulate_transport_compensated<VF, CF, TVF>(
    name: &str,
    velocity: VF,
    curvature: CF,
    transport_velocity: TVF,
    interpolation: Interpolation,
    trajectory: Trajectory,
    xc: &[f64],
    yc: &[f64],
    times: &[f64],
    geometry: &Geometry,
    characteristic_length: f64,
    config: &SimConfig,
) -> Result<SimulationResult>
where
    VF: Fn(&[f64], &[f64], f64) -> (Field2, Field2),
    CF: Fn(&[f64], &[f64], f64) -> Field2,
    TVF: Fn(f64, f64, f64) -> (f64, f64),
{
    if config.boundary != BoundaryMode::Periodic
    {
        return Err(ItdError::UnsupportedBoundary(
            "transport compensation requires the periodic boundary mode".into(),
        ));
    }
    let transport = TransportCompensation {
        velocity: &transport_velocity,
        interpolation,
        trajectory,
    };
    run_core(
        name,
        velocity,
        curvature,
        xc,
        yc,
        times,
        geometry,
        characteristic_length,
        config,
        Some(&transport),
    )
}

#[allow(clippy::too_many_arguments)]
fn run_core<VF, CF>(
    name: &str,
    velocity: VF,
    curvature: CF,
    xc: &[f64],
    yc: &[f64],
    times: &[f64],
    geometry: &Geometry,
    characteristic_length: f64,
    config: &SimConfig,
    transport: Option<&TransportCompensation>,
) -> Result<SimulationResult>
where
    VF: Fn(&[f64], &[f64], f64) -> (Field2, Field2),
    CF: Fn(&[f64], &[f64], f64) -> Field2,
{
    let n = times.len();
    if n < 2
    {
        return Err(ItdError::TooFewPoints(
            "simulation needs at least two time samples".into(),
        ));
    }
    if !times.iter().all(|t| t.is_finite())
    {
        return Err(ItdError::NonFinite("time samples".into()));
    }
    if !times.windows(2).all(|w| w[1] > w[0])
    {
        return Err(ItdError::InvalidGeometry(
            "time samples must be strictly increasing".into(),
        ));
    }
    if !characteristic_length.is_finite()
    {
        return Err(ItdError::NonFinite("characteristic length".into()));
    }

    let duration = times[n - 1] - times[0];
    let char_sq = characteristic_length * characteristic_length;
    let weights = config.weights.normalized();

    let mut intensity_rate = vec![0.0; n];
    let mut het = vec![0.0; n];
    let mut loc = vec![0.0; n];
    let mut rough = vec![0.0; n];
    let mut sign = vec![0.0; n];
    // Raw (unbounded) eulerian and active deformation series. In eulerian mode
    // the active series equals the eulerian one; in transport mode it holds the
    // compensated deformation.
    let mut tdef_eulerian = vec![0.0; n];
    let mut tdef_active = vec![0.0; n];

    let mut previous_omega: Option<Field2> = None;
    let mut previous_time = f64::NAN;

    for i in 0..n
    {
        let t = times[i];
        let (vx, vy) = velocity(xc, yc, t);
        let omega = vorticity(&vx, &vy, geometry, config.boundary)?;

        let curv = curvature(xc, yc, t);
        let density = omega.zip_map(&curv, |w, k| w * w * (char_sq * k).exp())?;
        intensity_rate[i] = spatial_mean(&density, geometry, config.boundary)?;

        let dt = if i > 0 { Some(t - previous_time) } else { None };
        // Eulerian metrics give the mode-independent components (heterogeneity,
        // localization, roughness, sign-mixing) and the eulerian deformation.
        let eulerian = structural_metrics(
            &omega,
            geometry,
            previous_omega.as_ref(),
            dt,
            config.structural_length,
            config.weights,
            config.boundary,
        )?;
        het[i] = eulerian.heterogeneity;
        loc[i] = eulerian.localization;
        rough[i] = eulerian.roughness;
        sign[i] = eulerian.sign_mixing;
        tdef_eulerian[i] = eulerian.temporal_deformation;

        tdef_active[i] = match (transport, previous_omega.as_ref(), dt)
        {
            (Some(tc), Some(prev), Some(_)) =>
            {
                let transported = transport_previous_vorticity(
                    prev,
                    xc,
                    yc,
                    previous_time,
                    t,
                    tc.velocity,
                    tc.interpolation,
                    tc.trajectory,
                )?;
                let compensated = structural_metrics(
                    &omega,
                    geometry,
                    Some(&transported),
                    dt,
                    config.structural_length,
                    config.weights,
                    config.boundary,
                )?;
                compensated.temporal_deformation
            },
            // First step, or eulerian mode: compensated ≡ eulerian.
            _ => eulerian.temporal_deformation,
        };

        previous_omega = Some(omega);
        previous_time = t;
    }

    // Interval reduction (m = n - 1 intervals).
    let m = n - 1;
    let interval_dt: Vec<f64> = (0..m).map(|j| times[j + 1] - times[j]).collect();

    let bounded_het: Vec<f64> = het.iter().map(|&v| bounded(v)).collect();
    let bounded_loc: Vec<f64> = loc.iter().map(|&v| bounded(v)).collect();
    let bounded_rough: Vec<f64> = rough.iter().map(|&v| bounded(v)).collect();
    let bounded_sign: Vec<f64> = sign.iter().map(|&v| v.clamp(0.0, 1.0)).collect();
    // Deformation at node i belongs to interval [t_{i-1}, t_i]; drop node 0.
    let bounded_defo_iv: Vec<f64> = (0..m).map(|j| bounded(tdef_active[j + 1])).collect();

    let midpoint = |a: &[f64], j: usize| 0.5 * (a[j] + a[j + 1]);

    let mut component_intervals = [
        vec![0.0; m],
        vec![0.0; m],
        vec![0.0; m],
        vec![0.0; m],
        vec![0.0; m],
    ];
    for j in 0..m
    {
        component_intervals[0][j] = midpoint(&bounded_het, j);
        component_intervals[1][j] = midpoint(&bounded_loc, j);
        component_intervals[2][j] = midpoint(&bounded_rough, j);
        component_intervals[3][j] = midpoint(&bounded_sign, j);
        component_intervals[4][j] = bounded_defo_iv[j];
    }

    let mut interval_structure = vec![0.0; m];
    for j in 0..m
    {
        interval_structure[j] = weights[0] * component_intervals[0][j]
            + weights[1] * component_intervals[1][j]
            + weights[2] * component_intervals[2][j]
            + weights[3] * component_intervals[3][j]
            + weights[4] * component_intervals[4][j];
    }

    let intensity_interval: Vec<f64> = (0..m)
        .map(|j| 0.5 * (intensity_rate[j] + intensity_rate[j + 1]))
        .collect();
    let coupled_interval: Vec<f64> = (0..m)
        .map(|j| intensity_interval[j] * (1.0 + interval_structure[j]))
        .collect();

    let integrate = |values: &[f64]| -> f64 {
        let mut acc = 0.0;
        for j in 0..m
        {
            acc += values[j] * interval_dt[j];
        }
        acc / duration
    };
    // Raw deformation index over the intervals (node i+1 covers interval j).
    let integrate_raw = |series: &[f64]| -> f64 {
        let mut acc = 0.0;
        for j in 0..m
        {
            acc += series[j + 1] * interval_dt[j];
        }
        acc / duration
    };

    let intensity_index = integrate(&intensity_interval);
    let structure_index = integrate(&interval_structure);
    let coupled_index = integrate(&coupled_interval);
    let component_indices = [
        integrate(&component_intervals[0]),
        integrate(&component_intervals[1]),
        integrate(&component_intervals[2]),
        integrate(&component_intervals[3]),
        integrate(&component_intervals[4]),
    ];

    let temporal_deformation_eulerian_index = integrate_raw(&tdef_eulerian);
    let temporal_deformation_compensated_index = transport.map(|_| integrate_raw(&tdef_active));

    Ok(SimulationResult {
        name: name.to_string(),
        intensity_index,
        structure_index,
        coupled_index,
        component_indices,
        temporal_deformation_eulerian_index,
        temporal_deformation_compensated_index,
        intensity_rate,
        heterogeneity: het,
        localization: loc,
        roughness: rough,
        sign_mixing: sign,
        temporal_deformation: tdef_active,
    })
}

/// Runs a canonical scenario on the grid and time horizon given by `config`,
/// using the shared [`curvature_field`] weighting (eulerian mode).
pub fn simulate_canonical(
    scenario: Scenario,
    config: &Config,
    sim: &SimConfig,
) -> Result<SimulationResult> {
    let xc = config.coordinates();
    let yc = xc.clone();
    let times = config.times();
    let geometry = Geometry::isotropic(config.spacing())?;
    simulate(
        scenario.name(),
        |x, y, t| scenario.velocity(x, y, t),
        curvature_field,
        &xc,
        &yc,
        &times,
        &geometry,
        config.characteristic_length,
        sim,
    )
}

/// Runs a canonical scenario in transport-compensated mode, advecting the
/// vorticity by the scenario's own flow (its pointwise velocity). Requires
/// `sim.boundary == BoundaryMode::Periodic`.
pub fn simulate_canonical_transport(
    scenario: Scenario,
    config: &Config,
    sim: &SimConfig,
    interpolation: Interpolation,
    trajectory: Trajectory,
) -> Result<SimulationResult> {
    let xc = config.coordinates();
    let yc = xc.clone();
    let times = config.times();
    let geometry = Geometry::isotropic(config.spacing())?;
    simulate_transport_compensated(
        scenario.name(),
        |x, y, t| scenario.velocity(x, y, t),
        curvature_field,
        |x, y, t| scenario.velocity_at(x, y, t),
        interpolation,
        trajectory,
        &xc,
        &yc,
        &times,
        &geometry,
        config.characteristic_length,
        sim,
    )
}
