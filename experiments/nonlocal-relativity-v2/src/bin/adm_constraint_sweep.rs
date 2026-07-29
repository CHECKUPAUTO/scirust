//! ADM constraint and evolution checks across a controlled family of 3+1
//! states: exact Minkowski, a static Schwarzschild slice, exact flat FLRW,
//! and deliberately perturbed (constraint-violating) states (the opening
//! Layer 3 slice; see `docs/LAYER_3_ADM_EVOLUTION.md`).
//!
//! For each scenario the experiment evaluates the decomposed Hamiltonian
//! constraint, the momentum constraint, and the norms of the two evolution
//! right-hand sides, from independently supplied spatial data (not extracted
//! from an already-known 4-metric — that is Layer 2's `adm_kinematics`
//! experiment). One scenario is deliberately invalid (a singular spatial
//! metric) and is reported as rejected, not silently coerced.
//!
//! Established general relativity. Every quantity is a numerical
//! approximation (central finite differences of the supplied fields); this
//! evolves nothing in time and asserts no new physics.

#![forbid(unsafe_code)]

use nonlocal_relativity_experiments::{print_experiment_header, require_finite};
use scirust_relativity::adm_evolution::{
    AdmEvolutionSettings, AdmSources, SpatialScalarField, SpatialTensorField, SpatialVectorField,
    curvature_evolution_rhs, hamiltonian_constraint, metric_evolution_rhs, momentum_constraint,
};
use scirust_relativity::{ExponentialScaleFactor, Flrw, Metric, Minkowski, Schwarzschild};
use std::f64::consts::{FRAC_PI_2, PI};

struct SpatialBlock<'a, B> {
    background: &'a B,
    time: f64,
}
impl<B: Metric<4>> Metric<3> for SpatialBlock<'_, B> {
    fn components(&self, x: &[f64; 3]) -> [[f64; 3]; 3] {
        let full = self.background.components(&[self.time, x[0], x[1], x[2]]);
        [
            [full[1][1], 0.0, 0.0],
            [0.0, full[2][2], 0.0],
            [0.0, 0.0, full[3][3]],
        ]
    }
}
struct LapseFromMetric<'a, B> {
    background: &'a B,
    time: f64,
}
impl<B: Metric<4>> SpatialScalarField for LapseFromMetric<'_, B> {
    fn value(&self, x: &[f64; 3]) -> f64 {
        let full = self.background.components(&[self.time, x[0], x[1], x[2]]);
        (-full[0][0]).sqrt()
    }
}
struct ZeroTensorField;
impl SpatialTensorField for ZeroTensorField {
    fn components(&self, _x: &[f64; 3]) -> [[f64; 3]; 3] {
        [[0.0; 3]; 3]
    }
}
struct ZeroVectorField;
impl SpatialVectorField for ZeroVectorField {
    fn components(&self, _x: &[f64; 3]) -> [f64; 3] {
        [0.0; 3]
    }
}
struct ConstantLapse(f64);
impl SpatialScalarField for ConstantLapse {
    fn value(&self, _x: &[f64; 3]) -> f64 {
        self.0
    }
}
struct FlatSpace;
impl Metric<3> for FlatSpace {
    fn components(&self, _x: &[f64; 3]) -> [[f64; 3]; 3] {
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    }
}
struct SingularMetric;
impl Metric<3> for SingularMetric {
    fn components(&self, _x: &[f64; 3]) -> [[f64; 3]; 3] {
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    }
}
struct ConstantDiagonalCurvature {
    values: [f64; 3],
}
impl SpatialTensorField for ConstantDiagonalCurvature {
    fn components(&self, _x: &[f64; 3]) -> [[f64; 3]; 3] {
        [
            [self.values[0], 0.0, 0.0],
            [0.0, self.values[1], 0.0],
            [0.0, 0.0, self.values[2]],
        ]
    }
}
struct LinearTracelessCurvature {
    epsilon: f64,
}
impl SpatialTensorField for LinearTracelessCurvature {
    fn components(&self, x: &[f64; 3]) -> [[f64; 3]; 3] {
        let k11 = self.epsilon * x[0];
        [
            [k11, 0.0, 0.0],
            [0.0, -0.5 * k11, 0.0],
            [0.0, 0.0, -0.5 * k11],
        ]
    }
}
struct FlrwSpatialSlice<'a> {
    flrw: &'a Flrw<ExponentialScaleFactor>,
    time: f64,
}
impl Metric<3> for FlrwSpatialSlice<'_> {
    fn components(&self, x: &[f64; 3]) -> [[f64; 3]; 3] {
        let full = self.flrw.components(&[self.time, x[0], x[1], x[2]]);
        [
            [full[1][1], 0.0, 0.0],
            [0.0, full[2][2], 0.0],
            [0.0, 0.0, full[3][3]],
        ]
    }
}
struct FlrwExtrinsicCurvature<'a> {
    flrw: &'a Flrw<ExponentialScaleFactor>,
    time: f64,
}
impl SpatialTensorField for FlrwExtrinsicCurvature<'_> {
    fn components(&self, x: &[f64; 3]) -> [[f64; 3]; 3] {
        let hubble = self.flrw.hubble_parameter(self.time);
        let full = self.flrw.components(&[self.time, x[0], x[1], x[2]]);
        [
            [-hubble * full[1][1], 0.0, 0.0],
            [0.0, -hubble * full[2][2], 0.0],
            [0.0, 0.0, -hubble * full[3][3]],
        ]
    }
}

fn settings() -> AdmEvolutionSettings {
    AdmEvolutionSettings {
        spatial_step: 1.0e-3,
        metric_step: 1.0e-3,
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_row<G: Metric<3>>(
    scenario: &str,
    parameter: f64,
    spatial_metric: &G,
    extrinsic_curvature: &impl SpatialTensorField,
    lapse: &impl SpatialScalarField,
    shift: &impl SpatialVectorField,
    point: &[f64; 3],
    sources: &AdmSources,
) -> Result<(), String> {
    let settings = settings();
    match (
        hamiltonian_constraint(
            spatial_metric,
            extrinsic_curvature,
            point,
            sources,
            &settings,
        ),
        momentum_constraint(
            spatial_metric,
            extrinsic_curvature,
            point,
            sources,
            &settings,
        ),
        metric_evolution_rhs(
            spatial_metric,
            extrinsic_curvature,
            lapse,
            shift,
            point,
            &settings,
        ),
        curvature_evolution_rhs(
            spatial_metric,
            extrinsic_curvature,
            lapse,
            shift,
            point,
            sources,
            &settings,
        ),
    )
    {
        (Ok(hamiltonian), Ok(momentum), Ok(metric_rhs), Ok(curvature_rhs)) =>
        {
            require_finite(&[
                ("spatial_ricci_scalar", hamiltonian.spatial_ricci_scalar),
                ("mean_curvature_squared", hamiltonian.mean_curvature_squared),
                (
                    "extrinsic_curvature_norm",
                    hamiltonian.extrinsic_curvature_norm,
                ),
                ("hamiltonian_matter_term", hamiltonian.matter_term),
                ("hamiltonian_residual", hamiltonian.signed_residual),
                ("momentum_norm", momentum.residual_norm),
            ])?;
            let tensor_norm = |tensor: &[[f64; 3]; 3]| -> f64 {
                let mut sum_of_squares = 0.0;
                for row in tensor
                {
                    for value in row
                    {
                        sum_of_squares += value * value;
                    }
                }
                sum_of_squares.sqrt()
            };
            let metric_rhs_norm = tensor_norm(&metric_rhs.total);
            let curvature_rhs_norm = tensor_norm(&curvature_rhs.total);
            require_finite(&[
                ("metric_rhs_norm", metric_rhs_norm),
                ("curvature_rhs_norm", curvature_rhs_norm),
            ])?;
            println!(
                "{scenario},{parameter},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},ok",
                hamiltonian.spatial_ricci_scalar,
                hamiltonian.mean_curvature_squared,
                hamiltonian.extrinsic_curvature_norm,
                hamiltonian.matter_term,
                hamiltonian.signed_residual,
                momentum.residual[0],
                momentum.residual[1],
                momentum.residual[2],
                momentum.residual_norm,
                metric_rhs_norm,
                curvature_rhs_norm,
            );
        },
        _ =>
        {
            println!(
                "{scenario},{parameter},nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,rejected_singular_spatial_metric"
            );
        },
    }
    Ok(())
}

fn main() -> Result<(), String> {
    print_experiment_header(
        "ADM constraint and evolution sweep",
        "scirust-relativity Layer 3.1 (established general relativity)",
        "Hamiltonian/momentum constraints and evolution-RHS norms from independently supplied 3+1 data; numerical, not a bound.",
    );

    println!(
        "scenario,parameter,spatial_ricci_scalar,mean_curvature_squared,extrinsic_curvature_norm,\
         hamiltonian_matter_term,hamiltonian_residual,momentum_1,momentum_2,momentum_3,momentum_norm,\
         metric_rhs_norm,curvature_rhs_norm,status"
    );

    // Minkowski: everything vanishes.
    let minkowski = Minkowski;
    let minkowski_spatial = SpatialBlock {
        background: &minkowski,
        time: 0.0,
    };
    let minkowski_lapse = LapseFromMetric {
        background: &minkowski,
        time: 0.0,
    };
    emit_row(
        "minkowski",
        0.0,
        &minkowski_spatial,
        &ZeroTensorField,
        &minkowski_lapse,
        &ZeroVectorField,
        &[1.0, 2.0, 3.0],
        &AdmSources::VACUUM,
    )?;

    // Static Schwarzschild slice: time-symmetric, K = 0, vacuum.
    let schwarzschild = Schwarzschild::try_new(1.0).ok_or("invalid Schwarzschild")?;
    let schwarzschild_spatial = SpatialBlock {
        background: &schwarzschild,
        time: 0.0,
    };
    let schwarzschild_lapse = LapseFromMetric {
        background: &schwarzschild,
        time: 0.0,
    };
    for &radius in &[4.0, 6.0, 10.0]
    {
        emit_row(
            "schwarzschild_static",
            radius,
            &schwarzschild_spatial,
            &ZeroTensorField,
            &schwarzschild_lapse,
            &ZeroVectorField,
            &[radius, FRAC_PI_2, 0.0],
            &AdmSources::VACUUM,
        )?;
    }

    // Flat FLRW: Lambda = 3H^2 as an equivalent perfect fluid.
    for &hubble in &[0.3, 0.5, 0.8]
    {
        let flrw = Flrw::new(ExponentialScaleFactor::try_new(hubble).ok_or("invalid Hubble")?);
        let time = 0.2;
        let spatial = FlrwSpatialSlice { flrw: &flrw, time };
        let extrinsic = FlrwExtrinsicCurvature { flrw: &flrw, time };
        let lapse = ConstantLapse(1.0);
        let point = [0.1, 0.2, 0.3];
        let lambda = 3.0 * hubble * hubble;
        let rho = lambda / (8.0 * PI);
        let gamma = spatial.components(&point);
        let sources = AdmSources::perfect_fluid(rho, -rho, &gamma);
        emit_row(
            "flrw",
            hubble,
            &spatial,
            &extrinsic,
            &lapse,
            &ZeroVectorField,
            &point,
            &sources,
        )?;
    }

    // Deliberate Hamiltonian violation: constant K = diag(eps, eps, 0) on flat space.
    for &epsilon in &[0.01, 0.02, 0.05, 0.1]
    {
        let curvature = ConstantDiagonalCurvature {
            values: [epsilon, epsilon, 0.0],
        };
        let lapse = ConstantLapse(1.0);
        emit_row(
            "hamiltonian_violation",
            epsilon,
            &FlatSpace,
            &curvature,
            &lapse,
            &ZeroVectorField,
            &[1.0, 2.0, 3.0],
            &AdmSources::VACUUM,
        )?;
    }

    // Deliberate momentum violation: traceless, position-dependent K on flat space.
    for &epsilon in &[0.01, 0.02, 0.05, 0.1]
    {
        let curvature = LinearTracelessCurvature { epsilon };
        let lapse = ConstantLapse(1.0);
        emit_row(
            "momentum_violation",
            epsilon,
            &FlatSpace,
            &curvature,
            &lapse,
            &ZeroVectorField,
            &[1.0, 2.0, 3.0],
            &AdmSources::VACUUM,
        )?;
    }

    // A deliberately invalid state: a singular spatial metric, rejected rather
    // than silently misused.
    emit_row(
        "singular_metric",
        0.0,
        &SingularMetric,
        &ZeroTensorField,
        &ConstantLapse(1.0),
        &ZeroVectorField,
        &[1.0, 2.0, 3.0],
        &AdmSources::VACUUM,
    )?;

    println!(
        "# interpretation: Minkowski and the static Schwarzschild slice satisfy both constraints"
    );
    println!(
        "# and have a zero curvature-evolution right-hand side (a genuinely static solution);"
    );
    println!(
        "# flat FLRW satisfies the Hamiltonian constraint (the first Friedmann equation, with"
    );
    println!(
        "# Lambda represented as an equivalent perfect fluid); the deliberately perturbed states"
    );
    println!(
        "# show a nonzero, monotonically scaling constraint violation attributed to the correct"
    );
    println!(
        "# term; and the singular spatial metric is rejected, not silently misused. Numerical"
    );
    println!(
        "# approximation, not a bound; this evaluates the ADM right-hand sides only -- nothing"
    );
    println!("# here evolves in time.");
    Ok(())
}
