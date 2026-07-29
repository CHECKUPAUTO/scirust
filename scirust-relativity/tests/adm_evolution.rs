//! Oracles for the ADM constraint and evolution core (Layer 3.1; see
//! `docs/LAYER_3_ADM_EVOLUTION.md`).
//!
//! Oracle A (Minkowski): every constraint and evolution right-hand side
//! vanishes. Oracle B (static Schwarzschild slice): isolates the
//! lapse-Hessian/spatial-Ricci-tensor combination, which must cancel exactly
//! for any genuinely time-independent vacuum solution. Oracle C (flat FLRW):
//! isolates the extrinsic-curvature-quadratic/matter-term combination and is
//! checked directly against a time finite difference of Layer 2's already-
//! validated extraction (this is the permanent regression test for the
//! extrinsic-curvature evolution equation's corrected matter-term sign — see
//! the design note §4). Oracle D: deliberate constraint violations, checked
//! against closed forms and for monotonic scaling. Two further tests isolate
//! the differentiation machinery itself (quadratic lapse, linear shift) with
//! zero-truncation closed forms, independent of any GR solution.
//!
//! Established general relativity; a numerical approximation validated
//! against exact and independently-computed oracles.

use scirust_relativity::adm::{AdmSettings as KinematicSettings, adm_decomposition};
use scirust_relativity::adm_evolution::{
    AdmEvolutionError, AdmEvolutionSettings, AdmSources, SpatialScalarField, SpatialTensorField,
    SpatialVectorField, curvature_evolution_rhs, hamiltonian_constraint, metric_evolution_rhs,
    momentum_constraint,
};
use scirust_relativity::{ExponentialScaleFactor, Flrw, Metric, Minkowski, Schwarzschild};
use std::f64::consts::{FRAC_PI_2, PI};

fn settings() -> AdmEvolutionSettings {
    AdmEvolutionSettings {
        spatial_step: 1.0e-3,
        metric_step: 1.0e-3,
    }
}

// ---------------------------------------------------------------------------
// Shared field adapters. Each reuses an existing, already-validated background
// rather than re-deriving a formula by hand.
// ---------------------------------------------------------------------------

/// The spatial 3-block of `background`'s 4-metric at fixed `time`.
struct SpatialBlock<'a, B> {
    background: &'a B,
    time: f64,
}

impl<B: Metric<4>> Metric<3> for SpatialBlock<'_, B> {
    fn components(&self, x: &[f64; 3]) -> [[f64; 3]; 3] {
        let full = self.background.components(&[self.time, x[0], x[1], x[2]]);
        [
            [full[1][1], full[1][2], full[1][3]],
            [full[2][1], full[2][2], full[2][3]],
            [full[3][1], full[3][2], full[3][3]],
        ]
    }
}

/// The lapse `sqrt(-g_00)` of `background` at fixed `time` (zero-shift charts).
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
    fn components(&self, _coordinates: &[f64; 3]) -> [[f64; 3]; 3] {
        [[0.0; 3]; 3]
    }
}

struct ZeroVectorField;
impl SpatialVectorField for ZeroVectorField {
    fn components(&self, _coordinates: &[f64; 3]) -> [f64; 3] {
        [0.0; 3]
    }
}

struct ConstantLapse(f64);
impl SpatialScalarField for ConstantLapse {
    fn value(&self, _coordinates: &[f64; 3]) -> f64 {
        self.0
    }
}

struct FlatSpace;
impl Metric<3> for FlatSpace {
    fn components(&self, _coordinates: &[f64; 3]) -> [[f64; 3]; 3] {
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    }
}

/// A constant diagonal extrinsic curvature `K_ij = diag(values)`.
struct ConstantDiagonalCurvature {
    values: [f64; 3],
}
impl SpatialTensorField for ConstantDiagonalCurvature {
    fn components(&self, _coordinates: &[f64; 3]) -> [[f64; 3]; 3] {
        [
            [self.values[0], 0.0, 0.0],
            [0.0, self.values[1], 0.0],
            [0.0, 0.0, self.values[2]],
        ]
    }
}

/// A traceless, position-dependent (linear in `x^1`) extrinsic curvature
/// `K_ij = diag(eps x^1, -eps x^1 / 2, -eps x^1 / 2)`.
struct LinearTracelessCurvature {
    epsilon: f64,
}
impl SpatialTensorField for LinearTracelessCurvature {
    fn components(&self, coordinates: &[f64; 3]) -> [[f64; 3]; 3] {
        let k11 = self.epsilon * coordinates[0];
        [
            [k11, 0.0, 0.0],
            [0.0, -0.5 * k11, 0.0],
            [0.0, 0.0, -0.5 * k11],
        ]
    }
}

/// A quadratic lapse `alpha = 1 + a (x1^2 + x2^2 + x3^2)`: its covariant Hessian
/// on flat space is exactly `2a * delta_ij` (zero truncation under central
/// differences, since the field is quadratic).
struct QuadraticLapse {
    coefficient: f64,
}
impl SpatialScalarField for QuadraticLapse {
    fn value(&self, coordinates: &[f64; 3]) -> f64 {
        let [x, y, z] = *coordinates;
        1.0 + self.coefficient * (x * x + y * y + z * z)
    }
}

/// A linear shift `beta^i = M^i_j x^j` for a constant matrix `M`: its first
/// derivatives are exactly `M` under central differences (zero truncation).
struct LinearShift {
    matrix: [[f64; 3]; 3],
}
impl SpatialVectorField for LinearShift {
    #[allow(clippy::needless_range_loop)]
    fn components(&self, coordinates: &[f64; 3]) -> [f64; 3] {
        let mut value = [0.0_f64; 3];
        for i in 0..3
        {
            for j in 0..3
            {
                value[i] += self.matrix[i][j] * coordinates[j];
            }
        }
        value
    }
}

// ---------------------------------------------------------------------------
// Oracle A -- Minkowski: every constraint and evolution right-hand side is
// zero to the finite-difference floor.
// ---------------------------------------------------------------------------

#[test]
fn oracle_a_minkowski_constraints_and_evolution_vanish() {
    let minkowski = Minkowski;
    let spatial = SpatialBlock {
        background: &minkowski,
        time: 0.0,
    };
    let lapse = LapseFromMetric {
        background: &minkowski,
        time: 0.0,
    };
    let point = [1.0, 2.0, 3.0];
    let settings = settings();

    let hamiltonian = hamiltonian_constraint(
        &spatial,
        &ZeroTensorField,
        &point,
        &AdmSources::VACUUM,
        &settings,
    )
    .unwrap();
    assert!(
        hamiltonian.absolute_residual < 1.0e-8,
        "H = {}",
        hamiltonian.signed_residual
    );

    let momentum = momentum_constraint(
        &spatial,
        &ZeroTensorField,
        &point,
        &AdmSources::VACUUM,
        &settings,
    )
    .unwrap();
    assert!(
        momentum.residual_norm < 1.0e-8,
        "|M| = {}",
        momentum.residual_norm
    );

    let metric_rhs = metric_evolution_rhs(
        &spatial,
        &ZeroTensorField,
        &lapse,
        &ZeroVectorField,
        &point,
        &settings,
    )
    .unwrap();
    for row in metric_rhs.total
    {
        for value in row
        {
            assert!(value.abs() < 1.0e-8, "metric RHS component {value}");
        }
    }

    let curvature_rhs = curvature_evolution_rhs(
        &spatial,
        &ZeroTensorField,
        &lapse,
        &ZeroVectorField,
        &point,
        &AdmSources::VACUUM,
        &settings,
    )
    .unwrap();
    for row in curvature_rhs.total
    {
        for value in row
        {
            assert!(value.abs() < 1.0e-6, "curvature RHS component {value}");
        }
    }
}

// ---------------------------------------------------------------------------
// Oracle B -- static Schwarzschild slice (time-symmetric): the Hessian/Ricci
// combination must cancel exactly for a genuinely time-independent solution.
// ---------------------------------------------------------------------------

#[test]
fn oracle_b_static_schwarzschild_hamiltonian_and_curvature_rhs_vanish() {
    let schwarzschild = Schwarzschild::try_new(1.0).unwrap();
    let spatial = SpatialBlock {
        background: &schwarzschild,
        time: 0.0,
    };
    let lapse = LapseFromMetric {
        background: &schwarzschild,
        time: 0.0,
    };
    let point = [6.0, FRAC_PI_2, 0.0];
    let settings = settings();

    // K = 0 (time-symmetric slicing): Hamiltonian residual is exactly R^(3),
    // known scalar-flat (Layer 2).
    let hamiltonian = hamiltonian_constraint(
        &spatial,
        &ZeroTensorField,
        &point,
        &AdmSources::VACUUM,
        &settings,
    )
    .unwrap();
    assert!(
        hamiltonian.absolute_residual < 1.0e-4,
        "H = {}",
        hamiltonian.signed_residual
    );

    // K = 0 identically -> P^{ij} = 0 identically -> momentum residual is exact zero.
    let momentum = momentum_constraint(
        &spatial,
        &ZeroTensorField,
        &point,
        &AdmSources::VACUUM,
        &settings,
    )
    .unwrap();
    assert!(
        momentum.residual_norm < 1.0e-9,
        "|M| = {}",
        momentum.residual_norm
    );

    // A genuinely time-independent vacuum solution has partial_t K_ij == 0
    // identically, which (with K = 0 and vacuum, so the quadratic-K and matter
    // terms vanish identically) forces -D_iD_j(alpha) + alpha R_ij = 0.
    let curvature_rhs = curvature_evolution_rhs(
        &spatial,
        &ZeroTensorField,
        &lapse,
        &ZeroVectorField,
        &point,
        &AdmSources::VACUUM,
        &settings,
    )
    .unwrap();
    for row in curvature_rhs.total
    {
        for value in row
        {
            assert!(value.abs() < 1.0e-4, "curvature RHS component {value}");
        }
    }
    // Confirm this is a nontrivial cancellation: the Ricci term and the lapse
    // Hessian are each individually nonzero (Schwarzschild's spatial slice is
    // curved and anisotropic, even though it is scalar-flat).
    let ricci_nonzero = curvature_rhs
        .ricci_term
        .iter()
        .flatten()
        .any(|value| value.abs() > 1.0e-3);
    let hessian_nonzero = curvature_rhs
        .lapse_hessian
        .iter()
        .flatten()
        .any(|value| value.abs() > 1.0e-3);
    assert!(
        ricci_nonzero,
        "ricci_term should be individually nonzero: {:?}",
        curvature_rhs.ricci_term
    );
    assert!(
        hessian_nonzero,
        "lapse_hessian should be individually nonzero: {:?}",
        curvature_rhs.lapse_hessian
    );
}

// ---------------------------------------------------------------------------
// Oracle C -- flat FLRW: reduces to the first Friedmann equation, and its
// curvature-evolution RHS is checked directly against a time finite difference
// of Layer 2's already-validated extraction (the permanent regression test for
// the corrected matter-term sign).
// ---------------------------------------------------------------------------

struct FlrwSpatialSlice<'a> {
    flrw: &'a Flrw<ExponentialScaleFactor>,
    time: f64,
}
impl Metric<3> for FlrwSpatialSlice<'_> {
    fn components(&self, coordinates: &[f64; 3]) -> [[f64; 3]; 3] {
        let full =
            self.flrw
                .components(&[self.time, coordinates[0], coordinates[1], coordinates[2]]);
        [
            [full[1][1], 0.0, 0.0],
            [0.0, full[2][2], 0.0],
            [0.0, 0.0, full[3][3]],
        ]
    }
}

/// `K_ij = -H(t) gamma_ij`, using the existing `Flrw::hubble_parameter` (not
/// re-derived by hand).
struct FlrwExtrinsicCurvature<'a> {
    flrw: &'a Flrw<ExponentialScaleFactor>,
    time: f64,
}
impl SpatialTensorField for FlrwExtrinsicCurvature<'_> {
    fn components(&self, coordinates: &[f64; 3]) -> [[f64; 3]; 3] {
        let hubble = self.flrw.hubble_parameter(self.time);
        let full =
            self.flrw
                .components(&[self.time, coordinates[0], coordinates[1], coordinates[2]]);
        [
            [-hubble * full[1][1], 0.0, 0.0],
            [0.0, -hubble * full[2][2], 0.0],
            [0.0, 0.0, -hubble * full[3][3]],
        ]
    }
}

#[test]
fn oracle_c_flat_flrw_reduces_to_friedmann_hamiltonian() {
    let hubble = 0.5_f64;
    let flrw = Flrw::new(ExponentialScaleFactor::try_new(hubble).unwrap());
    let time = 0.2;
    let spatial = FlrwSpatialSlice { flrw: &flrw, time };
    let extrinsic = FlrwExtrinsicCurvature { flrw: &flrw, time };
    let point = [0.1, 0.2, 0.3];
    let settings = settings();

    // A cosmological constant Lambda = 3H^2 is exactly equivalent to a perfect
    // fluid with rho = Lambda / 8pi, p = -rho (vacuum energy).
    let lambda = 3.0 * hubble * hubble;
    let rho = lambda / (8.0 * PI);
    let gamma = spatial.components(&point);
    let sources = AdmSources::perfect_fluid(rho, -rho, &gamma);

    let hamiltonian =
        hamiltonian_constraint(&spatial, &extrinsic, &point, &sources, &settings).unwrap();
    assert!(
        hamiltonian.absolute_residual < 1.0e-5,
        "H = {}",
        hamiltonian.signed_residual
    );
    assert!((hamiltonian.mean_curvature_squared - 9.0 * hubble * hubble).abs() < 1.0e-9);
    assert!((hamiltonian.extrinsic_curvature_norm - 3.0 * hubble * hubble).abs() < 1.0e-9);

    // Homogeneous configuration (gamma_ij and K_ij are position-independent at
    // fixed t): momentum vanishes exactly, not just approximately.
    let momentum = momentum_constraint(&spatial, &extrinsic, &point, &sources, &settings).unwrap();
    assert!(
        momentum.residual_norm < 1.0e-6,
        "|M| = {}",
        momentum.residual_norm
    );
}

// Explicit tensor-index loops read most clearly here (matching the crate's
// curvature, action, and adm modules).
#[allow(clippy::needless_range_loop)]
#[test]
fn oracle_c_curvature_rhs_matches_direct_time_derivative_of_exact_flrw() {
    let hubble = 0.5_f64;
    let flrw = Flrw::new(ExponentialScaleFactor::try_new(hubble).unwrap());
    let time = 0.2;
    let point4 = [time, 0.1, 0.2, 0.3];

    // Ground truth: central time-difference of Layer 2's already-validated
    // adm_decomposition, entirely independent of this module's new code.
    let kinematic_settings = KinematicSettings {
        time_step: 1.0e-4,
        spatial_step: 1.0e-3,
    };
    let dt = 1.0e-4;
    let mut forward = point4;
    forward[0] += dt;
    let mut backward = point4;
    backward[0] -= dt;
    let k_plus = adm_decomposition(&flrw, &forward, &kinematic_settings)
        .unwrap()
        .extrinsic_curvature;
    let k_minus = adm_decomposition(&flrw, &backward, &kinematic_settings)
        .unwrap()
        .extrinsic_curvature;
    let mut direct = [[0.0_f64; 3]; 3];
    for i in 0..3
    {
        for j in 0..3
        {
            direct[i][j] = (k_plus[i][j] - k_minus[i][j]) / (2.0 * dt);
        }
    }

    // Candidate: this module's closed-form evolution RHS.
    let spatial = FlrwSpatialSlice { flrw: &flrw, time };
    let extrinsic = FlrwExtrinsicCurvature { flrw: &flrw, time };
    let lapse = ConstantLapse(1.0);
    let point3 = [point4[1], point4[2], point4[3]];
    let lambda = 3.0 * hubble * hubble;
    let rho = lambda / (8.0 * PI);
    let gamma = spatial.components(&point3);
    let sources = AdmSources::perfect_fluid(rho, -rho, &gamma);
    let rhs = curvature_evolution_rhs(
        &spatial,
        &extrinsic,
        &lapse,
        &ZeroVectorField,
        &point3,
        &sources,
        &settings(),
    )
    .unwrap();

    let mut max_error = 0.0_f64;
    for i in 0..3
    {
        for j in 0..3
        {
            max_error = max_error.max((direct[i][j] - rhs.total[i][j]).abs());
        }
    }
    assert!(
        max_error < 1.0e-6,
        "max error between direct dK/dt and the closed-form RHS: {max_error}"
    );
}

// ---------------------------------------------------------------------------
// Oracle D -- deliberate constraint violations: nonzero, closed-form,
// monotonically scaling residuals, correctly attributed.
// ---------------------------------------------------------------------------

#[test]
fn oracle_d_deliberate_hamiltonian_violation_is_monotonic_and_matches_closed_form() {
    let point = [1.0, 2.0, 3.0];
    let settings = settings();
    let mut previous = 0.0_f64;
    for &epsilon in &[0.01, 0.02, 0.05, 0.1]
    {
        let curvature = ConstantDiagonalCurvature {
            values: [epsilon, epsilon, 0.0],
        };
        let hamiltonian = hamiltonian_constraint(
            &FlatSpace,
            &curvature,
            &point,
            &AdmSources::VACUUM,
            &settings,
        )
        .unwrap();
        let expected = 2.0 * epsilon * epsilon;
        assert!(
            (hamiltonian.signed_residual - expected).abs() < 1.0e-6,
            "epsilon={epsilon}: {} vs {expected}",
            hamiltonian.signed_residual
        );
        assert!(
            hamiltonian.absolute_residual > previous,
            "residual should increase monotonically with epsilon"
        );
        previous = hamiltonian.absolute_residual;

        // A position-independent (constant) K_ij has zero divergence: momentum
        // stays exactly zero regardless of the Hamiltonian violation.
        let momentum = momentum_constraint(
            &FlatSpace,
            &curvature,
            &point,
            &AdmSources::VACUUM,
            &settings,
        )
        .unwrap();
        assert!(
            momentum.residual_norm < 1.0e-6,
            "momentum should stay ~0 for constant K"
        );
    }
}

#[test]
fn oracle_d_deliberate_momentum_violation_is_monotonic_and_matches_closed_form() {
    let point = [1.0, 2.0, 3.0];
    let settings = settings();
    let mut previous = 0.0_f64;
    for &epsilon in &[0.01, 0.02, 0.05, 0.1]
    {
        let curvature = LinearTracelessCurvature { epsilon };
        let momentum = momentum_constraint(
            &FlatSpace,
            &curvature,
            &point,
            &AdmSources::VACUUM,
            &settings,
        )
        .unwrap();
        assert!(
            (momentum.residual[0] - epsilon).abs() < 1.0e-6,
            "epsilon={epsilon}: M^1={} vs {epsilon}",
            momentum.residual[0]
        );
        assert!(momentum.residual[1].abs() < 1.0e-6);
        assert!(momentum.residual[2].abs() < 1.0e-6);
        assert!(
            momentum.residual_norm > previous,
            "residual norm should increase monotonically"
        );
        previous = momentum.residual_norm;

        // Traceless K makes K^2 = 0, so the Hamiltonian residual is exactly
        // -K_ij K^{ij} = -1.5 epsilon^2 x1^2 here (a genuine, closed-form check
        // that the diagnostic decomposition attributes this correctly).
        let hamiltonian = hamiltonian_constraint(
            &FlatSpace,
            &curvature,
            &point,
            &AdmSources::VACUUM,
            &settings,
        )
        .unwrap();
        let expected_hamiltonian = -1.5 * epsilon * epsilon * point[0] * point[0];
        assert!(
            (hamiltonian.signed_residual - expected_hamiltonian).abs() < 1.0e-6,
            "epsilon={epsilon}: H={} vs {expected_hamiltonian}",
            hamiltonian.signed_residual
        );
        assert!(
            (hamiltonian.mean_curvature_squared).abs() < 1.0e-9,
            "K should be exactly traceless"
        );
    }
}

// ---------------------------------------------------------------------------
// Gauge unit tests: isolate the differentiation machinery with zero-truncation
// closed forms, independent of any particular GR solution.
// ---------------------------------------------------------------------------

#[test]
fn gauge_quadratic_lapse_gives_exact_hessian_on_flat_space() {
    // The central second-derivative stencil has zero *truncation* error for a
    // quadratic field, but subtracting O(1)-magnitude evaluations to recover an
    // O(h^2)-scale second difference costs floating-point precision: the
    // rounding floor here is ~epsilon / step^2 ~ 2e-16 / 1e-6 ~ 1e-9-1e-10,
    // matching the measured residual. This is the well-known rounding/truncation
    // trade-off of second-derivative finite differences, not a defect.
    let coefficient = 0.3;
    let lapse = QuadraticLapse { coefficient };
    let point = [1.0, 2.0, 3.0];
    let rhs = curvature_evolution_rhs(
        &FlatSpace,
        &ZeroTensorField,
        &lapse,
        &ZeroVectorField,
        &point,
        &AdmSources::VACUUM,
        &settings(),
    )
    .unwrap();
    for i in 0..3
    {
        for j in 0..3
        {
            let expected = if i == j { 2.0 * coefficient } else { 0.0 };
            assert!(
                (rhs.lapse_hessian[i][j] - expected).abs() < 1.0e-6,
                "({i},{j}): {} vs {expected}",
                rhs.lapse_hessian[i][j]
            );
        }
    }
    // R_ij = 0 on flat space, so total = -lapse_hessian exactly here.
    for i in 0..3
    {
        for j in 0..3
        {
            assert!((rhs.total[i][j] + rhs.lapse_hessian[i][j]).abs() < 1.0e-9);
        }
    }
}

// Explicit tensor-index loops read most clearly here (matching the crate's
// curvature, action, and adm modules).
#[allow(clippy::needless_range_loop)]
#[test]
fn gauge_linear_shift_gives_exact_lie_derivative_of_metric_and_curvature() {
    let mut matrix = [[0.0_f64; 3]; 3];
    matrix[0][1] = 0.4;
    let shift = LinearShift { matrix };
    let point = [1.0, 2.0, 3.0];
    let proportionality = 0.7;
    let curvature = ConstantDiagonalCurvature {
        values: [proportionality, proportionality, proportionality],
    };
    let lapse = ConstantLapse(1.0);

    let metric_rhs =
        metric_evolution_rhs(&FlatSpace, &curvature, &lapse, &shift, &point, &settings()).unwrap();
    let mut expected_lie = [[0.0_f64; 3]; 3];
    for i in 0..3
    {
        for j in 0..3
        {
            expected_lie[i][j] = matrix[j][i] + matrix[i][j];
        }
    }
    for i in 0..3
    {
        for j in 0..3
        {
            assert!(
                (metric_rhs.lie_derivative[i][j] - expected_lie[i][j]).abs() < 1.0e-9,
                "({i},{j}): {} vs {}",
                metric_rhs.lie_derivative[i][j],
                expected_lie[i][j]
            );
        }
    }

    let curvature_rhs = curvature_evolution_rhs(
        &FlatSpace,
        &curvature,
        &lapse,
        &shift,
        &point,
        &AdmSources::VACUUM,
        &settings(),
    )
    .unwrap();
    for i in 0..3
    {
        for j in 0..3
        {
            let expected = proportionality * expected_lie[i][j];
            assert!(
                (curvature_rhs.lie_derivative[i][j] - expected).abs() < 1.0e-9,
                "({i},{j}): {} vs {expected}",
                curvature_rhs.lie_derivative[i][j]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Determinism and error rejection.
// ---------------------------------------------------------------------------

#[test]
fn evaluators_are_deterministic() {
    let point = [1.0, 2.0, 3.0];
    let settings = settings();
    let curvature = LinearTracelessCurvature { epsilon: 0.05 };
    let lapse = QuadraticLapse { coefficient: 0.1 };
    let mut matrix = [[0.0_f64; 3]; 3];
    matrix[0][1] = 0.2;
    let shift = LinearShift { matrix };

    let first = hamiltonian_constraint(
        &FlatSpace,
        &curvature,
        &point,
        &AdmSources::VACUUM,
        &settings,
    )
    .unwrap();
    let second = hamiltonian_constraint(
        &FlatSpace,
        &curvature,
        &point,
        &AdmSources::VACUUM,
        &settings,
    )
    .unwrap();
    assert_eq!(first, second);

    let first = momentum_constraint(
        &FlatSpace,
        &curvature,
        &point,
        &AdmSources::VACUUM,
        &settings,
    )
    .unwrap();
    let second = momentum_constraint(
        &FlatSpace,
        &curvature,
        &point,
        &AdmSources::VACUUM,
        &settings,
    )
    .unwrap();
    assert_eq!(first, second);

    let first =
        metric_evolution_rhs(&FlatSpace, &curvature, &lapse, &shift, &point, &settings).unwrap();
    let second =
        metric_evolution_rhs(&FlatSpace, &curvature, &lapse, &shift, &point, &settings).unwrap();
    assert_eq!(first, second);

    let first = curvature_evolution_rhs(
        &FlatSpace,
        &curvature,
        &lapse,
        &shift,
        &point,
        &AdmSources::VACUUM,
        &settings,
    )
    .unwrap();
    let second = curvature_evolution_rhs(
        &FlatSpace,
        &curvature,
        &lapse,
        &shift,
        &point,
        &AdmSources::VACUUM,
        &settings,
    )
    .unwrap();
    assert_eq!(first, second);
}

struct SingularMetric;
impl Metric<3> for SingularMetric {
    fn components(&self, _coordinates: &[f64; 3]) -> [[f64; 3]; 3] {
        // Rank-deficient: the third row is a duplicate of the first.
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    }
}

struct NonFiniteLapse;
impl SpatialScalarField for NonFiniteLapse {
    fn value(&self, _coordinates: &[f64; 3]) -> f64 {
        f64::NAN
    }
}

struct NonFiniteShift;
impl SpatialVectorField for NonFiniteShift {
    fn components(&self, _coordinates: &[f64; 3]) -> [f64; 3] {
        [f64::INFINITY, 0.0, 0.0]
    }
}

#[test]
fn rejects_invalid_requests() {
    let point = [1.0, 2.0, 3.0];
    let settings = settings();

    assert!(matches!(
        hamiltonian_constraint(
            &FlatSpace,
            &ZeroTensorField,
            &[f64::NAN, 2.0, 3.0],
            &AdmSources::VACUUM,
            &settings
        ),
        Err(AdmEvolutionError::NonFiniteCoordinate(0))
    ));

    assert!(matches!(
        hamiltonian_constraint(
            &FlatSpace,
            &ZeroTensorField,
            &point,
            &AdmSources::VACUUM,
            &AdmEvolutionSettings {
                spatial_step: 0.0,
                metric_step: 1.0e-3
            }
        ),
        Err(AdmEvolutionError::InvalidStep(_))
    ));
    assert!(matches!(
        hamiltonian_constraint(
            &FlatSpace,
            &ZeroTensorField,
            &point,
            &AdmSources::VACUUM,
            &AdmEvolutionSettings {
                spatial_step: 1.0e-3,
                metric_step: -1.0
            }
        ),
        Err(AdmEvolutionError::InvalidStep(_))
    ));

    assert!(matches!(
        hamiltonian_constraint(
            &SingularMetric,
            &ZeroTensorField,
            &point,
            &AdmSources::VACUUM,
            &settings
        ),
        Err(AdmEvolutionError::SingularSpatialMetric)
    ));
    assert!(matches!(
        momentum_constraint(
            &SingularMetric,
            &ZeroTensorField,
            &point,
            &AdmSources::VACUUM,
            &settings
        ),
        Err(AdmEvolutionError::SingularSpatialMetric)
    ));

    assert!(matches!(
        metric_evolution_rhs(
            &FlatSpace,
            &ZeroTensorField,
            &NonFiniteLapse,
            &ZeroVectorField,
            &point,
            &settings
        ),
        Err(AdmEvolutionError::NonFiniteField { field: "lapse" })
    ));
    assert!(matches!(
        metric_evolution_rhs(
            &FlatSpace,
            &ZeroTensorField,
            &ConstantLapse(1.0),
            &NonFiniteShift,
            &point,
            &settings
        ),
        Err(AdmEvolutionError::NonFiniteField { field: "shift" })
    ));
    assert!(matches!(
        curvature_evolution_rhs(
            &FlatSpace,
            &ZeroTensorField,
            &NonFiniteLapse,
            &ZeroVectorField,
            &point,
            &AdmSources::VACUUM,
            &settings
        ),
        Err(AdmEvolutionError::NonFiniteField { field: "lapse" })
    ));
}
