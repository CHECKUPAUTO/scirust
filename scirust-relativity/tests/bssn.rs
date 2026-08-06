//! Oracles for the BSSN formulation core (Layer 3.3; see `docs/LAYER_3_BSSN.md`).
//!
//! Oracle A (Minkowski): the trivial BSSN state, vanishing right-hand sides,
//! exact round trip. Oracle B (flat FLRW): expansion carried entirely by `phi`
//! and `K`, with ADM/BSSN right-hand-side equivalence. Oracle C (anisotropic
//! manufactured state -- an algebraic oracle, **not** a claimed cosmological
//! solution): nonzero `Atilde_ij`, trace removal, determinant normalization.
//! Oracle D (static Schwarzschild slice): the conformal Ricci decomposition
//! against the independently computed physical Ricci. Oracle E: deliberate
//! algebraic violations and their explicit projections.
//!
//! Established general relativity; a numerical approximation validated against
//! exact and independently-computed oracles. Implementing BSSN does **not**
//! demonstrate numerical stability, and no test here claims it does.

use scirust_relativity::adm_evolution::{AdmEvolutionSettings, AdmSources, SpatialTensorField};
use scirust_relativity::bssn::{
    BssnError, BssnGauge, adm_bssn_equivalence, adm_to_bssn, bssn_algebraic_constraints,
    bssn_constraints_from_adm, bssn_evolution_rhs, bssn_to_adm, conformal_ricci,
    project_trace_free, project_unit_determinant,
};
use scirust_relativity::{Metric, determinant, ricci_tensor_from_metric};
use std::f64::consts::PI;

fn settings() -> AdmEvolutionSettings {
    AdmEvolutionSettings {
        spatial_step: 1.0e-3,
        metric_step: 1.0e-3,
    }
}

const ORIGIN: [f64; 3] = [0.0, 0.0, 0.0];

struct ConstantMetric([[f64; 3]; 3]);
impl Metric<3> for ConstantMetric {
    fn components(&self, _x: &[f64; 3]) -> [[f64; 3]; 3] {
        self.0
    }
}
struct ConstantTensor([[f64; 3]; 3]);
impl SpatialTensorField for ConstantTensor {
    fn components(&self, _x: &[f64; 3]) -> [[f64; 3]; 3] {
        self.0
    }
}

fn identity() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}
fn zeros() -> [[f64; 3]; 3] {
    [[0.0; 3]; 3]
}
fn diagonal(a: f64, b: f64, c: f64) -> [[f64; 3]; 3] {
    [[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c]]
}
fn max_abs_difference(left: &[[f64; 3]; 3], right: &[[f64; 3]; 3]) -> f64 {
    let mut worst = 0.0_f64;
    for i in 0..3
    {
        for j in 0..3
        {
            worst = worst.max((left[i][j] - right[i][j]).abs());
        }
    }
    worst
}

/// The Schwarzschild `t = const` slice in `(r, theta, phi)` -- curved,
/// inhomogeneous, and not conformally flat, so both `Rtilde` and `Rphi` are
/// nonzero (a genuinely nontrivial decomposition test).
struct SchwarzschildSlice {
    mass: f64,
}
impl Metric<3> for SchwarzschildSlice {
    fn components(&self, x: &[f64; 3]) -> [[f64; 3]; 3] {
        let (radius, polar) = (x[0], x[1]);
        let lapse = 1.0 - 2.0 * self.mass / radius;
        let sine = polar.sin();
        diagonal(1.0 / lapse, radius * radius, radius * radius * sine * sine)
    }
}

// ---------------------------------------------------------------------------
// Oracle A -- Minkowski
// ---------------------------------------------------------------------------

#[test]
fn oracle_a_minkowski_converts_to_the_trivial_bssn_state() {
    let state = adm_to_bssn(
        &ConstantMetric(identity()),
        &ConstantTensor(zeros()),
        &ORIGIN,
        &settings(),
    )
    .expect("converts");

    assert!(
        state.conformal_factor.abs() < 1.0e-14,
        "phi = {}",
        state.conformal_factor
    );
    assert!((state.chi() - 1.0).abs() < 1.0e-14, "chi = {}", state.chi());
    assert!(
        state.mean_curvature.abs() < 1.0e-14,
        "K = {}",
        state.mean_curvature
    );
    assert!(max_abs_difference(&state.conformal_metric, &identity()) < 1.0e-14);
    assert!(max_abs_difference(&state.conformal_curvature, &zeros()) < 1.0e-14);
    for component in state.conformal_connection
    {
        // A constant metric has exactly zero Christoffel symbols.
        assert!(component.abs() < 1.0e-12, "Gammatilde = {component}");
    }
}

#[test]
fn oracle_a_minkowski_evolution_rhs_vanishes() {
    let (_, rhs) = bssn_evolution_rhs(
        &ConstantMetric(identity()),
        &ConstantTensor(zeros()),
        &ORIGIN,
        &BssnGauge::SYNCHRONOUS,
        &AdmSources::VACUUM,
        &settings(),
    )
    .expect("rhs");

    assert!(rhs.conformal_factor.abs() < 1.0e-14);
    assert!(rhs.mean_curvature.abs() < 1.0e-14);
    assert!(max_abs_difference(&rhs.conformal_metric, &zeros()) < 1.0e-14);
    assert!(max_abs_difference(&rhs.conformal_curvature, &zeros()) < 1.0e-10);
    for component in rhs.conformal_connection
    {
        assert!(component.abs() < 1.0e-14);
    }
}

// ---------------------------------------------------------------------------
// Round trip
// ---------------------------------------------------------------------------

/// Scale-aware round-trip check: the reconstruction error is compared against
/// the magnitude of the quantity being reconstructed, not an absolute tolerance.
// Explicit tensor-index loops read most clearly here.
#[allow(clippy::needless_range_loop)]
fn assert_round_trip(metric: [[f64; 3]; 3], curvature: [[f64; 3]; 3], label: &str) {
    let state = adm_to_bssn(
        &ConstantMetric(metric),
        &ConstantTensor(curvature),
        &ORIGIN,
        &settings(),
    )
    .expect("converts");
    let back = bssn_to_adm(&state).expect("reconstructs");

    let metric_scale = metric
        .iter()
        .flatten()
        .fold(0.0_f64, |a, v| a.max(v.abs()))
        .max(1.0);
    let curvature_scale = curvature
        .iter()
        .flatten()
        .fold(0.0_f64, |a, v| a.max(v.abs()))
        .max(1.0);

    let metric_error = max_abs_difference(&back.spatial_metric, &metric) / metric_scale;
    let curvature_error =
        max_abs_difference(&back.extrinsic_curvature, &curvature) / curvature_scale;
    assert!(
        metric_error < 1.0e-14,
        "{label}: relative metric error {metric_error}"
    );
    assert!(
        curvature_error < 1.0e-14,
        "{label}: relative curvature error {curvature_error}"
    );

    // Symmetry and determinant are preserved.
    let mut asymmetry = 0.0_f64;
    for i in 0..3
    {
        for j in 0..3
        {
            asymmetry =
                asymmetry.max((back.spatial_metric[i][j] - back.spatial_metric[j][i]).abs());
        }
    }
    assert!(
        asymmetry < 1.0e-15,
        "{label}: reconstructed metric is not symmetric"
    );
    let original_det = determinant::<3>(&metric).expect("det");
    let reconstructed_det = determinant::<3>(&back.spatial_metric).expect("det");
    assert!(
        ((reconstructed_det - original_det) / original_det).abs() < 1.0e-13,
        "{label}: determinant drifted"
    );
}

#[test]
fn round_trip_preserves_adm_states() {
    assert_round_trip(identity(), zeros(), "minkowski");
    assert_round_trip(
        diagonal(1.44, 1.44, 1.44),
        diagonal(-0.5, -0.5, -0.5),
        "flrw-like",
    );
    assert_round_trip(
        diagonal(1.3, 0.8, 1.1),
        diagonal(-0.21, -0.09, -0.15),
        "anisotropic",
    );
    assert_round_trip(
        [[1.2, 0.1, 0.0], [0.1, 0.9, 0.05], [0.0, 0.05, 1.05]],
        [[-0.2, 0.03, 0.0], [0.03, -0.11, 0.01], [0.0, 0.01, -0.14]],
        "off-diagonal",
    );
}

// ---------------------------------------------------------------------------
// Algebraic constraints
// ---------------------------------------------------------------------------

#[test]
fn conversion_satisfies_the_algebraic_constraints_by_construction() {
    for (metric, curvature) in [
        (identity(), zeros()),
        (diagonal(1.44, 1.44, 1.44), diagonal(-0.5, -0.5, -0.5)),
        (diagonal(1.3, 0.8, 1.1), diagonal(-0.21, -0.09, -0.15)),
    ]
    {
        let (_, constraints) = bssn_constraints_from_adm(
            &ConstantMetric(metric),
            &ConstantTensor(curvature),
            &ORIGIN,
            &settings(),
        )
        .expect("constraints");
        assert!(constraints.finite);
        assert!(
            constraints.determinant_absolute < 1.0e-14,
            "det residual {}",
            constraints.determinant_residual
        );
        assert!(
            constraints.trace_absolute < 1.0e-14,
            "trace residual {}",
            constraints.trace_residual
        );
        assert!(
            constraints.connection_absolute < 1.0e-12,
            "connection residual {:?}",
            constraints.connection_residual
        );
    }
}

#[test]
fn normalized_trace_residual_is_none_when_the_curvature_vanishes() {
    // An honest "undefined" rather than a fabricated zero.
    let (_, constraints) = bssn_constraints_from_adm(
        &ConstantMetric(identity()),
        &ConstantTensor(zeros()),
        &ORIGIN,
        &settings(),
    )
    .expect("constraints");
    assert!(constraints.trace_normalized.is_none());

    let (_, nonzero) = bssn_constraints_from_adm(
        &ConstantMetric(diagonal(1.3, 0.8, 1.1)),
        &ConstantTensor(diagonal(-0.21, -0.09, -0.15)),
        &ORIGIN,
        &settings(),
    )
    .expect("constraints");
    assert!(nonzero.trace_normalized.is_some());
}

// ---------------------------------------------------------------------------
// Oracle E -- deliberate violations and explicit projection
// ---------------------------------------------------------------------------

#[test]
fn oracle_e_determinant_violation_responds_and_projects() {
    let mut previous = 0.0_f64;
    for &excess in &[0.01, 0.02, 0.05, 0.1]
    {
        let mut state = adm_to_bssn(
            &ConstantMetric(diagonal(1.3, 0.8, 1.1)),
            &ConstantTensor(diagonal(-0.21, -0.09, -0.15)),
            &ORIGIN,
            &settings(),
        )
        .expect("converts");
        // Scale the conformal metric so det gammatilde = (1 + excess)^3.
        let factor = 1.0 + excess;
        for i in 0..3
        {
            for j in 0..3
            {
                state.conformal_metric[i][j] *= factor;
            }
        }
        let reconstructed = state.conformal_connection;
        let before = bssn_algebraic_constraints(&state, &reconstructed);
        let expected = factor.powi(3) - 1.0;
        assert!(
            (before.determinant_residual - expected).abs() < 1.0e-12,
            "excess {excess}: {} vs {expected}",
            before.determinant_residual
        );
        assert!(
            before.determinant_absolute > previous,
            "residual must grow with the violation"
        );
        previous = before.determinant_absolute;

        // Explicit projection repairs exactly this constraint.
        let projection = project_unit_determinant(&mut state).expect("projects");
        assert!((projection.residual_before - expected).abs() < 1.0e-12);
        assert!(
            projection.residual_after.abs() < 1.0e-13,
            "after {}",
            projection.residual_after
        );
        assert!(projection.correction_magnitude > 0.0);

        // Idempotence: projecting again changes essentially nothing.
        let again = project_unit_determinant(&mut state).expect("projects");
        assert!(again.residual_after.abs() < 1.0e-13);
        assert!(
            again.correction_magnitude < 1.0e-13,
            "not idempotent: {}",
            again.correction_magnitude
        );
    }
}

#[test]
fn oracle_e_trace_violation_responds_and_projects() {
    let mut previous = 0.0_f64;
    for &amplitude in &[0.01, 0.02, 0.05, 0.1]
    {
        let mut state = adm_to_bssn(
            &ConstantMetric(diagonal(1.3, 0.8, 1.1)),
            &ConstantTensor(diagonal(-0.21, -0.09, -0.15)),
            &ORIGIN,
            &settings(),
        )
        .expect("converts");
        // Add a pure-trace piece: Atilde += amplitude * gammatilde, so the
        // conformal trace becomes exactly 3 * amplitude.
        for i in 0..3
        {
            for j in 0..3
            {
                state.conformal_curvature[i][j] += amplitude * state.conformal_metric[i][j];
            }
        }
        let reconstructed = state.conformal_connection;
        let before = bssn_algebraic_constraints(&state, &reconstructed);
        assert!(
            (before.trace_residual - 3.0 * amplitude).abs() < 1.0e-12,
            "amplitude {amplitude}: {} vs {}",
            before.trace_residual,
            3.0 * amplitude
        );
        assert!(before.trace_absolute > previous);
        previous = before.trace_absolute;
        // The determinant constraint is untouched by a trace perturbation.
        assert!(
            before.determinant_absolute < 1.0e-13,
            "determinant should stay clean"
        );

        let projection = project_trace_free(&mut state).expect("projects");
        assert!((projection.residual_before - 3.0 * amplitude).abs() < 1.0e-12);
        assert!(projection.residual_after.abs() < 1.0e-13);
        let again = project_trace_free(&mut state).expect("projects");
        assert!(again.correction_magnitude < 1.0e-14, "not idempotent");
    }
}

#[test]
fn oracle_e_connection_violation_is_detected() {
    let mut state = adm_to_bssn(
        &ConstantMetric(diagonal(1.3, 0.8, 1.1)),
        &ConstantTensor(diagonal(-0.21, -0.09, -0.15)),
        &ORIGIN,
        &settings(),
    )
    .expect("converts");
    let reconstructed = state.conformal_connection;
    let mut previous = 0.0_f64;
    for &offset in &[0.01, 0.05, 0.2]
    {
        state.conformal_connection = [
            reconstructed[0] + offset,
            reconstructed[1],
            reconstructed[2],
        ];
        let constraints = bssn_algebraic_constraints(&state, &reconstructed);
        assert!(
            (constraints.connection_absolute - offset).abs() < 1.0e-14,
            "offset {offset}: {}",
            constraints.connection_absolute
        );
        assert!(constraints.connection_absolute > previous);
        previous = constraints.connection_absolute;
        // Determinant and trace are unaffected by a connection perturbation.
        assert!(constraints.determinant_absolute < 1.0e-13);
        assert!(constraints.trace_absolute < 1.0e-13);
    }
}

// ---------------------------------------------------------------------------
// Oracle D -- conformal Ricci decomposition on a curved slice
// ---------------------------------------------------------------------------

#[test]
fn oracle_d_conformal_ricci_reconstructs_the_physical_ricci() {
    let slice = SchwarzschildSlice { mass: 1.0 };
    // Off the equator so the geometry is genuinely anisotropic in all directions.
    for point in [[6.0, 1.27, 0.0], [8.0, 1.0, 0.0], [12.0, 1.4, 0.0]]
    {
        let decomposition = conformal_ricci(&slice, &point, &settings()).expect("decomposes");
        let physical = ricci_tensor_from_metric::<_, 3>(&slice, &point, 1.0e-3, 1.0e-3).expect("R");

        let scale = physical
            .iter()
            .flatten()
            .fold(0.0_f64, |a, v| a.max(v.abs()));
        let mismatch = max_abs_difference(&decomposition.total, &physical);
        // The two paths share only the Ricci engine: `Rtilde` is the Ricci of
        // the conformal metric and `Rphi` is new, so agreement validates `Rphi`.
        // Tolerance is the nested finite-difference floor, ~1e-6 relative.
        assert!(
            mismatch / scale.max(1.0e-12) < 1.0e-4,
            "point {point:?}: relative mismatch {}",
            mismatch / scale
        );

        // Both pieces must be individually nonzero: the Schwarzschild slice is
        // not conformally flat, so this is a nontrivial cancellation.
        let conformal_scale = decomposition
            .conformal
            .iter()
            .flatten()
            .fold(0.0_f64, |a, v| a.max(v.abs()));
        let phi_scale = decomposition
            .conformal_factor_part
            .iter()
            .flatten()
            .fold(0.0_f64, |a, v| a.max(v.abs()));
        assert!(
            conformal_scale > 1.0e-3,
            "Rtilde should be nonzero: {conformal_scale}"
        );
        assert!(phi_scale > 1.0e-3, "Rphi should be nonzero: {phi_scale}");
    }
}

#[test]
fn oracle_d_time_symmetric_slice_has_vanishing_k_and_atilde() {
    let slice = SchwarzschildSlice { mass: 1.0 };
    let point = [6.0, 1.27, 0.0];
    let state =
        adm_to_bssn(&slice, &ConstantTensor(zeros()), &point, &settings()).expect("converts");
    assert!(state.mean_curvature.abs() < 1.0e-14);
    assert!(max_abs_difference(&state.conformal_curvature, &zeros()) < 1.0e-14);
    // The conformal metric still has unit determinant on a curved slice.
    let det = determinant::<3>(&state.conformal_metric).expect("det");
    assert!((det - 1.0).abs() < 1.0e-12, "det gammatilde = {det}");
}

// ---------------------------------------------------------------------------
// Oracle B / C -- ADM/BSSN right-hand-side equivalence
// ---------------------------------------------------------------------------

/// Flat-FLRW data that sits exactly on the constraint surface.
fn flrw_state(hubble: f64, scale: f64, w: f64) -> ([[f64; 3]; 3], [[f64; 3]; 3], AdmSources) {
    let spatial = scale * scale;
    let metric = diagonal(spatial, spatial, spatial);
    let curvature = diagonal(-hubble * spatial, -hubble * spatial, -hubble * spatial);
    let energy_density = 3.0 * hubble * hubble / (8.0 * PI);
    let sources = AdmSources::perfect_fluid(energy_density, w * energy_density, &metric);
    (metric, curvature, sources)
}

#[test]
fn oracle_b_flrw_expansion_is_carried_by_phi_and_k() {
    let hubble = 0.5;
    let (metric, curvature, _) = flrw_state(hubble, 1.2, -1.0);
    let (state, constraints) = bssn_constraints_from_adm(
        &ConstantMetric(metric),
        &ConstantTensor(curvature),
        &ORIGIN,
        &settings(),
    )
    .expect("converts");

    // gammatilde is the unit-determinant Euclidean metric; Atilde vanishes; the
    // expansion lives entirely in phi and K.
    assert!(max_abs_difference(&state.conformal_metric, &identity()) < 1.0e-13);
    assert!(max_abs_difference(&state.conformal_curvature, &zeros()) < 1.0e-13);
    for component in state.conformal_connection
    {
        assert!(component.abs() < 1.0e-12);
    }
    assert!((state.mean_curvature - (-3.0 * hubble)).abs() < 1.0e-13);
    // phi = (1/12) ln(a^6) = (1/2) ln a
    assert!((state.conformal_factor - 0.5 * 1.2_f64.ln()).abs() < 1.0e-13);
    assert!(constraints.determinant_absolute < 1.0e-13);
    assert!(constraints.trace_absolute < 1.0e-13);
}

#[test]
fn oracle_b_adm_bssn_rhs_equivalence_on_the_constraint_surface() {
    // On the constraint surface the reconstructed ADM RHS must match Layer 3.1
    // exactly (up to floating point), for both d_t gamma and d_t K.
    for &(hubble, scale, w) in &[(0.5, 1.0, -1.0), (0.37, 1.44, 0.0), (0.6, 0.9, 1.0 / 3.0)]
    {
        let (metric, curvature, sources) = flrw_state(hubble, scale, w);
        let equivalence = adm_bssn_equivalence(
            &ConstantMetric(metric),
            &ConstantTensor(curvature),
            &ORIGIN,
            &BssnGauge::SYNCHRONOUS,
            &sources,
            &settings(),
        )
        .expect("equivalence");

        assert!(
            equivalence.hamiltonian_residual.abs() < 1.0e-14,
            "data should be on the constraint surface: H = {}",
            equivalence.hamiltonian_residual
        );
        assert!(
            equivalence.metric_difference < 1.0e-14,
            "w={w}: d_t gamma differs by {}",
            equivalence.metric_difference
        );
        assert!(
            equivalence.curvature_difference < 1.0e-12,
            "w={w}: d_t K differs by {}",
            equivalence.curvature_difference
        );
    }
}

#[test]
fn oracle_c_anisotropic_state_exercises_atilde_and_equivalence() {
    // A manufactured algebraic oracle: a diagonal anisotropic metric with an
    // anisotropic extrinsic curvature. This is NOT claimed to be a physical
    // cosmological solution -- it exists to exercise nonzero Atilde_ij, the
    // trace decomposition, and the round trip.
    let metric = diagonal(1.3, 0.8, 1.1);
    let curvature = diagonal(-0.21, -0.09, -0.15);
    let sources = AdmSources::perfect_fluid(0.02, 0.007, &metric);

    let state = adm_to_bssn(
        &ConstantMetric(metric),
        &ConstantTensor(curvature),
        &ORIGIN,
        &settings(),
    )
    .expect("converts");
    let atilde_scale = state
        .conformal_curvature
        .iter()
        .flatten()
        .fold(0.0_f64, |a, v| a.max(v.abs()));
    assert!(
        atilde_scale > 1.0e-3,
        "Atilde should be genuinely nonzero: {atilde_scale}"
    );

    let equivalence = adm_bssn_equivalence(
        &ConstantMetric(metric),
        &ConstantTensor(curvature),
        &ORIGIN,
        &BssnGauge::SYNCHRONOUS,
        &sources,
        &settings(),
    )
    .expect("equivalence");

    // d_t gamma is equivalent unconditionally.
    assert!(
        equivalence.metric_difference < 1.0e-14,
        "d_t gamma differs by {}",
        equivalence.metric_difference
    );
    // This state is deliberately OFF the constraint surface, so the raw d_t K
    // difference is large -- and is explained exactly by alpha * H.
    assert!(
        equivalence.hamiltonian_residual.abs() > 1.0e-2,
        "expected an off-constraint state, H = {}",
        equivalence.hamiltonian_residual
    );
    assert!(
        equivalence.curvature_difference > 1.0e-2,
        "raw difference should be large off-constraint: {}",
        equivalence.curvature_difference
    );
    assert!(
        equivalence.curvature_difference_constraint_corrected < 1.0e-12,
        "constraint-corrected difference should vanish: {}",
        equivalence.curvature_difference_constraint_corrected
    );
}

// ---------------------------------------------------------------------------
// Determinism and rejection
// ---------------------------------------------------------------------------

#[test]
fn conversion_and_rhs_are_deterministic() {
    let metric = diagonal(1.3, 0.8, 1.1);
    let curvature = diagonal(-0.21, -0.09, -0.15);
    let sources = AdmSources::perfect_fluid(0.02, 0.007, &metric);

    let first = adm_to_bssn(
        &ConstantMetric(metric),
        &ConstantTensor(curvature),
        &ORIGIN,
        &settings(),
    )
    .expect("converts");
    let second = adm_to_bssn(
        &ConstantMetric(metric),
        &ConstantTensor(curvature),
        &ORIGIN,
        &settings(),
    )
    .expect("converts");
    assert_eq!(first, second);

    let a = bssn_evolution_rhs(
        &ConstantMetric(metric),
        &ConstantTensor(curvature),
        &ORIGIN,
        &BssnGauge::SYNCHRONOUS,
        &sources,
        &settings(),
    )
    .expect("rhs");
    let b = bssn_evolution_rhs(
        &ConstantMetric(metric),
        &ConstantTensor(curvature),
        &ORIGIN,
        &BssnGauge::SYNCHRONOUS,
        &sources,
        &settings(),
    )
    .expect("rhs");
    assert_eq!(a, b);
}

#[test]
fn rejects_singular_and_non_finite_states() {
    // Rank-deficient spatial metric.
    let singular = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
    assert!(matches!(
        adm_to_bssn(
            &ConstantMetric(singular),
            &ConstantTensor(zeros()),
            &ORIGIN,
            &settings()
        ),
        Err(BssnError::SingularAdmMetric | BssnError::InvalidDeterminant(_))
    ));

    // Negative-determinant (non-Riemannian) spatial metric.
    let negative = diagonal(-1.0, 1.0, 1.0);
    assert!(matches!(
        adm_to_bssn(
            &ConstantMetric(negative),
            &ConstantTensor(zeros()),
            &ORIGIN,
            &settings()
        ),
        Err(BssnError::InvalidDeterminant(_))
    ));

    // Non-finite extrinsic curvature.
    let bad_curvature = diagonal(f64::NAN, 0.0, 0.0);
    assert!(matches!(
        adm_to_bssn(
            &ConstantMetric(identity()),
            &ConstantTensor(bad_curvature),
            &ORIGIN,
            &settings()
        ),
        Err(BssnError::NonFiniteState { .. })
    ));

    // Non-finite spatial metric.
    let bad_metric = diagonal(f64::INFINITY, 1.0, 1.0);
    assert!(matches!(
        adm_to_bssn(
            &ConstantMetric(bad_metric),
            &ConstantTensor(zeros()),
            &ORIGIN,
            &settings()
        ),
        Err(BssnError::NonFiniteState { .. } | BssnError::InvalidDeterminant(_))
    ));

    // Non-finite gauge.
    assert!(matches!(
        bssn_evolution_rhs(
            &ConstantMetric(identity()),
            &ConstantTensor(zeros()),
            &ORIGIN,
            &BssnGauge {
                lapse: f64::NAN,
                shift: [0.0; 3]
            },
            &AdmSources::VACUUM,
            &settings(),
        ),
        Err(BssnError::NonFiniteState { quantity: "gauge" })
    ));

    // Projection rejects a singular conformal metric.
    let mut broken = adm_to_bssn(
        &ConstantMetric(identity()),
        &ConstantTensor(zeros()),
        &ORIGIN,
        &settings(),
    )
    .expect("converts");
    broken.conformal_metric = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
    assert!(matches!(
        project_unit_determinant(&mut broken),
        Err(BssnError::FailedProjection { .. })
    ));
}
