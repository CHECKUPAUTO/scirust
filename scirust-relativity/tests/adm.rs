//! Oracles for the 3+1 (ADM) decomposition (Layer 2, fourth slice; see
//! `docs/LAYER_2_ADM.md`).
//!
//! O0 the extracted lapse/shift/spatial-metric reconstruct the 4-metric; O1 the
//! spatial Ricci scalar hits its analytic value; O2 the extrinsic curvature and
//! mean curvature match; O3/O4 the Hamiltonian and momentum constraints vanish
//! for four exact vacuum-with-Lambda foliations (Schwarzschild, static de
//! Sitter, FLRW, and horizon-penetrating Painlevé–Gullstrand). Established
//! general relativity; a numerical approximation validated against exact oracles.

use scirust_relativity::adm::{AdmSettings, adm_constraints, adm_decomposition};
use scirust_relativity::{
    DeSitter, ExponentialScaleFactor, Flrw, Metric, PainleveGullstrand, Schwarzschild,
};
use std::f64::consts::FRAC_PI_2;

fn settings() -> AdmSettings {
    AdmSettings {
        time_step: 1.0e-3,
        spatial_step: 1.0e-3,
    }
}

/// O0: the ADM variables rebuild the 4-metric to `tolerance`.
// Explicit tensor-index loops read most clearly here (matching the crate's
// curvature and linearized modules).
#[allow(clippy::needless_range_loop)]
fn assert_reconstructs<B: Metric<4>>(background: &B, coordinates: &[f64; 4], tolerance: f64) {
    let decomposition = adm_decomposition(background, coordinates, &settings()).unwrap();
    let metric = background.components(coordinates);

    let mut shift_lower = [0.0_f64; 3];
    for i in 0..3
    {
        for j in 0..3
        {
            shift_lower[i] += decomposition.spatial_metric[i][j] * decomposition.shift[j];
        }
    }
    let mut shift_norm = 0.0;
    for i in 0..3
    {
        shift_norm += shift_lower[i] * decomposition.shift[i];
    }

    let g00 = -decomposition.lapse * decomposition.lapse + shift_norm;
    assert!(
        (g00 - metric[0][0]).abs() < tolerance,
        "g00: {g00} vs {}",
        metric[0][0]
    );
    for i in 0..3
    {
        assert!(
            (shift_lower[i] - metric[0][i + 1]).abs() < tolerance,
            "g0{}: {} vs {}",
            i + 1,
            shift_lower[i],
            metric[0][i + 1]
        );
        for j in 0..3
        {
            assert!(
                (decomposition.spatial_metric[i][j] - metric[i + 1][j + 1]).abs() < tolerance,
                "g{}{}",
                i + 1,
                j + 1
            );
        }
    }
}

#[test]
fn o0_reconstruction_on_every_background() {
    let schwarzschild = Schwarzschild::try_new(1.0).unwrap();
    let de_sitter = DeSitter::try_new(0.03).unwrap();
    let flrw = Flrw::new(ExponentialScaleFactor::try_new(0.5).unwrap());
    let painleve = PainleveGullstrand::try_new(1.0).unwrap();

    assert_reconstructs(&schwarzschild, &[0.0, 6.0, FRAC_PI_2, 0.0], 1.0e-10);
    assert_reconstructs(&de_sitter, &[0.0, 3.0, FRAC_PI_2, 0.0], 1.0e-10);
    assert_reconstructs(&flrw, &[0.0, 0.1, 0.2, 0.3], 1.0e-10);
    assert_reconstructs(&painleve, &[0.0, 4.0, FRAC_PI_2, 0.0], 1.0e-10);
}

#[test]
fn o1_o2_kinematics_match_analytic_values() {
    // Schwarzschild: static, time-symmetric — zero extrinsic curvature, scalar-flat slice.
    let schwarzschild = Schwarzschild::try_new(1.0).unwrap();
    let sch = adm_decomposition(&schwarzschild, &[0.0, 6.0, FRAC_PI_2, 0.0], &settings()).unwrap();
    assert!(
        sch.spatial_ricci_scalar.abs() < 1.0e-5,
        "R3 = {}",
        sch.spatial_ricci_scalar
    );
    assert!(
        sch.mean_curvature.abs() < 1.0e-6,
        "K = {}",
        sch.mean_curvature
    );
    assert!((sch.lapse - (1.0f64 - 2.0 / 6.0).sqrt()).abs() < 1.0e-12);

    // de Sitter: static, curved space — R^(3) = 2 Lambda, zero extrinsic curvature.
    let lambda = 0.03;
    let de_sitter = DeSitter::try_new(lambda).unwrap();
    let ds = adm_decomposition(&de_sitter, &[0.0, 3.0, FRAC_PI_2, 0.0], &settings()).unwrap();
    assert!(
        (ds.spatial_ricci_scalar - 2.0 * lambda).abs() < 1.0e-5,
        "R3 = {} vs {}",
        ds.spatial_ricci_scalar,
        2.0 * lambda
    );
    assert!(
        ds.mean_curvature.abs() < 1.0e-6,
        "K = {}",
        ds.mean_curvature
    );

    // FLRW: time-dependent, flat space — K = -3H, K_ij K^ij = 3 H^2.
    let hubble = 0.5;
    let flrw = Flrw::new(ExponentialScaleFactor::try_new(hubble).unwrap());
    let fl = adm_decomposition(&flrw, &[0.0, 0.1, 0.2, 0.3], &settings()).unwrap();
    assert!(
        fl.spatial_ricci_scalar.abs() < 1.0e-6,
        "R3 = {}",
        fl.spatial_ricci_scalar
    );
    assert!(
        (fl.mean_curvature - (-3.0 * hubble)).abs() < 1.0e-5,
        "K = {}",
        fl.mean_curvature
    );
    assert!(
        (fl.extrinsic_curvature_norm - 3.0 * hubble * hubble).abs() < 1.0e-5,
        "KK = {}",
        fl.extrinsic_curvature_norm
    );
}

#[test]
fn o3_o4_constraints_vanish_on_vacuum_solutions() {
    let cases: [(&str, f64); 4] = [
        ("schwarzschild", 0.0),
        ("de_sitter", 0.03),
        ("flrw", 0.75),
        ("painleve", 0.0),
    ];
    for (name, lambda) in cases
    {
        let (hamiltonian, momentum) = match name
        {
            "schwarzschild" =>
            {
                let bg = Schwarzschild::try_new(1.0).unwrap();
                let c =
                    adm_constraints(&bg, &[0.0, 6.0, FRAC_PI_2, 0.0], lambda, &settings()).unwrap();
                (c.hamiltonian, c.momentum)
            },
            "de_sitter" =>
            {
                let bg = DeSitter::try_new(lambda).unwrap();
                let c =
                    adm_constraints(&bg, &[0.0, 3.0, FRAC_PI_2, 0.0], lambda, &settings()).unwrap();
                (c.hamiltonian, c.momentum)
            },
            "flrw" =>
            {
                let bg = Flrw::new(ExponentialScaleFactor::try_new(0.5).unwrap());
                let c = adm_constraints(&bg, &[0.0, 0.1, 0.2, 0.3], lambda, &settings()).unwrap();
                (c.hamiltonian, c.momentum)
            },
            _ =>
            {
                let bg = PainleveGullstrand::try_new(1.0).unwrap();
                let c =
                    adm_constraints(&bg, &[0.0, 4.0, FRAC_PI_2, 0.0], lambda, &settings()).unwrap();
                (c.hamiltonian, c.momentum)
            },
        };
        let momentum_max = momentum.iter().fold(0.0_f64, |acc, &m| acc.max(m.abs()));
        println!("{name}: hamiltonian {hamiltonian:.3e}  momentum_max {momentum_max:.3e}");
        assert!(
            hamiltonian.abs() < 1.0e-4,
            "{name} hamiltonian = {hamiltonian}"
        );
        assert!(
            momentum_max < 1.0e-4,
            "{name} momentum_max = {momentum_max}"
        );
    }
}

#[test]
fn painleve_gullstrand_has_nonzero_shift_and_curvature() {
    let painleve = PainleveGullstrand::try_new(1.0).unwrap();
    let decomposition =
        adm_decomposition(&painleve, &[0.0, 4.0, FRAC_PI_2, 0.0], &settings()).unwrap();
    // Unit lapse, radial shift sqrt(2M/r), flat slice, nonzero extrinsic curvature.
    assert!(
        (decomposition.lapse - 1.0).abs() < 1.0e-9,
        "N = {}",
        decomposition.lapse
    );
    assert!(
        (decomposition.shift[0] - (0.5f64).sqrt()).abs() < 1.0e-9,
        "N^r = {}",
        decomposition.shift[0]
    );
    assert!(decomposition.spatial_ricci_scalar.abs() < 1.0e-5);
    assert!(
        decomposition.extrinsic_curvature_norm > 1.0e-3,
        "K_ij K^ij should be nonzero"
    );
}

#[test]
fn decomposition_is_deterministic() {
    let flrw = Flrw::new(ExponentialScaleFactor::try_new(0.5).unwrap());
    let first = adm_decomposition(&flrw, &[0.0, 0.1, 0.2, 0.3], &settings()).unwrap();
    let second = adm_decomposition(&flrw, &[0.0, 0.1, 0.2, 0.3], &settings()).unwrap();
    assert_eq!(first, second);
}

#[test]
fn rejects_invalid_requests() {
    use scirust_relativity::adm::AdmError;
    let flrw = Flrw::new(ExponentialScaleFactor::try_new(0.5).unwrap());

    assert!(matches!(
        adm_decomposition(&flrw, &[f64::NAN, 0.1, 0.2, 0.3], &settings()),
        Err(AdmError::NonFiniteCoordinate(0))
    ));
    assert!(matches!(
        adm_decomposition(
            &flrw,
            &[0.0, 0.1, 0.2, 0.3],
            &AdmSettings {
                time_step: 0.0,
                spatial_step: 1.0e-3
            }
        ),
        Err(AdmError::InvalidStep(_))
    ));
    assert!(matches!(
        adm_decomposition(
            &flrw,
            &[0.0, 0.1, 0.2, 0.3],
            &AdmSettings {
                time_step: 1.0e-3,
                spatial_step: -1.0
            }
        ),
        Err(AdmError::InvalidStep(_))
    ));
    assert!(matches!(
        adm_constraints(&flrw, &[0.0, 0.1, 0.2, 0.3], f64::INFINITY, &settings()),
        Err(AdmError::InvalidCosmologicalConstant(_))
    ));
}
