//! Oracles for the Einstein-Hilbert action variation (Layer 2, third slice; see
//! `docs/LAYER_2_ACTION_VARIATION.md`).
//!
//! O1 metric-only Ricci scalar recovers `4 Lambda` (de Sitter) and `0`
//! (Schwarzschild); O2 vacuum stationarity — the variation vanishes for
//! Schwarzschild and Lambda-matched de Sitter; O3 a mismatched action `Lambda`
//! reproduces the known nonzero prediction from the Einstein tensor; O4 the
//! residual falls under grid refinement. Established general relativity;
//! numerical approximation validated against exact and independently-computed
//! oracles.

use scirust_relativity::action::{
    ActionDomain, ActionError, ActionPerturbation, ActionSettings,
    einstein_hilbert_action_variation,
};
use scirust_relativity::{DeSitter, Schwarzschild, ricci_scalar_from_metric};
use std::f64::consts::FRAC_PI_2;

const LAMBDA: f64 = 0.03;

fn de_sitter_domain(grid: usize) -> ActionDomain {
    ActionDomain {
        radial_range: (2.0, 4.0),
        polar_range: (FRAC_PI_2 - 1.0, FRAC_PI_2 + 1.0),
        grid,
    }
}

fn settings(cosmological_constant: f64) -> ActionSettings {
    ActionSettings {
        amplitude: 1.0e-3,
        connection_step: 1.0e-3,
        metric_step: 1.0e-3,
        cosmological_constant,
    }
}

fn centered_perturbation(component: (usize, usize)) -> ActionPerturbation {
    ActionPerturbation {
        component,
        center: (3.0, FRAC_PI_2),
        half_widths: (1.0, 1.0),
    }
}

#[test]
fn o1_metric_only_ricci_scalar_recovers_oracles() {
    let de_sitter = DeSitter::try_new(LAMBDA).expect("valid de Sitter");
    let scalar =
        ricci_scalar_from_metric(&de_sitter, &[0.0, 3.0, FRAC_PI_2, 0.0], 1.0e-3, 1.0e-3).unwrap();
    assert!(
        (scalar - 4.0 * LAMBDA).abs() < 1.0e-5,
        "de Sitter R = {scalar}, expected {}",
        4.0 * LAMBDA
    );

    let schwarzschild = Schwarzschild::try_new(1.0).expect("valid Schwarzschild");
    let vacuum =
        ricci_scalar_from_metric(&schwarzschild, &[0.0, 6.0, FRAC_PI_2, 0.0], 1.0e-3, 1.0e-3)
            .unwrap();
    assert!(
        vacuum.abs() < 1.0e-5,
        "Schwarzschild R = {vacuum}, expected 0"
    );
}

#[test]
fn o2_de_sitter_matched_lambda_is_stationary_across_components() {
    // The variation vanishes for every perturbed component of a solution. At
    // n = 41 the slowest-converging (timelike) bump is ~2e-3; the tighter
    // convergence toward zero is shown by `o4_residual_falls_under_grid_refinement`
    // and the `action_variation` experiment (which reaches ~4e-4 at n = 61).
    let de_sitter = DeSitter::try_new(LAMBDA).expect("valid de Sitter");
    for component in [(0, 0), (1, 1), (2, 2)]
    {
        let variation = einstein_hilbert_action_variation(
            &de_sitter,
            &centered_perturbation(component),
            &de_sitter_domain(41),
            &settings(LAMBDA),
        )
        .expect("valid variation");
        assert!(
            variation.residual < 5.0e-3,
            "component {component:?}: residual {} (numeric {}, predicted {})",
            variation.residual,
            variation.numeric,
            variation.predicted
        );
    }
}

#[test]
fn o2_schwarzschild_vacuum_is_stationary() {
    let schwarzschild = Schwarzschild::try_new(1.0).expect("valid Schwarzschild");
    let perturbation = ActionPerturbation {
        component: (1, 1),
        center: (6.0, FRAC_PI_2),
        half_widths: (2.0, 1.0),
    };
    let domain = ActionDomain {
        radial_range: (4.0, 8.0),
        polar_range: (FRAC_PI_2 - 1.0, FRAC_PI_2 + 1.0),
        grid: 41,
    };
    let variation =
        einstein_hilbert_action_variation(&schwarzschild, &perturbation, &domain, &settings(0.0))
            .expect("valid variation");
    assert!(
        variation.residual < 1.0e-3,
        "Schwarzschild vacuum residual {} (numeric {}, predicted {})",
        variation.residual,
        variation.numeric,
        variation.predicted
    );
}

#[test]
fn o3_mismatched_lambda_reproduces_the_nonzero_einstein_prediction() {
    let de_sitter = DeSitter::try_new(LAMBDA).expect("valid de Sitter");
    // Action Lambda = 0 on a de Sitter background: E_{mu nu} = -Lambda g_{mu nu} != 0.
    let variation = einstein_hilbert_action_variation(
        &de_sitter,
        &centered_perturbation((1, 1)),
        &de_sitter_domain(41),
        &settings(0.0),
    )
    .expect("valid variation");
    assert!(
        variation.predicted.abs() > 5.0e-2,
        "prediction {} should be genuinely nonzero",
        variation.predicted
    );
    let relative = variation.residual / variation.predicted.abs();
    assert!(
        relative < 5.0e-3,
        "numeric {} vs predicted {} (relative {relative})",
        variation.numeric,
        variation.predicted
    );
}

#[test]
fn o4_residual_falls_under_grid_refinement() {
    let de_sitter = DeSitter::try_new(LAMBDA).expect("valid de Sitter");

    // Matched (stationary): the residual toward zero shrinks with resolution.
    let coarse = einstein_hilbert_action_variation(
        &de_sitter,
        &centered_perturbation((1, 1)),
        &de_sitter_domain(21),
        &settings(LAMBDA),
    )
    .unwrap();
    let fine = einstein_hilbert_action_variation(
        &de_sitter,
        &centered_perturbation((1, 1)),
        &de_sitter_domain(41),
        &settings(LAMBDA),
    )
    .unwrap();
    assert!(
        fine.residual < coarse.residual,
        "matched residual did not fall: coarse {} fine {}",
        coarse.residual,
        fine.residual
    );

    // Mismatched (nonzero target): the relative error shrinks with resolution.
    let coarse_mismatch = einstein_hilbert_action_variation(
        &de_sitter,
        &centered_perturbation((1, 1)),
        &de_sitter_domain(21),
        &settings(0.0),
    )
    .unwrap();
    let fine_mismatch = einstein_hilbert_action_variation(
        &de_sitter,
        &centered_perturbation((1, 1)),
        &de_sitter_domain(41),
        &settings(0.0),
    )
    .unwrap();
    let coarse_rel = coarse_mismatch.residual / coarse_mismatch.predicted.abs();
    let fine_rel = fine_mismatch.residual / fine_mismatch.predicted.abs();
    assert!(
        fine_rel < coarse_rel,
        "mismatch relative error did not fall: coarse {coarse_rel} fine {fine_rel}"
    );
}

#[test]
fn extraction_is_deterministic() {
    let de_sitter = DeSitter::try_new(LAMBDA).expect("valid de Sitter");
    let first = einstein_hilbert_action_variation(
        &de_sitter,
        &centered_perturbation((1, 1)),
        &de_sitter_domain(31),
        &settings(LAMBDA),
    )
    .unwrap();
    let second = einstein_hilbert_action_variation(
        &de_sitter,
        &centered_perturbation((1, 1)),
        &de_sitter_domain(31),
        &settings(LAMBDA),
    )
    .unwrap();
    assert_eq!(first, second);
}

#[test]
fn rejects_invalid_requests() {
    let de_sitter = DeSitter::try_new(LAMBDA).expect("valid de Sitter");
    let base_perturbation = centered_perturbation((1, 1));
    let base_domain = de_sitter_domain(31);
    let base_settings = settings(LAMBDA);

    let vary = |perturbation: ActionPerturbation, domain: ActionDomain, s: ActionSettings| {
        einstein_hilbert_action_variation(&de_sitter, &perturbation, &domain, &s)
    };

    // Even grid.
    assert!(matches!(
        vary(
            base_perturbation,
            ActionDomain {
                grid: 30,
                ..base_domain
            },
            base_settings
        ),
        Err(ActionError::InvalidGridResolution(30))
    ));
    // Grid below the Simpson minimum.
    assert!(matches!(
        vary(
            base_perturbation,
            ActionDomain {
                grid: 3,
                ..base_domain
            },
            base_settings
        ),
        Err(ActionError::InvalidGridResolution(3))
    ));
    // Non-positive radius.
    assert!(matches!(
        vary(
            base_perturbation,
            ActionDomain {
                radial_range: (-1.0, 4.0),
                ..base_domain
            },
            base_settings
        ),
        Err(ActionError::InvalidRadialRange { .. })
    ));
    // Polar range past pi.
    assert!(matches!(
        vary(
            base_perturbation,
            ActionDomain {
                polar_range: (0.1, 4.0),
                ..base_domain
            },
            base_settings
        ),
        Err(ActionError::InvalidPolarRange { .. })
    ));
    // Component index out of range.
    assert!(matches!(
        vary(
            ActionPerturbation {
                component: (4, 0),
                ..base_perturbation
            },
            base_domain,
            base_settings
        ),
        Err(ActionError::InvalidComponent { row: 4, col: 0 })
    ));
    // Non-positive width.
    assert!(matches!(
        vary(
            ActionPerturbation {
                half_widths: (-1.0, 1.0),
                ..base_perturbation
            },
            base_domain,
            base_settings
        ),
        Err(ActionError::InvalidPerturbationWidth(_))
    ));
    // Support spilling out of the domain.
    assert!(matches!(
        vary(
            ActionPerturbation {
                half_widths: (1.5, 1.0),
                ..base_perturbation
            },
            base_domain,
            base_settings
        ),
        Err(ActionError::SupportOutsideDomain)
    ));
    // Non-positive amplitude.
    assert!(matches!(
        vary(
            base_perturbation,
            base_domain,
            ActionSettings {
                amplitude: 0.0,
                ..base_settings
            }
        ),
        Err(ActionError::InvalidAmplitude(_))
    ));
    // Non-positive step.
    assert!(matches!(
        vary(
            base_perturbation,
            base_domain,
            ActionSettings {
                connection_step: 0.0,
                ..base_settings
            }
        ),
        Err(ActionError::InvalidStep(_))
    ));
    // Non-finite cosmological constant.
    assert!(matches!(
        vary(
            base_perturbation,
            base_domain,
            ActionSettings {
                cosmological_constant: f64::NAN,
                ..base_settings
            }
        ),
        Err(ActionError::InvalidCosmologicalConstant(_))
    ));
}
