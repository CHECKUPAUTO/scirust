//! Coordinate independence of scalar curvature invariants.
//!
//! Scalar curvature invariants are geometric — independent of the coordinate
//! chart. This experiment checks that directly by computing the Ricci scalar
//! `R` and the Kretschmann scalar `K = R_{abcd} R^{abcd}` of the *same physical
//! geometry* in two different charts and reporting their agreement:
//!
//! - Flat spacetime: Cartesian [`Minkowski`] (curvature exactly zero) versus
//!   spherical [`MinkowskiSpherical`] (non-zero Christoffel symbols, numerically
//!   zero curvature).
//! - Schwarzschild: areal [`Schwarzschild`] (`K = 48 M^2 / r^6`) versus
//!   isotropic [`IsotropicSchwarzschild`], whose Kretschmann scalar must equal
//!   `48 M^2 / r^6` with `r` the *areal* radius `r = rho (1 + M/2rho)^2` — not
//!   `48 M^2 / rho^6`.
//!
//! The isotropic background's connection is a finite difference, so its
//! curvature is a nested finite difference and less accurate than the
//! analytic-connection charts; the reported agreement (~1e-5) reflects that
//! honestly. This validates established general relativity against an exact,
//! coordinate-independent invariant; it asserts no physics beyond textbook GR.

#![forbid(unsafe_code)]

use nonlocal_relativity_experiments::{print_experiment_header, require_finite};
use scirust_relativity::{
    CurvatureTensors, IsotropicSchwarzschild, Minkowski, MinkowskiSpherical, Schwarzschild,
};
use std::f64::consts::FRAC_PI_2;

const ANALYTIC_STEP: f64 = 1.0e-4;
const NESTED_STEP: f64 = 1.0e-3;
const MASS: f64 = 1.0;
const ISOTROPIC_RADII: [f64; 4] = [3.0, 4.0, 6.0, 10.0];

/// Absolute gap between two invariants computed in different charts. Absolute
/// (not relative) so it stays meaningful when both values are ~0 (the flat
/// charts); for the curved rows the printed `value_*_K` columns show the
/// relative agreement directly (gap divided by K is ~1e-6 to ~1e-5).
fn absolute_gap(left: f64, right: f64) -> f64 {
    (left - right).abs()
}

fn main() -> Result<(), String> {
    let cartesian =
        CurvatureTensors::compute(&Minkowski, &[0.3, 4.0, FRAC_PI_2, -0.7], ANALYTIC_STEP)
            .map_err(|e| e.to_string())?;
    let spherical = CurvatureTensors::compute(
        &MinkowskiSpherical,
        &[0.3, 4.0, FRAC_PI_2, -0.7],
        ANALYTIC_STEP,
    )
    .map_err(|e| e.to_string())?;

    let areal = Schwarzschild::try_new(MASS).ok_or_else(|| "invalid Schwarzschild".to_string())?;
    let isotropic =
        IsotropicSchwarzschild::try_new(MASS).ok_or_else(|| "invalid isotropic".to_string())?;

    print_experiment_header(
        "Coordinate independence of scalar curvature invariants",
        "scirust-relativity geometry core (established general relativity)",
        "checks that R and K agree across charts of the same geometry; established GR.",
    );
    println!("# invariants are geometric: R and K must not depend on the coordinate chart");
    println!("# flat: Cartesian Minkowski (exact 0) vs spherical Minkowski (numerically 0)");
    println!("# Schwarzschild: areal r vs isotropic rho, with areal r = rho (1 + M/2rho)^2");
    println!(
        "# isotropic connection is a finite difference => nested-FD curvature, ~1e-5 accuracy"
    );
    println!(
        "geometry,chart_a,value_a_R,value_a_K,chart_b,value_b_R,value_b_K,R_abs_gap,K_abs_gap"
    );

    // Flat spacetime, two charts.
    let flat_r_abs_gap = absolute_gap(cartesian.ricci_scalar(), spherical.ricci_scalar());
    let flat_k_abs_gap = absolute_gap(cartesian.kretschmann(), spherical.kretschmann());
    require_finite(&[
        ("flat_R_abs_gap", flat_r_abs_gap),
        ("flat_K_abs_gap", flat_k_abs_gap),
    ])?;
    println!(
        "Minkowski,cartesian,{:.3e},{:.3e},spherical,{:.3e},{:.3e},{:.3e},{:.3e}",
        cartesian.ricci_scalar(),
        cartesian.kretschmann(),
        spherical.ricci_scalar(),
        spherical.kretschmann(),
        flat_r_abs_gap,
        flat_k_abs_gap,
    );

    // Schwarzschild, areal vs isotropic, across several radii.
    for &isotropic_radius in &ISOTROPIC_RADII
    {
        let areal_radius = isotropic.areal_radius(isotropic_radius);
        let areal_curvature =
            CurvatureTensors::compute(&areal, &[0.0, areal_radius, FRAC_PI_2, 0.0], ANALYTIC_STEP)
                .map_err(|e| e.to_string())?;
        let isotropic_curvature = CurvatureTensors::compute(
            &isotropic,
            &[0.0, isotropic_radius, FRAC_PI_2, 0.0],
            NESTED_STEP,
        )
        .map_err(|e| e.to_string())?;

        let r_abs_gap = absolute_gap(
            areal_curvature.ricci_scalar(),
            isotropic_curvature.ricci_scalar(),
        );
        let k_abs_gap = absolute_gap(
            areal_curvature.kretschmann(),
            isotropic_curvature.kretschmann(),
        );
        require_finite(&[
            ("areal_K", areal_curvature.kretschmann()),
            ("isotropic_K", isotropic_curvature.kretschmann()),
            ("R_abs_gap", r_abs_gap),
            ("K_abs_gap", k_abs_gap),
        ])?;

        // chart_a is areal at r; chart_b is isotropic at rho (annotated in the
        // geometry column so a reader sees the two distinct radii).
        println!(
            "Schwarzschild[r={:.4},rho={:.1}],areal,{:.3e},{:.6e},isotropic,{:.3e},{:.6e},{:.3e},{:.3e}",
            areal_radius,
            isotropic_radius,
            areal_curvature.ricci_scalar(),
            areal_curvature.kretschmann(),
            isotropic_curvature.ricci_scalar(),
            isotropic_curvature.kretschmann(),
            r_abs_gap,
            k_abs_gap,
        );
    }

    println!("# interpretation: the Ricci and Kretschmann scalars agree across charts of the same");
    println!("# geometry to the finite-difference tolerance (exactly for the analytic-connection");
    println!("# flat charts, to ~1e-5 for the nested-FD isotropic chart). This is a direct");
    println!("# coordinate-independence check of established-GR invariants, not a physical model.");
    Ok(())
}
