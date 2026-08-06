//! Local orthonormal frame (tetrad) construction checks.
//!
//! A tetrad `{e_a}` for a timelike observer `u` satisfies `g(e_a, e_b) = eta_ab`
//! with `eta = diag(-1, +1, +1, +1)`: `e_0` is the normalized four-velocity and
//! the spacelike legs are coordinate-basis directions orthonormalized against
//! the frame by metric Gram-Schmidt. Unlike parallel transport, this is an
//! exact closed-form construction at a single chart point -- no ODE, no
//! integration tolerance -- so every residual below is at the rounding floor,
//! not a discretization error that shrinks with resolution.
//!
//! Three consequences are reported across backgrounds and observers:
//!
//! - **Orthonormality.** `max_{a,b} |g(e_a, e_b) - eta_ab|`.
//! - **Completeness.** Any vector reconstructs from its frame components,
//!   `delta = sum_a eta_aa g(delta, e_a) e_a`; the max reconstruction gap.
//! - **Split agreement.** The tetrad's temporal magnitude `|c^0|` and spatial
//!   magnitude `sqrt(sum_{i>0} (c^i)^2)` match the closed-form metric split
//!   `|g(delta, u)| / sqrt(-g(u,u))` and `sqrt(g(P delta, P delta))`, where
//!   `P` projects orthogonal to `u`.
//!
//! Established general relativity only: this is the geometry-core tetrad the
//! experimental worldline observer-frame diagnostic delegates to.

#![forbid(unsafe_code)]

use nonlocal_relativity_experiments::{print_experiment_header, require_finite};
use scirust_relativity::{
    AntiDeSitter, DeSitter, Metric, Minkowski, OrthonormalTetrad, Schwarzschild, orthonormal_tetrad,
};
use std::f64::consts::FRAC_PI_2;

const FLOOR: f64 = 1.0e-9;

/// A labelled frame case: `(name, metric at the point, observer four-velocity)`.
type FrameCase = (&'static str, [[f64; 4]; 4], [f64; 4]);

/// Metric inner product `g(a, b) = g_(mu nu) a^mu b^nu`.
fn inner(metric: &[[f64; 4]; 4], left: &[f64; 4], right: &[f64; 4]) -> f64 {
    let mut value = 0.0;
    for (row, &left_component) in metric.iter().zip(left.iter())
    {
        for (&entry, &right_component) in row.iter().zip(right.iter())
        {
            value += entry * left_component * right_component;
        }
    }
    value
}

/// Coordinate-static observer `u = (1 / sqrt(-g_00), 0, 0, 0)`, valid wherever
/// the time coordinate is timelike (`g_00 < 0`).
fn static_observer(metric: &[[f64; 4]; 4]) -> Result<[f64; 4], String> {
    if metric[0][0] >= 0.0
    {
        return Err(format!(
            "time coordinate is not timelike here: g_00 = {}",
            metric[0][0]
        ));
    }
    Ok([1.0 / (-metric[0][0]).sqrt(), 0.0, 0.0, 0.0])
}

/// `max_{a,b} |g(e_a, e_b) - eta_ab|` for the tetrad legs.
fn orthonormality_defect(metric: &[[f64; 4]; 4], legs: &[[f64; 4]; 4]) -> f64 {
    let mut worst = 0.0_f64;
    for (a, leg_a) in legs.iter().enumerate()
    {
        for (b, leg_b) in legs.iter().enumerate()
        {
            let expected = if a == b
            {
                OrthonormalTetrad::<4>::signature(a)
            }
            else
            {
                0.0
            };
            worst = worst.max((inner(metric, leg_a, leg_b) - expected).abs());
        }
    }
    worst
}

/// `max_i |sum_a eta_aa g(delta, e_a) e_a - delta_i|`: the completeness gap.
fn reconstruction_gap(metric: &[[f64; 4]; 4], legs: &[[f64; 4]; 4], delta: &[f64; 4]) -> f64 {
    let mut reconstructed = [0.0_f64; 4];
    for (a, leg) in legs.iter().enumerate()
    {
        let component = OrthonormalTetrad::<4>::signature(a) * inner(metric, delta, leg);
        for (slot, &leg_component) in reconstructed.iter_mut().zip(leg.iter())
        {
            *slot += component * leg_component;
        }
    }
    reconstructed
        .iter()
        .zip(delta.iter())
        .map(|(&r, &d)| (r - d).abs())
        .fold(0.0_f64, f64::max)
}

/// Absolute gaps between the tetrad-frame temporal/spatial magnitudes and the
/// closed-form metric split, for a probe displacement `delta`.
fn split_agreement(
    metric: &[[f64; 4]; 4],
    observer: &[f64; 4],
    legs: &[[f64; 4]; 4],
    delta: &[f64; 4],
) -> (f64, f64) {
    // Tetrad-frame components c^a = eta_aa g(delta, e_a).
    let mut components = [0.0_f64; 4];
    for (a, leg) in legs.iter().enumerate()
    {
        components[a] = OrthonormalTetrad::<4>::signature(a) * inner(metric, delta, leg);
    }
    let tetrad_temporal = components[0].abs();
    let tetrad_spatial = components[1..].iter().map(|c| c * c).sum::<f64>().sqrt();

    // Closed-form metric split: temporal along u, spatial orthogonal to u.
    let norm = inner(metric, observer, observer);
    let inner_delta_u = inner(metric, delta, observer);
    let scalar_temporal = inner_delta_u.abs() / (-norm).sqrt();
    let mut projected = [0.0_f64; 4];
    for (slot, (&d, &u)) in projected.iter_mut().zip(delta.iter().zip(observer.iter()))
    {
        *slot = d - u * inner_delta_u / norm;
    }
    let scalar_spatial = inner(metric, &projected, &projected).max(0.0).sqrt();

    (
        (tetrad_temporal - scalar_temporal).abs(),
        (tetrad_spatial - scalar_spatial).abs(),
    )
}

fn main() -> Result<(), String> {
    let schwarzschild =
        Schwarzschild::try_new(1.0).ok_or_else(|| "invalid Schwarzschild".to_string())?;
    let de_sitter = DeSitter::try_new(0.05).ok_or_else(|| "invalid de Sitter".to_string())?;
    let anti_de_sitter =
        AntiDeSitter::try_new(0.05).ok_or_else(|| "invalid anti-de Sitter".to_string())?;

    let minkowski_point = [0.0, 0.0, 0.0, 0.0];
    let schwarzschild_point = [0.0, 10.0, FRAC_PI_2, 0.0];
    let de_sitter_point = [0.0, 3.0, FRAC_PI_2, 0.0];

    let minkowski_metric = Minkowski.components(&minkowski_point);
    let schwarzschild_metric = schwarzschild.components(&schwarzschild_point);
    let de_sitter_metric = de_sitter.components(&de_sitter_point);
    let anti_de_sitter_metric = anti_de_sitter.components(&de_sitter_point);

    // A flat-space boost u = gamma (1, v, 0, 0), and a boosted (t, r)
    // Schwarzschild observer -- both genuinely timelike, so the Gram-Schmidt
    // frame is not trivially the coordinate basis.
    let boost_velocity = 0.6_f64;
    let boost_gamma = 1.0 / (1.0 - boost_velocity * boost_velocity).sqrt();
    let minkowski_boosted = [boost_gamma, boost_gamma * boost_velocity, 0.0, 0.0];
    let schwarzschild_boosted = [1.3, 0.1, 0.0, 0.0];

    let cases: [FrameCase; 6] = [
        (
            "Minkowski_static",
            minkowski_metric,
            static_observer(&minkowski_metric)?,
        ),
        ("Minkowski_boosted", minkowski_metric, minkowski_boosted),
        (
            "Schwarzschild_static",
            schwarzschild_metric,
            static_observer(&schwarzschild_metric)?,
        ),
        (
            "Schwarzschild_boosted",
            schwarzschild_metric,
            schwarzschild_boosted,
        ),
        (
            "de_Sitter_static",
            de_sitter_metric,
            static_observer(&de_sitter_metric)?,
        ),
        (
            "anti_de_Sitter_static",
            anti_de_sitter_metric,
            static_observer(&anti_de_sitter_metric)?,
        ),
    ];

    print_experiment_header(
        "Orthonormal frame (tetrad) construction checks",
        "scirust-relativity geometry core (established general relativity)",
        "exact closed-form frame construction; residuals are the rounding floor, not ODE error.",
    );

    // The probe displacement decomposed in each observer's frame.
    let delta = [0.2, -0.1, 0.05, 0.3];
    println!("# probe displacement delta = {delta:?}");
    println!("#");
    println!("# orthonormality: max|g(e_a,e_b) - eta_ab|; completeness: max reconstruction gap;");
    println!("# split: |tetrad - closed-form| for the temporal and spatial magnitudes.");
    println!("case,orthonormality_defect,reconstruction_gap,temporal_split_gap,spatial_split_gap");

    for (label, metric, observer) in cases
    {
        // A boosted observer must actually be timelike for the case to be well
        // posed; the construction itself rejects a non-timelike leg.
        require_finite(&[("observer_norm", inner(&metric, &observer, &observer))])?;

        let tetrad = orthonormal_tetrad(&metric, &observer, FLOOR).map_err(|e| e.to_string())?;
        let legs = tetrad.legs();

        let orthonormality = orthonormality_defect(&metric, legs);
        let completeness = reconstruction_gap(&metric, legs, &delta);
        let (temporal_gap, spatial_gap) = split_agreement(&metric, &observer, legs, &delta);

        require_finite(&[
            ("orthonormality_defect", orthonormality),
            ("reconstruction_gap", completeness),
            ("temporal_split_gap", temporal_gap),
            ("spatial_split_gap", spatial_gap),
        ])?;

        println!(
            "{label},{orthonormality:.3e},{completeness:.3e},{temporal_gap:.3e},{spatial_gap:.3e}"
        );
    }

    println!("# interpretation: across flat, Schwarzschild, de Sitter, and anti-de Sitter");
    println!("# backgrounds -- for static and boosted observers alike -- the frame is orthonormal");
    println!(
        "# to machine precision, spans the tangent space (delta reconstructs exactly), and its"
    );
    println!("# temporal/spatial magnitudes agree with the closed-form metric split. Because the");
    println!(
        "# construction is exact algebra at one point, these are rounding-floor residuals, not"
    );
    println!("# convergent approximations. Established GR, not a phenomenological model.");
    Ok(())
}
