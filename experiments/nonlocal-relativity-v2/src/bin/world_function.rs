//! Synge world-function checks: flat exactness and curved convention-free
//! identities.
//!
//! Synge's world function `sigma(x', x)` is one-half the signed squared geodesic
//! distance between a base point `x'` and a field point `x`; its gradients
//! `sigma^{mu'} = -log_{x'}(x)` and `sigma^mu = -log_x(x')` are the tangents to
//! the connecting geodesic at the two ends. Everything here is built on the
//! geodesic logarithm map, so there is no independent distance formula to check
//! against on a general background; instead the experiment reports the identities
//! that any correct world function must satisfy:
//!
//! - **Flat exactness.** In Minkowski `sigma = (1/2) eta(Δ, Δ)`, `sigma^mu = Δ`,
//!   and `sigma^{mu'} = -Δ` with `Δ = x - x'`, to rounding.
//! - **Symmetry.** `sigma(x', x) = sigma(x, x')` — computed from independent
//!   shootings at each end.
//! - **Fundamental identity.** `2 sigma = g(x) sigma^mu sigma^mu`, the
//!   field-point gradient measured with an independent shooting from `x`.
//! - **Gradient round trip.** `exp_{x'}(-sigma^{mu'}) = x` and
//!   `exp_x(-sigma^mu) = x'`.
//!
//! Established general relativity only; no new geodesic solver is introduced.

#![forbid(unsafe_code)]

use nonlocal_relativity_experiments::{print_experiment_header, require_finite};
use scirust_relativity::{
    AntiDeSitter, Connection, DeSitter, Metric, Minkowski, Schwarzschild, WorldFunction,
    WorldFunctionSettings, geodesic_exponential, metric_norm, world_function,
    world_function_with_gradients,
};
use std::f64::consts::FRAC_PI_2;

fn max_abs_difference(left: &[f64; 4], right: &[f64; 4]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

/// The four convention-free identity residuals for a curved background.
fn curved_identity_gaps<B>(
    background: &B,
    base: [f64; 4],
    field: [f64; 4],
) -> Result<[f64; 4], String>
where
    B: Metric<4> + Connection<4> + Copy,
{
    let settings = WorldFunctionSettings::default();
    let WorldFunction {
        sigma,
        gradient_base,
        gradient_field,
    } = world_function_with_gradients(background, &base, &field, &settings)
        .map_err(|e| e.to_string())?;

    let sigma_reversed =
        world_function(background, &field, &base, &settings).map_err(|e| e.to_string())?;
    let symmetry_gap = (sigma - sigma_reversed).abs();

    let metric_field = background.components(&field);
    let fundamental_gap = (2.0 * sigma - metric_norm(&metric_field, &gradient_field)).abs();

    let recovered_field =
        geodesic_exponential(background, &base, &gradient_base.map(|c| -c), settings.step)
            .map_err(|e| e.to_string())?;
    let recovered_base = geodesic_exponential(
        background,
        &field,
        &gradient_field.map(|c| -c),
        settings.step,
    )
    .map_err(|e| e.to_string())?;

    Ok([
        symmetry_gap,
        fundamental_gap,
        max_abs_difference(&recovered_field, &field),
        max_abs_difference(&recovered_base, &base),
    ])
}

fn main() -> Result<(), String> {
    let schwarzschild =
        Schwarzschild::try_new(1.0).ok_or_else(|| "invalid Schwarzschild".to_string())?;
    let de_sitter = DeSitter::try_new(0.05).ok_or_else(|| "invalid de Sitter".to_string())?;
    let anti_de_sitter =
        AntiDeSitter::try_new(0.05).ok_or_else(|| "invalid anti-de Sitter".to_string())?;
    let settings = WorldFunctionSettings::default();

    print_experiment_header(
        "Synge world-function checks",
        "scirust-relativity geometry core (established general relativity)",
        "world function and gradients from the geodesic log map; convention-free identities.",
    );

    // Part A: flat exactness, both spacelike and timelike separations.
    println!("# Part A: Minkowski -- sigma vs (1/2) eta(Δ,Δ), and gradient errors");
    println!("separation_kind,sigma,half_eta_norm,sigma_error,grad_field_error,grad_base_error");
    let base = [0.0, 1.0, 2.0, 0.5];
    for (kind, field) in [
        ("spacelike", [0.0, 4.0, 6.0, 0.5]),
        ("timelike", [3.0, 1.2, 2.1, 0.6]),
    ]
    {
        let separation = [
            field[0] - base[0],
            field[1] - base[1],
            field[2] - base[2],
            field[3] - base[3],
        ];
        let minkowski_metric = Minkowski.components(&base);
        let half_eta_norm = 0.5 * metric_norm(&minkowski_metric, &separation);

        let result = world_function_with_gradients(&Minkowski, &base, &field, &settings)
            .map_err(|e| e.to_string())?;
        let sigma_error = (result.sigma - half_eta_norm).abs();
        let grad_field_error = max_abs_difference(&result.gradient_field, &separation);
        let grad_base_error = max_abs_difference(&result.gradient_base, &separation.map(|c| -c));

        require_finite(&[
            ("sigma", result.sigma),
            ("sigma_error", sigma_error),
            ("grad_field_error", grad_field_error),
            ("grad_base_error", grad_base_error),
        ])?;
        println!(
            "{kind},{:.6e},{half_eta_norm:.6e},{sigma_error:.3e},{grad_field_error:.3e},{grad_base_error:.3e}",
            result.sigma
        );
    }

    // Part B: curved convention-free identities.
    println!("#");
    println!("# Part B: curved backgrounds -- convention-free identity residuals");
    println!(
        "background,symmetry_gap,fundamental_identity_gap,roundtrip_field_gap,roundtrip_base_gap"
    );
    for &(label, gaps) in &[
        (
            "Schwarzschild",
            curved_identity_gaps(
                &schwarzschild,
                [0.0, 12.0, FRAC_PI_2, 0.0],
                [0.15, 12.4, FRAC_PI_2 + 0.03, 0.05],
            )?,
        ),
        (
            "de_Sitter",
            curved_identity_gaps(
                &de_sitter,
                [0.0, 3.0, FRAC_PI_2, 0.0],
                [0.1, 3.3, FRAC_PI_2 + 0.04, 0.03],
            )?,
        ),
        (
            "anti_de_Sitter",
            curved_identity_gaps(
                &anti_de_sitter,
                [0.0, 3.0, FRAC_PI_2, 0.0],
                [0.1, 3.3, FRAC_PI_2 + 0.04, 0.03],
            )?,
        ),
    ]
    {
        require_finite(&[
            ("symmetry_gap", gaps[0]),
            ("fundamental_identity_gap", gaps[1]),
            ("roundtrip_field_gap", gaps[2]),
            ("roundtrip_base_gap", gaps[3]),
        ])?;
        println!(
            "{label},{:.3e},{:.3e},{:.3e},{:.3e}",
            gaps[0], gaps[1], gaps[2], gaps[3]
        );
    }

    // Part C: coincidence -- sigma vanishes quadratically in the separation.
    println!("#");
    println!(
        "# Part C: Schwarzschild -- sigma ~ (1/2)|Δ|^2 as the field point approaches the base"
    );
    println!("scale,sigma,half_metric_norm,ratio");
    let coincidence_base = [0.0, 10.0, FRAC_PI_2, 0.0];
    let direction = [0.05, 0.2, 0.02, 0.03];
    for scale in [1.0, 0.5, 0.25, 0.125]
    {
        let field = [
            coincidence_base[0] + scale * direction[0],
            coincidence_base[1] + scale * direction[1],
            coincidence_base[2] + scale * direction[2],
            coincidence_base[3] + scale * direction[3],
        ];
        let sigma = world_function(&schwarzschild, &coincidence_base, &field, &settings)
            .map_err(|e| e.to_string())?;
        let scaled_direction = direction.map(|c| scale * c);
        let half_metric_norm = 0.5
            * metric_norm(
                &schwarzschild.components(&coincidence_base),
                &scaled_direction,
            );
        let ratio = sigma / half_metric_norm;
        require_finite(&[("sigma", sigma), ("ratio", ratio)])?;
        println!("{scale:.3},{sigma:.6e},{half_metric_norm:.6e},{ratio:.6}");
    }

    println!("# interpretation: in flat spacetime the world function and its gradients are exact;");
    println!("# on curved backgrounds the symmetry, fundamental identity, and gradient round trip");
    println!(
        "# hold to the geodesic-shooting tolerance; and near coincidence sigma approaches the"
    );
    println!("# leading (1/2)|Δ|^2 tangent-space form (ratio -> 1). Established GR, not a model.");
    Ok(())
}
