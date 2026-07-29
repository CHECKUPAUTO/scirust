//! Van Vleck–Morette determinant checks: flat exactness, symmetry, and the
//! near-coincidence known expansion.
//!
//! The van Vleck determinant `Delta(x', x)` measures the focusing of the
//! geodesics leaving `x'`: it is `1` in flat spacetime and at coincidence, above
//! `1` where the congruence focuses, below where it defocuses. It is computed
//! here from the Jacobian of the exponential map,
//! `Delta = sqrt|g(x')| / (sqrt|g(x)| det J)` with
//! `J = d exp_{x'}(v)/dv` at `v = log_{x'}(x)`. This experiment reports:
//!
//! - **Flat exactness.** In Minkowski `Delta = 1` to the finite-difference
//!   Jacobian floor.
//! - **Symmetry.** `Delta(x', x) = Delta(x, x')` on curved backgrounds — an
//!   independent check, since the two evaluations use different exponential-map
//!   Jacobians.
//! - **Known coincidence expansion.** `Delta = 1 + (1/6) R_{a'b'} sigma^{a'}
//!   sigma^{b'} + O(sigma^2)`. For a maximally symmetric background
//!   `R_{ab} = Lambda g_{ab}`, so `R_{a'b'} sigma^{a'} sigma^{b'} = 2 Lambda
//!   sigma` and `(Delta - 1) / sigma -> Lambda / 3` as the points approach. For
//!   Ricci-flat Schwarzschild the coefficient is `0`, so `(Delta - 1)/sigma -> 0`.
//!
//! Established general relativity only; built on the geodesic exponential /
//! logarithm maps.

#![forbid(unsafe_code)]

use nonlocal_relativity_experiments::{print_experiment_header, require_finite};
use scirust_relativity::{
    AntiDeSitter, DeSitter, Minkowski, Schwarzschild, WorldFunctionSettings, van_vleck_determinant,
    world_function,
};
use std::f64::consts::FRAC_PI_2;

fn main() -> Result<(), String> {
    let schwarzschild =
        Schwarzschild::try_new(1.0).ok_or_else(|| "invalid Schwarzschild".to_string())?;
    let de_sitter_lambda = 0.05;
    let de_sitter =
        DeSitter::try_new(de_sitter_lambda).ok_or_else(|| "invalid de Sitter".to_string())?;
    let anti_de_sitter = AntiDeSitter::try_new(de_sitter_lambda)
        .ok_or_else(|| "invalid anti-de Sitter".to_string())?;
    let settings = WorldFunctionSettings::default();

    print_experiment_header(
        "Van Vleck–Morette determinant checks",
        "scirust-relativity geometry core (established general relativity)",
        "van Vleck determinant from the exponential-map Jacobian; convention-free identities.",
    );

    // Part A: flat exactness -- Delta = 1 in Minkowski.
    println!("# Part A: Minkowski -- |Delta - 1| for several separations");
    println!("separation_kind,delta,abs_delta_minus_one");
    let flat_base = [0.0, 1.0, 2.0, 0.5];
    for (kind, field) in [
        ("spacelike", [0.0, 2.0, 3.0, 1.0]),
        ("timelike", [1.5, 1.2, 2.1, 0.6]),
    ]
    {
        let delta = van_vleck_determinant(&Minkowski, &flat_base, &field, &settings)
            .map_err(|e| e.to_string())?;
        let error = (delta - 1.0).abs();
        require_finite(&[("delta", delta), ("error", error)])?;
        println!("{kind},{delta:.12e},{error:.3e}");
    }

    // Part B: symmetry Delta(x', x) = Delta(x, x') on curved backgrounds.
    println!("#");
    println!("# Part B: curved backgrounds -- symmetry residual |Delta(x',x) - Delta(x,x')|");
    println!("background,delta_forward,delta_reversed,symmetry_gap");
    let curved_base = [0.0, 12.0, FRAC_PI_2, 0.0];
    let curved_field = [0.15, 12.4, FRAC_PI_2 + 0.03, 0.05];
    let de_sitter_base = [0.0, 3.0, FRAC_PI_2, 0.0];
    let de_sitter_field = [0.1, 3.3, FRAC_PI_2 + 0.04, 0.03];
    for (label, forward, reversed) in [
        (
            "Schwarzschild",
            van_vleck_determinant(&schwarzschild, &curved_base, &curved_field, &settings)
                .map_err(|e| e.to_string())?,
            van_vleck_determinant(&schwarzschild, &curved_field, &curved_base, &settings)
                .map_err(|e| e.to_string())?,
        ),
        (
            "de_Sitter",
            van_vleck_determinant(&de_sitter, &de_sitter_base, &de_sitter_field, &settings)
                .map_err(|e| e.to_string())?,
            van_vleck_determinant(&de_sitter, &de_sitter_field, &de_sitter_base, &settings)
                .map_err(|e| e.to_string())?,
        ),
        (
            "anti_de_Sitter",
            van_vleck_determinant(
                &anti_de_sitter,
                &de_sitter_base,
                &de_sitter_field,
                &settings,
            )
            .map_err(|e| e.to_string())?,
            van_vleck_determinant(
                &anti_de_sitter,
                &de_sitter_field,
                &de_sitter_base,
                &settings,
            )
            .map_err(|e| e.to_string())?,
        ),
    ]
    {
        let gap = (forward - reversed).abs();
        require_finite(&[("forward", forward), ("reversed", reversed), ("gap", gap)])?;
        println!("{label},{forward:.12e},{reversed:.12e},{gap:.3e}");
    }

    // Part C: known coincidence expansion (Delta - 1)/sigma -> Lambda/3.
    println!("#");
    println!(
        "# Part C: de Sitter -- (Delta - 1)/sigma -> Lambda/3 = {:.6e} as sigma -> 0",
        de_sitter_lambda / 3.0
    );
    println!("scale,sigma,delta,ratio,target_lambda_over_3");
    let expansion_base = [0.0, 3.0, FRAC_PI_2, 0.0];
    let direction = [0.02, 0.15, 0.03, 0.02];
    for scale in [1.0, 0.5, 0.25, 0.125]
    {
        let field = [
            expansion_base[0] + scale * direction[0],
            expansion_base[1] + scale * direction[1],
            expansion_base[2] + scale * direction[2],
            expansion_base[3] + scale * direction[3],
        ];
        let sigma = world_function(&de_sitter, &expansion_base, &field, &settings)
            .map_err(|e| e.to_string())?;
        let delta = van_vleck_determinant(&de_sitter, &expansion_base, &field, &settings)
            .map_err(|e| e.to_string())?;
        let ratio = (delta - 1.0) / sigma;
        require_finite(&[("sigma", sigma), ("delta", delta), ("ratio", ratio)])?;
        println!(
            "{scale:.3},{sigma:.6e},{delta:.12e},{ratio:.6e},{:.6e}",
            de_sitter_lambda / 3.0
        );
    }

    println!("# interpretation: the van Vleck determinant is 1 in flat spacetime and symmetric in");
    println!(
        "# its two arguments on curved backgrounds (an independent-Jacobian cross-check), and"
    );
    println!("# near coincidence (Delta - 1)/sigma approaches the maximally-symmetric leading");
    println!("# coefficient Lambda/3. Established GR, not a phenomenological model.");
    Ok(())
}
