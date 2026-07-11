//! Imprime la preuve formelle a priori (arithmétique rationnelle exacte) de
//! [`scirust_core::formal_proof`] : bornes exactes (fractions) et leur
//! valeur décimale approchée (pour la lecture humaine — la preuve elle-même
//! ne dépend que des fractions exactes, imprimées aussi).

use num_rational::BigRational;
use num_traits::ToPrimitive;
use scirust_core::formal_proof::{BoundProof, prove_cos, prove_exp_family, prove_ln, prove_sin};
use std::process::ExitCode;

fn f(x: &BigRational) -> f64 {
    x.to_f64().unwrap()
}

fn print_bound_proof(proof: &BoundProof) -> bool {
    println!("famille : {}", proof.name);
    println!(
        "R (borne de la plage réduite)      = {} ≈ {:.12}",
        proof.range_bound,
        f(&proof.range_bound)
    );
    println!(
        "borne de troncature (Lagrange)      ≈ {:.3e}",
        f(&proof.truncation_bound)
    );
    println!(
        "borne d'arrondi flottant propagée   ≈ {:.3e}",
        f(&proof.rounding_bound)
    );
    println!(
        "borne d'erreur relative totale      ≈ {:.3e} (2^{:.2})",
        f(&proof.relative_bound),
        f(&proof.relative_bound).log2()
    );
    println!(
        "seuil d'arrondi correct             = 2^-25 ≈ {:.3e}",
        f(&proof.threshold)
    );
    let ok = proof.holds();
    println!(
        "marge (seuil / borne)               ≈ {:.3e}",
        f(&proof.threshold) / f(&proof.relative_bound)
    );
    println!("verdict={}", if ok { "PASS" } else { "FAIL" });
    println!();
    ok
}

fn main() -> ExitCode {
    println!("PROOF-FORMAL-BOUNDS v1");

    let mut all_ok = true;

    all_ok &= print_bound_proof(&prove_exp_family());
    all_ok &= print_bound_proof(&prove_sin());
    all_ok &= print_bound_proof(&prove_cos());

    let ln_proof = prove_ln();
    println!("famille : {}", ln_proof.name);
    println!(
        "|s| max (s = (m−1)/(m+1))           ≈ {:.6e}",
        f(&ln_proof.s_max)
    );
    println!(
        "borne relative cas e=0 (x≈1)        ≈ {:.3e} (2^{:.2})",
        f(&ln_proof.e0_relative_bound),
        f(&ln_proof.e0_relative_bound).log2()
    );
    println!(
        "borne relative cas e≠0              ≈ {:.3e} (2^{:.2})",
        f(&ln_proof.ene0_relative_bound),
        f(&ln_proof.ene0_relative_bound).log2()
    );
    println!(
        "borne d'erreur relative totale       ≈ {:.3e} (2^{:.2})",
        f(&ln_proof.relative_bound),
        f(&ln_proof.relative_bound).log2()
    );
    println!(
        "seuil d'arrondi correct              = 2^-25 ≈ {:.3e}",
        f(&ln_proof.threshold)
    );
    let ln_ok = ln_proof.holds();
    println!(
        "marge (seuil / borne)                ≈ {:.3e}",
        f(&ln_proof.threshold) / f(&ln_proof.relative_bound)
    );
    println!("verdict={}", if ln_ok { "PASS" } else { "FAIL" });
    all_ok &= ln_ok;

    println!();
    println!("verdict_global={}", if all_ok { "PASS" } else { "FAIL" });
    if all_ok
    {
        ExitCode::SUCCESS
    }
    else
    {
        ExitCode::FAILURE
    }
}
