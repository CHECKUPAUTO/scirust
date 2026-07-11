//! Imprime la preuve formelle a priori (arithmétique rationnelle exacte) de
//! [`scirust_core::formal_proof`] : bornes exactes (fractions) et leur
//! valeur décimale approchée (pour la lecture humaine — la preuve elle-même
//! ne dépend que des fractions exactes, imprimées aussi).

use num_traits::ToPrimitive;
use scirust_core::formal_proof::prove_exp_family;
use std::process::ExitCode;

fn main() -> ExitCode {
    println!("PROOF-FORMAL-BOUNDS v1");
    let proof = prove_exp_family();
    println!("famille : {}", proof.name);
    println!(
        "R (borne de la plage réduite)      = {} ≈ {:.12}",
        proof.range_bound,
        proof.range_bound.to_f64().unwrap()
    );
    println!(
        "borne de troncature (Lagrange)     = {} ≈ {:.3e}",
        proof.truncation_bound,
        proof.truncation_bound.to_f64().unwrap()
    );
    println!(
        "borne d'arrondi Horner (Higham γ)  = {} ≈ {:.3e}",
        proof.rounding_bound,
        proof.rounding_bound.to_f64().unwrap()
    );
    println!(
        "borne d'erreur relative totale     ≈ {:.3e} (2^{:.2})",
        proof.relative_bound.to_f64().unwrap(),
        proof.relative_bound.to_f64().unwrap().log2()
    );
    println!(
        "seuil d'arrondi correct            = {} = 2^-25 ≈ {:.3e}",
        proof.threshold,
        proof.threshold.to_f64().unwrap()
    );
    let ok = proof.holds();
    println!(
        "marge (seuil / borne)              ≈ {:.3e}",
        (proof.threshold.to_f64().unwrap() / proof.relative_bound.to_f64().unwrap())
    );
    println!("verdict={}", if ok { "PASS" } else { "FAIL" });
    if ok
    {
        ExitCode::SUCCESS
    }
    else
    {
        ExitCode::FAILURE
    }
}
