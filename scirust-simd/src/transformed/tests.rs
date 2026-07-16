// scirust-simd/src/transformed/tests.rs
//
// Validation du sous-système TSHA. On vérifie, dans l'ordre :
//  * les fonctions spéciales (contre des valeurs exactes connues) ;
//  * chaque transformation (encode/décode/domaine/branches/ambiguïté) ;
//  * le scalaire transformé et l'algèbre de Cayley–Dickson générique ;
//  * les deux modèles d'exécution et leur défaut (dont Identity ≡ contrôle) ;
//  * les métriques et les expériences (déterminisme).

use super::branch::{GAMMA_TURN_X, GammaBranch};
use super::hypercomplex::{
    Hypercomplex, Octonion, Quaternion, Sedenion, model_a_product, model_b_product,
};
use super::identity::Identity;
use super::log_gamma::LogGamma;
use super::metrics::{abs_error, associator_norm, defect_report, norm};
use super::reciprocal_gamma::ReciprocalGamma;
use super::scalar::TransformedScalar;
use super::special::{GAMMA_ARGMIN, digamma, gamma, ln_gamma};
use super::transform::{DomainError, InverseError, ScalarTransform};
use super::{experiments, special};

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

// ------------------------------------------------------------------ //
//  Fonctions spéciales (contre valeurs exactes)                        //
// ------------------------------------------------------------------ //

#[test]
fn special_gamma_matches_factorials_and_sqrt_pi() {
    // Γ(n) = (n−1)!
    assert!(close(gamma(1.0), 1.0, 1e-10));
    assert!(close(gamma(2.0), 1.0, 1e-10));
    assert!(close(gamma(3.0), 2.0, 1e-10));
    assert!(close(gamma(5.0), 24.0, 1e-8));
    assert!(close(gamma(11.0), 3_628_800.0, 1e-2)); // 10!
    // Γ(½) = √π.
    assert!(close(gamma(0.5), std::f64::consts::PI.sqrt(), 1e-10));
    // ln Γ cohérent.
    assert!(close(ln_gamma(5.0), 24.0f64.ln(), 1e-10));
    assert!(close(ln_gamma(1.0), 0.0, 1e-12));
}

#[test]
fn special_digamma_matches_known_values() {
    let euler = 0.577_215_664_901_532_9;
    assert!(close(digamma(1.0), -euler, 1e-9)); // ψ(1) = −γ
    assert!(close(digamma(2.0), 1.0 - euler, 1e-9)); // ψ(2) = 1 − γ
    // ψ(z*) = 0 au minimum de Γ.
    assert!(close(digamma(GAMMA_ARGMIN), 0.0, 1e-8));
    assert!(close(special::GAMMA_ARGMIN, 1.461_632_1, 1e-6));
}

// ------------------------------------------------------------------ //
//  Identity — cas de contrôle                                          //
// ------------------------------------------------------------------ //

#[test]
fn identity_is_transparent() {
    assert!(<Identity as ScalarTransform<f64>>::is_globally_invertible());
    assert_eq!(<Identity as ScalarTransform<f64>>::encode(3.5), Ok(3.5));
    assert_eq!(<Identity as ScalarTransform<f64>>::decode(3.5, ()), Ok(3.5));
    assert_eq!(<Identity as ScalarTransform<f64>>::derivative(9.9), Ok(1.0));
}

// ------------------------------------------------------------------ //
//  ReciprocalGamma                                                     //
// ------------------------------------------------------------------ //

#[test]
fn reciprocal_gamma_known_values_and_domain() {
    let enc = |x| ReciprocalGamma::encode(x).unwrap();
    assert!(close(enc(0.0), 1.0, 1e-10)); // 1/Γ(1)
    assert!(close(enc(1.0), 1.0, 1e-10)); // 1/Γ(2)
    assert!(close(enc(2.0), 0.5, 1e-10)); // 1/Γ(3)
    assert!(close(enc(4.0), 1.0 / 24.0, 1e-10)); // 1/Γ(5)
    // Maximum ≈ 1.1292 au point de retournement.
    assert!(close(ReciprocalGamma::max_value(), 1.129_173_9, 1e-6));
    assert!(!<ReciprocalGamma as ScalarTransform<f64>>::is_globally_invertible());
    // Domaine : x ≤ −1 rejeté.
    assert_eq!(
        ReciprocalGamma::encode(-1.0),
        Err(DomainError::BelowDomain {
            value: -1.0,
            lower_bound: -1.0
        })
    );
    assert!(matches!(
        ReciprocalGamma::encode(f64::NAN),
        Err(DomainError::NotFinite { .. })
    ));
    // Dérivée : au sommet, φ'(x*) = 0 (car ψ(x*+1)=0).
    assert!(close(
        ReciprocalGamma::derivative(GAMMA_TURN_X).unwrap(),
        0.0,
        1e-6
    ));
}

#[test]
fn reciprocal_gamma_branch_ambiguity_is_explicit() {
    // φ(0) = φ(1) = 1 : deux antécédents, un par branche.
    let y = 1.0;
    let lo = ReciprocalGamma::decode(y, GammaBranch::Lower).unwrap();
    let up = ReciprocalGamma::decode(y, GammaBranch::Upper).unwrap();
    assert!(close(lo, 0.0, 1e-6), "branche basse ≈ 0, obtenu {lo}");
    assert!(close(up, 1.0, 1e-6), "branche haute ≈ 1, obtenu {up}");
    assert!(lo < GAMMA_TURN_X && up > GAMMA_TURN_X);
    // Les deux ré-encodent vers la même valeur (ambiguïté réelle).
    assert!(close(ReciprocalGamma::encode(lo).unwrap(), y, 1e-9));
    assert!(close(ReciprocalGamma::encode(up).unwrap(), y, 1e-9));
}

#[test]
fn reciprocal_gamma_roundtrip_and_out_of_range() {
    // Round-trip sur la branche correcte pour plusieurs x.
    for &x in &[-0.5, 0.0, 0.2, 2.0, 3.5, 6.0]
    {
        let y = ReciprocalGamma::encode(x).unwrap();
        let branch = if x >= GAMMA_TURN_X
        {
            GammaBranch::Upper
        }
        else
        {
            GammaBranch::Lower
        };
        let back = ReciprocalGamma::decode(y, branch).unwrap();
        assert!(close(back, x, 1e-6), "round-trip x={x} → {back}");
    }
    // Hors image : y ≤ 0 ou y > φ_max.
    assert!(matches!(
        ReciprocalGamma::decode(0.0, GammaBranch::Upper),
        Err(InverseError::OutOfRange { .. })
    ));
    assert!(matches!(
        ReciprocalGamma::decode(2.0, GammaBranch::Upper),
        Err(InverseError::OutOfRange { .. })
    ));
    assert!(matches!(
        ReciprocalGamma::decode(f64::INFINITY, GammaBranch::Upper),
        Err(InverseError::NotFinite { .. })
    ));
}

// ------------------------------------------------------------------ //
//  LogGamma                                                            //
// ------------------------------------------------------------------ //

#[test]
fn log_gamma_known_values_and_branches() {
    let enc = |x| LogGamma::encode(x).unwrap();
    assert!(close(enc(0.0), 0.0, 1e-10)); // ln Γ(1) = 0
    assert!(close(enc(1.0), 0.0, 1e-10)); // ln Γ(2) = 0
    assert!(close(enc(4.0), 24.0f64.ln(), 1e-9)); // ln 4! = ln 24
    assert!(close(LogGamma::min_value(), -0.121_486_29, 1e-6));
    // φ(0) = φ(1) = 0 : branches distinctes.
    let lo = LogGamma::decode(0.0, GammaBranch::Lower).unwrap();
    let up = LogGamma::decode(0.0, GammaBranch::Upper).unwrap();
    assert!(close(lo, 0.0, 1e-6) && close(up, 1.0, 1e-6));
    // Dérivée = digamma.
    assert!(close(
        LogGamma::derivative(3.0).unwrap(),
        digamma(4.0),
        1e-12
    ));
    // Sous le minimum : hors image.
    assert!(matches!(
        LogGamma::decode(LogGamma::min_value() - 0.01, GammaBranch::Upper),
        Err(InverseError::OutOfRange { .. })
    ));
}

// ------------------------------------------------------------------ //
//  TransformedScalar                                                   //
// ------------------------------------------------------------------ //

#[test]
fn transformed_scalar_api_and_arithmetic() {
    type Ts = TransformedScalar<f64, ReciprocalGamma>;
    let a = Ts::from_latent(2.0);
    assert_eq!(a.latent(), 2.0);
    assert!(close(a.encoded().unwrap(), 0.5, 1e-10)); // 1/Γ(3)
    // Décodage → scalaire (branche principale = Upper).
    let b = Ts::try_from_encoded(0.5, GammaBranch::Upper).unwrap();
    assert!(close(b.latent(), 2.0, 1e-6));
    // Arithmétique : opère sur le LATENT.
    use crate::fixed::NumericScalar;
    let s = a + Ts::from_latent(3.0);
    assert_eq!(s.latent(), 5.0);
    assert_eq!((a * Ts::from_latent(4.0)).latent(), 8.0);
    assert_eq!(Ts::zero().latent(), 0.0);
    assert_eq!(Ts::one().latent(), 1.0);
    assert_eq!(Ts::from_i32(-3).abs().latent(), 3.0);
}

// ------------------------------------------------------------------ //
//  Algèbre de Cayley–Dickson générique                                 //
// ------------------------------------------------------------------ //

#[test]
fn quaternion_hamilton_table_exact() {
    type Q = Quaternion<f64>;
    let (i, j, k) = (Q::basis(1), Q::basis(2), Q::basis(3));
    assert_eq!(i * j, k); // i·j = k
    assert_eq!(j * k, i); // j·k = i
    assert_eq!(k * i, j); // k·i = j
    assert_eq!(i * i, Q::real(-1.0)); // i² = −1
    assert_eq!(j * i, -k); // non-commutatif : j·i = −k
    // Quaternions associatifs : associateur nul pour tout triplet de base.
    for a in 0..4
    {
        for b in 0..4
        {
            for c in 0..4
            {
                let n = associator_norm(&Q::basis(a), &Q::basis(b), &Q::basis(c));
                assert!(n < 1e-12, "quaternion associatif violé ({a},{b},{c})");
            }
        }
    }
}

#[test]
fn octonion_is_nonassociative() {
    type O = Octonion<f64>;
    // Il existe un triplet de base à associateur non nul.
    let mut found = false;
    for a in 1..8
    {
        for b in 1..8
        {
            for c in 1..8
            {
                if associator_norm(&O::basis(a), &O::basis(b), &O::basis(c)) > 0.5
                {
                    found = true;
                }
            }
        }
    }
    assert!(found, "octonion devrait être non associatif");
    // Mais alternatif : associateur nul si deux arguments égaux.
    for a in 1..8
    {
        for b in 1..8
        {
            assert!(associator_norm(&O::basis(a), &O::basis(a), &O::basis(b)) < 1e-12);
        }
    }
}

#[test]
fn sedenion_has_zero_divisors() {
    type S = Sedenion<f64>;
    // Cherche une paire non nulle (eᵢ+eⱼ)(eₖ+eₗ) = 0 (diviseurs de zéro).
    let mut found = false;
    'outer: for i in 1..16
    {
        for j in (i + 1)..16
        {
            for k in 1..16
            {
                for l in (k + 1)..16
                {
                    let a = S::basis(i) + S::basis(j);
                    let b = S::basis(k) + S::basis(l);
                    if norm(&(a * b)) < 1e-12
                    {
                        found = true;
                        break 'outer;
                    }
                }
            }
        }
    }
    assert!(found, "les sédénions devraient avoir des diviseurs de zéro");
}

#[test]
fn hypercomplex_conj_norm_map() {
    type Q = Quaternion<f64>;
    let q = Q::new([1.0, 2.0, -3.0, 4.0]);
    assert_eq!(q.conj(), Q::new([1.0, -2.0, 3.0, -4.0]));
    // ‖q‖² = q·q̄ (composante réelle), et = Σcᵢ².
    assert!(close(q.norm_sqr(), 1.0 + 4.0 + 9.0 + 16.0, 1e-12));
    assert!(close((q * q.conj()).components()[0], q.norm_sqr(), 1e-10));
    // map : transporte vers un autre scalaire.
    let doubled = q.map(|x| x * 2.0);
    assert_eq!(doubled, Q::new([2.0, 4.0, -6.0, 8.0]));
}

// ------------------------------------------------------------------ //
//  Modèles A / B et défaut                                             //
// ------------------------------------------------------------------ //

fn ts_quat<F>(c: [f64; 4]) -> Quaternion<TransformedScalar<f64, F>> {
    Hypercomplex(c.map(TransformedScalar::from_latent))
}

#[test]
fn identity_makes_models_equivalent() {
    // Sous Identity, φ(A⋆B) == φ(A)⋆φ(B) au bit près : défaut nul.
    let a = ts_quat::<Identity>([0.1, 0.2, -0.3, 0.05]);
    let b = ts_quat::<Identity>([-0.2, 0.15, 0.1, -0.25]);
    let ma = model_a_product(a, b).unwrap();
    let mb = model_b_product(a, b).unwrap();
    assert_eq!(ma, mb, "Identity : Modèle A doit égaler Modèle B");
    assert!(abs_error(&ma, &mb) == 0.0);
}

#[test]
fn reciprocal_gamma_models_differ() {
    // Sous une transformation non linéaire, les deux modèles diffèrent.
    let a = ts_quat::<ReciprocalGamma>([0.1, 0.2, -0.15, 0.05]);
    let b = ts_quat::<ReciprocalGamma>([-0.1, 0.12, 0.08, -0.2]);
    let ma = model_a_product(a, b).unwrap();
    let mb = model_b_product(a, b).unwrap();
    let report = defect_report(&ma, &mb);
    assert!(report.abs_l2 > 1e-6, "défaut attendu non nul: {report:?}");
    // Déterminisme bit-à-bit : recalcul identique.
    let ma2 = model_a_product(a, b).unwrap();
    let mb2 = model_b_product(a, b).unwrap();
    assert_eq!(ma.components(), ma2.components());
    assert_eq!(mb.components(), mb2.components());
}

// ------------------------------------------------------------------ //
//  Expériences (déterminisme + tendances)                              //
// ------------------------------------------------------------------ //

/// Rapport à la demande : `cargo test -p scirust-simd --features portable-simd
/// -- --ignored --nocapture transformed::tests::print_experiment_csv`.
#[test]
#[ignore = "rapport CSV à la demande (non un test de correction)"]
fn print_experiment_csv() {
    let stats = experiments::run_suite(2000);
    eprintln!("{}", experiments::suite_csv(&stats));
}

#[test]
fn experiment_suite_is_deterministic() {
    let a = experiments::run_suite(64);
    let b = experiments::run_suite(64);
    assert_eq!(a, b, "les expériences doivent être déterministes");
    // CSV bien formé.
    let csv = experiments::suite_csv(&a);
    assert!(csv.starts_with("transform,dim,samples,"));
    assert_eq!(csv.lines().count(), a.len() + 1);
    // Identity : défaut nul ; Gamma : défaut non nul.
    for s in &a
    {
        if s.transform == "Identity"
        {
            assert!(s.max_abs == 0.0, "Identity sans défaut: {s:?}");
        }
        else
        {
            assert!(s.mean_abs > 0.0, "Gamma avec défaut: {s:?}");
        }
        assert!(s.samples > 0);
    }
}

// ------------------------------------------------------------------ //
//  Cas limites / singularités                                          //
// ------------------------------------------------------------------ //

#[test]
fn edge_cases_near_singularity_and_large() {
    // Proche de la singularité x → −1⁺ : encodage fini, décodable (branche basse).
    let x = -0.999;
    let y = ReciprocalGamma::encode(x).unwrap();
    assert!(y.is_finite() && y > 0.0);
    let back = ReciprocalGamma::decode(y, GammaBranch::Lower).unwrap();
    assert!(close(back, x, 1e-4), "près de −1 : {x} → {back}");
    // Grand x : 1/Γ décroît vers 0 (branche haute).
    let y = ReciprocalGamma::encode(8.0).unwrap();
    assert!(y > 0.0 && y < 1e-3);
    assert!(close(
        ReciprocalGamma::decode(y, GammaBranch::Upper).unwrap(),
        8.0,
        1e-4
    ));
    // LogGamma grand argument croît sans borne.
    assert!(LogGamma::encode(20.0).unwrap() > 30.0);
}
