//! **Voilement d'une plaque plane comprimée ou cisaillée** (Eurocode 3, EN
//! 1993-1-5) : contrainte critique de voilement élastique d'une plaque rectangulaire,
//! élancement de plaque réduit, facteur de réduction `ρ` (paroi interne) et largeur
//! efficace résultante prenant en compte la réserve post-critique.
//!
//! ```text
//! contrainte critique     σcr = kσ · π²·E / (12·(1 − ν²)) · (t / b)²
//! élancement de plaque     λ̄p  = √(fy / σcr)
//! facteur de réduction     ρ   = 1                              si λ̄p ≤ 0,673
//!                          ρ   = (λ̄p − 0,055·(3 + ψ)) / λ̄p²     si λ̄p > 0,673
//! largeur efficace         beff = ρ · b
//! ```
//!
//! `kσ` coefficient de voilement (sans dimension), `E` module d'élasticité (Pa),
//! `ν` coefficient de Poisson (sans dimension), `t` épaisseur de la plaque (m),
//! `b` largeur de la plaque entre appuis / raidisseurs (m), `σcr` contrainte
//! critique de voilement (Pa), `fy` limite d'élasticité caractéristique (Pa),
//! `λ̄p` élancement de plaque réduit (sans dimension), `ψ` rapport des contraintes
//! aux bords de la plaque (sans dimension), `ρ` facteur de réduction (sans
//! dimension, `ρ ≤ 1`), `beff` largeur efficace (m).
//!
//! **Convention** : unités **SI** cohérentes — contraintes et modules en **Pa**
//! (1 MPa = 10⁶ Pa), longueurs et épaisseurs en **m**. Le rapport `t / b` est sans
//! dimension : `σcr` et `E` partagent donc la même unité. Types `f64`.
//!
//! **Limite honnête** : le **coefficient de voilement `kσ`** est **fourni par
//! l'appelant** selon les **conditions d'appui** de la plaque et le **rapport de
//! contraintes `ψ`** (tableaux 4.1 et 4.2 de l'EN 1993-1-5) — aucune valeur n'est
//! choisie ici. Le **facteur de réduction** [`platebk_reduction_factor`] applique la
//! formule de l'**EN 1993-1-5 §4.4** pour une **paroi interne** (élément comprimé
//! sur ses deux bords longitudinaux) en prenant `3 + ψ = 4`, c.-à-d. une
//! **compression uniforme `ψ = 1`** ; pour une paroi en console ou un `ψ ≠ 1`,
//! l'expression de `ρ` diffère et **n'est pas** couverte ici. La plaque est supposée
//! **plane et de comportement élastique linéaire** ; la réserve **post-critique**
//! est modélisée par la **largeur efficace** (méthode des largeurs efficaces). Les
//! **résistances caractéristiques** (`fy`), le **module** (`E`) et le **coefficient
//! `kσ`** sont **fournis par l'appelant** d'après l'**Eurocode 3** et son **Annexe
//! Nationale** — aucune valeur « par défaut » n'est inventée.

use core::f64::consts::PI;

/// Contrainte critique de voilement élastique d'une plaque rectangulaire
/// `σcr = kσ · π²·E / (12·(1 − ν²)) · (t / b)²` (Pa) (EN 1993-1-5 §4.4, éq. de la
/// plaque de Bryan).
///
/// `buckling_coefficient` = `kσ` (sans dimension, fourni selon les conditions
/// d'appui et le rapport de contraintes `ψ`), `elastic_modulus` = `E` (Pa),
/// `poisson_ratio` = `ν` (sans dimension), `thickness` = `t` (m), `width` = `b`
/// (m) ; renvoie une contrainte (Pa, même unité que `E`).
///
/// Panique si `buckling_coefficient < 0`, si `elastic_modulus < 0`, si
/// `poisson_ratio` n'est pas dans `]-1 ; 0,5[` (le facteur `1 − ν²` doit rester
/// strictement positif et physiquement admissible), si `thickness < 0` ou si
/// `width <= 0` (division par zéro).
pub fn platebk_critical_stress(
    buckling_coefficient: f64,
    elastic_modulus: f64,
    poisson_ratio: f64,
    thickness: f64,
    width: f64,
) -> f64 {
    assert!(
        buckling_coefficient >= 0.0,
        "le coefficient de voilement kσ doit être ≥ 0"
    );
    assert!(
        elastic_modulus >= 0.0,
        "le module d'élasticité E doit être ≥ 0"
    );
    assert!(
        poisson_ratio > -1.0 && poisson_ratio < 0.5,
        "le coefficient de Poisson ν doit être dans ]-1 ; 0,5["
    );
    assert!(thickness >= 0.0, "l'épaisseur t doit être ≥ 0");
    assert!(
        width > 0.0,
        "la largeur b doit être strictement positive (division par zéro)"
    );
    buckling_coefficient * PI * PI * elastic_modulus
        / (12.0 * (1.0 - poisson_ratio * poisson_ratio))
        * (thickness / width).powi(2)
}

/// Élancement de plaque réduit `λ̄p = √(fy / σcr)` (sans dimension) (EN 1993-1-5
/// §4.4).
///
/// `yield_strength` = `fy` limite d'élasticité caractéristique (Pa), `critical_stress`
/// = `σcr` contrainte critique de voilement (Pa, même unité que `fy`) ; renvoie un
/// élancement (sans dimension).
///
/// Panique si `yield_strength < 0` ou si `critical_stress <= 0` (racine d'un négatif
/// ou division par zéro).
pub fn platebk_relative_slenderness(yield_strength: f64, critical_stress: f64) -> f64 {
    assert!(
        yield_strength >= 0.0,
        "la limite d'élasticité fy doit être ≥ 0"
    );
    assert!(
        critical_stress > 0.0,
        "la contrainte critique σcr doit être strictement positive"
    );
    (yield_strength / critical_stress).sqrt()
}

/// Facteur de réduction `ρ` d'une **paroi interne comprimée uniformément**
/// (EN 1993-1-5 §4.4, `3 + ψ = 4`) : `ρ = 1` si `λ̄p ≤ 0,673`, sinon
/// `ρ = (λ̄p − 0,22) / λ̄p²` (sans dimension).
///
/// `relative_slenderness` = `λ̄p` élancement de plaque réduit (sans dimension) ;
/// renvoie le facteur de réduction `ρ` (sans dimension, `ρ ≤ 1`). Le terme `0,22`
/// vaut `0,055·(3 + ψ)` avec `ψ = 1` (compression uniforme) : pour une paroi en
/// console ou `ψ ≠ 1`, cette expression **ne s'applique pas**.
///
/// Panique si `relative_slenderness <= 0` (division par zéro dans la branche
/// post-critique et grandeur physiquement non admissible).
pub fn platebk_reduction_factor(relative_slenderness: f64) -> f64 {
    assert!(
        relative_slenderness > 0.0,
        "l'élancement de plaque λ̄p doit être strictement positif"
    );
    if relative_slenderness <= 0.673
    {
        1.0
    }
    else
    {
        (relative_slenderness - 0.055 * 4.0) / (relative_slenderness * relative_slenderness)
    }
}

/// Largeur efficace d'une plaque `beff = ρ · b` (m) (EN 1993-1-5 §4.4, méthode
/// des largeurs efficaces).
///
/// `reduction_factor` = `ρ` (sans dimension, `0 ≤ ρ ≤ 1`), `width` = `b` largeur
/// brute de la plaque (m) ; renvoie la largeur efficace (m, même unité que `b`).
///
/// Panique si `reduction_factor` n'est pas dans `[0 ; 1]` ou si `width < 0`.
pub fn platebk_effective_width(reduction_factor: f64, width: f64) -> f64 {
    assert!(
        (0.0..=1.0).contains(&reduction_factor),
        "le facteur de réduction ρ doit être dans [0 ; 1]"
    );
    assert!(width >= 0.0, "la largeur b doit être ≥ 0");
    reduction_factor * width
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn critical_stress_proportionnel_carre_du_rapport_t_sur_b() {
        // σcr ∝ (t/b)² : doubler t (b fixe) quadruple la contrainte critique.
        let base = platebk_critical_stress(4.0, 210.0e9, 0.3, 0.010, 1.0);
        let double_t = platebk_critical_stress(4.0, 210.0e9, 0.3, 0.020, 1.0);
        assert_relative_eq!(double_t, 4.0 * base, epsilon = 1e-3, max_relative = 1e-3);
    }

    #[test]
    fn critical_stress_proportionnel_au_coefficient_de_voilement() {
        // σcr ∝ kσ : tripler kσ triple la contrainte critique.
        let k4 = platebk_critical_stress(4.0, 210.0e9, 0.3, 0.012, 0.8);
        let k12 = platebk_critical_stress(12.0, 210.0e9, 0.3, 0.012, 0.8);
        assert_relative_eq!(k12, 3.0 * k4, epsilon = 1e-3, max_relative = 1e-3);
    }

    #[test]
    fn relative_slenderness_vaut_un_a_l_egalite_fy_egale_sigma_cr() {
        // λ̄p = √(fy/σcr) = 1 quand fy = σcr (réciprocité / cas limite).
        let lambda = platebk_relative_slenderness(355.0e6, 355.0e6);
        assert_relative_eq!(lambda, 1.0, epsilon = 1e-3, max_relative = 1e-3);
    }

    #[test]
    fn reduction_factor_vaut_un_pour_plaque_trapue() {
        // λ̄p ≤ 0,673 : plaque non voilée, aucune réduction (ρ = 1).
        assert_relative_eq!(
            platebk_reduction_factor(0.5),
            1.0,
            epsilon = 1e-3,
            max_relative = 1e-3
        );
        // Juste à la borne : encore ρ = 1.
        assert_relative_eq!(
            platebk_reduction_factor(0.673),
            1.0,
            epsilon = 1e-3,
            max_relative = 1e-3
        );
    }

    #[test]
    fn effective_width_proportionnelle_au_facteur_de_reduction() {
        // beff = ρ·b : demi-facteur → demi-largeur efficace.
        let b = 1.2;
        assert_relative_eq!(
            platebk_effective_width(0.5, b),
            0.5 * b,
            epsilon = 1e-3,
            max_relative = 1e-3
        );
        // ρ = 1 restitue la largeur brute.
        assert_relative_eq!(
            platebk_effective_width(1.0, b),
            b,
            epsilon = 1e-3,
            max_relative = 1e-3
        );
    }

    #[test]
    fn cas_chiffre_chaine_complete() {
        // Plaque interne comprimée uniformément (ψ = 1) : kσ = 4, E = 210 GPa,
        // ν = 0,3, t = 10 mm, b = 1,0 m, fy = 355 MPa.
        //
        // σcr = 4·π²·210e9 / (12·(1 − 0,3²)) · (0,010/1,0)²
        //     = 4·9,8696044011·2,10e11 / 10,92 · 1e-4
        //     = 8,29046769684e12 / 10,92 · 1e-4
        //     = 7,59200338539e11 · 1e-4 = 7,59200338539e7 Pa ≈ 75 920 034 Pa
        let sigma_cr = platebk_critical_stress(4.0, 210.0e9, 0.3, 0.010, 1.0);
        assert_relative_eq!(sigma_cr, 75_920_033.85, epsilon = 1.0, max_relative = 1e-3);

        // λ̄p = √(355e6 / 7,59200338539e7) = √(4,675937…) = 2,162391…
        let lambda = platebk_relative_slenderness(355.0e6, sigma_cr);
        assert_relative_eq!(lambda, 2.162391, epsilon = 1e-3, max_relative = 1e-3);

        // ρ = (2,162391 − 0,22) / 2,162391² = 1,942391 / 4,675935 = 0,415405…
        let rho = platebk_reduction_factor(lambda);
        assert_relative_eq!(rho, 0.415405, epsilon = 1e-3, max_relative = 1e-3);

        // beff = 0,415405 · 1,0 = 0,415405 m
        let beff = platebk_effective_width(rho, 1.0);
        assert_relative_eq!(beff, 0.415405, epsilon = 1e-3, max_relative = 1e-3);
    }

    #[test]
    #[should_panic(expected = "la largeur b doit être strictement positive")]
    fn critical_stress_panique_si_largeur_nulle() {
        let _ = platebk_critical_stress(4.0, 210.0e9, 0.3, 0.010, 0.0);
    }
}
