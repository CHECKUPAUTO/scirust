//! **Structure bois — compression axiale et flambement par flexion**
//! (Eurocode 5, EN 1995-1-1 §6.3.2) : résistance de calcul en compression
//! `fc,0,d`, élancement relatif `λrel`, coefficient de flambement `kc` et
//! effort normal résistant `Nc,Rd` d'un poteau bois comprimé.
//!
//! ```text
//! résistance de calcul   fc,0,d = kmod · fc,0,k / γ_M
//! élancement relatif     λrel   = (λ / π) · √(fc,0,k / E0,05)
//! coefficient interméd.  k      = 0,5·(1 + βc·(λrel − 0,3) + λrel²)
//! coefficient flambement kc     = 1 / (k + √(k² − λrel²))   (plafonné à 1,0)
//! effort résistant       Nc,Rd  = kc · A · fc,0,d
//! ```
//!
//! `kmod` coefficient de modification (classe de service × durée de charge, sans
//! dimension), `fc,0,k` résistance caractéristique en compression axiale (MPa),
//! `γ_M` coefficient partiel de sécurité sur le matériau (sans dimension),
//! `fc,0,d` résistance de calcul en compression (MPa) ; `λ` élancement mécanique
//! `Lcr/i` (sans dimension), `E0,05` module d'élasticité au 5ᵉ centile (module
//! caractéristique, MPa), `λrel` élancement relatif (sans dimension) ; `βc`
//! facteur d'imperfection (sans dimension, p. ex. 0,2 pour le bois massif, 0,1
//! pour le lamellé-collé et le LVC), `k` grandeur intermédiaire (sans
//! dimension), `kc` coefficient de flambement (sans dimension, `0 < kc ≤ 1`) ;
//! `A` aire brute de la section comprimée (mm²), `Nc,Rd` effort normal résistant
//! de calcul (N).
//!
//! **Convention** : unités **N, mm, MPa** (`1 MPa = 1 N/mm²`), cohérentes entre
//! elles (Eurocode) ; `fc,0,k`, `fc,0,d` et `E0,05` sont en **MPa**, `A` en
//! **mm²**, donc `Nc,Rd` est en **N** (1 kN = 10³ N) ; `λ`, `λrel`, `βc`, `k`,
//! `kc`, `kmod` et `γ_M` sont **sans dimension**. Types `f64`.
//!
//! **Limite honnête** : ce module traite la seule **compression axiale avec
//! flambement par flexion** d'un poteau bois (EN 1995-1-1 §6.3.2), autour d'un
//! seul axe. Le **déversement** des poutres, la **flexion composée**
//! (compression + flexion, formules d'interaction §6.3.2(3)) et le voilement ne
//! sont **pas** traités. L'**élancement mécanique** `λ = Lcr/i` (longueur de
//! flambement et rayon de giration) est **fourni par l'appelant**. Le
//! coefficient `kmod`, la **résistance caractéristique** `fc,0,k`, le **module au
//! 5ᵉ centile** `E0,05` et le **facteur d'imperfection** `βc` sont eux aussi
//! **fournis par l'appelant** d'après l'**Eurocode 5 (EN 1995-1-1)** et son
//! **Annexe Nationale** — aucune valeur « par défaut » n'est inventée. Le
//! coefficient `kc` est **plafonné à 1,0** : il n'y a **pas d'instabilité** tant
//! que `λrel ≤ 0,3`.

/// Résistance de calcul en compression axiale `fc,0,d = kmod · fc,0,k / γ_M`
/// (MPa) (Eurocode 5, EN 1995-1-1 §2.4.1 et §6.1.4).
///
/// `kmod` coefficient de modification (sans dimension), `characteristic_strength`
/// = `fc,0,k` résistance caractéristique en compression (MPa), `gamma_m` = `γ_M`
/// coefficient partiel de sécurité (sans dimension) fourni par l'Eurocode 5 et
/// son Annexe Nationale ; renvoie la résistance de calcul (MPa).
///
/// Panique si `kmod < 0`, si `characteristic_strength < 0` ou si `gamma_m <= 0`
/// (division par zéro).
pub fn timbercomp_design_strength(kmod: f64, characteristic_strength: f64, gamma_m: f64) -> f64 {
    assert!(
        kmod >= 0.0,
        "le coefficient de modification kmod doit être ≥ 0"
    );
    assert!(
        characteristic_strength >= 0.0,
        "la résistance caractéristique fc,0,k doit être ≥ 0 (MPa)"
    );
    assert!(
        gamma_m > 0.0,
        "le coefficient partiel γ_M doit être strictement positif"
    );
    kmod * characteristic_strength / gamma_m
}

/// Élancement relatif `λrel = (λ / π) · √(fc,0,k / E0,05)` (sans dimension)
/// (Eurocode 5, EN 1995-1-1 §6.3.2, éq. 6.21).
///
/// `slenderness` = `λ` élancement mécanique `Lcr/i` (sans dimension, fourni),
/// `characteristic_strength` = `fc,0,k` résistance caractéristique en compression
/// (MPa), `fifth_percentile_modulus` = `E0,05` module d'élasticité au 5ᵉ centile
/// (MPa) ; renvoie l'élancement relatif `λrel`.
///
/// Panique si `slenderness < 0`, si `characteristic_strength < 0` ou si
/// `fifth_percentile_modulus <= 0` (division par zéro sous la racine).
pub fn timbercomp_relative_slenderness(
    slenderness: f64,
    characteristic_strength: f64,
    fifth_percentile_modulus: f64,
) -> f64 {
    assert!(slenderness >= 0.0, "l'élancement λ doit être ≥ 0");
    assert!(
        characteristic_strength >= 0.0,
        "la résistance caractéristique fc,0,k doit être ≥ 0 (MPa)"
    );
    assert!(
        fifth_percentile_modulus > 0.0,
        "le module au 5ᵉ centile E0,05 doit être strictement positif (MPa)"
    );
    (slenderness / core::f64::consts::PI)
        * (characteristic_strength / fifth_percentile_modulus).sqrt()
}

/// Coefficient de flambement `kc` (sans dimension, plafonné à `1,0`) :
/// `k = 0,5·(1 + βc·(λrel − 0,3) + λrel²)` puis
/// `kc = 1 / (k + √(k² − λrel²))` (Eurocode 5, EN 1995-1-1 §6.3.2, éq. 6.25 et
/// 6.27).
///
/// `relative_slenderness` = `λrel` élancement relatif (sans dimension),
/// `imperfection_factor` = `βc` facteur d'imperfection de l'élément (sans
/// dimension, p. ex. 0,2 pour le bois massif, 0,1 pour le lamellé-collé),
/// **fourni par l'Eurocode 5** ; renvoie le coefficient `kc`. Le plafond `1,0`
/// traduit l'absence d'instabilité tant que `λrel ≤ 0,3`.
///
/// Panique si `relative_slenderness < 0` ou si `imperfection_factor < 0`.
pub fn timbercomp_instability_factor(relative_slenderness: f64, imperfection_factor: f64) -> f64 {
    assert!(
        relative_slenderness >= 0.0,
        "l'élancement relatif λrel doit être ≥ 0"
    );
    assert!(
        imperfection_factor >= 0.0,
        "le facteur d'imperfection βc doit être ≥ 0"
    );
    let lambda_rel = relative_slenderness;
    let k = 0.5 * (1.0 + imperfection_factor * (lambda_rel - 0.3) + lambda_rel * lambda_rel);
    let kc = 1.0 / (k + (k * k - lambda_rel * lambda_rel).sqrt());
    kc.min(1.0)
}

/// Effort normal résistant de calcul au flambement
/// `Nc,Rd = kc · A · fc,0,d` (N) (Eurocode 5, EN 1995-1-1 §6.3.2, éq. 6.23).
///
/// `instability_factor` = `kc` coefficient de flambement (sans dimension),
/// `cross_section_area` = `A` aire brute de la section comprimée (mm²),
/// `design_strength` = `fc,0,d` résistance de calcul en compression (MPa) ;
/// renvoie l'effort résistant (N, car mm²·MPa = mm²·N/mm² = N).
///
/// Panique si `instability_factor < 0`, si `cross_section_area <= 0` ou si
/// `design_strength < 0`.
pub fn timbercomp_buckling_resistance(
    instability_factor: f64,
    cross_section_area: f64,
    design_strength: f64,
) -> f64 {
    assert!(
        instability_factor >= 0.0,
        "le coefficient de flambement kc doit être ≥ 0"
    );
    assert!(
        cross_section_area > 0.0,
        "l'aire A de la section doit être strictement positive (mm²)"
    );
    assert!(
        design_strength >= 0.0,
        "la résistance de calcul fc,0,d doit être ≥ 0 (MPa)"
    );
    instability_factor * cross_section_area * design_strength
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn design_strength_reciprocity() {
        // Réciprocité : fc,0,d · γ_M / kmod restitue la résistance
        // caractéristique fc,0,k.
        let (kmod, fc0k, gamma_m) = (0.8_f64, 21.0, 1.3);
        let fc0d = timbercomp_design_strength(kmod, fc0k, gamma_m);
        assert_relative_eq!(fc0d * gamma_m / kmod, fc0k, epsilon = 1e-9);
        // Valeur chiffrée : 0,8·21/1,3 = 16,8/1,3 = 12,923076923… MPa.
        assert_relative_eq!(fc0d, 12.923_076_923, epsilon = 1e-6);
    }

    #[test]
    fn relative_slenderness_proportional_to_slenderness() {
        // λrel ∝ λ à matériau donné : doubler λ double λrel.
        let (fc0k, e05) = (21.0_f64, 7400.0);
        let l1 = timbercomp_relative_slenderness(60.0, fc0k, e05);
        let l2 = timbercomp_relative_slenderness(120.0, fc0k, e05);
        assert_relative_eq!(l2, 2.0 * l1, epsilon = 1e-9);
        // λ = 0 → λrel = 0 (pas d'effet de second ordre).
        assert_relative_eq!(
            timbercomp_relative_slenderness(0.0, fc0k, e05),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn relative_slenderness_worked_value() {
        // C24 : fc,0,k = 21 MPa, E0,05 = 7400 MPa, λ = 60.
        // λrel = (60/π)·√(21/7400) = 19,098593…·√0,00283784
        //      = 19,098593…·0,0532714 = 1,017371…
        let lr = timbercomp_relative_slenderness(60.0, 21.0, 7400.0);
        assert_relative_eq!(lr, 1.017_371, epsilon = 1e-3);
    }

    #[test]
    fn instability_factor_unity_below_threshold() {
        // À λrel = 0,3 : k = 0,5·(1 + 0 + 0,09) = 0,545 et
        // √(0,545² − 0,3²) = √(0,297025 − 0,09) = √0,207025 = 0,455 exactement,
        // donc kc = 1/(0,545 + 0,455) = 1/1,0 = 1,0 quel que soit βc.
        assert_relative_eq!(timbercomp_instability_factor(0.3, 0.2), 1.0, epsilon = 1e-9);
        assert_relative_eq!(timbercomp_instability_factor(0.3, 0.1), 1.0, epsilon = 1e-9);
        // En deçà du seuil (λrel < 0,3), kc reste plafonné à 1,0.
        assert_relative_eq!(timbercomp_instability_factor(0.2, 0.2), 1.0, epsilon = 1e-9);
    }

    #[test]
    fn instability_factor_worked_value() {
        // Bois massif βc = 0,2, λrel = 1,017371 (cas C24 ci-dessus).
        // k  = 0,5·(1 + 0,2·(1,017371 − 0,3) + 1,017371²)
        //    = 0,5·(1 + 0,2·0,717371 + 1,035044)
        //    = 0,5·(1 + 0,143474 + 1,035044) = 0,5·2,178518 = 1,089259
        // kc = 1/(1,089259 + √(1,089259² − 1,035044))
        //    = 1/(1,089259 + √(1,186485 − 1,035044))
        //    = 1/(1,089259 + √0,151441) = 1/(1,089259 + 0,389154)
        //    = 1/1,478413 = 0,676402
        let kc = timbercomp_instability_factor(1.017_371, 0.2);
        assert_relative_eq!(kc, 0.676_402, epsilon = 1e-3);
        assert!(kc < 1.0); // barre élancée : instabilité effective
    }

    #[test]
    fn buckling_resistance_scales_and_worked_case() {
        // Nc,Rd = kc·A·fc,0,d : proportionnel à kc et à l'aire.
        // Poteau C24 100×100 mm : A = 10000 mm², fc,0,d = 12,923077 MPa,
        // kc = 0,676402 → Nc,Rd = 0,676402·10000·12,923077
        //    = 6764,02·12,923077 = 87411,9 N ≈ 87,4 kN.
        let fc0d = timbercomp_design_strength(0.8, 21.0, 1.3);
        let kc = 0.676_402_f64;
        let area = 10_000.0_f64;
        let n_crd = timbercomp_buckling_resistance(kc, area, fc0d);
        assert_relative_eq!(n_crd, 87_411.9, epsilon = 1.0);
        // À kc = 1 on retrouve la charge de compression pure A·fc,0,d.
        let n_pure = timbercomp_buckling_resistance(1.0, area, fc0d);
        assert_relative_eq!(n_pure, area * fc0d, epsilon = 1e-6);
        // Nc,Rd est proportionnel à kc.
        assert_relative_eq!(n_crd, kc * n_pure, epsilon = 1e-6);
    }

    #[test]
    #[should_panic(expected = "le coefficient partiel γ_M doit être strictement positif")]
    fn zero_gamma_m_panics() {
        timbercomp_design_strength(0.8, 21.0, 0.0);
    }
}
