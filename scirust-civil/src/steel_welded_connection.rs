//! **Charpente métallique — soudure d'angle** (Eurocode 3, EN 1993-1-8) :
//! gorge d'un cordon d'angle isocèle, **méthode directionnelle** (contrainte de
//! comparaison sur la section de gorge et contrainte limite) et **méthode
//! simplifiée** (résistance au cisaillement par unité de longueur).
//!
//! ```text
//! gorge (cordon isocèle)      a  = z / √2
//! contrainte de comparaison   σcomp = √( σ⊥² + 3·(τ⊥² + τ∥²) )
//! critère directionnel        σcomp ≤ fu / (βw · γ_M2)   et   σ⊥ ≤ 0,9·fu/γ_M2
//! contrainte limite           σlim  = fu / (βw · γ_M2)
//! méthode simplifiée          Fw,Rd/l = fvw,d · a = a · fu / (√2 · βw · γ_M2)
//! ```
//!
//! `z` côté (leg) du cordon d'angle isocèle (mm), `a` gorge utile (mm) ; `σ⊥`
//! contrainte normale **perpendiculaire** au plan de la gorge (MPa), `τ⊥`
//! contrainte de cisaillement **perpendiculaire** à l'axe du cordon dans le plan de
//! la gorge (MPa), `τ∥` contrainte de cisaillement **parallèle** à l'axe du cordon
//! (MPa), `σcomp` contrainte de comparaison (von Mises pondérée) sur la section de
//! gorge (MPa) ; `fu` résistance ultime à la traction de la **pièce la plus faible**
//! assemblée (MPa), `βw` facteur de corrélation propre à la nuance d'acier (sans
//! dimension), `γ_M2` coefficient partiel des assemblages (sans dimension) ;
//! `fvw,d` résistance de calcul au cisaillement de la soudure (MPa), `Fw,Rd/l`
//! résistance de calcul par unité de longueur (N/mm).
//!
//! **Convention** : unités **N, mm, MPa** (1 MPa = 1 N/mm²), cohérentes entre elles
//! (Eurocode) ; les contraintes s'expriment en **MPa**, les longueurs en **mm**, les
//! résistances par unité de longueur en **N/mm**. Types `f64`.
//!
//! **Limite honnête** : ce module traite un **cordon d'angle** par la **méthode
//! directionnelle** (contrainte de comparaison sur la section de gorge) et par la
//! **méthode simplifiée**. La **résistance caractéristique** `fu` (de la pièce la
//! plus faible), le **facteur de corrélation** `βw` (EN 1993-1-8, tableau 4.1) et le
//! **coefficient partiel** `γ_M2` sont **fournis par l'appelant** d'après
//! l'**Eurocode 3 (EN 1993-1-8)** et son **Annexe Nationale** — aucune valeur « par
//! défaut » n'est inventée. Le module **calcule les contraintes de comparaison et
//! limite** mais **ne vérifie pas** lui-même l'inégalité `σcomp ≤ σlim` ni la
//! condition annexe `σ⊥ ≤ 0,9·fu/γ_M2` : l'appelant compare les deux grandeurs. La
//! **décomposition** de l'effort appliqué en composantes `σ⊥`, `τ⊥`, `τ∥` est **à la
//! charge de l'appelant** (projection sur la section de gorge). Ne sont **pas**
//! traités ici : la **longueur efficace** (déductions d'amorçage/cratère), les
//! **cordons longs** (facteur réducteur `βLw`), les **soudures bout à bout**, ni la
//! résistance des **pièces assemblées** ; distinct de `steel_bolted_connection`.

/// Gorge utile `a = z / √2` (mm) d'un **cordon d'angle isocèle** de côté `z`
/// (Eurocode 3, EN 1993-1-8, § 4.5.2).
///
/// `leg_size` = `z` côté du cordon d'angle isocèle (mm) ; renvoie la gorge utile `a`
/// (mm), hauteur du triangle rectangle isocèle inscrit dans le cordon.
///
/// Panique si `leg_size < 0`.
pub fn steelweld_throat(leg_size: f64) -> f64 {
    assert!(leg_size >= 0.0, "le côté du cordon z doit être ≥ 0");
    leg_size / core::f64::consts::SQRT_2
}

/// Contrainte de comparaison (von Mises pondérée)
/// `σcomp = √( σ⊥² + 3·(τ⊥² + τ∥²) )` (MPa) sur la section de gorge, **méthode
/// directionnelle** (Eurocode 3, EN 1993-1-8, § 4.5.3.2).
///
/// `sigma_perp` = `σ⊥` contrainte normale perpendiculaire au plan de la gorge (MPa,
/// signée : traction > 0), `tau_perp` = `τ⊥` cisaillement perpendiculaire à l'axe du
/// cordon (MPa), `tau_par` = `τ∥` cisaillement parallèle à l'axe du cordon (MPa) ;
/// renvoie la contrainte de comparaison (MPa, toujours ≥ 0). Le résultat ne dépend
/// que des **carrés** des composantes : les signes n'influent pas.
///
/// Panique si l'une des composantes n'est pas finie (`NaN` ou infinie).
pub fn steelweld_von_mises_stress(sigma_perp: f64, tau_perp: f64, tau_par: f64) -> f64 {
    assert!(
        sigma_perp.is_finite(),
        "la contrainte normale σ⊥ doit être finie"
    );
    assert!(
        tau_perp.is_finite(),
        "le cisaillement perpendiculaire τ⊥ doit être fini"
    );
    assert!(
        tau_par.is_finite(),
        "le cisaillement parallèle τ∥ doit être fini"
    );
    (sigma_perp * sigma_perp + 3.0 * (tau_perp * tau_perp + tau_par * tau_par)).sqrt()
}

/// Contrainte limite `σlim = fu / (βw · γ_M2)` (MPa) de la **méthode directionnelle**
/// (Eurocode 3, EN 1993-1-8, § 4.5.3.2) : borne supérieure de la contrainte de
/// comparaison `σcomp`.
///
/// `ultimate_strength` = `fu` résistance ultime de la pièce la plus faible (MPa),
/// `correlation_factor` = `βw` facteur de corrélation (sans dimension, EN 1993-1-8
/// tableau 4.1), `gamma_m2` = `γ_M2` (sans dimension) ; tous fournis par l'Eurocode 3
/// et son Annexe Nationale ; renvoie la contrainte limite (MPa).
///
/// Panique si `ultimate_strength < 0`, si `correlation_factor <= 0` ou si
/// `gamma_m2 <= 0` (division par zéro).
pub fn steelweld_limit_stress(
    ultimate_strength: f64,
    correlation_factor: f64,
    gamma_m2: f64,
) -> f64 {
    assert!(
        ultimate_strength >= 0.0,
        "la résistance ultime fu doit être ≥ 0"
    );
    assert!(
        correlation_factor > 0.0,
        "le facteur de corrélation βw doit être strictement positif"
    );
    assert!(
        gamma_m2 > 0.0,
        "le coefficient partiel γ_M2 doit être strictement positif"
    );
    ultimate_strength / (correlation_factor * gamma_m2)
}

/// Résistance de calcul au cisaillement **par unité de longueur** de la **méthode
/// simplifiée** `Fw,Rd/l = fvw,d · a = a · fu / (√2 · βw · γ_M2)` (N/mm)
/// (Eurocode 3, EN 1993-1-8, § 4.5.3.3).
///
/// `throat` = `a` gorge utile (mm), `ultimate_strength` = `fu` résistance ultime de
/// la pièce la plus faible (MPa), `correlation_factor` = `βw` facteur de corrélation
/// (sans dimension), `gamma_m2` = `γ_M2` (sans dimension) ; `fu`, `βw` et `γ_M2`
/// fournis par l'Eurocode 3 et son Annexe Nationale ; renvoie la résistance par unité
/// de longueur (N/mm), à comparer à la résultante des efforts par unité de longueur
/// **quelle que soit** leur direction dans le plan de la gorge.
///
/// Panique si `throat < 0`, si `ultimate_strength < 0`, si `correlation_factor <= 0`
/// ou si `gamma_m2 <= 0` (division par zéro).
pub fn steelweld_simplified_resistance_per_length(
    throat: f64,
    ultimate_strength: f64,
    correlation_factor: f64,
    gamma_m2: f64,
) -> f64 {
    assert!(throat >= 0.0, "la gorge a doit être ≥ 0");
    assert!(
        ultimate_strength >= 0.0,
        "la résistance ultime fu doit être ≥ 0"
    );
    assert!(
        correlation_factor > 0.0,
        "le facteur de corrélation βw doit être strictement positif"
    );
    assert!(
        gamma_m2 > 0.0,
        "le coefficient partiel γ_M2 doit être strictement positif"
    );
    throat * ultimate_strength / (core::f64::consts::SQRT_2 * correlation_factor * gamma_m2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn throat_reciprocity() {
        // a = z/√2 : réciproquement a·√2 restitue le côté z du cordon.
        let z = 8.0_f64;
        let a = steelweld_throat(z);
        assert_relative_eq!(a * core::f64::consts::SQRT_2, z, epsilon = 1e-12);
        // Cas connu z = 6 mm → a ≈ 4,2426 mm.
        assert_relative_eq!(steelweld_throat(6.0), 4.242_640_687_119_285, epsilon = 1e-9);
    }

    #[test]
    fn von_mises_pure_components() {
        // σ⊥ seul : σcomp = |σ⊥| (les cisaillements sont nuls).
        assert_relative_eq!(
            steelweld_von_mises_stress(150.0, 0.0, 0.0),
            150.0,
            epsilon = 1e-9
        );
        // τ∥ seul : σcomp = √3·τ∥ (facteur 3 sur le cisaillement).
        assert_relative_eq!(
            steelweld_von_mises_stress(0.0, 0.0, 100.0),
            3.0_f64.sqrt() * 100.0,
            epsilon = 1e-9
        );
        // Insensibilité au signe : σ⊥ = -120 donne le même résultat que +120.
        assert_relative_eq!(
            steelweld_von_mises_stress(-120.0, 0.0, 0.0),
            steelweld_von_mises_stress(120.0, 0.0, 0.0),
            epsilon = 1e-12
        );
    }

    #[test]
    fn von_mises_realistic_case() {
        // Décomposition σ⊥ = 120, τ⊥ = 120, τ∥ = 80 MPa :
        //   σcomp = √(120² + 3·(120² + 80²))
        //         = √(14 400 + 3·(14 400 + 6 400))
        //         = √(14 400 + 3·20 800) = √(14 400 + 62 400) = √76 800 ≈ 277,128 MPa.
        let sigma_comp = steelweld_von_mises_stress(120.0, 120.0, 80.0);
        assert_relative_eq!(sigma_comp, 277.128_129_211_020_4, epsilon = 1e-3);
    }

    #[test]
    fn limit_stress_s235_case() {
        // Nuance S235 : fu = 360 MPa, βw = 0,8, γ_M2 = 1,25 (EN 1993-1-8) :
        //   σlim = 360 / (0,8 · 1,25) = 360 / 1,0 = 360 MPa.
        let sigma_lim = steelweld_limit_stress(360.0, 0.8, 1.25);
        assert_relative_eq!(sigma_lim, 360.0, epsilon = 1e-6);
    }

    #[test]
    fn simplified_resistance_proportional_to_throat() {
        // Fw,Rd/l ∝ a : doubler la gorge double la résistance par unité de longueur.
        let single = steelweld_simplified_resistance_per_length(4.0, 360.0, 0.8, 1.25);
        let double = steelweld_simplified_resistance_per_length(8.0, 360.0, 0.8, 1.25);
        assert_relative_eq!(double, 2.0 * single, epsilon = 1e-9);
    }

    #[test]
    fn simplified_resistance_realistic_case() {
        // Cordon z = 6 mm (a = 6/√2 mm), S235 : fu = 360 MPa, βw = 0,8, γ_M2 = 1,25 :
        //   Fw,Rd/l = a · fu / (√2 · βw · γ_M2)
        //           = (6/√2) · 360 / (√2 · 0,8 · 1,25)
        //           = 6 · 360 / (2 · 0,8 · 1,25) = 2 160 / 2,0 = 1 080 N/mm.
        let a = steelweld_throat(6.0);
        let fvw = steelweld_simplified_resistance_per_length(a, 360.0, 0.8, 1.25);
        assert_relative_eq!(fvw, 1080.0, epsilon = 1e-3);
    }

    #[test]
    #[should_panic(expected = "le coefficient partiel γ_M2 doit être strictement positif")]
    fn zero_gamma_m2_panics() {
        steelweld_simplified_resistance_per_length(4.24, 360.0, 0.8, 0.0);
    }
}
