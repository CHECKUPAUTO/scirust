//! Écoulement diphasique gaz–liquide en conduite — corrélation de
//! Lockhart-Martinelli (paramètre `X`, multiplicateur diphasique `φ_L²`,
//! gradient de pression diphasique) et modèle homogène sans glissement
//! (masse volumique du mélange, taux de vide).
//!
//! ```text
//! paramètre de Lockhart-Martinelli
//!   X    = sqrt( (dp/dz)_L / (dp/dz)_G )                        [-]
//! multiplicateur diphasique côté liquide (Chisholm)
//!   φ_L² = 1 + C/X + 1/X²                                       [-]
//! gradient de pression diphasique
//!   (dp/dz)_TP = (dp/dz)_L · φ_L²                               [Pa·m⁻¹]
//! masse volumique du mélange (modèle homogène)
//!   ρ_h  = 1 / [ x/ρ_G + (1 − x)/ρ_L ]                          [kg·m⁻³]
//! taux de vide (modèle homogène, sans glissement)
//!   α    = 1 / [ 1 + ((1 − x)/x)·(ρ_G/ρ_L) ]                    [-]
//! ```
//!
//! `(dp/dz)_L` gradient de pression du liquide s'écoulant **seul** dans la
//! conduite [Pa·m⁻¹], `(dp/dz)_G` gradient de pression du gaz s'écoulant
//! **seul** [Pa·m⁻¹], `X` paramètre de Lockhart-Martinelli [sans dimension],
//! `C` constante de Chisholm [sans dimension], `φ_L²` multiplicateur diphasique
//! rapporté au gradient liquide seul [sans dimension], `(dp/dz)_TP` gradient de
//! pression de l'écoulement diphasique [Pa·m⁻¹], `x` titre massique en vapeur
//! (fraction massique de gaz) [sans dimension, 0 ≤ x ≤ 1], `ρ_L` masse
//! volumique du liquide [kg·m⁻³], `ρ_G` masse volumique du gaz [kg·m⁻³], `ρ_h`
//! masse volumique du mélange homogène [kg·m⁻³], `α` taux de vide (fraction
//! volumique occupée par le gaz) [sans dimension].
//!
//! **Limite honnête** : corrélation **empirique** de Lockhart-Martinelli. Les
//! gradients de pression **monophasiques** `(dp/dz)_L` et `(dp/dz)_G` (liquide
//! seul, gaz seul) sont **FOURNIS** par l'appelant — calculés en amont à partir
//! des débits et propriétés, jamais estimés ici. La constante de Chisholm `C`
//! est **FOURNIE** selon les régimes d'écoulement des deux phases (turbulent–
//! turbulent, laminaire–turbulent, etc.) d'après l'abaque de Chisholm ; elle
//! n'est ni supposée ni inventée. Les masses volumiques `ρ_L` et `ρ_G` sont
//! elles aussi **FOURNIES** : aucune propriété physique n'est calculée ici. Le
//! modèle homogène suppose l'**absence de glissement** entre phases (vitesses
//! égales), hypothèse rarement vérifiée : ce sont des **estimations à forte
//! incertitude**, à recouper avec des données expérimentales ou des méthodes
//! plus fines (Friedel, drift-flux) pour un dimensionnement.

/// Paramètre de Lockhart-Martinelli
/// `X = sqrt( (dp/dz)_L / (dp/dz)_G )` (sans dimension), racine du rapport des
/// gradients de pression du liquide seul et du gaz seul dans la conduite.
///
/// `liquid_pressure_gradient` ((dp/dz)_L) gradient du liquide seul [Pa·m⁻¹],
/// `gas_pressure_gradient` ((dp/dz)_G) gradient du gaz seul [Pa·m⁻¹]. Tous deux
/// exprimés en magnitude (valeur positive).
///
/// Panique si `(dp/dz)_L ≤ 0` ou si `(dp/dz)_G ≤ 0`.
pub fn twop_martinelli_parameter(liquid_pressure_gradient: f64, gas_pressure_gradient: f64) -> f64 {
    assert!(
        liquid_pressure_gradient > 0.0,
        "(dp/dz)_L > 0 requis (gradient du liquide seul)"
    );
    assert!(
        gas_pressure_gradient > 0.0,
        "(dp/dz)_G > 0 requis (gradient du gaz seul)"
    );
    (liquid_pressure_gradient / gas_pressure_gradient).sqrt()
}

/// Multiplicateur diphasique côté liquide (corrélation de Chisholm)
/// `φ_L² = 1 + C/X + 1/X²` (sans dimension), facteur par lequel le gradient de
/// pression du liquide seul est amplifié par la présence du gaz.
///
/// `martinelli_parameter` (X) paramètre de Lockhart-Martinelli [sans dimension],
/// `chisholm_constant` (C) constante de Chisholm [sans dimension], **FOURNIE**
/// selon les régimes des deux phases (ex. 5, 10, 12, 20).
///
/// Panique si `X ≤ 0` ou si `C < 0`.
pub fn twop_two_phase_multiplier_liquid(martinelli_parameter: f64, chisholm_constant: f64) -> f64 {
    assert!(
        martinelli_parameter > 0.0,
        "X > 0 requis (paramètre de Lockhart-Martinelli)"
    );
    assert!(
        chisholm_constant >= 0.0,
        "C ≥ 0 requis (constante de Chisholm)"
    );
    1.0 + chisholm_constant / martinelli_parameter
        + 1.0 / (martinelli_parameter * martinelli_parameter)
}

/// Gradient de pression de l'écoulement diphasique
/// `(dp/dz)_TP = (dp/dz)_L · φ_L²` (Pa·m⁻¹), gradient du liquide seul multiplié
/// par le multiplicateur diphasique côté liquide.
///
/// `liquid_pressure_gradient` ((dp/dz)_L) gradient du liquide seul [Pa·m⁻¹],
/// `two_phase_multiplier_liquid` (φ_L²) multiplicateur diphasique [sans
/// dimension].
///
/// Panique si `(dp/dz)_L ≤ 0` ou si `φ_L² < 1` (le multiplicateur diphasique est
/// toujours supérieur ou égal à 1).
pub fn twop_pressure_gradient(
    liquid_pressure_gradient: f64,
    two_phase_multiplier_liquid: f64,
) -> f64 {
    assert!(
        liquid_pressure_gradient > 0.0,
        "(dp/dz)_L > 0 requis (gradient du liquide seul)"
    );
    assert!(
        two_phase_multiplier_liquid >= 1.0,
        "φ_L² ≥ 1 requis (multiplicateur diphasique)"
    );
    liquid_pressure_gradient * two_phase_multiplier_liquid
}

/// Masse volumique du mélange selon le modèle homogène
/// `ρ_h = 1 / [ x/ρ_G + (1 − x)/ρ_L ]` (kg·m⁻³), moyenne harmonique des masses
/// volumiques pondérée par le titre massique (volume spécifique du mélange).
///
/// `quality` (x) titre massique en vapeur [sans dimension], `liquid_density`
/// (ρ_L) [kg·m⁻³], `gas_density` (ρ_G) [kg·m⁻³].
///
/// Panique si `x` hors de `[0, 1]`, si `ρ_L ≤ 0`, ou si `ρ_G ≤ 0`.
pub fn twop_homogeneous_density(quality: f64, liquid_density: f64, gas_density: f64) -> f64 {
    assert!(
        (0.0..=1.0).contains(&quality),
        "0 ≤ x ≤ 1 requis (titre massique en vapeur)"
    );
    assert!(
        liquid_density > 0.0,
        "ρ_L > 0 requis (masse volumique du liquide)"
    );
    assert!(gas_density > 0.0, "ρ_G > 0 requis (masse volumique du gaz)");
    1.0 / (quality / gas_density + (1.0 - quality) / liquid_density)
}

/// Taux de vide selon le modèle homogène (sans glissement)
/// `α = 1 / [ 1 + ((1 − x)/x)·(ρ_G/ρ_L) ]` (sans dimension), fraction de section
/// occupée par le gaz sous l'hypothèse de vitesses de phases égales.
///
/// `quality` (x) titre massique en vapeur [sans dimension], `liquid_density`
/// (ρ_L) [kg·m⁻³], `gas_density` (ρ_G) [kg·m⁻³].
///
/// Panique si `x` hors de `]0, 1]` (le titre doit être strictement positif : le
/// terme `(1 − x)/x` diverge en `x = 0`), si `ρ_L ≤ 0`, ou si `ρ_G ≤ 0`.
pub fn twop_void_fraction_homogeneous(quality: f64, liquid_density: f64, gas_density: f64) -> f64 {
    assert!(
        quality > 0.0 && quality <= 1.0,
        "0 < x ≤ 1 requis (titre massique en vapeur)"
    );
    assert!(
        liquid_density > 0.0,
        "ρ_L > 0 requis (masse volumique du liquide)"
    );
    assert!(gas_density > 0.0, "ρ_G > 0 requis (masse volumique du gaz)");
    1.0 / (1.0 + ((1.0 - quality) / quality) * (gas_density / liquid_density))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn martinelli_reciprocity_and_unit() {
        // X² = (dp/dz)_L / (dp/dz)_G : le carré redonne le rapport des gradients.
        let x = twop_martinelli_parameter(400.0_f64, 100.0_f64);
        assert_relative_eq!(x * x, 4.0, max_relative = 1e-12);
        // Gradients égaux ⇒ X = 1 (frontière des régimes dominés par une phase).
        assert_relative_eq!(
            twop_martinelli_parameter(250.0_f64, 250.0_f64),
            1.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn multiplier_computed_case_and_limit() {
        // X = 2, C = 20 : φ_L² = 1 + 20/2 + 1/(2·2) = 1 + 10 + 0.25 = 11.25.
        // Recalcul : 20/2 = 10 ; 1/4 = 0.25 ; 1 + 10 + 0.25 = 11.25.
        let phi = twop_two_phase_multiplier_liquid(2.0_f64, 20.0_f64);
        assert_relative_eq!(phi, 11.25, max_relative = 1e-12);
        // Limite X → ∞ : les termes C/X et 1/X² s'annulent ⇒ φ_L² → 1.
        let phi_large = twop_two_phase_multiplier_liquid(1.0e6_f64, 20.0_f64);
        assert_relative_eq!(phi_large, 1.0, max_relative = 1e-4);
    }

    #[test]
    fn pressure_gradient_case_and_proportionality() {
        // (dp/dz)_L = 100 Pa/m, φ_L² = 11.25 ⇒ (dp/dz)_TP = 100·11.25 = 1125 Pa/m.
        let dp_tp = twop_pressure_gradient(100.0_f64, 11.25_f64);
        assert_relative_eq!(dp_tp, 1125.0, max_relative = 1e-12);
        // Linéarité en (dp/dz)_L : doubler le gradient liquide double le résultat.
        let dp_tp2 = twop_pressure_gradient(200.0_f64, 11.25_f64);
        assert_relative_eq!(dp_tp2, 2.0 * dp_tp, max_relative = 1e-12);
        // φ_L² = 1 (pas de gaz) ⇒ le gradient diphasique se réduit au liquide seul.
        assert_relative_eq!(
            twop_pressure_gradient(100.0_f64, 1.0_f64),
            100.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn homogeneous_density_case_and_extremes() {
        // x = 0.1, ρ_L = 1000, ρ_G = 5 :
        //   x/ρ_G = 0.1/5 = 0.02 ; (1−x)/ρ_L = 0.9/1000 = 0.0009
        //   somme = 0.0209 ; ρ_h = 1/0.0209 ≈ 47.84689 kg/m³.
        // Recalcul : 0.02 + 0.0009 = 0.0209 ; 1/0.0209 = 47.846890.
        let rho_h = twop_homogeneous_density(0.1_f64, 1000.0_f64, 5.0_f64);
        assert_relative_eq!(rho_h, 47.84689, max_relative = 1e-3);
        // x = 0 ⇒ ρ_h = ρ_L (liquide pur) ; x = 1 ⇒ ρ_h = ρ_G (gaz pur).
        assert_relative_eq!(
            twop_homogeneous_density(0.0_f64, 1000.0_f64, 5.0_f64),
            1000.0,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            twop_homogeneous_density(1.0_f64, 1000.0_f64, 5.0_f64),
            5.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn void_fraction_case_and_density_identity() {
        // x = 0.1, ρ_L = 1000, ρ_G = 5 :
        //   (1−x)/x = 0.9/0.1 = 9 ; ρ_G/ρ_L = 5/1000 = 0.005
        //   9·0.005 = 0.045 ; α = 1/(1+0.045) = 1/1.045 ≈ 0.9569378.
        // Recalcul : 9·0.005 = 0.045 ; 1/1.045 = 0.95693780.
        let alpha = twop_void_fraction_homogeneous(0.1_f64, 1000.0_f64, 5.0_f64);
        assert_relative_eq!(alpha, 0.9569378, max_relative = 1e-3);
        // Identité modèle homogène : ρ_h = ρ_G·α + ρ_L·(1 − α).
        let rho_h = twop_homogeneous_density(0.1_f64, 1000.0_f64, 5.0_f64);
        assert_relative_eq!(
            5.0 * alpha + 1000.0 * (1.0 - alpha),
            rho_h,
            max_relative = 1e-9
        );
        // x = 1 (gaz pur) ⇒ α = 1 (toute la section occupée par le gaz).
        assert_relative_eq!(
            twop_void_fraction_homogeneous(1.0_f64, 1000.0_f64, 5.0_f64),
            1.0,
            max_relative = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "0 < x ≤ 1 requis")]
    fn void_fraction_panics_on_zero_quality() {
        // x = 0 : le terme (1 − x)/x diverge ⇒ entrée rejetée.
        let _ = twop_void_fraction_homogeneous(0.0_f64, 1000.0_f64, 5.0_f64);
    }
}
