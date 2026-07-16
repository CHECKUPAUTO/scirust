//! **Béton armé — ouverture de fissure à l'ELS (Eurocode 2, §7.3.4)** :
//! espacement maximal des fissures `sr,max`, différence de déformation moyenne
//! acier/béton `εsm − εcm`, ouverture de fissure caractéristique `wk` et ratio
//! d'armature effectif `ρp,eff` dans la zone tendue.
//!
//! ```text
//! espacement fissures  sr,max = 3,4 · c + 0,425 · k1 · k2 · φ / ρp,eff
//! diff. déformation    εsm−εcm = max( [σs − kt·(fct,eff/ρp,eff)·(1 + αe·ρp,eff)] / Es ,
//!                                      0,6 · σs / Es )
//! ouverture fissure    wk     = sr,max · (εsm − εcm)
//! ratio effectif       ρp,eff = As / Ac,eff
//! ```
//!
//! `sr,max` espacement maximal des fissures (mm), `c` enrobage des armatures
//! (mm), `k1` coefficient d'adhérence des barres (sans dimension, EC2 : `0,8`
//! barres à haute adhérence, `1,6` barres lisses), `k2` coefficient de
//! répartition des déformations (sans dimension : `0,5` en flexion, `1,0` en
//! traction pure), `φ` diamètre des barres (mm), `ρp,eff` ratio d'armature
//! effectif dans la zone tendue (sans dimension), `εsm − εcm` différence entre
//! déformation moyenne de l'acier et déformation moyenne du béton entre
//! fissures (sans dimension), `σs` contrainte de traction dans l'acier tendu
//! sous la combinaison ELS considérée (MPa), `kt` coefficient dépendant de la
//! durée du chargement (sans dimension, EC2 : `0,6` court terme, `0,4` long
//! terme), `fct,eff` résistance moyenne effective en traction du béton au
//! moment considéré (MPa), `αe = Es/Ecm` rapport modulaire acier/béton (sans
//! dimension), `Es` module d'élasticité de l'acier (MPa), `wk` ouverture de
//! fissure caractéristique (mm), `As` aire d'acier tendu (mm²), `Ac,eff` aire
//! effective de béton tendu entourant les armatures (mm²).
//!
//! **Convention** : N, mm, MPa (avec `1 MPa = 1 N/mm²`). Les contraintes et le
//! module `Es` sont en **MPa**, les longueurs (enrobage, diamètre, espacement,
//! ouverture) en **mm**, les aires en **mm²**, les déformations, ratios et
//! coefficients **sans dimension**.
//!
//! **Limite honnête** : vérification ELS **réglementaire et empirique**. La
//! résistance effective en traction `fct,eff`, la contrainte d'acier `σs`, les
//! coefficients `k1`, `k2`, `kt` et le rapport modulaire `αe` sont **fournis
//! par l'appelant** d'après l'Eurocode 2 et son Annexe Nationale ; aucune valeur
//! « par défaut » n'est inventée. L'aire tendue effective `Ac,eff` (et donc la
//! géométrie de la section) est **fournie**, elle n'est pas calculée ici. La
//! comparaison de `wk` à l'ouverture admissible `wmax` et la conclusion
//! réglementaire restent à la charge de l'ingénieur.

/// Espacement maximal des fissures `sr,max = 3,4 · c + 0,425 · k1 · k2 · φ / ρp,eff`
/// (mm), d'après l'Eurocode 2 §7.3.4 (7.11), avec `c` enrobage (mm), `φ`
/// diamètre des barres (mm), `k1`, `k2` coefficients réglementaires et `ρp,eff`
/// ratio d'armature effectif (sans dimension).
///
/// Panique si `cover < 0`, `bar_diameter <= 0`, `reinforcement_ratio_eff <= 0`,
/// `k1 <= 0` ou `k2 <= 0`.
pub fn crack_max_spacing(
    cover: f64,
    bar_diameter: f64,
    reinforcement_ratio_eff: f64,
    k1: f64,
    k2: f64,
) -> f64 {
    assert!(cover >= 0.0, "l'enrobage cover doit être positif ou nul");
    assert!(
        bar_diameter > 0.0,
        "le diamètre bar_diameter doit être strictement positif"
    );
    assert!(
        reinforcement_ratio_eff > 0.0,
        "le ratio d'armature effectif reinforcement_ratio_eff doit être strictement positif"
    );
    assert!(k1 > 0.0, "le coefficient k1 doit être strictement positif");
    assert!(k2 > 0.0, "le coefficient k2 doit être strictement positif");
    3.4 * cover + 0.425 * k1 * k2 * bar_diameter / reinforcement_ratio_eff
}

/// Différence de déformation moyenne acier/béton
/// `εsm − εcm = max( [σs − kt·(fct,eff/ρp,eff)·(1 + αe·ρp,eff)] / Es , 0,6·σs/Es )`
/// (sans dimension), d'après l'Eurocode 2 §7.3.4 (7.9), avec `σs`
/// (`steel_stress`) et `fct,eff` (`tension_strength_eff`) en MPa, `Es`
/// (`elastic_modulus_steel`) en MPa, `kt`, `αe` (`alpha_e`) et `ρp,eff`
/// (`reinforcement_ratio_eff`) sans dimension. Le second terme `0,6·σs/Es`
/// impose la valeur minimale réglementaire.
///
/// Panique si `steel_stress <= 0`, `kt <= 0`, `tension_strength_eff <= 0`,
/// `reinforcement_ratio_eff <= 0`, `alpha_e <= 0` ou
/// `elastic_modulus_steel <= 0`.
pub fn crack_mean_strain_difference(
    steel_stress: f64,
    kt: f64,
    tension_strength_eff: f64,
    reinforcement_ratio_eff: f64,
    alpha_e: f64,
    elastic_modulus_steel: f64,
) -> f64 {
    assert!(
        steel_stress > 0.0,
        "la contrainte d'acier steel_stress doit être strictement positive"
    );
    assert!(kt > 0.0, "le coefficient kt doit être strictement positif");
    assert!(
        tension_strength_eff > 0.0,
        "la résistance tension_strength_eff doit être strictement positive"
    );
    assert!(
        reinforcement_ratio_eff > 0.0,
        "le ratio d'armature effectif reinforcement_ratio_eff doit être strictement positif"
    );
    assert!(
        alpha_e > 0.0,
        "le rapport modulaire alpha_e doit être strictement positif"
    );
    assert!(
        elastic_modulus_steel > 0.0,
        "le module elastic_modulus_steel doit être strictement positif"
    );
    let main = (steel_stress
        - kt * (tension_strength_eff / reinforcement_ratio_eff)
            * (1.0 + alpha_e * reinforcement_ratio_eff))
        / elastic_modulus_steel;
    let floor = 0.6 * steel_stress / elastic_modulus_steel;
    main.max(floor)
}

/// Ouverture de fissure caractéristique `wk = sr,max · (εsm − εcm)` (mm),
/// d'après l'Eurocode 2 §7.3.4 (7.8), avec `sr,max` (`max_spacing`) en mm et
/// `εsm − εcm` (`mean_strain_difference`) sans dimension.
///
/// Panique si `max_spacing < 0` ou `mean_strain_difference < 0`.
pub fn crack_width(max_spacing: f64, mean_strain_difference: f64) -> f64 {
    assert!(
        max_spacing >= 0.0,
        "l'espacement max_spacing doit être positif ou nul"
    );
    assert!(
        mean_strain_difference >= 0.0,
        "la différence de déformation mean_strain_difference doit être positive ou nulle"
    );
    max_spacing * mean_strain_difference
}

/// Ratio d'armature effectif dans la zone tendue `ρp,eff = As / Ac,eff` (sans
/// dimension), d'après l'Eurocode 2 §7.3.2 (7.10), avec `As` (`steel_area`) et
/// `Ac,eff` (`effective_tension_area`) en mm².
///
/// Panique si `steel_area <= 0` ou `effective_tension_area <= 0`.
pub fn crack_effective_reinforcement_ratio(steel_area: f64, effective_tension_area: f64) -> f64 {
    assert!(
        steel_area > 0.0,
        "l'aire d'acier steel_area doit être strictement positive"
    );
    assert!(
        effective_tension_area > 0.0,
        "l'aire tendue effective effective_tension_area doit être strictement positive"
    );
    steel_area / effective_tension_area
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn max_spacing_clean_case() {
        // Cas chiffré (nombres choisis pour un résultat entier) :
        //   3,4 · c           = 3,4 · 25 = 85
        //   0,425 · k1 · k2   = 0,425 · 0,8 · 0,5 = 0,17
        //   0,17 · φ / ρ      = 0,17 · 16 / 0,02 = 2,72 / 0,02 = 136
        //   sr,max            = 85 + 136 = 221 mm
        let sr = crack_max_spacing(25.0, 16.0, 0.02, 0.8, 0.5);
        assert_relative_eq!(sr, 221.0, epsilon = 1e-9);
    }

    #[test]
    fn max_spacing_second_term_scales_with_diameter() {
        // À enrobage nul, seul le second terme subsiste ; il est linéaire en φ :
        // doubler le diamètre double l'espacement.
        let sr1 = crack_max_spacing(0.0, 16.0, 0.02, 0.8, 0.5);
        let sr2 = crack_max_spacing(0.0, 32.0, 0.02, 0.8, 0.5);
        assert_relative_eq!(sr2 / sr1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn mean_strain_difference_main_branch_case() {
        // Branche principale active (numérateur > plancher) :
        //   kt·(fct/ρ)          = 0,6 · (3,0 / 0,03) = 0,6 · 100 = 60
        //   1 + αe·ρ            = 1 + 6,0 · 0,03 = 1,18
        //   60 · 1,18           = 70,8
        //   σs − 70,8           = 280 − 70,8 = 209,2
        //   / Es                = 209,2 / 200000 = 0,001046
        //   plancher 0,6·σs/Es  = 0,6 · 280 / 200000 = 0,00084 (inférieur)
        let d = crack_mean_strain_difference(280.0, 0.6, 3.0, 0.03, 6.0, 200_000.0);
        assert_relative_eq!(d, 0.001046, epsilon = 1e-9);
    }

    #[test]
    fn mean_strain_difference_floor_branch() {
        // Contrainte d'acier faible : la branche plancher 0,6·σs/Es domine.
        //   numérateur principal = 100 − 70,8 = 29,2 → /Es = 0,000146
        //   plancher             = 0,6 · 100 / 200000 = 0,0003 (supérieur)
        let d = crack_mean_strain_difference(100.0, 0.6, 3.0, 0.03, 6.0, 200_000.0);
        assert_relative_eq!(d, 0.0003, epsilon = 1e-12);
        // Identité de la branche plancher : d = 0,6 · σs / Es.
        assert_relative_eq!(d, 0.6 * 100.0 / 200_000.0, epsilon = 1e-12);
    }

    #[test]
    fn crack_width_composes_and_is_linear() {
        // Identité de composition : wk = sr,max · (εsm − εcm).
        //   sr,max = 200 mm, εsm − εcm = 0,001 → wk = 0,2 mm
        assert_relative_eq!(crack_width(200.0, 0.001), 0.2, epsilon = 1e-12);
        // Un espacement nul donne une ouverture nulle.
        assert_relative_eq!(crack_width(0.0, 0.001), 0.0, epsilon = 1e-12);
        // Linéarité : doubler la déformation double l'ouverture.
        let w1 = crack_width(200.0, 0.001);
        let w2 = crack_width(200.0, 0.002);
        assert_relative_eq!(w2 / w1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn effective_reinforcement_ratio_case() {
        // ρp,eff = As / Ac,eff = 500 / 25000 = 0,02.
        let rho = crack_effective_reinforcement_ratio(500.0, 25_000.0);
        assert_relative_eq!(rho, 0.02, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "effective_tension_area doit être strictement positive")]
    fn effective_reinforcement_ratio_rejects_zero_area() {
        let _ = crack_effective_reinforcement_ratio(500.0, 0.0);
    }
}
