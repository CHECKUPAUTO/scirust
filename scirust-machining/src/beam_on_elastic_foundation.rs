//! Poutre sur **fondation élastique** — modèle de **Winkler** : cas de la
//! poutre infinie sous charge ponctuelle.
//!
//! ```text
//! facteur caractéristique     β = (k/(4·E·I))^(1/4)        (1/m)
//! longueur caractéristique    L_c = 1/β                    (m)
//! flèche max (charge P)       δ_max = P·β/(2·k)            (m)
//! moment max (charge P)       M_max = P/(4·β)              (N·m)
//! ```
//!
//! `k` module de réaction de la fondation (N/m² = N·m⁻¹ par mètre de longueur),
//! `E` module de Young de la poutre (Pa = N/m²), `I` moment quadratique de la
//! section (m⁴), `β` facteur caractéristique (1/m), `L_c` longueur
//! caractéristique (m), `P` charge ponctuelle (N), `δ_max` flèche maximale sous
//! la charge (m), `M_max` moment fléchissant maximal sous la charge (N·m).
//!
//! **Convention** : SI cohérent, flèches et charges comptées positives dans le
//! sens de la charge. **Limite honnête** : modèle de **Winkler** (fondation =
//! ressorts **indépendants**, sans continuité du sol — ni Pasternak ni
//! élasticité de milieu continu), poutre supposée **infinie**, comportement
//! **élastique linéaire**, petites déformations. Le module de réaction `k`
//! (procédé/sol), le module `E` (matériau) et le moment quadratique `I`
//! (géométrie de section) sont **fournis par l'appelant** ; aucune valeur
//! « par défaut » de matériau, de sol ou de procédé n'est inventée ici.

/// Facteur caractéristique de la poutre sur fondation élastique
/// `β = (k/(4·E·I))^(1/4)` (1/m).
///
/// Panique si `foundation_modulus <= 0`, `youngs_modulus <= 0`
/// ou `second_moment <= 0`.
pub fn bef_characteristic_factor(
    foundation_modulus: f64,
    youngs_modulus: f64,
    second_moment: f64,
) -> f64 {
    assert!(
        foundation_modulus > 0.0,
        "le module de réaction k doit être strictement positif"
    );
    assert!(
        youngs_modulus > 0.0,
        "le module de Young E doit être strictement positif"
    );
    assert!(
        second_moment > 0.0,
        "le moment quadratique I doit être strictement positif"
    );
    (foundation_modulus / (4.0 * youngs_modulus * second_moment)).powf(0.25)
}

/// Longueur caractéristique de la poutre sur fondation élastique
/// `L_c = 1/β` (m).
///
/// Panique si `characteristic_factor <= 0`.
pub fn bef_characteristic_length(characteristic_factor: f64) -> f64 {
    assert!(
        characteristic_factor > 0.0,
        "le facteur caractéristique β doit être strictement positif"
    );
    1.0 / characteristic_factor
}

/// Flèche maximale sous une charge ponctuelle, poutre infinie
/// `δ_max = P·β/(2·k)` (m). Maximum atteint au droit de la charge.
///
/// Panique si `characteristic_factor <= 0` ou `foundation_modulus <= 0`.
pub fn bef_max_deflection_point_load(
    point_load: f64,
    characteristic_factor: f64,
    foundation_modulus: f64,
) -> f64 {
    assert!(
        characteristic_factor > 0.0,
        "le facteur caractéristique β doit être strictement positif"
    );
    assert!(
        foundation_modulus > 0.0,
        "le module de réaction k doit être strictement positif"
    );
    point_load * characteristic_factor / (2.0 * foundation_modulus)
}

/// Moment fléchissant maximal sous une charge ponctuelle, poutre infinie
/// `M_max = P/(4·β)` (N·m). Maximum atteint au droit de la charge.
///
/// Panique si `characteristic_factor <= 0`.
pub fn bef_max_moment_point_load(point_load: f64, characteristic_factor: f64) -> f64 {
    assert!(
        characteristic_factor > 0.0,
        "le facteur caractéristique β doit être strictement positif"
    );
    point_load / (4.0 * characteristic_factor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Jeu d'essai réaliste et « rond » : acier E = 200 GPa, section
    // I = 1·10⁻⁶ m⁴, module de réaction k = 1.28·10⁷ N/m².
    // 4·E·I = 8.0·10⁵, k/(4·E·I) = 16, donc β = 16^(1/4) = 2.0 exactement.
    const E: f64 = 2.0e11_f64;
    const I: f64 = 1.0e-6_f64;
    const K: f64 = 1.28e7_f64;

    #[test]
    fn characteristic_factor_valeur_ronde() {
        // β = (1.28e7 / (4·2e11·1e-6))^(1/4) = 16^(1/4) = 2.0.
        let beta = bef_characteristic_factor(K, E, I);
        assert_relative_eq!(beta, 2.0, max_relative = 1e-12);
    }

    #[test]
    fn characteristic_factor_echelle_quart() {
        // β ∝ k^(1/4) : multiplier k par 16 double β.
        let beta = bef_characteristic_factor(K, E, I);
        let beta16 = bef_characteristic_factor(16.0 * K, E, I);
        assert_relative_eq!(beta16, 2.0 * beta, max_relative = 1e-12);
    }

    #[test]
    fn length_reciproque_du_facteur() {
        // Aller-retour : 1/(1/β) = β.
        let beta = bef_characteristic_factor(K, E, I);
        let l_c = bef_characteristic_length(beta);
        assert_relative_eq!(l_c, 0.5, max_relative = 1e-12);
        assert_relative_eq!(bef_characteristic_length(l_c), beta, max_relative = 1e-12);
    }

    #[test]
    fn deflection_cas_chiffre() {
        // δ = P·β/(2·k) = 1e5·2 / (2·1.28e7) = 2e5/2.56e7 = 7.8125e-3 m.
        let beta = bef_characteristic_factor(K, E, I);
        let delta = bef_max_deflection_point_load(1.0e5, beta, K);
        assert_relative_eq!(delta, 7.8125e-3, max_relative = 1e-12);
    }

    #[test]
    fn moment_cas_chiffre_et_proportionnalite() {
        // M = P/(4·β) = 1e5/(4·2) = 12500 N·m ; linéaire en P.
        let beta = bef_characteristic_factor(K, E, I);
        let m = bef_max_moment_point_load(1.0e5, beta);
        assert_relative_eq!(m, 12500.0, max_relative = 1e-12);
        let m2 = bef_max_moment_point_load(2.0e5, beta);
        assert_relative_eq!(m2, 2.0 * m, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "le module de réaction k doit être strictement positif")]
    fn characteristic_factor_panique_k_negatif() {
        let _ = bef_characteristic_factor(-1.0, E, I);
    }
}
