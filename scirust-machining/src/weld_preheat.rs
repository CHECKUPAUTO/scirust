//! **Température de préchauffage en soudage** — méthode empirique de Séférian
//! reliant le carbone équivalent et l'épaisseur à la température de préchauffage.
//!
//! ```text
//! carbone équivalent corrigé   CEt = CE·(1 + 0,005·e)
//! température de préchauffage   Tp  = 350·√(CEt − 0,25)      (°C, si CEt > 0,25)
//! épaisseur combinée           ec  = e1 + e2                (mm)
//! préchauffage requis          CE > seuil                   (booléen)
//! ```
//!
//! `CE` carbone équivalent chimique (sans dimension), `e` épaisseur soudée (mm),
//! `CEt` carbone équivalent corrigé de l'épaisseur (sans dimension), `Tp`
//! température de préchauffage (°C), `ec` épaisseur combinée de deux tôles (mm),
//! `seuil` valeur de CE au-delà de laquelle un préchauffage est jugé nécessaire.
//!
//! **Convention** : épaisseurs en mm, température en °C, carbone équivalent sans
//! dimension. **Limite honnête** : la relation `Tp = 350·√(CEt − 0,25)` est une
//! **formule empirique de Séférian** dont les coefficients fixes (350, 0,25,
//! 0,005/mm) font partie du modèle lui-même ; le carbone équivalent `CE` et les
//! épaisseurs sont **fournis par l'appelant** (aucune composition d'acier ni
//! valeur « par défaut » n'est inventée). Le résultat est une **borne
//! indicative** : l'apport d'hydrogène du métal d'apport, le procédé, le bridage,
//! l'apport de chaleur et la vitesse de refroidissement ne sont **pas** pris en
//! compte. Voir [`crate::weld_heat_input`] et [`crate::welds`].

/// Température de préchauffage `Tp = 350·√(CEt − 0,25)` (°C), avec correction
/// d'épaisseur de Séférian `CEt = CE·(1 + 0,005·e)`.
///
/// `carbon_equivalent` est le carbone équivalent chimique `CE` (sans dimension),
/// `thickness` l'épaisseur soudée `e` en mm.
///
/// Panique si `carbon_equivalent <= 0`, si `thickness < 0`, ou si le carbone
/// équivalent corrigé `CEt` ne dépasse pas 0,25 (hors du domaine de la formule
/// de Séférian, où la racine deviendrait négative).
pub fn preheat_from_carbon_equivalent(carbon_equivalent: f64, thickness: f64) -> f64 {
    assert!(
        carbon_equivalent > 0.0,
        "le carbone équivalent CE doit être strictement positif"
    );
    assert!(
        thickness >= 0.0,
        "l'épaisseur e doit être positive ou nulle"
    );
    let corrected = carbon_equivalent * (1.0 + 0.005 * thickness);
    assert!(
        corrected > 0.25,
        "le carbone équivalent corrigé CEt doit dépasser 0,25 (domaine de la formule de Séférian)"
    );
    350.0 * (corrected - 0.25).sqrt()
}

/// Épaisseur combinée `ec = e1 + e2` (mm), somme des épaisseurs des deux tôles
/// contribuant à l'évacuation de la chaleur du joint.
///
/// Panique si `thickness1 < 0` ou `thickness2 < 0`.
pub fn preheat_combined_thickness(thickness1: f64, thickness2: f64) -> f64 {
    assert!(
        thickness1 >= 0.0 && thickness2 >= 0.0,
        "les épaisseurs e1 et e2 doivent être positives ou nulles"
    );
    thickness1 + thickness2
}

/// Indique si un préchauffage est requis, c.-à-d. si `CE > seuil` (booléen).
///
/// `carbon_equivalent` est le carbone équivalent `CE` (sans dimension) et
/// `threshold` le seuil de décision (typiquement 0,4, à fournir par l'appelant).
///
/// Panique si `carbon_equivalent < 0` ou `threshold < 0`.
pub fn preheat_is_required(carbon_equivalent: f64, threshold: f64) -> bool {
    assert!(
        carbon_equivalent >= 0.0 && threshold >= 0.0,
        "le carbone équivalent CE et le seuil doivent être positifs ou nuls"
    );
    carbon_equivalent > threshold
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn reference_point_gives_round_temperature() {
        // CEt = 0,5 → Tp = 350·√(0,5 − 0,25) = 350·√0,25 = 350·0,5 = 175 °C.
        // Ici CE = 0,5 et e = 0 → CEt = CE = 0,5.
        assert_relative_eq!(
            preheat_from_carbon_equivalent(0.5, 0.0),
            175.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn thickness_correction_reaches_same_corrected_ce() {
        // La correction d'épaisseur agit multiplicativement : CE = 0,4 avec
        // e = 50 mm donne CEt = 0,4·(1 + 0,005·50) = 0,4·1,25 = 0,5, soit le
        // même CEt (et donc le même Tp = 175 °C) que CE = 0,5 sans épaisseur.
        assert_relative_eq!(
            preheat_from_carbon_equivalent(0.4, 50.0),
            preheat_from_carbon_equivalent(0.5, 0.0),
            epsilon = 1e-9
        );
    }

    #[test]
    fn preheat_increases_with_thickness() {
        // À CE fixé, augmenter l'épaisseur augmente CEt donc Tp (monotonie).
        let thin = preheat_from_carbon_equivalent(0.45, 10.0);
        let thick = preheat_from_carbon_equivalent(0.45, 40.0);
        assert!(thick > thin);
    }

    #[test]
    fn realistic_carbon_steel_case() {
        // Acier au carbone : CE = 0,45 ; e = 30 mm.
        // CEt = 0,45·(1 + 0,005·30) = 0,45·1,15 = 0,5175.
        // Tp = 350·√(0,5175 − 0,25) = 350·√0,2675 ≈ 181,0214 °C.
        assert_relative_eq!(
            preheat_from_carbon_equivalent(0.45, 30.0),
            181.021_407_573_8,
            epsilon = 1e-6
        );
    }

    #[test]
    fn combined_thickness_is_symmetric_sum() {
        // ec = e1 + e2, commutatif et égal à la somme arithmétique.
        assert_relative_eq!(
            preheat_combined_thickness(12.0, 8.0),
            preheat_combined_thickness(8.0, 12.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(preheat_combined_thickness(12.0, 8.0), 20.0, epsilon = 1e-12);
    }

    #[test]
    fn requirement_brackets_the_threshold() {
        // Le préchauffage est requis strictement au-dessus du seuil.
        assert!(preheat_is_required(0.45, 0.4));
        assert!(!preheat_is_required(0.35, 0.4));
        assert!(!preheat_is_required(0.4, 0.4));
    }

    #[test]
    #[should_panic(expected = "domaine de la formule de Séférian")]
    fn low_carbon_equivalent_panics() {
        // CEt = 0,20 ≤ 0,25 : hors domaine, la racine serait négative.
        preheat_from_carbon_equivalent(0.2, 0.0);
    }
}
