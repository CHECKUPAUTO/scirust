//! Usinage — **rectification plane** : modèle cinématique du débit de matière,
//! débit spécifique, épaisseur de copeau équivalente et rapport de rectification.
//!
//! ```text
//! débit de matière         Qw  = b·ae·vw
//! débit spécifique         Q'  = ae·vw
//! épaisseur équivalente    heq = vw·ae/vs
//! rapport de rectification G   = Vw/Vs
//! ```
//!
//! `b` largeur rectifiée (mm), `ae` profondeur de passe (mm), `vw` vitesse de la
//! table / pièce (mm/s), `vs` vitesse périphérique de la meule (mm/s), `Qw` débit
//! volumique de matière (mm³/s), `Q'` débit spécifique par unité de largeur
//! (mm²/s), `heq` épaisseur de copeau équivalente (mm), `Vw` volume de matière
//! enlevé (mm³), `Vs` volume de meule usé (mm³), `G` rapport de rectification
//! (sans dimension). L'épaisseur équivalente `heq` représente l'épaisseur d'un
//! ruban continu qui s'écoulerait à la vitesse `vs` pour le même débit spécifique.
//!
//! **Convention** : unités cohérentes de fiche (mm, mm/s) ; `G` sans dimension.
//! **Limite honnête** : modèle **cinématique** idéalisé (contact rectiligne,
//! vitesses supposées constantes) ; ne traite ni les forces, ni la thermique, ni
//! l'usure réelle de la meule. Le rapport de rectification `G` et les vitesses
//! `vw`, `vs` sont **fournis par l'appelant** (mesurés ou issus d'essais) : aucune
//! valeur « par défaut » de matériau ou d'abrasif n'est inventée ici.

/// Débit volumique de matière `Qw = b·ae·vw` (mm³/s).
///
/// Panique si l'un des arguments est négatif.
pub fn grinding_material_removal_rate(width: f64, depth_of_cut: f64, table_speed: f64) -> f64 {
    assert!(width >= 0.0, "la largeur rectifiée b doit être positive");
    assert!(
        depth_of_cut >= 0.0,
        "la profondeur de passe ae doit être positive"
    );
    assert!(
        table_speed >= 0.0,
        "la vitesse de table vw doit être positive"
    );
    width * depth_of_cut * table_speed
}

/// Débit spécifique par unité de largeur `Q' = ae·vw` (mm²/s).
///
/// Panique si l'un des arguments est négatif.
pub fn specific_removal_rate(depth_of_cut: f64, table_speed: f64) -> f64 {
    assert!(
        depth_of_cut >= 0.0,
        "la profondeur de passe ae doit être positive"
    );
    assert!(
        table_speed >= 0.0,
        "la vitesse de table vw doit être positive"
    );
    depth_of_cut * table_speed
}

/// Épaisseur de copeau équivalente `heq = vw·ae/vs` (mm).
///
/// Panique si `vs <= 0` ou si `vw`/`ae` est négatif.
pub fn equivalent_chip_thickness(table_speed: f64, depth_of_cut: f64, wheel_speed: f64) -> f64 {
    assert!(
        wheel_speed > 0.0,
        "la vitesse de meule vs doit être strictement positive"
    );
    assert!(
        table_speed >= 0.0,
        "la vitesse de table vw doit être positive"
    );
    assert!(
        depth_of_cut >= 0.0,
        "la profondeur de passe ae doit être positive"
    );
    table_speed * depth_of_cut / wheel_speed
}

/// Rapport de rectification `G = Vw/Vs` (sans dimension).
///
/// Panique si `wheel_wear_volume <= 0` ou si `volume_removed` est négatif.
pub fn grinding_ratio(volume_removed: f64, wheel_wear_volume: f64) -> f64 {
    assert!(
        wheel_wear_volume > 0.0,
        "le volume de meule usé Vs doit être strictement positif"
    );
    assert!(
        volume_removed >= 0.0,
        "le volume de matière enlevé Vw doit être positif"
    );
    volume_removed / wheel_wear_volume
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn removal_rate_factors_as_width_times_specific() {
        // Identité : Qw = b·Q' (le débit total est la largeur × débit spécifique).
        let b = 20.0;
        let ae = 0.03;
        let vw = 200.0;
        let qw = grinding_material_removal_rate(b, ae, vw);
        let q_prime = specific_removal_rate(ae, vw);
        assert_relative_eq!(qw, b * q_prime, epsilon = 1e-12);
    }

    #[test]
    fn specific_rate_realistic_value() {
        // Cas chiffré : ae = 0,02 mm, vw = 250 mm/s → Q' = 5 mm²/s.
        assert_relative_eq!(specific_removal_rate(0.02, 250.0), 5.0, epsilon = 1e-12);
    }

    #[test]
    fn equivalent_thickness_scales_with_speed_ratio() {
        // heq = Q'/vs : proportionnel à vw·ae et inversement à vs.
        // vw=300, ae=0,04, vs=30000 → heq = 300·0,04/30000 = 4e-4 mm.
        assert_relative_eq!(
            equivalent_chip_thickness(300.0, 0.04, 30000.0),
            4.0e-4,
            epsilon = 1e-15
        );
        // Doubler la vitesse de meule divise heq par deux.
        let heq1 = equivalent_chip_thickness(300.0, 0.04, 30000.0);
        let heq2 = equivalent_chip_thickness(300.0, 0.04, 60000.0);
        assert_relative_eq!(heq2, heq1 / 2.0, epsilon = 1e-15);
    }

    #[test]
    fn equivalent_thickness_conserves_specific_rate() {
        // Conservation du débit : Q' = heq·vs (ruban équivalent à la vitesse vs).
        let vw = 180.0;
        let ae = 0.05;
        let vs = 25000.0;
        let heq = equivalent_chip_thickness(vw, ae, vs);
        assert_relative_eq!(heq * vs, specific_removal_rate(ae, vw), epsilon = 1e-9);
    }

    #[test]
    fn grinding_ratio_is_volume_quotient() {
        // G = Vw/Vs : 6000 mm³ enlevés pour 20 mm³ d'usure → G = 300.
        assert_relative_eq!(grinding_ratio(6000.0, 20.0), 300.0, epsilon = 1e-12);
        // Volume enlevé nul → G = 0 (aucune matière, meule intacte comptée).
        assert_relative_eq!(grinding_ratio(0.0, 20.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "vitesse de meule vs")]
    fn zero_wheel_speed_panics() {
        equivalent_chip_thickness(300.0, 0.04, 0.0);
    }
}
