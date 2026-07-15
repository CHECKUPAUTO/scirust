//! Fabrication additive (impression 3D) — nombre de couches, débit de dépôt et
//! temps de construction idéalisés.
//!
//! ```text
//! nombre de couches   n  = ceil(H / t)
//! débit volumique     Q  = t · h · v
//! temps de dépôt      T  = V / Q
//! ```
//!
//! `H` hauteur de la pièce (m), `t` épaisseur de couche (m), `n` nombre entier
//! de couches (—), `h` espacement des cordons (« hatch spacing », m), `v`
//! vitesse de balayage (m/s), `Q` débit volumique de matière déposée (m³/s),
//! `V` volume de matière à déposer (m³), `T` temps de dépôt (s).
//!
//! **Convention** : SI cohérent (mètres, secondes), toutes grandeurs en f64.
//! **Limite honnête** : temps de dépôt **idéalisé**, hors préchauffe, recoating
//! (étalement de poudre), temps morts inter-couches, supports et rendement
//! machine. Le débit volumique `Q` (ou, à défaut, l'épaisseur de couche,
//! l'espacement des cordons et la vitesse de balayage propres au procédé/à la
//! machine) est **fourni par l'appelant** — aucune valeur « par défaut » n'est
//! inventée ici.

/// Nombre de couches nécessaires `n = ceil(H / t)` (—).
///
/// Panique si `part_height < 0` ou si `layer_thickness <= 0`.
pub fn am_number_of_layers(part_height: f64, layer_thickness: f64) -> u32 {
    assert!(
        part_height >= 0.0,
        "la hauteur de la pièce doit être positive"
    );
    assert!(
        layer_thickness > 0.0,
        "l'épaisseur de couche doit être strictement positive"
    );
    (part_height / layer_thickness).ceil() as u32
}

/// Débit volumique de dépôt `Q = t · h · v` (m³/s).
///
/// Panique si `layer_thickness <= 0`, `hatch_spacing <= 0` ou `scan_speed < 0`.
pub fn am_deposition_rate(layer_thickness: f64, hatch_spacing: f64, scan_speed: f64) -> f64 {
    assert!(
        layer_thickness > 0.0,
        "l'épaisseur de couche doit être strictement positive"
    );
    assert!(
        hatch_spacing > 0.0,
        "l'espacement des cordons doit être strictement positif"
    );
    assert!(
        scan_speed >= 0.0,
        "la vitesse de balayage doit être positive"
    );
    layer_thickness * hatch_spacing * scan_speed
}

/// Temps de dépôt idéalisé `T = V / Q` (s).
///
/// Panique si `part_volume < 0` ou si `volumetric_rate <= 0`.
pub fn am_build_time(part_volume: f64, volumetric_rate: f64) -> f64 {
    assert!(
        part_volume >= 0.0,
        "le volume de la pièce doit être positif"
    );
    assert!(
        volumetric_rate > 0.0,
        "le débit volumique doit être strictement positif"
    );
    part_volume / volumetric_rate
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn number_of_layers_rounds_up() {
        // Hauteur non multiple de l'épaisseur : arrondi supérieur.
        // 10 mm / 0,3 mm = 33,33… → 34 couches.
        assert_eq!(am_number_of_layers(0.010, 0.000_3), 34);
    }

    #[test]
    fn number_of_layers_exact_division() {
        // Division exacte : 1,0 mm / 0,05 mm = 20 couches, sans couche en trop.
        assert_eq!(am_number_of_layers(0.001, 0.000_05), 20);
    }

    #[test]
    fn deposition_rate_is_product_of_factors() {
        // Q = t·h·v ; multiplier la vitesse par k multiplie le débit par k.
        let q1 = am_deposition_rate(0.000_04, 0.000_1, 0.8);
        let q2 = am_deposition_rate(0.000_04, 0.000_1, 1.6);
        assert_relative_eq!(q2 / q1, 2.0, epsilon = 1e-12);
        // Cas chiffré : 40 µm · 100 µm · 0,8 m/s = 3,2e-9 m³/s.
        assert_relative_eq!(q1, 3.2e-9, epsilon = 1e-21);
    }

    #[test]
    fn build_time_is_volume_over_rate() {
        // T = V/Q, réciproque : V = Q·T.
        let v = 1.0e-5_f64;
        let q = 3.2e-9_f64;
        let t = am_build_time(v, q);
        assert_relative_eq!(q * t, v, epsilon = 1e-18);
        // Cas chiffré : 1e-5 / 3,2e-9 = 3125 s.
        assert_relative_eq!(t, 3125.0, epsilon = 1e-6);
    }

    #[test]
    fn build_time_is_inversely_proportional_to_rate() {
        // À volume fixé, doubler le débit divise le temps par deux.
        let t1 = am_build_time(2.0e-6, 5.0e-9);
        let t2 = am_build_time(2.0e-6, 1.0e-8);
        assert_relative_eq!(t1 / t2, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn zero_height_needs_no_layer() {
        // Cas limite : pièce de hauteur nulle → aucune couche.
        assert_eq!(am_number_of_layers(0.0, 0.000_1), 0);
    }

    #[test]
    #[should_panic(expected = "épaisseur de couche")]
    fn zero_layer_thickness_panics() {
        am_number_of_layers(0.010, 0.0);
    }
}
