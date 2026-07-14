//! Joints toriques (**O-rings**) en **étanchéité statique** — ratios de montage
//! dans la gorge : écrasement, étirement et taux de remplissage.
//!
//! ```text
//! écrasement (%)   Sq = (Dc − Hg) / Dc · 100          (Dc = Ø de tore, Hg = profondeur de gorge)
//! étirement  (%)   St = (Dg − Did) / Did · 100        (Dg = Ø int. de gorge, Did = Ø int. joint)
//! remplissage(%)   Fg = A_o / A_g · 100               (A_o = section joint, A_g = section gorge)
//! ```
//!
//! `Dc` diamètre de tore (cordon) du joint (m), `Hg` profondeur de gorge (m),
//! `Dg` diamètre intérieur de la gorge (m), `Did` diamètre intérieur libre du
//! joint torique (m), `A_o` aire de la section du tore (m²), `A_g` aire de la
//! section de la gorge (m²). Les trois grandeurs sont retournées en **pourcent**.
//!
//! **Convention** : SI cohérent (longueurs en m, aires en m²) ; il suffit que
//! numérateur et dénominateur partagent la même unité pour un ratio correct.
//! **Limite honnête** : étanchéité **statique** seulement, élastomère supposé
//! **quasi-incompressible** (le remplissage laisse une marge de dilatation, d'où
//! une cible usuelle < 90 %). Les fourchettes cibles — écrasement typique 15–30 %,
//! remplissage < 90 % — sont des **repères de conception fournis par l'appelant**
//! et ne sont pas imposées ici ; aucune valeur matériau n'est inventée.

/// Taux d'**écrasement** `Sq = (Dc − Hg)/Dc · 100` (%), compression radiale du tore.
///
/// Panique si `cord_diameter <= 0`.
pub fn oring_squeeze_ratio(cord_diameter: f64, groove_depth: f64) -> f64 {
    assert!(
        cord_diameter > 0.0,
        "le diamètre de tore Dc doit être strictement positif"
    );
    assert!(
        groove_depth >= 0.0,
        "la profondeur de gorge Hg ne peut être négative"
    );
    (cord_diameter - groove_depth) / cord_diameter * 100.0
}

/// Taux d'**étirement** `St = (Dg − Did)/Did · 100` (%) du joint sur sa gorge.
///
/// Panique si `oring_id <= 0`.
pub fn oring_stretch_ratio(groove_id: f64, oring_id: f64) -> f64 {
    assert!(
        oring_id > 0.0,
        "le diamètre intérieur du joint Did doit être strictement positif"
    );
    assert!(
        groove_id >= 0.0,
        "le diamètre intérieur de gorge Dg ne peut être négatif"
    );
    (groove_id - oring_id) / oring_id * 100.0
}

/// Taux de **remplissage** de gorge `Fg = A_o/A_g · 100` (%).
///
/// Panique si `groove_section_area <= 0` ou `oring_section_area < 0`.
pub fn gland_fill_percent(oring_section_area: f64, groove_section_area: f64) -> f64 {
    assert!(
        groove_section_area > 0.0,
        "l'aire de section de gorge A_g doit être strictement positive"
    );
    assert!(
        oring_section_area >= 0.0,
        "l'aire de section du joint A_o ne peut être négative"
    );
    oring_section_area / groove_section_area * 100.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn squeeze_zero_depth_means_full_cord() {
        // Gorge de profondeur nulle : le tore n'est pas écrasé du tout → 100 %.
        assert_relative_eq!(oring_squeeze_ratio(0.0035, 0.0035), 0.0, epsilon = 1e-9);
        assert_relative_eq!(oring_squeeze_ratio(0.0035, 0.0), 100.0, epsilon = 1e-9);
    }

    #[test]
    fn squeeze_realistic_value_in_typical_band() {
        // Cordon 3,53 mm dans une gorge de 2,80 mm → écrasement ≈ 20,7 % (bande 15–30 %).
        let sq = oring_squeeze_ratio(0.00353, 0.00280);
        assert_relative_eq!(
            sq,
            (0.00353 - 0.00280) / 0.00353 * 100.0,
            max_relative = 1e-12
        );
        assert!(sq > 15.0 && sq < 30.0);
    }

    #[test]
    fn stretch_zero_when_groove_matches_oring() {
        // Gorge au même diamètre que le joint libre : aucun étirement.
        assert_relative_eq!(oring_stretch_ratio(0.050, 0.050), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn stretch_scales_inversely_with_oring_id() {
        // Même surcote absolue sur un joint deux fois plus petit → étirement doublé.
        let big = oring_stretch_ratio(0.0510, 0.0500);
        let small = oring_stretch_ratio(0.0260, 0.0250);
        assert_relative_eq!(small, 2.0 * big, max_relative = 1e-12);
    }

    #[test]
    fn fill_is_ratio_of_areas() {
        // Section joint = 80 % de la section de gorge → remplissage 80 % (< 90 %).
        let fg = gland_fill_percent(8.0e-6, 1.0e-5);
        assert_relative_eq!(fg, 80.0, max_relative = 1e-12);
        assert!(fg < 90.0);
    }

    #[test]
    fn fill_proportional_to_oring_area() {
        // Le remplissage est linéaire en l'aire du tore, à gorge fixée.
        let f1 = gland_fill_percent(4.0e-6, 1.0e-5);
        let f2 = gland_fill_percent(8.0e-6, 1.0e-5);
        assert_relative_eq!(f2, 2.0 * f1, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "A_g doit être strictement positive")]
    fn zero_groove_area_panics() {
        gland_fill_percent(5.0e-6, 0.0);
    }
}
