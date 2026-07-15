//! Grippage d'engrenage — critère de **température de contact** de Blok, via la
//! **température éclair** (flash temperature) au contact des dentures.
//!
//! ```text
//! T_flash   = 1,11 · mu · w · |v_sliding| / (Bm · √a)
//! T_contact = T_bulk + T_flash
//! S_scuff   = T_scuff / T_contact
//! ```
//!
//! avec :
//! - `mu` coefficient de frottement au contact (sans dimension) ;
//! - `w` charge linéique normale (N/m) rapportée à la largeur de contact ;
//! - `v_sliding` vitesse de glissement au point considéré (m/s) ;
//! - `Bm` coefficient de contact thermique du couple de matériaux
//!   (W·s^0,5·m^-2·K^-1), `Bm = √(λ·ρ·c)` combiné des deux corps ;
//! - `a` demi-largeur de contact de Hertz (m) ;
//! - `T_flash` élévation de température éclair (K) ;
//! - `T_bulk` température de masse (volumique) de la denture (K ou °C) ;
//! - `T_contact` température de contact totale (même échelle que `T_bulk`) ;
//! - `T_scuff` température limite de grippage (même échelle) ;
//! - `S_scuff` coefficient de sécurité au grippage (sans dimension).
//!
//! **Limite honnête** : le critère de température de contact (Blok) suppose
//! FOURNIS par l'appelant le coefficient de frottement `mu`, la charge linéique
//! `w`, la vitesse de glissement, les propriétés thermiques (`Bm`) et la
//! géométrie de contact (`a`) issues d'une analyse de denture (p. ex. Hertz).
//! La température limite de grippage `T_scuff` dépend du **lubrifiant** et de
//! l'essai (FZG, etc.) : elle est FOURNIE, jamais inventée. Ce module est
//! distinct de [`crate::iso6336`], qui traite la pression de contact (pitting).

/// Élévation de **température éclair** de Blok (K) :
/// `T_flash = 1,11·mu·w·|v_sliding|/(Bm·√a)`.
///
/// `friction_coefficient` = `mu` (sans dimension), `normal_load_per_width` =
/// `w` (N/m), `sliding_velocity` = `v_sliding` (m/s, signe indifférent),
/// `thermal_contact_coefficient` = `Bm` (W·s^0,5·m^-2·K^-1),
/// `hertzian_half_width` = `a` (m).
///
/// Panique si `mu` < 0, si `w` < 0, si `Bm` ≤ 0 ou si `a` ≤ 0.
pub fn gearscuff_flash_temperature(
    friction_coefficient: f64,
    normal_load_per_width: f64,
    sliding_velocity: f64,
    thermal_contact_coefficient: f64,
    hertzian_half_width: f64,
) -> f64 {
    assert!(
        friction_coefficient >= 0.0,
        "le coefficient de frottement doit être positif ou nul"
    );
    assert!(
        normal_load_per_width >= 0.0,
        "la charge linéique doit être positive ou nulle"
    );
    assert!(
        thermal_contact_coefficient > 0.0,
        "le coefficient de contact thermique doit être strictement positif"
    );
    assert!(
        hertzian_half_width > 0.0,
        "la demi-largeur de contact doit être strictement positive"
    );
    1.11 * friction_coefficient * normal_load_per_width * sliding_velocity.abs()
        / (thermal_contact_coefficient * hertzian_half_width.sqrt())
}

/// **Température de contact** totale de Blok (K ou °C) :
/// `T_contact = T_bulk + T_flash`.
///
/// `bulk_temperature` = `T_bulk` température de masse de la denture,
/// `flash_temperature` = `T_flash` élévation éclair (K), dans la même échelle.
///
/// Panique si `flash_temperature` < 0 (une élévation éclair est positive ou nulle).
pub fn gearscuff_contact_temperature(bulk_temperature: f64, flash_temperature: f64) -> f64 {
    assert!(
        flash_temperature >= 0.0,
        "l'élévation de température éclair doit être positive ou nulle"
    );
    bulk_temperature + flash_temperature
}

/// **Coefficient de sécurité au grippage** (sans dimension) :
/// `S_scuff = T_scuff / T_contact`.
///
/// `scuffing_temperature` = `T_scuff` température limite de grippage (fournie,
/// dépend du lubrifiant), `contact_temperature` = `T_contact` température de
/// contact de service, dans la même échelle absolue.
///
/// Panique si `scuffing_temperature` ≤ 0 ou si `contact_temperature` ≤ 0.
pub fn gearscuff_safety_factor(scuffing_temperature: f64, contact_temperature: f64) -> f64 {
    assert!(
        scuffing_temperature > 0.0,
        "la température limite de grippage doit être strictement positive"
    );
    assert!(
        contact_temperature > 0.0,
        "la température de contact doit être strictement positive"
    );
    scuffing_temperature / contact_temperature
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn flash_temperature_matches_the_formula() {
        // mu=0,05 ; w=2,0e4 N/m ; v=5 m/s ; Bm=1,0e4 ; a=1,0e-4 m.
        // 1,11·0,05·2e4·5 = 5550 ; Bm·√a = 1e4·0,01 = 100 → 55,5 K.
        let (mu, w, v, bm, a) = (0.05_f64, 2.0e4_f64, 5.0_f64, 1.0e4_f64, 1.0e-4_f64);
        let t = gearscuff_flash_temperature(mu, w, v, bm, a);
        let expected = 1.11 * mu * w * v.abs() / (bm * a.sqrt());
        assert_relative_eq!(t, expected, epsilon = 1e-9);
        assert_relative_eq!(t, 55.5, epsilon = 1e-9);
    }

    #[test]
    fn flash_temperature_is_insensitive_to_sliding_direction() {
        // La valeur absolue rend T_flash indépendante du signe de v.
        let forward = gearscuff_flash_temperature(0.06, 3.0e4, 4.0, 1.2e4, 2.0e-4);
        let backward = gearscuff_flash_temperature(0.06, 3.0e4, -4.0, 1.2e4, 2.0e-4);
        assert_relative_eq!(forward, backward, epsilon = 1e-12);
    }

    #[test]
    fn flash_temperature_scales_linearly_with_load_and_speed() {
        // T_flash ∝ w·|v| : doubler l'un ou l'autre double l'élévation.
        let base = gearscuff_flash_temperature(0.05, 2.0e4, 5.0, 1.0e4, 1.0e-4);
        let double_load = gearscuff_flash_temperature(0.05, 4.0e4, 5.0, 1.0e4, 1.0e-4);
        let double_speed = gearscuff_flash_temperature(0.05, 2.0e4, 10.0, 1.0e4, 1.0e-4);
        assert_relative_eq!(double_load, 2.0 * base, epsilon = 1e-9);
        assert_relative_eq!(double_speed, 2.0 * base, epsilon = 1e-9);
    }

    #[test]
    fn contact_temperature_adds_bulk_and_flash() {
        // T_bulk=80 K ; T_flash=55,5 K → T_contact=135,5 K.
        let flash = gearscuff_flash_temperature(0.05, 2.0e4, 5.0, 1.0e4, 1.0e-4);
        assert_relative_eq!(
            gearscuff_contact_temperature(80.0, flash),
            135.5,
            epsilon = 1e-9
        );
        // Sans glissement, T_flash=0 → T_contact = T_bulk.
        let no_slide = gearscuff_flash_temperature(0.05, 2.0e4, 0.0, 1.0e4, 1.0e-4);
        assert_relative_eq!(
            gearscuff_contact_temperature(80.0, no_slide),
            80.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn safety_factor_is_scuffing_over_contact() {
        // T_scuff=200 ; T_contact=135,5 → S = 1,47601...
        let s = gearscuff_safety_factor(200.0, 135.5);
        assert_relative_eq!(s, 200.0 / 135.5, epsilon = 1e-12);
        // À T_contact = T_scuff, la sécurité vaut exactement 1.
        assert_relative_eq!(gearscuff_safety_factor(150.0, 150.0), 1.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "coefficient de contact thermique")]
    fn zero_thermal_contact_coefficient_panics() {
        gearscuff_flash_temperature(0.05, 2.0e4, 5.0, 0.0, 1.0e-4);
    }
}
