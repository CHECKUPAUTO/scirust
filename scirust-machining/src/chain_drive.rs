//! Transmissions par **chaîne à rouleaux** — diamètre primitif, vitesse linéaire,
//! rapport de réduction et longueur de chaîne (en pas).
//!
//! ```text
//! diamètre primitif  d = p/sin(π/Z)
//! vitesse de chaîne  v = p·Z·n                 (n en tr/s)
//! rapport            i = Z1/Z2                  (ω2/ω1)
//! longueur (pas)     L = 2C + (Z1+Z2)/2 + ((Z2−Z1)/(2π))²/C
//! ```
//!
//! `p` pas de la chaîne (m), `Z`/`Z1`/`Z2` nombres de dents des pignons, `n`
//! fréquence de rotation (tr/s), `C` entraxe **exprimé en pas**, `L` longueur en
//! nombre de pas (à arrondir au nombre pair supérieur en pratique).
//!
//! **Convention** : SI cohérent, entraxe et longueur en **pas**. **Limite
//! honnête** : géométrie idéale ; l'effet de corde (variation de vitesse aux
//! faibles `Z`), l'allongement et la tension de montage ne sont pas modélisés. Le
//! pas et les dentures sont fournis par l'appelant.

use core::f64::consts::PI;

/// Diamètre primitif d'un pignon de chaîne `d = p/sin(π/Z)` (m).
///
/// Panique si `teeth < 3`.
pub fn sprocket_pitch_diameter(pitch: f64, teeth: u32) -> f64 {
    assert!(teeth >= 3, "un pignon de chaîne a au moins 3 dents");
    pitch / (PI / teeth as f64).sin()
}

/// Vitesse linéaire de la chaîne `v = p·Z·n` (m/s).
pub fn chain_velocity(pitch: f64, teeth: u32, rev_per_s: f64) -> f64 {
    pitch * teeth as f64 * rev_per_s
}

/// Rapport de réduction `i = Z1/Z2` (`ω2/ω1`).
///
/// Panique si `driven_teeth == 0`.
pub fn sprocket_speed_ratio(driver_teeth: u32, driven_teeth: u32) -> f64 {
    assert!(
        driven_teeth > 0,
        "le pignon mené doit avoir au moins une dent"
    );
    driver_teeth as f64 / driven_teeth as f64
}

/// Longueur de chaîne en **pas** `L = 2C + (Z1+Z2)/2 + ((Z2−Z1)/(2π))²/C`.
///
/// Panique si `center_distance_pitches <= 0`.
pub fn chain_length_pitches(teeth1: u32, teeth2: u32, center_distance_pitches: f64) -> f64 {
    assert!(
        center_distance_pitches > 0.0,
        "l'entraxe (en pas) doit être strictement positif"
    );
    let z1 = teeth1 as f64;
    let z2 = teeth2 as f64;
    let diff = (z2 - z1) / (2.0 * PI);
    2.0 * center_distance_pitches + (z1 + z2) / 2.0 + diff * diff / center_distance_pitches
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn pitch_diameter_grows_with_teeth() {
        // Plus de dents → plus grand diamètre primitif (à pas égal).
        let d17 = sprocket_pitch_diameter(0.0127, 17);
        let d38 = sprocket_pitch_diameter(0.0127, 38);
        assert!(d38 > d17);
        assert_relative_eq!(d17, 0.0127 / (PI / 17.0).sin(), epsilon = 1e-12);
    }

    #[test]
    fn chain_velocity_definition() {
        // p=12,7 mm, Z=17, n=25 tr/s → v = 0,0127·17·25.
        assert_relative_eq!(
            chain_velocity(0.0127, 17, 25.0),
            0.0127 * 17.0 * 25.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn reduction_ratio() {
        // 17 → 34 dents : i = 0,5 (réduction ×2).
        assert_relative_eq!(sprocket_speed_ratio(17, 34), 0.5, epsilon = 1e-12);
    }

    #[test]
    fn chain_length_equal_sprockets() {
        // Z1=Z2 : le terme de différence s'annule → L = 2C + Z.
        let l = chain_length_pitches(20, 20, 30.0);
        assert_relative_eq!(l, 2.0 * 30.0 + 20.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "au moins 3 dents")]
    fn too_few_teeth_panics() {
        sprocket_pitch_diameter(0.0127, 2);
    }
}
