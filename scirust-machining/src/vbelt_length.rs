//! Longueur d'une courroie sur deux poulies — **montage ouvert** (poulies
//! tournant dans le même sens) et **montage croisé** (sens inversé) — et angles
//! d'enroulement du montage ouvert.
//!
//! ```text
//! montage ouvert    L_o = 2·C + (π/2)·(D1 + D2) + (D2 − D1)² / (4·C)
//! montage croisé    L_x = 2·C + (π/2)·(D1 + D2) + (D1 + D2)² / (4·C)
//! enroulement petit β_s = π − 2·asin((D_g − D_p) / (2·C))
//! enroulement grand β_g = π + 2·asin((D_g − D_p) / (2·C))
//! ```
//!
//! `L_o`/`L_x` longueur de la courroie (m), `C` entraxe (m), `D1`/`D2` diamètres
//! des poulies (m), `D_p`/`D_g` diamètres de la petite et de la grande poulie (m),
//! `β_s`/`β_g` angles d'enroulement sur la petite et la grande poulie (rad). En
//! montage ouvert `β_s + β_g = 2π`.
//!
//! **Convention** : SI cohérent (mètres, radians), types `f64`. **Limite
//! honnête** : géométrie plane idéale, brins rectilignes tangents et courroie
//! d'épaisseur négligeable ; l'effet de la tension, du fluage, de la précharge et
//! de l'épaisseur réelle de la courroie n'est pas modélisé. L'entraxe et les
//! diamètres sont **fournis par l'appelant** — aucune valeur « par défaut » n'est
//! inventée.

use core::f64::consts::PI;

/// Longueur d'une courroie en **montage ouvert**
/// `L_o = 2·C + (π/2)·(D1 + D2) + (D2 − D1)² / (4·C)` (m).
///
/// La formule est symétrique en `D1`/`D2` (le terme au carré ne dépend que de
/// leur différence), l'ordre des poulies est donc indifférent.
///
/// `center_distance` entraxe, `pulley_diameter_1`/`pulley_diameter_2` diamètres
/// des poulies (tous en mètres).
///
/// Panique si `center_distance <= 0` ou si un diamètre est négatif.
pub fn open_belt_length(
    center_distance: f64,
    pulley_diameter_1: f64,
    pulley_diameter_2: f64,
) -> f64 {
    assert!(
        center_distance > 0.0,
        "l'entraxe doit être strictement positif"
    );
    assert!(
        pulley_diameter_1 >= 0.0 && pulley_diameter_2 >= 0.0,
        "les diamètres des poulies ne peuvent pas être négatifs"
    );
    let diff = pulley_diameter_2 - pulley_diameter_1;
    2.0 * center_distance
        + (PI / 2.0) * (pulley_diameter_1 + pulley_diameter_2)
        + diff * diff / (4.0 * center_distance)
}

/// Longueur d'une courroie en **montage croisé**
/// `L_x = 2·C + (π/2)·(D1 + D2) + (D1 + D2)² / (4·C)` (m).
///
/// Le croisement remplace la différence des diamètres par leur somme dans le
/// terme correctif : la courroie croisée est toujours plus longue que la courroie
/// ouverte de mêmes poulies.
///
/// `center_distance` entraxe, `pulley_diameter_1`/`pulley_diameter_2` diamètres
/// des poulies (tous en mètres).
///
/// Panique si `center_distance <= 0` ou si un diamètre est négatif.
pub fn crossed_belt_length(
    center_distance: f64,
    pulley_diameter_1: f64,
    pulley_diameter_2: f64,
) -> f64 {
    assert!(
        center_distance > 0.0,
        "l'entraxe doit être strictement positif"
    );
    assert!(
        pulley_diameter_1 >= 0.0 && pulley_diameter_2 >= 0.0,
        "les diamètres des poulies ne peuvent pas être négatifs"
    );
    let sum = pulley_diameter_1 + pulley_diameter_2;
    2.0 * center_distance + (PI / 2.0) * sum + sum * sum / (4.0 * center_distance)
}

/// Angle d'enroulement sur la **petite poulie** d'un montage ouvert
/// `β_s = π − 2·asin((D_g − D_p) / (2·C))` (rad).
///
/// `small_pulley_diameter` diamètre de la petite poulie `D_p`,
/// `large_pulley_diameter` diamètre de la grande poulie `D_g`,
/// `center_distance` entraxe `C` (tous en mètres).
///
/// Panique si `center_distance <= 0`, si `large_pulley_diameter <
/// small_pulley_diameter`, ou si `(D_g − D_p) > 2·C` (poulies qui se
/// chevaucheraient : brins non tangents).
pub fn open_belt_wrap_angle_small(
    small_pulley_diameter: f64,
    large_pulley_diameter: f64,
    center_distance: f64,
) -> f64 {
    assert!(
        center_distance > 0.0,
        "l'entraxe doit être strictement positif"
    );
    assert!(
        large_pulley_diameter >= small_pulley_diameter && small_pulley_diameter >= 0.0,
        "le grand diamètre doit être supérieur ou égal au petit, tous deux positifs"
    );
    let ratio = (large_pulley_diameter - small_pulley_diameter) / (2.0 * center_distance);
    assert!(
        ratio <= 1.0,
        "la différence des diamètres dépasse 2·C : les brins ne sont plus tangents"
    );
    PI - 2.0 * ratio.asin()
}

/// Angle d'enroulement sur la **grande poulie** d'un montage ouvert
/// `β_g = π + 2·asin((D_g − D_p) / (2·C))` (rad).
///
/// Complément du précédent : `β_s + β_g = 2π`.
///
/// `small_pulley_diameter` diamètre de la petite poulie `D_p`,
/// `large_pulley_diameter` diamètre de la grande poulie `D_g`,
/// `center_distance` entraxe `C` (tous en mètres).
///
/// Panique si `center_distance <= 0`, si `large_pulley_diameter <
/// small_pulley_diameter`, ou si `(D_g − D_p) > 2·C` (brins non tangents).
pub fn open_belt_wrap_angle_large(
    small_pulley_diameter: f64,
    large_pulley_diameter: f64,
    center_distance: f64,
) -> f64 {
    assert!(
        center_distance > 0.0,
        "l'entraxe doit être strictement positif"
    );
    assert!(
        large_pulley_diameter >= small_pulley_diameter && small_pulley_diameter >= 0.0,
        "le grand diamètre doit être supérieur ou égal au petit, tous deux positifs"
    );
    let ratio = (large_pulley_diameter - small_pulley_diameter) / (2.0 * center_distance);
    assert!(
        ratio <= 1.0,
        "la différence des diamètres dépasse 2·C : les brins ne sont plus tangents"
    );
    PI + 2.0 * ratio.asin()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn open_length_symmetric_in_diameters() {
        // La longueur ouverte est symétrique : échanger D1 et D2 ne change rien.
        let l_ab = open_belt_length(0.250, 0.060, 0.120);
        let l_ba = open_belt_length(0.250, 0.120, 0.060);
        assert_relative_eq!(l_ab, l_ba, epsilon = 1e-12);
    }

    #[test]
    fn equal_pulleys_open_reduces_to_two_c_plus_pi_d() {
        // Poulies identiques (D1 = D2 = D) : le terme (D2−D1)² disparaît et
        // (π/2)(D+D) = π·D → L = 2C + π·D.
        let c = 0.300_f64;
        let d = 0.080_f64;
        let l = open_belt_length(c, d, d);
        assert_relative_eq!(l, 2.0 * c + PI * d, epsilon = 1e-12);
    }

    #[test]
    fn crossed_longer_than_open_by_diameter_product_term() {
        // Écart exact courroie croisée − ouverte : (D1+D2)² − (D2−D1)² = 4·D1·D2,
        // divisé par 4C → D1·D2 / C.
        let c = 0.400_f64;
        let d1 = 0.090_f64;
        let d2 = 0.150_f64;
        let diff = crossed_belt_length(c, d1, d2) - open_belt_length(c, d1, d2);
        assert_relative_eq!(diff, d1 * d2 / c, epsilon = 1e-12);
    }

    #[test]
    fn wrap_angles_sum_to_full_turn() {
        // Réciprocité géométrique du montage ouvert : β_s + β_g = 2π.
        let dp = 0.070_f64;
        let dg = 0.200_f64;
        let c = 0.500_f64;
        let bs = open_belt_wrap_angle_small(dp, dg, c);
        let bg = open_belt_wrap_angle_large(dp, dg, c);
        assert_relative_eq!(bs + bg, 2.0 * PI, epsilon = 1e-12);
    }

    #[test]
    fn equal_pulleys_wrap_is_half_turn() {
        // Poulies identiques (D_p = D_g) : asin(0) = 0 → chaque enroulement vaut π.
        let d = 0.100_f64;
        let c = 0.350_f64;
        assert_relative_eq!(open_belt_wrap_angle_small(d, d, c), PI, epsilon = 1e-12);
        assert_relative_eq!(open_belt_wrap_angle_large(d, d, c), PI, epsilon = 1e-12);
    }

    #[test]
    fn wrap_small_matches_closed_form_case() {
        // Cas chiffré : D_g − D_p = C → asin(0,5) = π/6 → β_s = π − π/3 = 2π/3.
        let dp = 0.100_f64;
        let dg = 0.400_f64;
        let c = 0.300_f64; // D_g − D_p = 0,300 = C, donc ratio = 0,5
        assert_relative_eq!(
            open_belt_wrap_angle_small(dp, dg, c),
            2.0 * PI / 3.0,
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "l'entraxe doit être strictement positif")]
    fn zero_center_distance_panics() {
        open_belt_length(0.0, 0.060, 0.120);
    }
}
