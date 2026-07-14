//! Courroies **crantées (synchrones)** — nombre de dents de la courroie, dents en
//! prise sur la petite poulie et longueur primitive d'un montage à deux poulies.
//!
//! ```text
//! nombre de dents      Nb = round(L / p)
//! dents en prise       Nm = floor(z · β / (2π))
//! longueur primitive   L  = 2·C + (π/2)·(d1 + d2) + (d2 − d1)² / (4·C)
//! ```
//!
//! `L` longueur primitive de la courroie (m), `p` pas de la courroie (m), `Nb`
//! nombre entier de dents de la courroie, `z` nombre de dents de la petite poulie,
//! `β` angle d'enroulement sur cette poulie (rad), `Nm` nombre entier de dents en
//! prise, `C` entraxe (m), `d1`/`d2` diamètres primitifs des poulies (m).
//!
//! **Convention** : SI cohérent (mètres, radians). **Limite honnête** :
//! engrènement idéal sans glissement (courroie inextensible, denture parfaitement
//! conjuguée) ; les longueurs doivent être cohérentes entre elles (même pas
//! primitif) ; l'effet de la tension, du fluage et de la précharge n'est pas
//! modélisé. Le pas, les dentures et la géométrie du montage sont **fournis par
//! l'appelant** — aucune valeur « par défaut » n'est inventée.

use core::f64::consts::PI;

/// Nombre entier de dents d'une courroie crantée `Nb = round(L / p)`.
///
/// `pitch_length` et `belt_pitch` en mètres (même pas primitif).
///
/// Panique si `belt_pitch <= 0` ou `pitch_length <= 0`.
pub fn timing_belt_teeth(pitch_length: f64, belt_pitch: f64) -> u32 {
    assert!(
        belt_pitch > 0.0,
        "le pas de la courroie doit être strictement positif"
    );
    assert!(
        pitch_length > 0.0,
        "la longueur primitive doit être strictement positive"
    );
    (pitch_length / belt_pitch).round() as u32
}

/// Nombre entier de dents en prise sur la petite poulie
/// `Nm = floor(z · β / (2π))`.
///
/// `wrap_angle_rad` angle d'enroulement (rad), au plus `2π` (poulie entière).
///
/// Panique si `wrap_angle_rad < 0` ou `wrap_angle_rad > 2π`.
pub fn belt_teeth_in_mesh(small_pulley_teeth: u32, wrap_angle_rad: f64) -> u32 {
    assert!(
        wrap_angle_rad >= 0.0,
        "l'angle d'enroulement doit être positif ou nul"
    );
    assert!(
        wrap_angle_rad <= 2.0 * PI,
        "l'angle d'enroulement ne peut pas dépasser 2π (une poulie entière)"
    );
    (small_pulley_teeth as f64 * wrap_angle_rad / (2.0 * PI)).floor() as u32
}

/// Longueur primitive d'une courroie sur deux poulies
/// `L = 2·C + (π/2)·(d1 + d2) + (d2 − d1)² / (4·C)` (m).
///
/// `center_distance` entraxe, `pitch_diam_1`/`pitch_diam_2` diamètres primitifs
/// (tous en mètres).
///
/// Panique si `center_distance <= 0` ou si un diamètre est négatif.
pub fn belt_pitch_length_two_pulley(
    center_distance: f64,
    pitch_diam_1: f64,
    pitch_diam_2: f64,
) -> f64 {
    assert!(
        center_distance > 0.0,
        "l'entraxe doit être strictement positif"
    );
    assert!(
        pitch_diam_1 >= 0.0 && pitch_diam_2 >= 0.0,
        "les diamètres primitifs ne peuvent pas être négatifs"
    );
    let diff = pitch_diam_2 - pitch_diam_1;
    2.0 * center_distance
        + (PI / 2.0) * (pitch_diam_1 + pitch_diam_2)
        + diff * diff / (4.0 * center_distance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn teeth_count_is_exact_multiple() {
        // Une longueur multiple exact du pas donne le nombre de dents attendu.
        // p = 5 mm, L = 5 mm × 120 = 0,600 m → 120 dents.
        assert_eq!(timing_belt_teeth(0.600, 0.005), 120);
    }

    #[test]
    fn teeth_count_rounds_to_nearest() {
        // L = 0,6024 m, p = 5 mm → 120,48 → arrondi à 120 dents.
        assert_eq!(timing_belt_teeth(0.6024, 0.005), 120);
        // L = 0,6026 m → 120,52 → arrondi à 121 dents.
        assert_eq!(timing_belt_teeth(0.6026, 0.005), 121);
    }

    #[test]
    fn half_wrap_engages_half_the_teeth() {
        // Enroulement de 180° (β = π) sur une poulie de 20 dents → 10 dents en prise.
        assert_eq!(belt_teeth_in_mesh(20, PI), 10);
        // Enroulement complet (β = 2π) → toutes les dents.
        assert_eq!(belt_teeth_in_mesh(20, 2.0 * PI), 20);
    }

    #[test]
    fn mesh_is_proportional_to_wrap() {
        // À denture fixe, doubler l'angle double (au plancher près) les dents en prise.
        let z = 40;
        let quarter = belt_teeth_in_mesh(z, PI / 2.0);
        let half = belt_teeth_in_mesh(z, PI);
        assert_eq!(quarter, 10);
        assert_eq!(half, 20);
    }

    #[test]
    fn equal_pulleys_reduce_to_open_belt() {
        // Poulies identiques (d1 = d2 = d) : le terme (d2−d1)² disparaît et
        // (π/2)(d+d) = π·d → L = 2C + π·d.
        let c = 0.300_f64;
        let d = 0.080_f64;
        let l = belt_pitch_length_two_pulley(c, d, d);
        assert_relative_eq!(l, 2.0 * c + PI * d, epsilon = 1e-12);
    }

    #[test]
    fn pitch_length_symmetric_in_diameters() {
        // La longueur est symétrique : échanger d1 et d2 ne change rien.
        let l_ab = belt_pitch_length_two_pulley(0.250, 0.060, 0.120);
        let l_ba = belt_pitch_length_two_pulley(0.250, 0.120, 0.060);
        assert_relative_eq!(l_ab, l_ba, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "pas de la courroie doit être strictement positif")]
    fn zero_pitch_panics() {
        timing_belt_teeth(0.600, 0.0);
    }
}
