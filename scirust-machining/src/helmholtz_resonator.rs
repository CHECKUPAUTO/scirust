//! **Résonateur de Helmholtz** (acoustique) — fréquence propre d'oscillation
//! du bouchon d'air d'un col sur la raideur pneumatique d'une cavité fermée.
//!
//! ```text
//! fréquence propre       f = (c / (2·π)) · sqrt(S / (V · L))
//! longueur effective     L = L0 + 1.7 · r        (col à bride, un côté)
//! cavité réciproque      V = S · (c / (2·π·f))² / L
//! ```
//!
//! `f` fréquence de résonance (Hz), `c` célérité du son dans le fluide du col
//! (m/s), `S` aire de la section du col (m²), `V` volume de la cavité (m³),
//! `L` longueur **effective** (acoustique) du col (m), `L0` longueur physique
//! du col (m), `r` rayon du col (m). La correction d'embouchure `1.7·r`
//! correspond à un col cylindrique **bridé d'un seul côté** (bride côté cavité,
//! extrémité libre côté extérieur) ; ajuster le coefficient pour d'autres
//! géométries d'extrémités.
//!
//! **Convention** : SI cohérent (m, m², m³, m/s, Hz). **Limite honnête** :
//! cavité à **parois rigides** grande devant le col, longueur d'onde **très
//! supérieure** à toutes les dimensions (régime basses fréquences, air supposé
//! incompressible dans le col et purement compressible dans la cavité), pertes
//! **visqueuses et thermiques négligées**, rayonnement négligé. La célérité du
//! son `c` est une **donnée de l'appelant** (elle dépend du fluide et de la
//! température) et n'est jamais supposée par le module.

use core::f64::consts::PI;

/// Fréquence de résonance de Helmholtz `f = (c/(2·π))·sqrt(S/(V·L))` (Hz).
///
/// Fréquence propre du bouchon d'air du col oscillant sur la raideur de l'air
/// de la cavité. Croît avec la section du col, décroît quand la cavité ou la
/// longueur du col augmentent.
///
/// Panique si `speed_of_sound <= 0`, `neck_area <= 0`, `cavity_volume <= 0`
/// ou `neck_effective_length <= 0`.
pub fn helmholtz_resonant_frequency(
    speed_of_sound: f64,
    neck_area: f64,
    cavity_volume: f64,
    neck_effective_length: f64,
) -> f64 {
    assert!(speed_of_sound > 0.0, "la célérité du son c doit être > 0");
    assert!(neck_area > 0.0, "l'aire du col S doit être > 0");
    assert!(
        cavity_volume > 0.0,
        "le volume de la cavité V doit être > 0"
    );
    assert!(
        neck_effective_length > 0.0,
        "la longueur effective du col L doit être > 0"
    );
    (speed_of_sound / (2.0 * PI)) * (neck_area / (cavity_volume * neck_effective_length)).sqrt()
}

/// Longueur effective du col `L = L0 + 1.7·r` (m), correction d'embouchure.
///
/// Ajoute à la longueur physique la masse d'air rayonnée aux extrémités
/// (col bridé d'un seul côté). C'est cette longueur acoustique qui intervient
/// dans [`helmholtz_resonant_frequency`].
///
/// Panique si `physical_length < 0` ou `neck_radius < 0`.
pub fn helmholtz_effective_length(physical_length: f64, neck_radius: f64) -> f64 {
    assert!(
        physical_length >= 0.0,
        "la longueur physique du col L0 doit être ≥ 0"
    );
    assert!(neck_radius >= 0.0, "le rayon du col r doit être ≥ 0");
    physical_length + 1.7 * neck_radius
}

/// Volume de cavité pour une fréquence visée `V = S·(c/(2·π·f))²/L` (m³).
///
/// Réciproque de [`helmholtz_resonant_frequency`] : donne le volume qui place
/// la résonance à `target_frequency`, à col (section et longueur effective) et
/// fluide (`c`) fixés.
///
/// Panique si `target_frequency <= 0`, `speed_of_sound <= 0`, `neck_area <= 0`
/// ou `neck_effective_length <= 0`.
pub fn helmholtz_cavity_volume_for_frequency(
    target_frequency: f64,
    speed_of_sound: f64,
    neck_area: f64,
    neck_effective_length: f64,
) -> f64 {
    assert!(target_frequency > 0.0, "la fréquence visée f doit être > 0");
    assert!(speed_of_sound > 0.0, "la célérité du son c doit être > 0");
    assert!(neck_area > 0.0, "l'aire du col S doit être > 0");
    assert!(
        neck_effective_length > 0.0,
        "la longueur effective du col L doit être > 0"
    );
    let ratio = speed_of_sound / (2.0 * PI * target_frequency);
    neck_area * ratio * ratio / neck_effective_length
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn frequency_reciprocity_with_volume() {
        // Réciprocité : le volume calculé pour f0 redonne bien f0.
        let c = 343.0_f64;
        let s = 5.0e-4_f64;
        let l = 0.04_f64;
        let f0 = 150.0_f64;
        let v = helmholtz_cavity_volume_for_frequency(f0, c, s, l);
        let f = helmholtz_resonant_frequency(c, s, v, l);
        assert_relative_eq!(f, f0, epsilon = 1e-9);
    }

    #[test]
    fn frequency_proportional_to_speed_of_sound() {
        // f ∝ c à géométrie fixée : doubler c double la fréquence.
        let s = 5.0e-4_f64;
        let v = 2.0e-3_f64;
        let l = 0.04_f64;
        let f1 = helmholtz_resonant_frequency(340.0, s, v, l);
        let f2 = helmholtz_resonant_frequency(680.0, s, v, l);
        assert_relative_eq!(f2, 2.0 * f1, epsilon = 1e-9);
    }

    #[test]
    fn frequency_scales_as_inverse_sqrt_volume() {
        // f ∝ 1/sqrt(V) : quadrupler le volume divise la fréquence par deux.
        let c = 343.0_f64;
        let s = 5.0e-4_f64;
        let l = 0.04_f64;
        let f1 = helmholtz_resonant_frequency(c, s, 1.0e-3, l);
        let f4 = helmholtz_resonant_frequency(c, s, 4.0e-3, l);
        assert_relative_eq!(f1, 2.0 * f4, epsilon = 1e-9);
    }

    #[test]
    fn resonant_frequency_realistic_case() {
        // Air à 20 °C : c = 343 m/s, col S = 5e-4 m², cavité V = 2e-3 m³,
        // longueur effective L = 0,04 m.
        // S/(V·L) = 5e-4/(2e-3·0,04) = 5e-4/8e-5 = 6,25 ; sqrt = 2,5.
        // f = (343/(2π))·2,5 = 857,5/6,283185307 = 136,475364 Hz.
        let f = helmholtz_resonant_frequency(343.0, 5.0e-4, 2.0e-3, 0.04);
        assert_relative_eq!(f, 136.475_364, epsilon = 1e-4);
    }

    #[test]
    fn effective_length_adds_end_correction() {
        // L = L0 + 1,7·r : L0 = 0,03 m, r = 0,01 m → 0,03 + 0,017 = 0,047 m.
        let l = helmholtz_effective_length(0.03, 0.01);
        assert_relative_eq!(l, 0.047, epsilon = 1e-12);
        // Rayon nul : la longueur effective se réduit à la longueur physique.
        assert_relative_eq!(helmholtz_effective_length(0.03, 0.0), 0.03, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "le volume de la cavité V doit être > 0")]
    fn zero_volume_panics() {
        helmholtz_resonant_frequency(343.0, 5.0e-4, 0.0, 0.04);
    }
}
