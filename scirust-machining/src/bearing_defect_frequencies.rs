//! **Fréquences de défaut de roulement** — fréquences cinématiques
//! caractéristiques (BPFO, BPFI, BSF, FTF) utilisées en analyse vibratoire pour
//! localiser un défaut sur bague externe, bague interne, bille ou cage.
//!
//! ```text
//! passage bille bague ext.  BPFO = (n/2)·fr·(1 − (d/D)·cos α)
//! passage bille bague int.  BPFI = (n/2)·fr·(1 + (d/D)·cos α)
//! fréquence de bille        BSF  = (D/(2·d))·fr·(1 − ((d/D)·cos α)²)
//! fréquence de cage         FTF  = (fr/2)·(1 − (d/D)·cos α)
//! ```
//!
//! `fr` fréquence de rotation de l'arbre (bague interne tournante, bague externe
//! fixe) (Hz), `n` nombre de billes (—), `d` diamètre des billes (m), `D`
//! diamètre primitif du roulement (m), `α` angle de contact (rad), toutes les
//! fréquences de sortie en Hz. Seul le rapport `d/D` intervient, donc `d` et `D`
//! peuvent être fournis dans n'importe quelle unité de longueur commune.
//!
//! **Convention** : SI, fréquences en Hz, angle en rad. **Limite honnête** :
//! modèle cinématique d'un roulement à billes **idéal** en **roulement pur sans
//! glissement**, bague interne tournante et bague externe fixe. La géométrie
//! (`n`, `d`, `D`, `α`) est **fournie par le catalogue/constructeur** ; aucune
//! valeur n'est inventée par défaut. Le glissement réel des billes (typiquement
//! 1 à 2 %), les effets de charge/précharge sur l'angle de contact effectif, et
//! les modulations d'amplitude ne sont **pas** modélisés : ces formules donnent
//! les fréquences **théoriques** de répétition du choc, pas leur amplitude. Voir
//! [`crate::bearings`] et [`crate::bearing_preload`].

use core::f64::consts::FRAC_PI_2;

/// Valide la géométrie commune `(fr, n, d, D, α)` d'un roulement à billes.
///
/// Panique si `shaft_frequency < 0`, `ball_count < 1`, `ball_diameter <= 0`,
/// `ball_diameter >= pitch_diameter`, ou `|contact_angle| >= π/2`.
fn assert_bearing_geometry(
    shaft_frequency: f64,
    ball_count: f64,
    ball_diameter: f64,
    pitch_diameter: f64,
    contact_angle: f64,
) {
    assert!(
        shaft_frequency >= 0.0,
        "la fréquence de rotation fr doit être positive"
    );
    assert!(
        ball_count >= 1.0,
        "le nombre de billes n doit valoir au moins 1"
    );
    assert!(
        ball_diameter > 0.0,
        "le diamètre des billes d doit être strictement positif"
    );
    assert!(
        ball_diameter < pitch_diameter,
        "le diamètre des billes d doit être inférieur au diamètre primitif D"
    );
    assert!(
        contact_angle.abs() < FRAC_PI_2,
        "l'angle de contact α doit être strictement compris entre −π/2 et π/2"
    );
}

/// Fréquence de passage des billes sur la bague **externe** (BPFO) :
/// `BPFO = (n/2)·fr·(1 − (d/D)·cos α)`.
///
/// Fréquence de répétition du choc d'une bille sur un défaut localisé de la
/// piste **externe** (fixe).
///
/// Panique si la géométrie est invalide (voir `assert_bearing_geometry`).
pub fn bearing_bpfo(
    shaft_frequency: f64,
    ball_count: f64,
    ball_diameter: f64,
    pitch_diameter: f64,
    contact_angle: f64,
) -> f64 {
    assert_bearing_geometry(
        shaft_frequency,
        ball_count,
        ball_diameter,
        pitch_diameter,
        contact_angle,
    );
    let ratio = (ball_diameter / pitch_diameter) * contact_angle.cos();
    0.5_f64 * ball_count * shaft_frequency * (1.0 - ratio)
}

/// Fréquence de passage des billes sur la bague **interne** (BPFI) :
/// `BPFI = (n/2)·fr·(1 + (d/D)·cos α)`.
///
/// Fréquence de répétition du choc d'une bille sur un défaut localisé de la
/// piste **interne** (tournante).
///
/// Panique si la géométrie est invalide (voir `assert_bearing_geometry`).
pub fn bearing_bpfi(
    shaft_frequency: f64,
    ball_count: f64,
    ball_diameter: f64,
    pitch_diameter: f64,
    contact_angle: f64,
) -> f64 {
    assert_bearing_geometry(
        shaft_frequency,
        ball_count,
        ball_diameter,
        pitch_diameter,
        contact_angle,
    );
    let ratio = (ball_diameter / pitch_diameter) * contact_angle.cos();
    0.5_f64 * ball_count * shaft_frequency * (1.0 + ratio)
}

/// Fréquence de rotation d'une bille (BSF, *ball spin frequency*) :
/// `BSF = (D/(2·d))·fr·(1 − ((d/D)·cos α)²)`.
///
/// Fréquence à laquelle un défaut ponctuel de la **bille** revient frapper une
/// piste (le défaut heurte les deux pistes par tour, d'où l'usage courant de
/// `2·BSF` en diagnostic).
///
/// Panique si la géométrie est invalide (voir `assert_bearing_geometry`).
pub fn bearing_bsf(
    shaft_frequency: f64,
    ball_diameter: f64,
    pitch_diameter: f64,
    contact_angle: f64,
) -> f64 {
    // `ball_count` factice ≥ 1 : la BSF ne dépend pas du nombre de billes.
    assert_bearing_geometry(
        shaft_frequency,
        1.0,
        ball_diameter,
        pitch_diameter,
        contact_angle,
    );
    let ratio = (ball_diameter / pitch_diameter) * contact_angle.cos();
    0.5_f64 * (pitch_diameter / ball_diameter) * shaft_frequency * (1.0 - ratio * ratio)
}

/// Fréquence fondamentale de la cage (FTF, *fundamental train frequency*) :
/// `FTF = (fr/2)·(1 − (d/D)·cos α)`.
///
/// Fréquence de rotation du train de billes (cage). On a l'identité
/// `BPFO = n·FTF`.
///
/// Panique si la géométrie est invalide (voir `assert_bearing_geometry`).
pub fn bearing_ftf(
    shaft_frequency: f64,
    ball_diameter: f64,
    pitch_diameter: f64,
    contact_angle: f64,
) -> f64 {
    assert_bearing_geometry(
        shaft_frequency,
        1.0,
        ball_diameter,
        pitch_diameter,
        contact_angle,
    );
    let ratio = (ball_diameter / pitch_diameter) * contact_angle.cos();
    0.5_f64 * shaft_frequency * (1.0 - ratio)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Cas de référence : roulement à gorge profonde, angle de contact nul.
    // n = 8 billes, d = 8 mm, D = 40 mm → d/D = 0,2 ; cos 0 = 1 → ratio = 0,2.
    // fr = 30 Hz (1800 tr/min).
    //   BPFO = 4·30·(1−0,2)      = 96 Hz
    //   BPFI = 4·30·(1+0,2)      = 144 Hz
    //   BSF  = 2,5·30·(1−0,04)   = 72 Hz
    //   FTF  = 15·(1−0,2)        = 12 Hz
    const FR: f64 = 30.0;
    const N: f64 = 8.0;
    const D_BALL: f64 = 8.0;
    const D_PITCH: f64 = 40.0;
    const ALPHA: f64 = 0.0;

    #[test]
    fn reference_case_values() {
        assert_relative_eq!(
            bearing_bpfo(FR, N, D_BALL, D_PITCH, ALPHA),
            96.0,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            bearing_bpfi(FR, N, D_BALL, D_PITCH, ALPHA),
            144.0,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            bearing_bsf(FR, D_BALL, D_PITCH, ALPHA),
            72.0,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            bearing_ftf(FR, D_BALL, D_PITCH, ALPHA),
            12.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn bpfo_plus_bpfi_equals_n_times_shaft() {
        // Identité : BPFO + BPFI = (n/2)·fr·(1−r) + (n/2)·fr·(1+r) = n·fr.
        let bpfo = bearing_bpfo(FR, N, D_BALL, D_PITCH, ALPHA);
        let bpfi = bearing_bpfi(FR, N, D_BALL, D_PITCH, ALPHA);
        assert_relative_eq!(bpfo + bpfi, N * FR, epsilon = 1e-9);
    }

    #[test]
    fn bpfo_is_n_times_cage_frequency() {
        // Identité : BPFO = n·FTF (la cage entraîne les n billes sur la piste).
        let bpfo = bearing_bpfo(FR, N, D_BALL, D_PITCH, ALPHA);
        let ftf = bearing_ftf(FR, D_BALL, D_PITCH, ALPHA);
        assert_relative_eq!(bpfo, N * ftf, epsilon = 1e-9);
    }

    #[test]
    fn bsf_factors_as_product_of_train_terms() {
        // BSF = (D/2d)·fr·(1−r²) = (D/2d)·fr·(1−r)·(1+r).
        // Avec r = 0,2 : (1−r²) = 0,96 ; (1−r)(1+r) = 0,8·1,2 = 0,96.
        let r = (D_BALL / D_PITCH) * ALPHA.cos();
        let expected = 0.5 * (D_PITCH / D_BALL) * FR * (1.0 - r) * (1.0 + r);
        assert_relative_eq!(
            bearing_bsf(FR, D_BALL, D_PITCH, ALPHA),
            expected,
            epsilon = 1e-9
        );
    }

    #[test]
    fn all_frequencies_scale_linearly_with_shaft_speed() {
        // Toutes les fréquences sont proportionnelles à fr : doubler fr double tout.
        let k = 2.0;
        assert_relative_eq!(
            bearing_bpfo(k * FR, N, D_BALL, D_PITCH, ALPHA),
            k * bearing_bpfo(FR, N, D_BALL, D_PITCH, ALPHA),
            epsilon = 1e-9
        );
        assert_relative_eq!(
            bearing_bsf(k * FR, D_BALL, D_PITCH, ALPHA),
            k * bearing_bsf(FR, D_BALL, D_PITCH, ALPHA),
            epsilon = 1e-9
        );
    }

    #[test]
    fn nonzero_contact_angle_reduces_ratio() {
        // Un angle de contact non nul réduit cos α, donc rapproche BPFO et BPFI
        // de n·fr/2 : BPFI(α) < BPFI(0) et BPFO(α) > BPFO(0).
        let bpfi_0 = bearing_bpfi(FR, N, D_BALL, D_PITCH, 0.0);
        let bpfi_a = bearing_bpfi(FR, N, D_BALL, D_PITCH, 0.6);
        let bpfo_0 = bearing_bpfo(FR, N, D_BALL, D_PITCH, 0.0);
        let bpfo_a = bearing_bpfo(FR, N, D_BALL, D_PITCH, 0.6);
        assert!(bpfi_a < bpfi_0);
        assert!(bpfo_a > bpfo_0);
    }

    #[test]
    #[should_panic(
        expected = "le diamètre des billes d doit être inférieur au diamètre primitif D"
    )]
    fn ball_larger_than_pitch_panics() {
        bearing_bpfo(FR, N, 50.0, D_PITCH, ALPHA);
    }
}
