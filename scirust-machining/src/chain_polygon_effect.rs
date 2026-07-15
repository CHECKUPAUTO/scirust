//! **Effet polygonal** d'une transmission par chaîne à rouleaux : la chaîne
//! s'enroule sur le pignon en formant un polygone, d'où une variation cyclique
//! de la vitesse linéaire (et de la tension) à chaque engagement de dent.
//!
//! ```text
//! diamètre primitif     d = p / sin(π/Z)
//! pas angulaire (dent)  α = 2π/Z
//! ondulation vitesse    r = 1 − cos(π/Z)          (crête relative, sans dimension)
//! vitesse moyenne       v = p·Z·N/60              (N en tr/min)
//! ```
//!
//! `p` pas de la chaîne (m), `Z` nombre de dents du pignon, `N` fréquence de
//! rotation (tr/min), `d` diamètre primitif (m), `α` pas angulaire (rad), `r`
//! ondulation relative crête de la vitesse linéaire (sans dimension), `v` vitesse
//! linéaire moyenne de la chaîne (m/s).
//!
//! **Convention** : SI cohérent (mètres, secondes, radians), la vitesse de
//! rotation étant fournie en tr/min. L'ondulation naît de ce que la vitesse
//! linéaire passe de `ω·R` (dent tangente) à `ω·R·cos(π/Z)` (brin à mi-course),
//! `R = d/2` étant le rayon primitif ; augmenter `Z` réduit l'effet.
//!
//! **Limite honnête** : effet polygonal purement **géométrique** (variation
//! cyclique de vitesse/tension par dent) ; la dynamique de la chaîne (masse,
//! élasticité, choc à l'engagement, amortissement) n'est pas modélisée. Le pas et
//! la denture sont **fournis par l'appelant** — aucune valeur « par défaut » de
//! chaîne ou de procédé n'est supposée.

use core::f64::consts::PI;

/// Diamètre primitif du pignon `d = p / sin(π/Z)` (m).
///
/// Panique si `pitch <= 0` ou `teeth < 3`.
pub fn chain_pitch_diameter(pitch: f64, teeth: u32) -> f64 {
    assert!(
        pitch > 0.0,
        "le pas de la chaîne doit être strictement positif"
    );
    assert!(teeth >= 3, "un pignon de chaîne a au moins 3 dents");
    pitch / (PI / teeth as f64).sin()
}

/// Pas angulaire (angle entre deux dents) `α = 2π/Z` (rad).
///
/// Panique si `teeth < 3`.
pub fn chain_tooth_angle(teeth: u32) -> f64 {
    assert!(teeth >= 3, "un pignon de chaîne a au moins 3 dents");
    2.0 * PI / teeth as f64
}

/// Ondulation relative crête de la vitesse linéaire `r = 1 − cos(π/Z)`
/// (sans dimension) due à l'effet polygonal.
///
/// Panique si `teeth < 3`.
pub fn chain_speed_ripple(teeth: u32) -> f64 {
    assert!(teeth >= 3, "un pignon de chaîne a au moins 3 dents");
    1.0 - (PI / teeth as f64).cos()
}

/// Vitesse linéaire moyenne de la chaîne `v = p·Z·N/60` (m/s), `N` en tr/min.
///
/// Panique si `pitch <= 0`, `teeth < 3` ou `rpm < 0`.
pub fn chain_mean_speed(pitch: f64, teeth: u32, rpm: f64) -> f64 {
    assert!(
        pitch > 0.0,
        "le pas de la chaîne doit être strictement positif"
    );
    assert!(teeth >= 3, "un pignon de chaîne a au moins 3 dents");
    assert!(
        rpm >= 0.0,
        "la fréquence de rotation ne peut pas être négative"
    );
    pitch * teeth as f64 * rpm / 60.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn pitch_diameter_matches_pitch_definition() {
        // Réciprocité géométrique : p = d·sin(π/Z), donc d·sin(π/Z) doit
        // redonner le pas de départ.
        let pitch = 0.0127; // pas 12,7 mm (chaîne ANSI 40 / pas 1/2")
        let teeth = 17;
        let d = chain_pitch_diameter(pitch, teeth);
        assert_relative_eq!(d * (PI / teeth as f64).sin(), pitch, epsilon = 1e-15);
    }

    #[test]
    fn pitch_diameter_grows_with_teeth() {
        // À pas égal, plus de dents → plus grand diamètre primitif.
        let d13 = chain_pitch_diameter(0.0127, 13);
        let d25 = chain_pitch_diameter(0.0127, 25);
        assert!(d25 > d13);
    }

    #[test]
    fn ripple_from_half_tooth_angle() {
        // Identité : l'ondulation ne dépend que du demi-pas angulaire π/Z,
        // soit la moitié du pas angulaire α = 2π/Z.
        for &z in &[9_u32, 17, 38]
        {
            let r = chain_speed_ripple(z);
            let half_angle = chain_tooth_angle(z) / 2.0;
            assert_relative_eq!(r, 1.0 - half_angle.cos(), epsilon = 1e-15);
        }
    }

    #[test]
    fn ripple_decreases_and_vanishes_with_teeth() {
        // Monotonie : augmenter Z atténue l'effet polygonal, et pour un très
        // grand nombre de dents l'ondulation tend vers 0.
        assert!(chain_speed_ripple(11) > chain_speed_ripple(23));
        assert!(chain_speed_ripple(1000) < 1e-4);
    }

    #[test]
    fn mean_speed_worked_case() {
        // p=12,7 mm, Z=17, N=1500 tr/min = 25 tr/s.
        // v = 0,0127·17·1500/60 = 0,0127·17·25 = 5,3975 m/s.
        assert_relative_eq!(
            chain_mean_speed(0.0127, 17, 1500.0),
            5.3975,
            epsilon = 1e-12
        );
    }

    #[test]
    fn mean_speed_proportional_to_rpm() {
        // Proportionnalité stricte à la vitesse de rotation (à p, Z fixés).
        let v1 = chain_mean_speed(0.0127, 17, 300.0);
        let v3 = chain_mean_speed(0.0127, 17, 900.0);
        assert_relative_eq!(v3, 3.0 * v1, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "au moins 3 dents")]
    fn too_few_teeth_panics() {
        chain_speed_ripple(2);
    }
}
