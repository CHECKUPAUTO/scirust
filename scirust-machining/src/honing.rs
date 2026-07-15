//! Usinage — **rodage (honing)** : cinématique combinée (rotation + va-et-vient),
//! angle de croisillon, vitesse périphérique et temps d'enlèvement radial.
//!
//! ```text
//! angle de croisillon    theta = 2·atan(vr / vp)
//! vitesse périphérique   vp = PI·d·N / 60
//! temps d'enlèvement     t = e / q
//! ```
//!
//! `vr` vitesse de va-et-vient (réciprocation) de la pierre le long de l'axe
//! (m/s), `vp` vitesse périphérique (tangentielle) due à la rotation (m/s),
//! `theta` angle de croisillon plein (angle total entre les deux familles de
//! rayures gravées, radians), `d` diamètre rodé (m), `N` vitesse de rotation
//! (tr/min), `e` surépaisseur radiale à enlever (m), `q` taux d'enlèvement
//! radial (m/s), `t` temps d'enlèvement (s). L'angle de croisillon est plein
//! (`2·atan`) car les deux flancs du va-et-vient (montée et descente) tracent des
//! rayures symétriques de part et d'autre de la circonférence.
//!
//! **Convention** : SI cohérent (m, m/s, s, rad) ; vitesses de rotation en tr/min
//! là où c'est indiqué. **Limite honnête** : cinématique **idéale** (rotation et
//! va-et-vient uniformes, sans inversion ni temps mort en bout de course). Les
//! vitesses `vr`, `vp` et `N` sont **fournies par l'appelant**. L'angle de
//! croisillon vise typiquement 30–60° en pratique, mais il reste ici **calculé**
//! depuis les vitesses fournies, sans valeur imposée. Le taux d'enlèvement radial
//! `q` dépend de l'abrasif, de la pression et du matériau : il est **fourni par
//! l'appelant** — aucune constante procédé/matériau n'est inventée ici.

use core::f64::consts::PI;

/// Angle de croisillon plein `theta = 2·atan(vr / vp)` (radians).
///
/// Rapport de la vitesse de va-et-vient à la vitesse périphérique : les deux
/// familles de rayures se croisent d'un angle total `theta`. À `vr = vp` l'angle
/// vaut `PI/2` (croisillon à 90°).
///
/// Panique si `reciprocation_speed < 0` ou si `rotation_peripheral_speed <= 0`.
pub fn honing_crosshatch_angle(reciprocation_speed: f64, rotation_peripheral_speed: f64) -> f64 {
    assert!(
        reciprocation_speed >= 0.0,
        "la vitesse de va-et-vient doit être positive"
    );
    assert!(
        rotation_peripheral_speed > 0.0,
        "la vitesse périphérique doit être strictement positive"
    );
    2.0 * (reciprocation_speed / rotation_peripheral_speed).atan()
}

/// Vitesse périphérique `vp = PI·d·N / 60` (m/s).
///
/// Vitesse tangentielle au diamètre rodé `d` (m) pour une rotation `N` (tr/min) ;
/// le facteur `1/60` convertit les tours par minute en tours par seconde.
///
/// Panique si `diameter < 0` ou si `rotational_speed_rpm < 0`.
pub fn honing_peripheral_speed(diameter: f64, rotational_speed_rpm: f64) -> f64 {
    assert!(diameter >= 0.0, "le diamètre doit être positif");
    assert!(
        rotational_speed_rpm >= 0.0,
        "la vitesse de rotation doit être positive"
    );
    PI * diameter * rotational_speed_rpm / 60.0
}

/// Temps d'enlèvement radial `t = e / q` (s).
///
/// Temps pour enlever une surépaisseur radiale `e` (m) à un taux d'enlèvement
/// radial constant `q` (m/s).
///
/// Panique si `radial_stock < 0` ou si `radial_removal_rate <= 0`.
pub fn honing_stock_removal_time(radial_stock: f64, radial_removal_rate: f64) -> f64 {
    assert!(
        radial_stock >= 0.0,
        "la surépaisseur radiale doit être positive"
    );
    assert!(
        radial_removal_rate > 0.0,
        "le taux d'enlèvement radial doit être strictement positif"
    );
    radial_stock / radial_removal_rate
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn crosshatch_angle_is_right_angle_when_speeds_equal() {
        // vr = vp → 2·atan(1) = 2·(PI/4) = PI/2 (croisillon à 90°).
        let theta = honing_crosshatch_angle(0.30, 0.30);
        assert_relative_eq!(theta, PI / 2.0, epsilon = 1e-12);
    }

    #[test]
    fn crosshatch_angle_depends_only_on_speed_ratio() {
        // theta dépend du seul rapport vr/vp : doubler les deux vitesses
        // laisse l'angle inchangé.
        let a = honing_crosshatch_angle(0.20, 0.35);
        let b = honing_crosshatch_angle(0.40, 0.70);
        assert_relative_eq!(a, b, epsilon = 1e-12);
    }

    #[test]
    fn crosshatch_angle_matches_sixty_degrees_case() {
        // Pour theta = 60° = PI/3, il faut atan(vr/vp) = 30°, soit
        // vr/vp = tan(30°) = 1/sqrt(3). On construit vr en conséquence.
        let vp = 0.40;
        let vr = vp * (1.0_f64 / 3.0_f64).sqrt();
        let theta = honing_crosshatch_angle(vr, vp);
        assert_relative_eq!(theta, PI / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn peripheral_speed_numeric_case() {
        // d = 80 mm, N = 300 tr/min → vp = PI·0,080·300/60 = PI·0,4 ≈ 1,256637 m/s.
        let vp = honing_peripheral_speed(0.080, 300.0);
        assert_relative_eq!(vp, PI * 0.4, epsilon = 1e-12);
        assert_relative_eq!(vp, 1.256_637_061, epsilon = 1e-9);
    }

    #[test]
    fn peripheral_speed_is_linear_in_diameter_and_rpm() {
        // vp ∝ d et vp ∝ N : doubler le diamètre ou la vitesse double vp.
        let base = honing_peripheral_speed(0.050, 200.0);
        assert_relative_eq!(
            honing_peripheral_speed(0.100, 200.0) / base,
            2.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            honing_peripheral_speed(0.050, 400.0) / base,
            2.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn stock_removal_time_numeric_and_reciprocal() {
        // e = 20 µm à q = 2 µm/s → t = 20e-6 / 2e-6 = 10 s.
        let t = honing_stock_removal_time(20e-6, 2e-6);
        assert_relative_eq!(t, 10.0, epsilon = 1e-12);
        // Réciprocité : q·t reconstitue la surépaisseur enlevée.
        assert_relative_eq!(2e-6 * t, 20e-6, epsilon = 1e-18);
    }

    #[test]
    #[should_panic(expected = "vitesse périphérique")]
    fn zero_peripheral_speed_panics() {
        honing_crosshatch_angle(0.30, 0.0);
    }
}
