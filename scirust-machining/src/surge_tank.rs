//! Cheminée d'équilibre — oscillation en **masse** rigide de la colonne d'eau
//! d'une conduite forcée (amplitude, période, aire de stabilité de Thoma).
//!
//! ```text
//! amplitude max (sans pertes)  z_max = v·√( L·A_t / (g·A_s) )
//! période d'oscillation        T     = 2·π·√( L·A_s / (g·A_t) )
//! aire de Thoma (stabilité)    A_th  = L·A_t / (2·k·g·H)
//! ```
//!
//! `v` vitesse dans la galerie (m/s), `L` longueur de galerie (m), `A_t` section
//! de galerie (tunnel, m²), `A_s` section de la cheminée (surge tank, m²), `g`
//! accélération de la pesanteur (m/s²), `k` coefficient de perte de charge de la
//! galerie (s²/m, tel que la perte = k·v²), `H` chute nette (m), `z_max`
//! amplitude de l'oscillation (m), `T` période (s), `A_th` aire minimale de
//! stabilité (m²).
//!
//! **Convention** : SI cohérent. **Limite honnête** : théorie de l'oscillation en
//! **masse** (colonne rigide, incompressible) — distincte du coup de bélier
//! **élastique** de [`crate::water_hammer`]. Les sections de galerie et de
//! cheminée, la chute nette et le coefficient de perte sont **fournis** par
//! l'appelant (aucune valeur « par défaut » inventée). L'amplitude maximale est
//! donnée **sans pertes** (elle surestime l'oscillation réelle) ; le critère de
//! Thoma donne l'aire **minimale** de stabilité, pas l'aire de projet.

use core::f64::consts::PI;

/// Amplitude maximale de l'oscillation en masse **sans pertes**
/// `z_max = v·√( L·A_t / (g·A_s) )` (m).
///
/// Panique si `tunnel_length < 0`, ou si `tunnel_area`, `tank_area` ou
/// `gravity` ne sont pas strictement positifs.
pub fn surgetank_max_surge_amplitude(
    tunnel_velocity: f64,
    tunnel_length: f64,
    tunnel_area: f64,
    tank_area: f64,
    gravity: f64,
) -> f64 {
    assert!(
        tunnel_length >= 0.0,
        "la longueur de galerie doit être positive ou nulle"
    );
    assert!(
        tunnel_area > 0.0,
        "la section de galerie doit être strictement positive"
    );
    assert!(
        tank_area > 0.0,
        "la section de cheminée doit être strictement positive"
    );
    assert!(gravity > 0.0, "la pesanteur doit être strictement positive");
    tunnel_velocity * (tunnel_length * tunnel_area / (gravity * tank_area)).sqrt()
}

/// Période de l'oscillation en masse `T = 2·π·√( L·A_s / (g·A_t) )` (s).
///
/// Panique si `tunnel_length < 0`, ou si `tunnel_area`, `tank_area` ou
/// `gravity` ne sont pas strictement positifs.
pub fn surgetank_oscillation_period(
    tunnel_length: f64,
    tunnel_area: f64,
    tank_area: f64,
    gravity: f64,
) -> f64 {
    assert!(
        tunnel_length >= 0.0,
        "la longueur de galerie doit être positive ou nulle"
    );
    assert!(
        tunnel_area > 0.0,
        "la section de galerie doit être strictement positive"
    );
    assert!(
        tank_area > 0.0,
        "la section de cheminée doit être strictement positive"
    );
    assert!(gravity > 0.0, "la pesanteur doit être strictement positive");
    2.0 * PI * (tunnel_length * tank_area / (gravity * tunnel_area)).sqrt()
}

/// Aire minimale de stabilité de **Thoma**
/// `A_th = L·A_t / (2·k·g·H)` (m²).
///
/// Panique si `tunnel_length < 0`, ou si `tunnel_area`,
/// `head_loss_coefficient`, `net_head` ou `gravity` ne sont pas strictement
/// positifs.
pub fn surgetank_thoma_area(
    tunnel_length: f64,
    tunnel_area: f64,
    head_loss_coefficient: f64,
    net_head: f64,
    gravity: f64,
) -> f64 {
    assert!(
        tunnel_length >= 0.0,
        "la longueur de galerie doit être positive ou nulle"
    );
    assert!(
        tunnel_area > 0.0,
        "la section de galerie doit être strictement positive"
    );
    assert!(
        head_loss_coefficient > 0.0,
        "le coefficient de perte de charge doit être strictement positif"
    );
    assert!(
        net_head > 0.0,
        "la chute nette doit être strictement positive"
    );
    assert!(gravity > 0.0, "la pesanteur doit être strictement positive");
    tunnel_length * tunnel_area / (2.0 * head_loss_coefficient * gravity * net_head)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn amplitude_matches_closed_form() {
        // v=2, L=1000, A_t=10, A_s=50, g=9,81 :
        // z = 2·√(1000·10/(9,81·50)) = 2·√20,38736… = 9,03047… m.
        let (v, l, at, as_, g) = (2.0_f64, 1000.0_f64, 10.0_f64, 50.0_f64, 9.81_f64);
        let z = surgetank_max_surge_amplitude(v, l, at, as_, g);
        assert_relative_eq!(z, v * (l * at / (g * as_)).sqrt(), epsilon = 1e-12);
        assert_relative_eq!(z, 9.030_472_8, epsilon = 1e-6);
    }

    #[test]
    fn amplitude_is_proportional_to_velocity() {
        // z ∝ v : doubler la vitesse double l'amplitude.
        let z1 = surgetank_max_surge_amplitude(2.0, 1000.0, 10.0, 50.0, 9.81);
        let z2 = surgetank_max_surge_amplitude(4.0, 1000.0, 10.0, 50.0, 9.81);
        assert_relative_eq!(z2, 2.0 * z1, epsilon = 1e-12);
    }

    #[test]
    fn amplitude_times_period_is_area_independent() {
        // z_max·T = v·2π·√(L²/g²) = 2π·v·L/g, indépendant des sections.
        let (v, l, g) = (2.0_f64, 1000.0_f64, 9.81_f64);
        let z = surgetank_max_surge_amplitude(v, l, 10.0, 50.0, g);
        let t = surgetank_oscillation_period(l, 10.0, 50.0, g);
        assert_relative_eq!(z * t, 2.0 * PI * v * l / g, epsilon = 1e-9);
    }

    #[test]
    fn period_grows_with_tank_area() {
        // T ∝ √A_s : quadrupler la section de cheminée double la période.
        let t1 = surgetank_oscillation_period(1000.0, 10.0, 50.0, 9.81);
        let t2 = surgetank_oscillation_period(1000.0, 10.0, 200.0, 9.81);
        assert_relative_eq!(t2, 2.0 * t1, epsilon = 1e-12);
    }

    #[test]
    fn thoma_area_matches_closed_form() {
        // L=1000, A_t=10, k=0,1, H=100, g=9,81 :
        // A_th = 1000·10/(2·0,1·9,81·100) = 10000/196,2 = 50,9683… m².
        let (l, at, k, h, g) = (1000.0_f64, 10.0_f64, 0.1_f64, 100.0_f64, 9.81_f64);
        let a = surgetank_thoma_area(l, at, k, h, g);
        assert_relative_eq!(a, l * at / (2.0 * k * g * h), epsilon = 1e-12);
        assert_relative_eq!(a, 50.968_399, epsilon = 1e-5);
    }

    #[test]
    #[should_panic(expected = "section de cheminée")]
    fn zero_tank_area_amplitude_panics() {
        surgetank_max_surge_amplitude(2.0, 1000.0, 10.0, 0.0, 9.81);
    }
}
