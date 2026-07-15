//! Vis différentielle : deux filetages coaxiaux de pas voisins et de **même
//! sens**, dont l'avance nette par tour vaut la différence des pas, offrant une
//! résolution très fine sans filet à pas minuscule.
//!
//! Une vis différentielle porte un filetage grossier de pas `p1` (mm) engagé
//! dans un bâti fixe et un filetage fin de pas `p2` (mm) engagé dans l'organe
//! mobile. À chaque tour, le corps avance de `p1` par rapport au bâti tandis
//! que l'organe recule de `p2` par rapport au corps ; l'avance nette de
//! l'organe est donc la **soustraction** des pas :
//!
//! ```text
//! L_net = p1 − p2                 (avance nette par tour, mm/tour)
//! s     = L_net · n               (avance pour n tours, mm)
//! MA    = 2π·R / L_net            (avantage mécanique idéal, sans dim.)
//! F     = 2π·T / L_net            (effort axial idéal, sans frottement)
//! ```
//!
//! Légende : `p1`, `p2`, `L_net`, `s`, `R` (rayon d'application du couple) en
//! **mm** ; `n` en tours ; `T` couple d'entrée en **N·m** ; `F` effort axial en
//! **N** ; `MA` sans dimension.
//!
//! **Limite honnête** : les deux filets sont supposés de **même sens** (d'où la
//! soustraction ; des filets de sens opposés donneraient une addition). Les
//! relations d'effort (`MA`, `F`) sont **idéales** : le rendement et le
//! frottement des deux filetages ne sont **pas** inclus et doivent être
//! appliqués par l'appelant, de même que toute donnée matériaux ou de
//! lubrification. Complète [`crate::power_screws`] pour le couple réel au filet.

use core::f64::consts::PI;

/// Avance nette par tour `L_net = p1 − p2` (mm/tour) d'une vis différentielle,
/// pas grossier `pitch_coarse` et pas fin `pitch_fine` (mm), de même sens.
///
/// Panique si un pas est négatif ou si `pitch_coarse <= pitch_fine`.
pub fn differential_lead(pitch_coarse_mm: f64, pitch_fine_mm: f64) -> f64 {
    assert!(
        pitch_fine_mm > 0.0,
        "le pas fin doit être strictement positif"
    );
    assert!(
        pitch_coarse_mm > pitch_fine_mm,
        "le pas grossier doit dépasser le pas fin (filets de même sens)"
    );
    pitch_coarse_mm - pitch_fine_mm
}

/// Avance axiale `s = L_net · n` (mm) pour `turns` tours, avance nette par tour
/// `net_lead` (mm/tour).
///
/// Panique si `net_lead <= 0`.
pub fn differential_advance(net_lead_mm: f64, turns: f64) -> f64 {
    assert!(
        net_lead_mm > 0.0,
        "l'avance nette par tour doit être strictement positive"
    );
    net_lead_mm * turns
}

/// Nombre de tours `n = s / L_net` requis pour une avance `advance` (mm),
/// réciproque de [`differential_advance`], avance nette `net_lead` (mm/tour).
///
/// Panique si `net_lead <= 0`.
pub fn differential_turns(net_lead_mm: f64, advance_mm: f64) -> f64 {
    assert!(
        net_lead_mm > 0.0,
        "l'avance nette par tour doit être strictement positive"
    );
    advance_mm / net_lead_mm
}

/// Avantage mécanique **idéal** (sans dimension) `MA = 2π·R / L_net`, rayon
/// d'application du couple `input_torque_radius` (mm) et avance nette
/// `net_lead` (mm/tour). Sans frottement ni rendement.
///
/// Panique si `input_torque_radius <= 0` ou si `net_lead <= 0`.
pub fn diffscrew_mechanical_advantage(input_torque_radius_mm: f64, net_lead_mm: f64) -> f64 {
    assert!(
        input_torque_radius_mm > 0.0,
        "le rayon d'application du couple doit être strictement positif"
    );
    assert!(
        net_lead_mm > 0.0,
        "l'avance nette par tour doit être strictement positive"
    );
    2.0 * PI * input_torque_radius_mm / net_lead_mm
}

/// Effort axial **idéal** `F = 2π·T / L_net` (N), couple d'entrée `input_torque`
/// (N·m) et avance nette `net_lead` (mm/tour). Sans frottement (borne
/// supérieure théorique de l'effort développé).
///
/// Panique si `input_torque <= 0` ou si `net_lead <= 0`.
pub fn diffscrew_ideal_axial_force(input_torque_nm: f64, net_lead_mm: f64) -> f64 {
    assert!(
        input_torque_nm > 0.0,
        "le couple d'entrée doit être strictement positif"
    );
    assert!(
        net_lead_mm > 0.0,
        "l'avance nette par tour doit être strictement positive"
    );
    // net_lead en mm → m pour cohérence SI avec le couple en N·m.
    2.0 * PI * input_torque_nm / (net_lead_mm / 1000.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn lead_is_difference_of_pitches() {
        // Vis d'ajustage fin classique : pas 1,00 mm et 0,75 mm de même sens.
        let l = differential_lead(1.0, 0.75);
        assert_relative_eq!(l, 0.25, epsilon = 1e-12);
    }

    #[test]
    fn advance_and_turns_are_reciprocal() {
        let lead = differential_lead(1.0, 0.9); // 0,1 mm/tour
        let n = 12.5_f64;
        let s = differential_advance(lead, n);
        assert_relative_eq!(differential_turns(lead, s), n, epsilon = 1e-12);
    }

    #[test]
    fn advance_is_proportional_to_turns() {
        let lead = 0.2_f64;
        let s1 = differential_advance(lead, 3.0);
        let s3 = differential_advance(lead, 9.0);
        // Tripler les tours triple l'avance.
        assert_relative_eq!(s3, 3.0 * s1, epsilon = 1e-12);
    }

    #[test]
    fn mechanical_advantage_known_value() {
        // R = 50 mm, L_net = 0,1 mm → MA = 2π·50/0,1 = 1000·π.
        let ma = diffscrew_mechanical_advantage(50.0, 0.1);
        assert_relative_eq!(ma, 1000.0 * PI, epsilon = 1e-9);
    }

    #[test]
    fn ideal_axial_force_known_value() {
        // T = 1 N·m, L_net = 0,1 mm = 1e-4 m → F = 2π/1e-4 = 20000·π N.
        let f = diffscrew_ideal_axial_force(1.0, 0.1);
        assert_relative_eq!(f, 20_000.0 * PI, epsilon = 1e-6);
    }

    #[test]
    fn realistic_fine_positioner() {
        // p1 = 0,80 mm, p2 = 0,75 mm → 0,05 mm/tour ; 20 tours → 1,00 mm.
        let lead = differential_lead(0.80, 0.75);
        assert_relative_eq!(lead, 0.05, epsilon = 1e-12);
        assert_relative_eq!(differential_advance(lead, 20.0), 1.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "le pas grossier doit dépasser le pas fin")]
    fn equal_pitches_panics() {
        let _ = differential_lead(0.75, 0.75);
    }
}
