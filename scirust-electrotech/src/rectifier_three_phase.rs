//! **Redresseur triphasé** — tension continue moyenne d'un montage à 3 pulses
//! (P3, demi-onde), d'un pont de Graetz à 6 pulses (non commandé puis commandé)
//! et fréquence de l'ondulation résiduelle en sortie.
//!
//! ```text
//! P3 demi-onde (3 pulses)     V_dc = 3·√3·V̂_ph / (2·π)
//! pont de Graetz (6 pulses)   V_dc = 3·V̂_ll / π
//! pont commandé (6 pulses)    V_dcα = (3·V̂_ll / π) · cos(α)
//! fréquence d'ondulation      f_r  = p · f_s
//! ```
//!
//! `V̂_ph` tension **crête** simple (phase-neutre, V), `V̂_ll` tension **crête**
//! composée (phase-phase, V), `V_dc` tension continue moyenne en sortie (V), `α`
//! angle d'amorçage des thyristors (rad), `f_s` fréquence du réseau
//! d'alimentation (Hz), `p` indice de pulsation du montage (sans dimension : 3
//! pour P3, 6 pour le pont de Graetz), `f_r` fréquence fondamentale de
//! l'ondulation de la tension redressée (Hz).
//!
//! **Convention** : SI ; tensions en V, fréquences en Hz, angles en **radians**.
//! **Limite honnête** : semi-conducteurs **idéaux** (chute nulle, commutation
//! instantanée), courant de sortie **continu et parfaitement lissé** (charge
//! fortement inductive), source triphasée **équilibrée sans empiètement**
//! (recouvrement de commutation négligé). Toutes les grandeurs réseau
//! (tensions crêtes, fréquence) et l'indice de pulsation sont **fournis par
//! l'appelant** ; aucune valeur « par défaut » n'est inventée. `V̂_ll` est la
//! valeur **crête** de la tension **composée**, à ne pas confondre avec sa
//! valeur efficace ni avec la tension simple.

use core::f64::consts::PI;

/// Tension continue moyenne d'un redresseur triphasé **P3 demi-onde** (3 pulses) :
/// `V_dc = 3·√3·V̂_ph / (2·π)`, où `V̂_ph` est la tension crête simple.
///
/// Panique si `peak_phase_voltage < 0`.
pub fn rect3ph_halfwave_average(peak_phase_voltage: f64) -> f64 {
    assert!(peak_phase_voltage >= 0.0, "V̂_ph ≥ 0 requis");
    3.0 * 3.0_f64.sqrt() * peak_phase_voltage / (2.0 * PI)
}

/// Tension continue moyenne d'un **pont de Graetz** triphasé (6 pulses) non
/// commandé : `V_dc = 3·V̂_ll / π`, où `V̂_ll` est la tension crête composée.
///
/// Panique si `peak_line_voltage < 0`.
pub fn rect3ph_bridge_average(peak_line_voltage: f64) -> f64 {
    assert!(peak_line_voltage >= 0.0, "V̂_ll ≥ 0 requis");
    3.0 * peak_line_voltage / PI
}

/// Fréquence fondamentale de l'ondulation en sortie : `f_r = p · f_s`, produit de
/// l'indice de pulsation `p` par la fréquence réseau `f_s`.
///
/// Panique si `supply_frequency < 0` ou si `pulse_number <= 0`.
pub fn rect3ph_ripple_frequency(supply_frequency: f64, pulse_number: f64) -> f64 {
    assert!(supply_frequency >= 0.0, "f_s ≥ 0 requis");
    assert!(pulse_number > 0.0, "p > 0 requis");
    pulse_number * supply_frequency
}

/// Tension continue moyenne d'un **pont de Graetz commandé** (6 pulses) :
/// `V_dcα = (3·V̂_ll / π) · cos(α)`, avec `α` l'angle d'amorçage des thyristors.
///
/// Panique si `peak_line_voltage < 0` ou si `firing_angle_rad ∉ [0, π]`.
pub fn rect3ph_controlled_bridge_average(peak_line_voltage: f64, firing_angle_rad: f64) -> f64 {
    assert!(peak_line_voltage >= 0.0, "V̂_ll ≥ 0 requis");
    assert!(
        (0.0..=PI).contains(&firing_angle_rad),
        "angle d'amorçage α ∈ [0, π] requis"
    );
    (3.0 * peak_line_voltage / PI) * firing_angle_rad.cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn bridge_average_normalises_on_pi() {
        // Identité : avec V̂_ll = π, V_dc = 3·π/π = 3 exactement.
        assert_relative_eq!(rect3ph_bridge_average(PI), 3.0, epsilon = 1e-12);
    }

    #[test]
    fn halfwave_average_normalises_on_two_pi() {
        // Identité : avec V̂_ph = 2π, V_dc = 3·√3·2π/(2π) = 3·√3.
        assert_relative_eq!(
            rect3ph_halfwave_average(2.0 * PI),
            3.0 * 3.0_f64.sqrt(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn controlled_bridge_reduces_to_uncontrolled_at_zero_angle() {
        // Cas limite α = 0 : cos(0) = 1, le pont commandé rejoint le pont non
        // commandé ; et à α = π/3 (cos = 1/2) la tension vaut la moitié.
        let peak = 565.685_f64;
        let uncontrolled = rect3ph_bridge_average(peak);
        assert_relative_eq!(
            rect3ph_controlled_bridge_average(peak, 0.0),
            uncontrolled,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            rect3ph_controlled_bridge_average(peak, PI / 3.0),
            0.5 * uncontrolled,
            epsilon = 1e-9
        );
        // À α = π/2 le pont est complètement décalé : V_dc = 0.
        assert_relative_eq!(
            rect3ph_controlled_bridge_average(peak, PI / 2.0),
            0.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn ripple_frequency_scales_with_pulse_number() {
        // Réseau 50 Hz : pont 6 pulses → 300 Hz, montage P3 → 150 Hz.
        assert_relative_eq!(rect3ph_ripple_frequency(50.0, 6.0), 300.0, epsilon = 1e-12);
        assert_relative_eq!(rect3ph_ripple_frequency(50.0, 3.0), 150.0, epsilon = 1e-12);
        // Proportionnalité : doubler la fréquence réseau double l'ondulation.
        let f1 = rect3ph_ripple_frequency(50.0, 6.0);
        let f2 = rect3ph_ripple_frequency(100.0, 6.0);
        assert_relative_eq!(f2 / f1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn realistic_400v_bridge_case() {
        // Cas chiffré : réseau 400 V efficace composé → V̂_ll = 400·√2 ≈ 565,685 V.
        // Pont de Graetz : V_dc = 3·565,685/π ≈ 540,19 V (facteur 3√2/π ≈ 1,3505).
        let peak_line = 400.0 * 2.0_f64.sqrt();
        assert_relative_eq!(
            rect3ph_bridge_average(peak_line),
            540.189_79,
            epsilon = 1e-2
        );
        // Pont commandé à α = 30° (π/6) : V_dcα = 540,19 · cos(30°) ≈ 467,82 V.
        assert_relative_eq!(
            rect3ph_controlled_bridge_average(peak_line, PI / 6.0),
            467.818_08,
            epsilon = 1e-2
        );
    }

    #[test]
    #[should_panic(expected = "angle d'amorçage α ∈ [0, π] requis")]
    fn negative_firing_angle_panics() {
        rect3ph_controlled_bridge_average(565.685, -0.1);
    }
}
