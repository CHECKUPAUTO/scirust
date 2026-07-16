//! **Redresseur monophasé sur charge résistive** — valeurs moyenne et efficace
//! des tensions redressées (simple et double alternance), facteur d'ondulation
//! et redressement commandé par pont à thyristors.
//!
//! ```text
//! simple alternance (moy.)     V_dc  = V_m / π
//! double alternance (moy.)     V_dc  = 2·V_m / π
//! double alternance (eff.)     V_rms = V_m / √2
//! facteur d'ondulation         r     = √((V_rms / V_dc)² − 1)
//! pont commandé (moy.)         V_dc  = (V_m / π)·(1 + cos α)
//! ```
//!
//! `V_m` tension crête de la source sinusoïdale (V), `V_dc` valeur moyenne
//! (composante continue) de la tension redressée (V), `V_rms` valeur efficace de
//! la tension redressée (V), `r` facteur d'ondulation (sans dimension), `α`
//! angle d'amorçage des thyristors (rad, `∈ [0, π]`). Le pont commandé à `α = 0`
//! se ramène au redressement double alternance non commandé (`2·V_m / π`) et
//! s'annule à `α = π`.
//!
//! **Convention** : SI ; tensions en V, angle d'amorçage en **radians** pour la
//! fonction trigonométrique. **Limite honnête** : diodes et thyristors supposés
//! **idéaux** (chute directe nulle, commutation instantanée), **charge
//! résistive pure** (pas de lissage capacitif ni inductif) et source
//! **sinusoïdale sans impédance interne** (pas d'empiétement) ; le redressement
//! commandé suppose un **amorçage franc** à l'angle `α` fourni. La tension crête
//! `V_m` et l'angle d'amorçage `α` sont **fournis par l'appelant** (réseau,
//! commande de gâchette) — aucune valeur « par défaut » n'est inventée.

use core::f64::consts::PI;

/// Valeur moyenne d'un redressement **simple alternance** `V_dc = V_m / π` (V).
///
/// Panique si `peak_voltage < 0`.
pub fn rect1ph_halfwave_average(peak_voltage: f64) -> f64 {
    assert!(peak_voltage >= 0.0, "la tension crête V_m doit être ≥ 0");
    peak_voltage / PI
}

/// Valeur moyenne d'un redressement **double alternance** `V_dc = 2·V_m / π`
/// (V).
///
/// Panique si `peak_voltage < 0`.
pub fn rect1ph_fullwave_average(peak_voltage: f64) -> f64 {
    assert!(peak_voltage >= 0.0, "la tension crête V_m doit être ≥ 0");
    2.0 * peak_voltage / PI
}

/// Valeur efficace d'un redressement **double alternance** `V_rms = V_m / √2`
/// (V) — identique à la valeur efficace de la sinusoïde source.
///
/// Panique si `peak_voltage < 0`.
pub fn rect1ph_fullwave_rms(peak_voltage: f64) -> f64 {
    assert!(peak_voltage >= 0.0, "la tension crête V_m doit être ≥ 0");
    peak_voltage / 2.0_f64.sqrt()
}

/// Facteur d'ondulation `r = √((V_rms / V_dc)² − 1)` (sans dimension) — mesure
/// de la part alternative résiduelle dans la tension redressée.
///
/// Panique si `average_voltage <= 0`, si `rms_voltage < 0` ou si
/// `rms_voltage < average_voltage` (radicande négatif, physiquement exclu).
pub fn rect1ph_ripple_factor(rms_voltage: f64, average_voltage: f64) -> f64 {
    assert!(
        average_voltage > 0.0,
        "la valeur moyenne V_dc doit être strictement positive"
    );
    assert!(rms_voltage >= 0.0, "la valeur efficace V_rms doit être ≥ 0");
    assert!(
        rms_voltage >= average_voltage,
        "V_rms ≥ V_dc requis (radicande négatif sinon)"
    );
    ((rms_voltage / average_voltage).powi(2) - 1.0).sqrt()
}

/// Valeur moyenne d'un **pont commandé double alternance** à thyristors sur
/// charge résistive `V_dc = (V_m / π)·(1 + cos α)` (V).
///
/// Panique si `peak_voltage < 0` ou si `firing_angle_rad` n'est pas dans
/// `[0, π]`.
pub fn rect1ph_controlled_fullwave_average(peak_voltage: f64, firing_angle_rad: f64) -> f64 {
    assert!(peak_voltage >= 0.0, "la tension crête V_m doit être ≥ 0");
    assert!(
        (0.0..=PI).contains(&firing_angle_rad),
        "l'angle d'amorçage α doit être dans [0, π] rad"
    );
    (peak_voltage / PI) * (1.0 + firing_angle_rad.cos())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn fullwave_average_is_twice_halfwave() {
        // Identité structurelle : 2·V_m/π = 2·(V_m/π), à V_m fixé.
        let v_m = 325.0_f64;
        let half = rect1ph_halfwave_average(v_m);
        let full = rect1ph_fullwave_average(v_m);
        assert_relative_eq!(full, 2.0 * half, epsilon = 1e-12);
    }

    #[test]
    fn averages_scale_linearly_with_peak() {
        // Proportionnalité : doubler V_m double la valeur moyenne.
        let v1 = rect1ph_fullwave_average(100.0);
        let v2 = rect1ph_fullwave_average(200.0);
        assert_relative_eq!(v2 / v1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn realistic_100v_peak_case() {
        // Cas chiffré, V_m = 100 V :
        //   simple alt. moy. = 100/π      ≈ 31,830989 V
        //   double alt. moy. = 200/π      ≈ 63,661977 V
        //   double alt. eff. = 100/√2     ≈ 70,710678 V
        let v_m = 100.0_f64;
        assert_relative_eq!(rect1ph_halfwave_average(v_m), 31.830_988_6, epsilon = 1e-4);
        assert_relative_eq!(rect1ph_fullwave_average(v_m), 63.661_977_2, epsilon = 1e-4);
        assert_relative_eq!(rect1ph_fullwave_rms(v_m), 70.710_678_1, epsilon = 1e-4);
    }

    #[test]
    fn fullwave_ripple_factor_reference_value() {
        // Facteur d'ondulation double alternance : r = √((π/(2√2))² − 1)
        //   V_rms/V_dc = (V_m/√2)/(2V_m/π) = π/(2√2) ≈ 1,110721
        //   r = √(1,233701 − 1) ≈ 0,483426  (valeur classique ≈ 0,48).
        let v_m = 240.0_f64;
        let v_rms = rect1ph_fullwave_rms(v_m);
        let v_dc = rect1ph_fullwave_average(v_m);
        let r = rect1ph_ripple_factor(v_rms, v_dc);
        assert_relative_eq!(r, 0.483_426, epsilon = 1e-3);
    }

    #[test]
    fn controlled_at_zero_firing_equals_uncontrolled_fullwave() {
        // Identité : à α = 0, (V_m/π)·(1+cos0) = 2V_m/π = double alternance non
        // commandée ; à α = π, (V_m/π)·(1+cosπ) = 0.
        let v_m = 311.0_f64;
        assert_relative_eq!(
            rect1ph_controlled_fullwave_average(v_m, 0.0),
            rect1ph_fullwave_average(v_m),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            rect1ph_controlled_fullwave_average(v_m, PI),
            0.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn controlled_fullwave_at_sixty_degrees() {
        // Cas chiffré : V_m = 100 V, α = 60° = π/3, cos = 0,5 →
        //   V_dc = (100/π)·1,5 = 150/π ≈ 47,746483 V.
        let v_dc = rect1ph_controlled_fullwave_average(100.0, PI / 3.0);
        assert_relative_eq!(v_dc, 47.746_482_9, epsilon = 1e-4);
    }

    #[test]
    #[should_panic(expected = "l'angle d'amorçage α doit être dans [0, π] rad")]
    fn firing_angle_out_of_range_panics() {
        rect1ph_controlled_fullwave_average(100.0, 4.0);
    }
}
