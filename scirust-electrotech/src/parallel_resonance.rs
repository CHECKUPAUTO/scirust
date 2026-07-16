//! **Résonance parallèle (circuit bouchon RLC)** — module de la fréquence de
//! résonance, de l'impédance dynamique, du facteur de qualité de la bobine et de
//! la largeur de bande d'un circuit bouchon (bobine `R`–`L` en parallèle avec un
//! condensateur `C`).
//!
//! ```text
//! fréquence de résonance   f_0 = 1 / (2·π·√(L·C))     (approximation R faible)
//! impédance dynamique      R_d = L / (C·R)            (maximale et résistive)
//! facteur de qualité       Q   = ω_0·L / R            (facteur de la bobine)
//! largeur de bande         Δf  = f_0 / Q
//! ```
//!
//! `R` résistance série de la bobine (Ω), `L` inductance (H), `C` capacité (F),
//! `f_0` fréquence propre de résonance (Hz), `ω_0 = 2·π·f_0` pulsation propre
//! (rad/s), `R_d` impédance dynamique à la résonance (Ω, réelle et maximale),
//! `Q` facteur de qualité de la bobine (sans dimension), `Δf` largeur de bande
//! à −3 dB (Hz).
//!
//! **Convention** : SI ; résistances et impédances en Ω, inductances en H,
//! capacités en F, fréquences en Hz, pulsations en **rad/s**. **Limite
//! honnête** : circuit **bouchon** (`L` et `C` en **parallèle**, `R` en **série
//! avec `L`**), résistance `R` **faible devant la réactance** ; à la résonance
//! l'impédance vue des bornes est **maximale et purement résistive** égale à
//! `L/(C·R)`, et le courant de ligne est **minimal**. C'est le dual du module
//! `rlc_series` (résonance série, impédance **minimale**). Les valeurs des
//! composants (`R`, `L`, `C`) et la pulsation sont **fournies par l'appelant**
//! (mesures, fiches composant) — aucune valeur « par défaut » n'est inventée.

/// Fréquence de résonance `f_0 = 1 / (2·π·√(L·C))` (Hz), approximation valable
/// pour une résistance `R` faible devant la réactance.
///
/// Panique si `inductance <= 0` ou si `capacitance <= 0`.
pub fn pres_resonant_frequency(inductance: f64, capacitance: f64) -> f64 {
    assert!(
        inductance > 0.0,
        "l'inductance L doit être strictement positive"
    );
    assert!(
        capacitance > 0.0,
        "la capacité C doit être strictement positive"
    );
    1.0 / (2.0 * core::f64::consts::PI * (inductance * capacitance).sqrt())
}

/// Impédance dynamique à la résonance `R_d = L / (C·R)` (Ω), maximale et
/// purement résistive vue des bornes du circuit bouchon.
///
/// Panique si `inductance <= 0`, si `capacitance <= 0` ou si `resistance <= 0`.
pub fn pres_dynamic_impedance(inductance: f64, capacitance: f64, resistance: f64) -> f64 {
    assert!(
        inductance > 0.0,
        "l'inductance L doit être strictement positive"
    );
    assert!(
        capacitance > 0.0,
        "la capacité C doit être strictement positive"
    );
    assert!(
        resistance > 0.0,
        "la résistance R doit être strictement positive"
    );
    inductance / (capacitance * resistance)
}

/// Facteur de qualité de la bobine `Q = ω_0·L / R` (sans dimension).
///
/// Panique si `resonant_angular_frequency < 0`, si `inductance < 0` ou si
/// `resistance <= 0` (division par zéro).
pub fn pres_quality_factor(
    resonant_angular_frequency: f64,
    inductance: f64,
    resistance: f64,
) -> f64 {
    assert!(
        resonant_angular_frequency >= 0.0,
        "la pulsation de résonance ω_0 doit être ≥ 0"
    );
    assert!(inductance >= 0.0, "l'inductance L doit être ≥ 0");
    assert!(
        resistance > 0.0,
        "la résistance R doit être strictement positive"
    );
    resonant_angular_frequency * inductance / resistance
}

/// Largeur de bande à −3 dB `Δf = f_0 / Q` (Hz).
///
/// Panique si `resonant_frequency < 0` ou si `quality_factor <= 0`.
pub fn pres_bandwidth(resonant_frequency: f64, quality_factor: f64) -> f64 {
    assert!(
        resonant_frequency >= 0.0,
        "la fréquence de résonance f_0 doit être ≥ 0"
    );
    assert!(
        quality_factor > 0.0,
        "le facteur de qualité Q doit être strictement positif"
    );
    resonant_frequency / quality_factor
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn resonant_frequency_matches_angular_definition() {
        // Identité : ω_0 = 2·π·f_0 = 1/√(L·C). On vérifie que la fréquence
        // renvoyée redonne bien la pulsation propre attendue.
        let l = 0.05_f64;
        let c = 2.0e-6_f64;
        let f0 = pres_resonant_frequency(l, c);
        let omega0 = 2.0 * core::f64::consts::PI * f0;
        assert_relative_eq!(omega0, 1.0 / (l * c).sqrt(), epsilon = 1e-9);
    }

    #[test]
    fn dynamic_impedance_maximal_over_series_resistance() {
        // À la résonance R_d = L/(C·R) est très supérieure à R quand R est
        // faible devant la réactance : proportionnalité en 1/R.
        let l = 0.1_f64;
        let c = 1.0e-5_f64;
        let rd_small = pres_dynamic_impedance(l, c, 5.0);
        let rd_large = pres_dynamic_impedance(l, c, 10.0);
        // Doubler R divise l'impédance dynamique par deux.
        assert_relative_eq!(rd_small, 2.0 * rd_large, epsilon = 1e-9);
    }

    #[test]
    fn dynamic_impedance_equals_q_squared_times_r() {
        // Identité physique : R_d = L/(C·R) = Q²·R, avec Q = ω_0·L/R et
        // ω_0 = 1/√(L·C). On combine les trois fonctions sur les mêmes
        // composants pour vérifier la cohérence.
        let l = 0.2_f64;
        let c = 5.0e-6_f64;
        let r = 8.0_f64;
        let omega0 = 1.0 / (l * c).sqrt();
        let q = pres_quality_factor(omega0, l, r);
        let rd = pres_dynamic_impedance(l, c, r);
        assert_relative_eq!(rd, q * q * r, epsilon = 1e-6);
    }

    #[test]
    fn realistic_tank_circuit_case() {
        // Cas chiffré réaliste, L = 0,1 H, C = 10 µF, R = 10 Ω :
        //   f_0 = 1/(2π·√(0,1·1e-5)) = 1/(2π·1e-3) ≈ 159,154 9 Hz
        //   ω_0 = 1/√(0,1·1e-5) = 1/√(1e-6) = 1000 rad/s
        //   R_d = L/(C·R) = 0,1/(1e-5·10) = 0,1/1e-4 = 1000 Ω
        //   Q   = ω_0·L/R = 1000·0,1/10 = 10
        //   Δf  = f_0/Q ≈ 15,915 5 Hz
        let l = 0.1_f64;
        let c = 1.0e-5_f64;
        let r = 10.0_f64;
        assert_relative_eq!(pres_resonant_frequency(l, c), 159.154_9, epsilon = 1e-3);
        assert_relative_eq!(pres_dynamic_impedance(l, c, r), 1000.0, epsilon = 1e-9);
        let omega0 = 1.0 / (l * c).sqrt();
        assert_relative_eq!(omega0, 1000.0, epsilon = 1e-6);
        assert_relative_eq!(pres_quality_factor(omega0, l, r), 10.0, epsilon = 1e-9);
        let f0 = pres_resonant_frequency(l, c);
        let q = pres_quality_factor(omega0, l, r);
        assert_relative_eq!(pres_bandwidth(f0, q), 15.915_49, epsilon = 1e-3);
    }

    #[test]
    fn bandwidth_matches_r_over_two_pi_l() {
        // Identité : Δf = f_0/Q = R/(2·π·L). On la vérifie en combinant les
        // fonctions sur les mêmes composants.
        let l = 0.1_f64;
        let c = 1.0e-5_f64;
        let r = 10.0_f64;
        let f0 = pres_resonant_frequency(l, c);
        let omega0 = 1.0 / (l * c).sqrt();
        let q = pres_quality_factor(omega0, l, r);
        let bw = pres_bandwidth(f0, q);
        let expected = r / (2.0 * core::f64::consts::PI * l);
        assert_relative_eq!(bw, expected, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "la résistance R doit être strictement positive")]
    fn zero_resistance_dynamic_impedance_panics() {
        pres_dynamic_impedance(0.1, 1.0e-5, 0.0);
    }
}
