//! **Circuit RLC série en régime sinusoïdal** — module d'impédance, fréquence de
//! résonance, facteur de qualité, largeur de bande et déphasage d'un circuit
//! résistance–inductance–capacité placés en série.
//!
//! ```text
//! module d'impédance      |Z| = √(R² + (X_L − X_C)²)
//! fréquence de résonance  f_0 = 1 / (2·π·√(L·C))
//! facteur de qualité      Q   = (1/R)·√(L/C)          (à la résonance)
//! largeur de bande        Δf  = f_0 / Q
//! déphasage               φ   = atan((X_L − X_C) / R)
//! ```
//!
//! `R` résistance série (Ω), `X_L = ω·L` réactance inductive (Ω), `X_C = 1/(ω·C)`
//! réactance capacitive (Ω), `L` inductance (H), `C` capacité (F), `|Z|` module
//! de l'impédance série (Ω), `f_0` fréquence propre de résonance (Hz), `Q`
//! facteur de qualité (sans dimension), `Δf` largeur de bande à −3 dB (Hz),
//! `φ` déphasage tension–courant (radians, positif pour une charge inductive
//! `X_L > X_C`, négatif pour une charge capacitive).
//!
//! **Convention** : SI ; résistances et réactances en Ω, inductances en H,
//! capacités en F, fréquences en Hz, angle `φ` en **radians**. **Limite
//! honnête** : circuit RLC **série linéaire** en **régime sinusoïdal permanent**
//! ; à la résonance `X_L = X_C`, le déphasage s'annule et l'impédance est
//! **minimale et égale à R**. Les grandeurs réseau (fréquence, réactances) et
//! les valeurs des composants (`R`, `L`, `C`) sont **fournies par l'appelant**
//! (mesures, fiches composant) — aucune valeur « par défaut » n'est inventée.

/// Module de l'impédance série `|Z| = √(R² + (X_L − X_C)²)` (Ω).
///
/// Panique si `resistance < 0`, si `inductive_reactance < 0` ou si
/// `capacitive_reactance < 0`.
pub fn rlc_impedance_magnitude(
    resistance: f64,
    inductive_reactance: f64,
    capacitive_reactance: f64,
) -> f64 {
    assert!(resistance >= 0.0, "la résistance R doit être ≥ 0");
    assert!(
        inductive_reactance >= 0.0,
        "la réactance inductive X_L doit être ≥ 0"
    );
    assert!(
        capacitive_reactance >= 0.0,
        "la réactance capacitive X_C doit être ≥ 0"
    );
    let net_reactance = inductive_reactance - capacitive_reactance;
    (resistance * resistance + net_reactance.powi(2)).sqrt()
}

/// Fréquence de résonance `f_0 = 1 / (2·π·√(L·C))` (Hz).
///
/// Panique si `inductance <= 0` ou si `capacitance <= 0`.
pub fn rlc_resonant_frequency(inductance: f64, capacitance: f64) -> f64 {
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

/// Facteur de qualité à la résonance `Q = (1/R)·√(L/C)` (sans dimension).
///
/// Panique si `inductance <= 0`, si `capacitance <= 0` ou si `resistance <= 0`.
pub fn rlc_quality_factor(inductance: f64, capacitance: f64, resistance: f64) -> f64 {
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
    (1.0 / resistance) * (inductance / capacitance).sqrt()
}

/// Largeur de bande à −3 dB `Δf = f_0 / Q` (Hz).
///
/// Panique si `resonant_frequency < 0` ou si `quality_factor <= 0`.
pub fn rlc_bandwidth(resonant_frequency: f64, quality_factor: f64) -> f64 {
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

/// Déphasage tension–courant `φ = atan((X_L − X_C) / R)` (radians).
///
/// Panique si `resistance <= 0` (division par zéro), si `inductive_reactance < 0`
/// ou si `capacitive_reactance < 0`.
pub fn rlc_phase_angle(
    resistance: f64,
    inductive_reactance: f64,
    capacitive_reactance: f64,
) -> f64 {
    assert!(
        resistance > 0.0,
        "la résistance R doit être strictement positive"
    );
    assert!(
        inductive_reactance >= 0.0,
        "la réactance inductive X_L doit être ≥ 0"
    );
    assert!(
        capacitive_reactance >= 0.0,
        "la réactance capacitive X_C doit être ≥ 0"
    );
    ((inductive_reactance - capacitive_reactance) / resistance).atan()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn impedance_minimal_at_resonance() {
        // Cas limite : à la résonance X_L = X_C, le terme réactif s'annule et
        // le module d'impédance se réduit à la résistance série R.
        let r = 47.0_f64;
        let x = 315.0_f64;
        assert_relative_eq!(rlc_impedance_magnitude(r, x, x), r, epsilon = 1e-12);
    }

    #[test]
    fn phase_null_at_resonance() {
        // Cas limite : à la résonance X_L = X_C, le déphasage tension–courant
        // est nul (charge purement résistive vue des bornes).
        let r = 12.0_f64;
        let x = 88.0_f64;
        assert_relative_eq!(rlc_phase_angle(r, x, x), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn impedance_three_four_five_triangle() {
        // Cas chiffré exact via le triangle 3-4-5 :
        //   X_L − X_C = 7 − 3 = 4 Ω, R = 3 Ω
        //   |Z| = √(3² + 4²) = √25 = 5 Ω
        //   φ   = atan(4/3) = 0,927 295 218… rad
        let z = rlc_impedance_magnitude(3.0, 7.0, 3.0);
        assert_relative_eq!(z, 5.0, epsilon = 1e-12);
        let phi = rlc_phase_angle(3.0, 7.0, 3.0);
        assert_relative_eq!(phi, 0.927_295_218, epsilon = 1e-6);
    }

    #[test]
    fn bandwidth_matches_r_over_two_pi_l() {
        // Identité physique : Δf = f_0 / Q = R / (2·π·L). On la vérifie en
        // combinant les trois fonctions sur les mêmes composants.
        let l = 0.1_f64;
        let c = 1.0e-5_f64;
        let r = 10.0_f64;
        let f0 = rlc_resonant_frequency(l, c);
        let q = rlc_quality_factor(l, c, r);
        let bw = rlc_bandwidth(f0, q);
        let expected = r / (2.0 * core::f64::consts::PI * l);
        assert_relative_eq!(bw, expected, epsilon = 1e-9);
    }

    #[test]
    fn realistic_resonant_circuit_case() {
        // Cas chiffré réaliste, L = 0,1 H, C = 10 µF, R = 10 Ω :
        //   f_0 = 1/(2π·√(0,1·1e-5)) = 1/(2π·1e-3) ≈ 159,154 9 Hz
        //   Q   = (1/10)·√(0,1/1e-5) = (1/10)·√(1e4) = (1/10)·100 = 10
        //   Δf  = f_0 / Q ≈ 15,915 5 Hz
        let l = 0.1_f64;
        let c = 1.0e-5_f64;
        let r = 10.0_f64;
        assert_relative_eq!(rlc_resonant_frequency(l, c), 159.154_9, epsilon = 1e-3);
        assert_relative_eq!(rlc_quality_factor(l, c, r), 10.0, epsilon = 1e-9);
        let f0 = rlc_resonant_frequency(l, c);
        let q = rlc_quality_factor(l, c, r);
        assert_relative_eq!(rlc_bandwidth(f0, q), 15.915_49, epsilon = 1e-3);
    }

    #[test]
    fn capacitive_load_gives_negative_phase() {
        // Cohérence de signe : une charge dominée par la capacité (X_C > X_L)
        // produit un déphasage négatif (le courant est en avance sur la
        // tension).
        let phi = rlc_phase_angle(10.0, 5.0, 20.0);
        assert!(phi < 0.0, "X_C > X_L doit donner un déphasage négatif");
        assert_relative_eq!(phi, (-15.0_f64 / 10.0).atan(), epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "la résistance R doit être strictement positive")]
    fn zero_resistance_phase_panics() {
        rlc_phase_angle(0.0, 5.0, 3.0);
    }
}
