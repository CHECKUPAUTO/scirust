//! **Régime transitoire d'un circuit RLC série (réponse libre)** — coefficient
//! d'amortissement, pulsation propre non amortie, facteur d'amortissement,
//! pulsation amortie et classification du régime d'un circuit résistance–
//! inductance–capacité en série lorsque les sources sont éteintes.
//!
//! ```text
//! coefficient d'amortissement   α  = R / (2·L)
//! pulsation propre non amortie  ω0 = 1 / √(L·C)
//! facteur d'amortissement       ζ  = α / ω0
//! pulsation amortie (ζ < 1)     ωd = ω0·√(1 − ζ²)
//! régime : ζ < 1 sous-amorti | ζ = 1 critique | ζ > 1 sur-amorti
//! ```
//!
//! `R` résistance série (Ω), `L` inductance (H), `C` capacité (F), `α`
//! coefficient d'amortissement (s⁻¹), `ω0` pulsation propre non amortie
//! (rad/s), `ζ` facteur d'amortissement (sans dimension), `ωd` pulsation des
//! oscillations amorties (rad/s, uniquement en régime sous-amorti `ζ < 1`).
//!
//! **Convention** : SI ; résistance en Ω, inductance en H, capacité en F,
//! pulsations en **rad/s**. **Limite honnête** : circuit RLC **série** en
//! **réponse libre** (sources éteintes, régime transitoire) — à distinguer du
//! module `rlc_series` qui traite le **régime sinusoïdal permanent établi**. La
//! pulsation amortie `ωd` n'a de sens que dans le cas **sous-amorti** `ζ < 1` ;
//! au-delà (`ζ ≥ 1`) la réponse est apériodique et n'oscille pas. Les valeurs
//! des composants (`R`, `L`, `C`) sont **fournies par l'appelant** (fiches
//! composant, mesures) — aucune valeur « par défaut » n'est inventée.

/// Régime transitoire d'un circuit RLC série selon le facteur d'amortissement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RlcRegime {
    /// Sous-amorti (`ζ < 1`) : la réponse oscille en s'atténuant.
    Underdamped,
    /// Critique (`ζ = 1`) : retour le plus rapide sans oscillation.
    Critical,
    /// Sur-amorti (`ζ > 1`) : réponse apériodique lente, sans oscillation.
    Overdamped,
}

/// Coefficient d'amortissement `α = R / (2·L)` (s⁻¹).
///
/// Panique si `resistance < 0` ou si `inductance <= 0`.
pub fn rlctr_damping_coefficient(resistance: f64, inductance: f64) -> f64 {
    assert!(resistance >= 0.0, "la résistance R doit être ≥ 0");
    assert!(
        inductance > 0.0,
        "l'inductance L doit être strictement positive"
    );
    resistance / (2.0 * inductance)
}

/// Pulsation propre non amortie `ω0 = 1 / √(L·C)` (rad/s).
///
/// Panique si `inductance <= 0` ou si `capacitance <= 0`.
pub fn rlctr_undamped_natural_frequency(inductance: f64, capacitance: f64) -> f64 {
    assert!(
        inductance > 0.0,
        "l'inductance L doit être strictement positive"
    );
    assert!(
        capacitance > 0.0,
        "la capacité C doit être strictement positive"
    );
    1.0 / (inductance * capacitance).sqrt()
}

/// Facteur d'amortissement `ζ = α / ω0` (sans dimension).
///
/// Panique si `damping_coefficient < 0` ou si `undamped_natural_frequency <= 0`.
pub fn rlctr_damping_ratio(damping_coefficient: f64, undamped_natural_frequency: f64) -> f64 {
    assert!(
        damping_coefficient >= 0.0,
        "le coefficient d'amortissement α doit être ≥ 0"
    );
    assert!(
        undamped_natural_frequency > 0.0,
        "la pulsation propre ω0 doit être strictement positive"
    );
    damping_coefficient / undamped_natural_frequency
}

/// Pulsation des oscillations amorties `ωd = ω0·√(1 − ζ²)` (rad/s).
///
/// Panique si `undamped_natural_frequency < 0`, si `damping_ratio < 0` ou si
/// `damping_ratio >= 1` (la pulsation amortie n'existe qu'en régime
/// sous-amorti).
pub fn rlctr_damped_frequency(undamped_natural_frequency: f64, damping_ratio: f64) -> f64 {
    assert!(
        undamped_natural_frequency >= 0.0,
        "la pulsation propre ω0 doit être ≥ 0"
    );
    assert!(
        damping_ratio >= 0.0,
        "le facteur d'amortissement ζ doit être ≥ 0"
    );
    assert!(
        damping_ratio < 1.0,
        "la pulsation amortie n'existe que pour ζ < 1 (régime sous-amorti)"
    );
    undamped_natural_frequency * (1.0 - damping_ratio * damping_ratio).sqrt()
}

/// Classement du régime transitoire à partir du facteur d'amortissement `ζ`.
///
/// Renvoie [`RlcRegime::Underdamped`] pour `ζ < 1`, [`RlcRegime::Critical`]
/// pour `ζ = 1` et [`RlcRegime::Overdamped`] pour `ζ > 1`.
///
/// Panique si `damping_ratio < 0`.
pub fn rlctr_regime(damping_ratio: f64) -> RlcRegime {
    assert!(
        damping_ratio >= 0.0,
        "le facteur d'amortissement ζ doit être ≥ 0"
    );
    if damping_ratio < 1.0
    {
        RlcRegime::Underdamped
    }
    else if damping_ratio > 1.0
    {
        RlcRegime::Overdamped
    }
    else
    {
        RlcRegime::Critical
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn damping_ratio_from_components_matches_direct_formula() {
        // Identité : ζ = α/ω0 = [R/(2L)] / [1/√(L·C)] = (R/2)·√(C/L).
        let r = 20.0_f64;
        let l = 0.5_f64;
        let c = 200.0e-6_f64;
        let alpha = rlctr_damping_coefficient(r, l);
        let w0 = rlctr_undamped_natural_frequency(l, c);
        let zeta = rlctr_damping_ratio(alpha, w0);
        let expected = (r / 2.0) * (c / l).sqrt();
        assert_relative_eq!(zeta, expected, epsilon = 1e-12);
    }

    #[test]
    fn damped_frequency_reduces_to_natural_when_undamped() {
        // Cas limite ζ = 0 (R = 0) : aucune perte, ωd = ω0·√(1) = ω0.
        let w0 = 100.0_f64;
        assert_relative_eq!(rlctr_damped_frequency(w0, 0.0), w0, epsilon = 1e-12);
    }

    #[test]
    fn damped_frequency_never_exceeds_natural_frequency() {
        // Proportionnalité/borne : pour 0 ≤ ζ < 1, ωd ≤ ω0 (le facteur
        // √(1−ζ²) est dans ]0, 1]).
        let w0 = 100.0_f64;
        let zeta = 0.6_f64;
        let wd = rlctr_damped_frequency(w0, zeta);
        assert!(wd < w0, "la pulsation amortie doit rester < ω0 pour ζ > 0");
        assert_relative_eq!(wd, w0 * 0.8, epsilon = 1e-12); // √(1−0,36)=√0,64=0,8
    }

    #[test]
    fn regime_classification_boundaries() {
        // Frontières du classement : sous-amorti / critique / sur-amorti.
        assert_eq!(rlctr_regime(0.2), RlcRegime::Underdamped);
        assert_eq!(rlctr_regime(1.0), RlcRegime::Critical);
        assert_eq!(rlctr_regime(2.5), RlcRegime::Overdamped);
    }

    #[test]
    fn realistic_underdamped_case() {
        // Cas chiffré réaliste : R = 20 Ω, L = 0,5 H, C = 200 µF.
        //   α  = R/(2L)        = 20/(2·0,5) = 20/1        = 20 s⁻¹
        //   ω0 = 1/√(L·C)      = 1/√(0,5·2e-4) = 1/√1e-4  = 1/0,01 = 100 rad/s
        //   ζ  = α/ω0          = 20/100                   = 0,2
        //   ωd = ω0·√(1−ζ²)    = 100·√(1−0,04) = 100·√0,96 ≈ 97,979 589 71 rad/s
        let r = 20.0_f64;
        let l = 0.5_f64;
        let c = 200.0e-6_f64;
        let alpha = rlctr_damping_coefficient(r, l);
        let w0 = rlctr_undamped_natural_frequency(l, c);
        let zeta = rlctr_damping_ratio(alpha, w0);
        let wd = rlctr_damped_frequency(w0, zeta);
        assert_relative_eq!(alpha, 20.0, epsilon = 1e-12);
        assert_relative_eq!(w0, 100.0, epsilon = 1e-9);
        assert_relative_eq!(zeta, 0.2, epsilon = 1e-12);
        assert_relative_eq!(wd, 97.979_589_711, epsilon = 1e-6);
        assert_eq!(rlctr_regime(zeta), RlcRegime::Underdamped);
    }

    #[test]
    #[should_panic(expected = "la pulsation amortie n'existe que pour ζ < 1")]
    fn damped_frequency_panics_when_overdamped() {
        rlctr_damped_frequency(100.0, 1.5);
    }
}
