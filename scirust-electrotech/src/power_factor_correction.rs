//! **Compensation du facteur de puissance par condensateur** — puissance
//! réactive à fournir, capacité, courant du condensateur et puissance apparente
//! corrigée, en régime sinusoïdal monophasé à puissance active constante.
//!
//! ```text
//! réactive à compenser  Q_c = P · (tan(acos(pf_i)) − tan(acos(pf_t)))
//! capacité (monophasé)  C   = Q_c / (2·π·f·V_rms²)
//! courant condensateur  I_c = Q_c / V_rms
//! apparente corrigée    S'  = P / pf_t
//! ```
//!
//! `P` puissance active de la charge (W, constante avant/après compensation),
//! `pf_i` facteur de puissance initial (`cos φ_i`, sans dimension),
//! `pf_t` facteur de puissance cible (`cos φ_t`, sans dimension),
//! `Q_c` puissance réactive fournie par le condensateur (var),
//! `C` capacité du condensateur de compensation (F), `f` fréquence réseau (Hz),
//! `V_rms` valeur efficace de la tension aux bornes du condensateur (V),
//! `I_c` valeur efficace du courant dans le condensateur (A),
//! `S'` puissance apparente après compensation (VA).
//!
//! **Convention** : SI ; tensions en V, courants en A, puissances en W/var/VA,
//! capacités en F, fréquences en Hz, angles en **radians** dans les fonctions
//! trigonométriques. **Limite honnête** : compensation par **condensateur** en
//! régime **sinusoïdal** permanent, puissance active supposée **constante** ;
//! les facteurs de puissance initial et cible (`0 < pf ≤ 1`), la tension
//! efficace, la fréquence réseau et la puissance active sont **fournis par
//! l'appelant** (mesures réseau, plaque signalétique de la charge) — aucune
//! valeur « par défaut » n'est inventée. Formules **monophasées** : le triphasé
//! adapte les grandeurs selon le montage étoile/triangle des condensateurs, à la
//! charge de l'appelant.

use core::f64::consts::PI;

/// Puissance réactive à fournir par le condensateur pour passer du facteur de
/// puissance `pf_i` au facteur cible `pf_t` :
/// `Q_c = P · (tan(acos(pf_i)) − tan(acos(pf_t)))` (var).
///
/// Panique si `active_power < 0`, ou si `initial_power_factor` ou
/// `target_power_factor` n'est pas dans `]0, 1]`.
pub fn pfc_required_reactive_power(
    active_power: f64,
    initial_power_factor: f64,
    target_power_factor: f64,
) -> f64 {
    assert!(active_power >= 0.0, "la puissance active P doit être ≥ 0");
    assert!(
        initial_power_factor > 0.0 && initial_power_factor <= 1.0,
        "le facteur de puissance initial pf_i doit être dans ]0, 1]"
    );
    assert!(
        target_power_factor > 0.0 && target_power_factor <= 1.0,
        "le facteur de puissance cible pf_t doit être dans ]0, 1]"
    );
    active_power * (initial_power_factor.acos().tan() - target_power_factor.acos().tan())
}

/// Capacité du condensateur de compensation monophasé fournissant la puissance
/// réactive `Q_c` sous la tension efficace `V_rms` à la fréquence `f` :
/// `C = Q_c / (2·π·f·V_rms²)` (F).
///
/// Panique si `reactive_power < 0`, si `voltage_rms <= 0` ou si
/// `frequency <= 0` (division par zéro).
pub fn pfc_capacitance(reactive_power: f64, voltage_rms: f64, frequency: f64) -> f64 {
    assert!(
        reactive_power >= 0.0,
        "la puissance réactive Q_c doit être ≥ 0"
    );
    assert!(
        voltage_rms > 0.0,
        "la tension efficace V_rms doit être strictement positive"
    );
    assert!(
        frequency > 0.0,
        "la fréquence f doit être strictement positive"
    );
    reactive_power / (2.0 * PI * frequency * voltage_rms * voltage_rms)
}

/// Valeur efficace du courant dans le condensateur de compensation :
/// `I_c = Q_c / V_rms` (A).
///
/// Panique si `reactive_power < 0` ou si `voltage_rms <= 0` (division par zéro).
pub fn pfc_capacitor_current(reactive_power: f64, voltage_rms: f64) -> f64 {
    assert!(
        reactive_power >= 0.0,
        "la puissance réactive Q_c doit être ≥ 0"
    );
    assert!(
        voltage_rms > 0.0,
        "la tension efficace V_rms doit être strictement positive"
    );
    reactive_power / voltage_rms
}

/// Puissance apparente de la charge après compensation, à puissance active
/// constante : `S' = P / pf_t` (VA).
///
/// Panique si `active_power < 0`, ou si `target_power_factor` n'est pas dans
/// `]0, 1]`.
pub fn pfc_corrected_apparent_power(active_power: f64, target_power_factor: f64) -> f64 {
    assert!(active_power >= 0.0, "la puissance active P doit être ≥ 0");
    assert!(
        target_power_factor > 0.0 && target_power_factor <= 1.0,
        "le facteur de puissance cible pf_t doit être dans ]0, 1]"
    );
    active_power / target_power_factor
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn required_reactive_realistic_case() {
        // Cas chiffré, P = 12000 W, pf_i = 0,6, pf_t = 0,8 :
        //   tan(acos 0,6) = 0,8/0,6 = 4/3
        //   tan(acos 0,8) = 0,6/0,8 = 3/4
        //   Q_c = 12000·(4/3 − 3/4) = 12000·(16/12 − 9/12) = 12000·7/12 = 7000 var
        let p = 12_000.0_f64;
        let q_c = pfc_required_reactive_power(p, 0.6, 0.8);
        assert_relative_eq!(q_c, 7000.0, epsilon = 1e-6);
    }

    #[test]
    fn no_reactive_when_target_equals_initial() {
        // Cas limite : si la cible est déjà atteinte, aucune compensation
        // n'est requise, Q_c = 0.
        let q_c = pfc_required_reactive_power(9_500.0, 0.85, 0.85);
        assert_relative_eq!(q_c, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn required_reactive_scales_with_active_power() {
        // Proportionnalité : à facteurs de puissance fixés, Q_c est linéaire
        // en P, donc doubler P double Q_c.
        let q1 = pfc_required_reactive_power(5_000.0, 0.7, 0.95);
        let q2 = pfc_required_reactive_power(10_000.0, 0.7, 0.95);
        assert_relative_eq!(q2 / q1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn capacitance_round_trip() {
        // Réciprocité : la capacité obtenue restitue bien Q_c sous
        // Q_c = 2·π·f·V_rms²·C.
        let q_c = 7_000.0_f64;
        let v = 230.0_f64;
        let f = 50.0_f64;
        let c = pfc_capacitance(q_c, v, f);
        assert_relative_eq!(2.0 * PI * f * v * v * c, q_c, epsilon = 1e-6);
    }

    #[test]
    fn capacitor_current_matches_reactive_definition() {
        // Réciprocité : I_c = Q_c/V_rms implique Q_c = V_rms·I_c (le
        // condensateur ne consomme que du réactif).
        let q_c = 7_000.0_f64;
        let v = 230.0_f64;
        let i_c = pfc_capacitor_current(q_c, v);
        assert_relative_eq!(v * i_c, q_c, epsilon = 1e-9);
    }

    #[test]
    fn corrected_apparent_power_consistency() {
        // Cas chiffré, P = 12000 W, pf_t = 0,8 : S' = 12000/0,8 = 15000 VA.
        // Cohérence avec le triangle des puissances : la réactive résiduelle
        // vaut Q' = P·tan(acos 0,8) = 12000·0,75 = 9000 var, et
        // √(P² + Q'²) = √(12000² + 9000²) = √(225·10⁶) = 15000 VA.
        let p = 12_000.0_f64;
        let pf_t = 0.8_f64;
        let s_prime = pfc_corrected_apparent_power(p, pf_t);
        assert_relative_eq!(s_prime, 15_000.0, epsilon = 1e-6);
        let q_residual = p * pf_t.acos().tan();
        assert_relative_eq!(
            (p * p + q_residual * q_residual).sqrt(),
            s_prime,
            epsilon = 1e-6
        );
    }

    #[test]
    #[should_panic(expected = "le facteur de puissance cible pf_t doit être dans ]0, 1]")]
    fn zero_target_power_factor_panics() {
        pfc_corrected_apparent_power(12_000.0, 0.0);
    }
}
