//! **Transformateurs de mesure (TC / TP)** — grandeurs secondaires idéales,
//! erreur de rapport, impédance de charge (burden) et facteur limite de précision
//! d'un transformateur de courant (TC) ou de tension (TP), à partir des rapports
//! de transformation et des puissances de précision fournis par la fiche.
//!
//! ```text
//! courant secondaire (TC)      I_s = I_p / K_n
//! tension secondaire (TP)      U_s = U_p / K_n
//! erreur de rapport            ε = (K_m − K_n) / K_n
//! impédance de charge (burden) Z_b = S_b / I_s²
//! facteur limite de précision  ALF = I_sat / I_pn
//! ```
//!
//! `I_p` courant primaire (efficace, A), `U_p` tension primaire (efficace, V),
//! `K_n` rapport de transformation nominal (primaire/secondaire, sans dimension,
//! `> 0`), `I_s` courant secondaire (efficace, A), `U_s` tension secondaire
//! (efficace, V), `K_m` rapport de transformation mesuré (sans dimension), `ε`
//! erreur de rapport (sans dimension), `S_b` puissance de précision imposée au
//! secondaire (burden, VA, `>= 0`), `Z_b` impédance de charge équivalente (Ω),
//! `I_sat` courant primaire de saturation (A), `I_pn` courant primaire assigné
//! (A, `> 0`) et `ALF` facteur limite de précision (accuracy limit factor, sans
//! dimension).
//!
//! **Convention** : SI ; courants efficaces en A, tensions efficaces en V,
//! puissance de précision en VA, impédance en Ω ; grandeurs sans dimension pour
//! les rapports, l'erreur de rapport et le facteur limite de précision ; **régime
//! établi** (grandeurs efficaces sinusoïdales, pas de représentation complexe).
//!
//! **Limite honnête** : transformateurs de mesure supposés **idéaux au rapport**
//! (le rapport de transformation `K_n`, les rapports mesurés `K_m` et les
//! puissances de précision `S_b` sont **fournis par la fiche**, pas déduits d'un
//! circuit magnétique). L'erreur de rapport et l'erreur de phase **réelles**
//! dépendent de la charge secondaire effective et de la saturation du noyau : ce
//! module ne les modélise pas, il ne fait qu'appliquer la définition de l'erreur
//! de rapport à un rapport mesuré **fourni**. Le facteur limite de précision
//! borne la mesure d'un TC de protection **en régime de défaut** (au-delà de
//! `ALF · I_pn`, le TC sature et la mesure n'est plus garantie), mais ce module ne
//! simule pas la saturation elle-même. Arithmétique **réelle** (f64).

/// Courant secondaire idéal d'un transformateur de courant (TC)
/// `I_s = I_p / K_n`.
///
/// `primary_current` est le courant primaire efficace (`I_p`, A) et `turns_ratio`
/// le rapport de transformation nominal primaire/secondaire (`K_n`, sans
/// dimension) ; le résultat est le courant secondaire efficace (A).
///
/// Panique si `primary_current < 0` ou si `turns_ratio <= 0`.
pub fn insttr_ct_secondary_current(primary_current: f64, turns_ratio: f64) -> f64 {
    assert!(primary_current >= 0.0, "I_p ≥ 0 requis");
    assert!(turns_ratio > 0.0, "K_n > 0 requis");
    primary_current / turns_ratio
}

/// Tension secondaire idéale d'un transformateur de tension (TP)
/// `U_s = U_p / K_n`.
///
/// `primary_voltage` est la tension primaire efficace (`U_p`, V) et `turns_ratio`
/// le rapport de transformation nominal primaire/secondaire (`K_n`, sans
/// dimension) ; le résultat est la tension secondaire efficace (V).
///
/// Panique si `primary_voltage < 0` ou si `turns_ratio <= 0`.
pub fn insttr_vt_secondary_voltage(primary_voltage: f64, turns_ratio: f64) -> f64 {
    assert!(primary_voltage >= 0.0, "U_p ≥ 0 requis");
    assert!(turns_ratio > 0.0, "K_n > 0 requis");
    primary_voltage / turns_ratio
}

/// Erreur de rapport d'un transformateur de mesure
/// `ε = (K_m − K_n) / K_n`.
///
/// `measured_ratio` est le rapport de transformation mesuré (`K_m`, sans
/// dimension) et `nominal_ratio` le rapport nominal (`K_n`, sans dimension) ; le
/// résultat est l'erreur de rapport (sans dimension ; multiplier par 100 pour un
/// pourcentage).
///
/// Panique si `nominal_ratio <= 0`.
pub fn insttr_ratio_error(measured_ratio: f64, nominal_ratio: f64) -> f64 {
    assert!(nominal_ratio > 0.0, "K_n > 0 requis");
    (measured_ratio - nominal_ratio) / nominal_ratio
}

/// Impédance de charge (burden) d'un TC pour une puissance de précision donnée
/// `Z_b = S_b / I_s²`.
///
/// `burden_va` est la puissance de précision imposée au secondaire (`S_b`, VA) et
/// `secondary_current` le courant secondaire assigné (`I_s`, A) ; le résultat est
/// l'impédance de charge équivalente (Ω).
///
/// Panique si `burden_va < 0` ou si `secondary_current <= 0`.
pub fn insttr_burden_impedance(burden_va: f64, secondary_current: f64) -> f64 {
    assert!(burden_va >= 0.0, "S_b ≥ 0 requis");
    assert!(secondary_current > 0.0, "I_s > 0 requis");
    burden_va / (secondary_current * secondary_current)
}

/// Facteur limite de précision d'un TC de protection
/// `ALF = I_sat / I_pn`.
///
/// `saturation_current` est le courant primaire de saturation (`I_sat`, A) et
/// `rated_primary_current` le courant primaire assigné (`I_pn`, A) ; le résultat
/// est le facteur limite de précision (sans dimension).
///
/// Panique si `saturation_current < 0` ou si `rated_primary_current <= 0`.
pub fn insttr_accuracy_limit_factor(saturation_current: f64, rated_primary_current: f64) -> f64 {
    assert!(saturation_current >= 0.0, "I_sat ≥ 0 requis");
    assert!(rated_primary_current > 0.0, "I_pn > 0 requis");
    saturation_current / rated_primary_current
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn ct_secondary_current_rated_case() {
        // TC 1000/5 : K_n = 1000/5 = 200 ; au primaire assigné I_p = 1000 A
        // → I_s = 1000 / 200 = 5 A (courant secondaire assigné).
        // Recalcul : 1000 / 200 = 5.0.
        let i_s = insttr_ct_secondary_current(1000.0, 200.0);
        assert_relative_eq!(i_s, 5.0, epsilon = 1e-9);
    }

    #[test]
    fn ct_secondary_proportional_to_primary() {
        // Le secondaire est proportionnel au primaire à rapport fixe : doubler le
        // courant primaire double le courant secondaire.
        let ratio = 200.0;
        let a = insttr_ct_secondary_current(300.0, ratio);
        let b = insttr_ct_secondary_current(600.0, ratio);
        assert_relative_eq!(b, 2.0 * a, epsilon = 1e-9);
    }

    #[test]
    fn vt_secondary_voltage_rated_case() {
        // TP 20000/100 : K_n = 20000/100 = 200 ; U_p = 20000 V
        // → U_s = 20000 / 200 = 100 V. Recalcul : 20000 / 200 = 100.0.
        let u_s = insttr_vt_secondary_voltage(20000.0, 200.0);
        assert_relative_eq!(u_s, 100.0, epsilon = 1e-9);
    }

    #[test]
    fn ratio_error_zero_when_measured_equals_nominal() {
        // Identité : si le rapport mesuré égale le rapport nominal, l'erreur de
        // rapport est nulle.
        let eps = insttr_ratio_error(200.0, 200.0);
        assert_relative_eq!(eps, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn ratio_error_computed_case() {
        // K_m = 201, K_n = 200 → ε = (201 − 200)/200 = 1/200 = 0,005 (soit +0,5 %).
        // Recalcul : (201 - 200) / 200 = 1 / 200 = 0.005.
        let eps = insttr_ratio_error(201.0, 200.0);
        assert_relative_eq!(eps, 0.005, epsilon = 1e-9);
    }

    #[test]
    fn burden_impedance_computed_case() {
        // TC à secondaire 5 A, burden 30 VA → Z_b = 30 / 5² = 30 / 25 = 1,2 Ω.
        // Recalcul : 30 / (5 * 5) = 30 / 25 = 1.2.
        let z_b = insttr_burden_impedance(30.0, 5.0);
        assert_relative_eq!(z_b, 1.2, epsilon = 1e-9);
    }

    #[test]
    fn accuracy_limit_factor_computed_case() {
        // TC de protection 100 A assigné, saturation à 2000 A
        // → ALF = 2000 / 100 = 20 (classe 5P20 typique).
        // Recalcul : 2000 / 100 = 20.0.
        let alf = insttr_accuracy_limit_factor(2000.0, 100.0);
        assert_relative_eq!(alf, 20.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "K_n > 0 requis")]
    fn ct_secondary_panics_on_nonpositive_ratio() {
        let _ = insttr_ct_secondary_current(1000.0, 0.0);
    }
}
