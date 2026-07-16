//! **Courant de court-circuit (méthode des impédances / per-unit)** — courant de
//! base triphasé, courant de court-circuit symétrique, puissance de court-circuit
//! et courant de défaut aux bornes d'un transformateur alimenté par un réseau
//! amont de puissance infinie.
//!
//! ```text
//! courant de base triphasé      I_base = S_base / (√3 · U_base)
//! courant de défaut symétrique  I_cc   = I_base / z_pu
//! puissance de court-circuit    S_cc   = S_base / z_pu
//! Icc aux bornes d'un transfo    I_cc   = I_n · 100 / u_cc%
//! ```
//!
//! `S_base` puissance apparente de base (VA), `U_base` tension composée
//! (entre phases) de base (V), `I_base` courant de ligne de base (A), `z_pu`
//! impédance de court-circuit **totale** ramenée sur la base, en per-unit
//! (sans dimension, `> 0`), `I_cc` courant de court-circuit symétrique efficace
//! (A), `S_cc` puissance de court-circuit apparente (VA), `I_n` courant nominal
//! (de ligne) du transformateur (A) et `u_cc%` tension de court-circuit du
//! transformateur en pourcent (sans dimension, `> 0`).
//!
//! **Convention** : SI ; tensions composées efficaces en V, courants efficaces
//! en A, puissances apparentes en VA ; **régime sinusoïdal permanent**.
//! **Limite honnête** : défaut **triphasé symétrique** uniquement ; les
//! impédances (source, transformateur, câbles) sont **fournies par l'appelant**
//! en per-unit sur la base choisie, ou en pourcent pour le transformateur, et
//! le réseau amont est supposé de **puissance infinie** sauf si son impédance
//! est incluse dans `z_pu` — aucune valeur « par défaut » n'est inventée. Ce
//! module ne traite **pas** les défauts dissymétriques (monophasé, biphasé) :
//! les composantes symétriques restent à la charge de l'appelant. Les
//! impédances sont traitées en **module réel** (pas de représentation complexe
//! R + jX).

/// Courant de ligne de base triphasé `I_base = S_base / (√3 · U_base)`.
///
/// `base_power` est la puissance apparente de base (VA), `base_line_voltage` la
/// tension composée (entre phases) de base (V) ; le résultat est en ampères.
///
/// Panique si `base_power < 0` ou si `base_line_voltage <= 0`.
pub fn scc_base_current(base_power: f64, base_line_voltage: f64) -> f64 {
    assert!(base_power >= 0.0, "S_base ≥ 0 requis");
    assert!(base_line_voltage > 0.0, "U_base > 0 requis");
    base_power / (3.0_f64.sqrt() * base_line_voltage)
}

/// Courant de court-circuit symétrique `I_cc = I_base / z_pu`.
///
/// `base_current` est le courant de base (A) et `per_unit_impedance` l'impédance
/// de court-circuit totale en per-unit sur la même base ; le résultat est en
/// ampères.
///
/// Panique si `base_current < 0` ou si `per_unit_impedance <= 0`.
pub fn scc_symmetrical_fault_current(base_current: f64, per_unit_impedance: f64) -> f64 {
    assert!(base_current >= 0.0, "I_base ≥ 0 requis");
    assert!(per_unit_impedance > 0.0, "z_pu > 0 requis");
    base_current / per_unit_impedance
}

/// Puissance de court-circuit apparente `S_cc = S_base / z_pu`.
///
/// `base_power` est la puissance apparente de base (VA) et `per_unit_impedance`
/// l'impédance de court-circuit totale en per-unit sur la même base ; le
/// résultat est en voltampères.
///
/// Panique si `base_power < 0` ou si `per_unit_impedance <= 0`.
pub fn scc_fault_power(base_power: f64, per_unit_impedance: f64) -> f64 {
    assert!(base_power >= 0.0, "S_base ≥ 0 requis");
    assert!(per_unit_impedance > 0.0, "z_pu > 0 requis");
    base_power / per_unit_impedance
}

/// Courant de court-circuit aux bornes secondaires d'un transformateur alimenté
/// par un réseau amont de puissance infinie `I_cc = I_n · 100 / u_cc%`.
///
/// `rated_current` est le courant nominal (de ligne) du transformateur (A) et
/// `percent_impedance` sa tension de court-circuit en pourcent (`u_cc%`) ; le
/// résultat est en ampères.
///
/// Panique si `rated_current < 0` ou si `percent_impedance <= 0`.
pub fn scc_transformer_secondary_fault(rated_current: f64, percent_impedance: f64) -> f64 {
    assert!(rated_current >= 0.0, "I_n ≥ 0 requis");
    assert!(percent_impedance > 0.0, "u_cc% > 0 requis");
    rated_current * 100.0 / percent_impedance
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn base_current_reconstructs_base_power() {
        // Réciprocité : S_base = √3 · U_base · I_base.
        let s_base = 100.0e6_f64;
        let u_base = 20.0e3_f64;
        let i_base = scc_base_current(s_base, u_base);
        assert_relative_eq!(3.0_f64.sqrt() * u_base * i_base, s_base, epsilon = 1e-3);
    }

    #[test]
    fn base_current_realistic_value() {
        // Cas chiffré : 100 MVA sous 20 kV → 100e6/(√3·20000) ≈ 2886,751 A.
        let i_base = scc_base_current(100.0e6, 20.0e3);
        assert_relative_eq!(i_base, 2886.751346, epsilon = 1e-3);
    }

    #[test]
    fn unit_impedance_gives_base_values() {
        // Cas limite z_pu = 1 : I_cc = I_base et S_cc = S_base.
        assert_relative_eq!(
            scc_symmetrical_fault_current(577.35, 1.0),
            577.35,
            epsilon = 1e-12
        );
        assert_relative_eq!(scc_fault_power(50.0e6, 1.0), 50.0e6, epsilon = 1e-3);
    }

    #[test]
    fn fault_current_and_power_share_ratio() {
        // Identité : I_cc/I_base = S_cc/S_base = 1/z_pu.
        let s_base = 10.0e6_f64;
        let u_base = 10.0e3_f64;
        let z_pu = 0.1_f64;
        let i_base = scc_base_current(s_base, u_base);
        let i_cc = scc_symmetrical_fault_current(i_base, z_pu);
        let s_cc = scc_fault_power(s_base, z_pu);
        assert_relative_eq!(i_cc / i_base, s_cc / s_base, epsilon = 1e-9);
        // Cohérence triphasée : S_cc = √3 · U_base · I_cc.
        assert_relative_eq!(3.0_f64.sqrt() * u_base * i_cc, s_cc, epsilon = 1e-3);
    }

    #[test]
    fn halving_impedance_doubles_fault_current() {
        // Proportionnalité inverse : diviser z_pu par 2 double I_cc.
        let i1 = scc_symmetrical_fault_current(1000.0, 0.2);
        let i2 = scc_symmetrical_fault_current(1000.0, 0.1);
        assert_relative_eq!(i2 / i1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn transformer_fault_matches_per_unit_form() {
        // Cas chiffré : I_n = 1000 A, u_cc% = 5 → 1000·100/5 = 20000 A.
        let i_cc = scc_transformer_secondary_fault(1000.0, 5.0);
        assert_relative_eq!(i_cc, 20_000.0, epsilon = 1e-9);
        // Identité pourcent ↔ per-unit : u_cc% = 5 équivaut à z_pu = 0,05.
        assert_relative_eq!(
            i_cc,
            scc_symmetrical_fault_current(1000.0, 0.05),
            epsilon = 1e-9
        );
    }

    #[test]
    #[should_panic(expected = "z_pu > 0 requis")]
    fn zero_impedance_panics() {
        scc_symmetrical_fault_current(1000.0, 0.0);
    }
}
