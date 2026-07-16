//! **Transformateur monophasé (schéma équivalent)** — rapport de transformation,
//! tension au secondaire, ramenée d'impédance au primaire, chute de tension
//! (régulation) et rendement à partir des pertes cuivre et fer.
//!
//! ```text
//! rapport de transformation   a      = N_p / N_s
//! tension secondaire          V_s    = V_p / a
//! impédance ramenée primaire  Z'     = a² · Z_s
//! régulation de tension       ε      = (V_nl − V_fl) / V_fl
//! rendement                   η      = P_out / (P_out + P_cu + P_fe)
//! ```
//!
//! `N_p`, `N_s` nombres de spires des enroulements primaire et secondaire
//! (sans dimension, `> 0`), `a` rapport de transformation (sans dimension),
//! `V_p`, `V_s` tensions primaire et secondaire (V), `Z_s` impédance côté
//! secondaire (Ω), `Z'` cette impédance ramenée au primaire (Ω), `V_nl` tension
//! secondaire à vide et `V_fl` en charge (V), `ε` régulation (sans dimension),
//! `P_out` puissance utile délivrée (W), `P_cu` pertes cuivre (effet Joule dans
//! les enroulements, W), `P_fe` pertes fer (hystérésis + courants de Foucault,
//! W), `η` rendement (sans dimension, `∈ [0, 1]`).
//!
//! **Convention** : SI ; tensions en V, impédances en Ω, puissances en W ;
//! grandeurs efficaces en **régime sinusoïdal permanent**. **Limite honnête** :
//! transformateur **monophasé** décrit par son **schéma équivalent** à
//! **magnétisation linéaire** (pas de saturation ni d'harmoniques) ; le rapport
//! de transformation et les pertes **cuivre/fer** sont **fournis par
//! l'appelant** (plaque signalétique, essais à vide et en court-circuit),
//! aucune valeur « par défaut » n'est inventée. Les impédances sont traitées en
//! module réel (Ω) ; ce module ne modélise pas le déphasage complexe.

/// Rapport de transformation `a = N_p / N_s` (spires primaire sur secondaire).
///
/// Panique si `primary_turns <= 0` ou si `secondary_turns <= 0`.
pub fn xfmr_turns_ratio(primary_turns: f64, secondary_turns: f64) -> f64 {
    assert!(primary_turns > 0.0, "N_p > 0 requis");
    assert!(secondary_turns > 0.0, "N_s > 0 requis");
    primary_turns / secondary_turns
}

/// Tension au secondaire `V_s = V_p / a` (transformateur idéal du schéma).
///
/// Panique si `primary_voltage < 0` ou si `turns_ratio <= 0`.
pub fn xfmr_secondary_voltage(primary_voltage: f64, turns_ratio: f64) -> f64 {
    assert!(primary_voltage >= 0.0, "V_p ≥ 0 requis");
    assert!(turns_ratio > 0.0, "a > 0 requis");
    primary_voltage / turns_ratio
}

/// Impédance secondaire ramenée au primaire `Z' = a² · Z_s`.
///
/// Panique si `secondary_impedance < 0` ou si `turns_ratio <= 0`.
pub fn xfmr_referred_impedance_to_primary(secondary_impedance: f64, turns_ratio: f64) -> f64 {
    assert!(secondary_impedance >= 0.0, "Z_s ≥ 0 requis");
    assert!(turns_ratio > 0.0, "a > 0 requis");
    turns_ratio * turns_ratio * secondary_impedance
}

/// Régulation de tension `ε = (V_nl − V_fl) / V_fl` (chute relative à vide → en
/// charge).
///
/// Panique si `no_load_voltage < 0` ou si `full_load_voltage <= 0`.
pub fn xfmr_voltage_regulation(no_load_voltage: f64, full_load_voltage: f64) -> f64 {
    assert!(no_load_voltage >= 0.0, "V_nl ≥ 0 requis");
    assert!(full_load_voltage > 0.0, "V_fl > 0 requis");
    (no_load_voltage - full_load_voltage) / full_load_voltage
}

/// Rendement `η = P_out / (P_out + P_cu + P_fe)` (pertes cuivre et fer).
///
/// Panique si `output_power <= 0`, si `copper_loss < 0` ou si `iron_loss < 0`.
pub fn xfmr_efficiency(output_power: f64, copper_loss: f64, iron_loss: f64) -> f64 {
    assert!(output_power > 0.0, "P_out > 0 requis");
    assert!(copper_loss >= 0.0, "P_cu ≥ 0 requis");
    assert!(iron_loss >= 0.0, "P_fe ≥ 0 requis");
    output_power / (output_power + copper_loss + iron_loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn ratio_and_secondary_voltage_are_reciprocal() {
        // Réciprocité : V_p = a · V_s, donc V_s · a redonne V_p.
        let a = xfmr_turns_ratio(2400.0, 240.0);
        let v_p = 2400.0_f64;
        let v_s = xfmr_secondary_voltage(v_p, a);
        assert_relative_eq!(v_s * a, v_p, epsilon = 1e-9);
    }

    #[test]
    fn unit_ratio_transformer_keeps_voltage() {
        // Cas limite : rapport 1:1 → V_s = V_p et Z' = Z_s.
        let a = xfmr_turns_ratio(500.0, 500.0);
        assert_relative_eq!(a, 1.0, epsilon = 1e-15);
        assert_relative_eq!(xfmr_secondary_voltage(230.0, a), 230.0, epsilon = 1e-12);
        assert_relative_eq!(
            xfmr_referred_impedance_to_primary(0.8, a),
            0.8,
            epsilon = 1e-12
        );
    }

    #[test]
    fn referred_impedance_scales_with_ratio_squared() {
        // Proportionnalité : doubler `a` quadruple l'impédance ramenée.
        let z_s = 0.5_f64;
        let z1 = xfmr_referred_impedance_to_primary(z_s, 5.0);
        let z2 = xfmr_referred_impedance_to_primary(z_s, 10.0);
        assert_relative_eq!(z2 / z1, 4.0, epsilon = 1e-12);
    }

    #[test]
    fn realistic_2400_to_240_transformer() {
        // Cas chiffré : abaisseur 2400 V / 240 V, a = 10.
        let a = xfmr_turns_ratio(2400.0, 240.0);
        assert_relative_eq!(a, 10.0, epsilon = 1e-12);
        // Tension secondaire à vide : 2400 / 10 = 240 V.
        assert_relative_eq!(xfmr_secondary_voltage(2400.0, a), 240.0, epsilon = 1e-9);
        // Impédance secondaire 0,5 Ω ramenée au primaire : 10² · 0,5 = 50 Ω.
        assert_relative_eq!(
            xfmr_referred_impedance_to_primary(0.5, a),
            50.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn regulation_and_efficiency_realistic_values() {
        // Régulation : à vide 252 V, en charge 240 V → (252 − 240)/240 = 0,05.
        assert_relative_eq!(xfmr_voltage_regulation(252.0, 240.0), 0.05, epsilon = 1e-12);
        // Rendement : 98 kW utiles, 1 kW cuivre + 1 kW fer → 98000/100000 = 0,98.
        assert_relative_eq!(
            xfmr_efficiency(98_000.0, 1_000.0, 1_000.0),
            0.98,
            epsilon = 1e-12
        );
    }

    #[test]
    fn lossless_transformer_has_unit_efficiency() {
        // Cas limite : sans pertes cuivre ni fer → η = 1.
        assert_relative_eq!(xfmr_efficiency(50_000.0, 0.0, 0.0), 1.0, epsilon = 1e-15);
    }

    #[test]
    #[should_panic(expected = "P_out > 0 requis")]
    fn zero_output_power_panics() {
        xfmr_efficiency(0.0, 100.0, 50.0);
    }
}
