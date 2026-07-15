//! Élément piézoélectrique (capteur / actionneur) — charge, tension, allongement,
//! énergie récupérée et effort bloqué dans l'approximation linéaire du coefficient
//! de charge.
//!
//! ```text
//! charge (mode direct)      Q  = d · F
//! tension aux bornes        V  = Q / C
//! allongement (mode inverse) ΔL = d · V · n
//! énergie récupérée         W  = 0.5 · Q · V
//! effort bloqué             F_b = d · V · k
//! ```
//!
//! `d` coefficient de charge piézoélectrique (C/N ≡ m/V), `F` force appliquée (N),
//! `Q` charge générée (C), `C` capacité de l'élément (F), `V` tension (V), `n`
//! nombre de couches de l'empilement (sans dimension), `ΔL` allongement (m), `W`
//! énergie (J), `k` raideur de l'actionneur (N/m), `F_b` effort bloqué (N). Le mode
//! **direct** couvre le capteur et la récupération d'énergie ; le mode **inverse**
//! couvre l'actionneur.
//!
//! **Convention** : SI cohérent (mètre, newton, coulomb, farad, volt, joule).
//! **Limite honnête** : comportement **linéaire** — le coefficient de charge `d`
//! est supposé **constant** (hors saturation et dépolarisation) et **fourni par
//! l'appelant** selon le matériau ; la capacité `C` et la raideur `k` sont
//! également **fournies**, aucune valeur « par défaut » n'est inventée. On néglige
//! l'**hystérésis**, le **fluage** et la dépendance en **température** et en
//! **fréquence** ; ce module ne remplace pas une caractérisation expérimentale ni
//! un modèle constitutif complet (ex. modèle de Preisach).

/// Charge générée en mode direct `Q = d · F` (C).
///
/// Panique si `charge_coefficient_d <= 0` ou si `applied_force` n'est pas fini.
pub fn piezo_generated_charge(charge_coefficient_d: f64, applied_force: f64) -> f64 {
    assert!(
        charge_coefficient_d > 0.0,
        "le coefficient de charge d doit être strictement positif (C/N)"
    );
    assert!(
        applied_force.is_finite(),
        "la force appliquée doit être finie (N)"
    );
    charge_coefficient_d * applied_force
}

/// Tension aux bornes `V = Q / C` (V).
///
/// Panique si `capacitance <= 0` ou si `generated_charge` n'est pas fini.
pub fn piezo_output_voltage(generated_charge: f64, capacitance: f64) -> f64 {
    assert!(
        generated_charge.is_finite(),
        "la charge générée doit être finie (C)"
    );
    assert!(
        capacitance > 0.0,
        "la capacité doit être strictement positive (F)"
    );
    generated_charge / capacitance
}

/// Allongement d'un empilement en mode inverse `ΔL = d · V · n` (m).
///
/// Panique si `charge_coefficient_d <= 0`, si `layer_count < 1` ou si
/// `applied_voltage` n'est pas fini.
pub fn piezo_actuator_displacement(
    charge_coefficient_d: f64,
    applied_voltage: f64,
    layer_count: f64,
) -> f64 {
    assert!(
        charge_coefficient_d > 0.0,
        "le coefficient de charge d doit être strictement positif (m/V)"
    );
    assert!(
        applied_voltage.is_finite(),
        "la tension appliquée doit être finie (V)"
    );
    assert!(
        layer_count >= 1.0,
        "le nombre de couches doit être supérieur ou égal à 1"
    );
    charge_coefficient_d * applied_voltage * layer_count
}

/// Énergie récupérée `W = 0.5 · Q · V` (J).
///
/// Panique si `generated_charge` ou `output_voltage` n'est pas fini.
pub fn piezo_generated_energy(generated_charge: f64, output_voltage: f64) -> f64 {
    assert!(
        generated_charge.is_finite(),
        "la charge générée doit être finie (C)"
    );
    assert!(
        output_voltage.is_finite(),
        "la tension de sortie doit être finie (V)"
    );
    0.5 * generated_charge * output_voltage
}

/// Effort bloqué d'un actionneur `F_b = d · V · k` (N) : force développée lorsque
/// l'allongement libre est entièrement empêché par une raideur `k`.
///
/// Panique si `charge_coefficient_d <= 0`, `stiffness <= 0` ou si `applied_voltage`
/// n'est pas fini.
pub fn piezo_blocking_force(
    charge_coefficient_d: f64,
    applied_voltage: f64,
    stiffness: f64,
) -> f64 {
    assert!(
        charge_coefficient_d > 0.0,
        "le coefficient de charge d doit être strictement positif (m/V)"
    );
    assert!(
        applied_voltage.is_finite(),
        "la tension appliquée doit être finie (V)"
    );
    assert!(
        stiffness > 0.0,
        "la raideur doit être strictement positive (N/m)"
    );
    charge_coefficient_d * applied_voltage * stiffness
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn direct_mode_reference_case() {
        // PZT d33 ≈ 300 pC/N, F = 100 N → Q = 300e-12 · 100 = 3e-8 C = 30 nC.
        let q = piezo_generated_charge(300e-12, 100.0);
        assert_relative_eq!(q, 3e-8, epsilon = 1e-18);
        // Avec C = 10 nF → V = 3e-8 / 10e-9 = 3 V.
        let v = piezo_output_voltage(q, 10e-9);
        assert_relative_eq!(v, 3.0, epsilon = 1e-12);
    }

    #[test]
    fn charge_proportional_to_force() {
        // À d fixé, doubler la force double la charge.
        let q1 = piezo_generated_charge(300e-12, 50.0);
        let q2 = piezo_generated_charge(300e-12, 100.0);
        assert_relative_eq!(q2 / q1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn energy_reference_case() {
        // Q = 3e-8 C, V = 3 V → W = 0.5 · 3e-8 · 3 = 4.5e-8 J = 45 nJ.
        let w = piezo_generated_energy(3e-8, 3.0);
        assert_relative_eq!(w, 4.5e-8, epsilon = 1e-18);
    }

    #[test]
    fn stack_scales_with_layer_count() {
        // ΔL ∝ n : un empilement de 10 couches allonge 10 fois plus qu'une seule.
        let one = piezo_actuator_displacement(300e-12, 100.0, 1.0);
        let ten = piezo_actuator_displacement(300e-12, 100.0, 10.0);
        assert_relative_eq!(ten / one, 10.0, epsilon = 1e-12);
        // Cas chiffré : d = 300 pm/V, V = 100 V, n = 10 → ΔL = 3e-7 m = 300 nm.
        assert_relative_eq!(ten, 3e-7, epsilon = 1e-18);
    }

    #[test]
    fn blocking_force_equals_free_displacement_times_stiffness() {
        // Identité : F_b = (d · V) · k = allongement libre monocouche · raideur.
        let d = 300e-12;
        let v = 100.0;
        let k = 100e6;
        let free_displacement = piezo_actuator_displacement(d, v, 1.0);
        let f_block = piezo_blocking_force(d, v, k);
        assert_relative_eq!(f_block, free_displacement * k, epsilon = 1e-15);
        // Cas chiffré : 300e-12 · 100 · 100e6 = 3 N.
        assert_relative_eq!(f_block, 3.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "coefficient de charge d")]
    fn zero_coefficient_panics() {
        piezo_generated_charge(0.0, 100.0);
    }
}
