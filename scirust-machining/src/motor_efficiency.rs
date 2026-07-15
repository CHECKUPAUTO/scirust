//! **Rendement et pertes globales d'un moteur** — rapport puissance
//! utile/puissance absorbée, puissance à fournir pour une sortie visée,
//! pertes totales dissipées et couple mécanique développé à l'arbre.
//!
//! ```text
//! rendement            η     = P_out / P_in
//! puissance absorbée   P_in  = P_out / η
//! pertes totales       P_loss = P_in − P_out
//! couple à l'arbre     T     = P_out / ω
//! ```
//!
//! `P_out` puissance mécanique utile (W), `P_in` puissance absorbée (W), `η`
//! rendement global (sans dimension, `∈ [0, 1]`), `P_loss` pertes totales
//! dissipées (W), `T` couple mécanique à l'arbre (N·m), `ω` vitesse angulaire
//! de l'arbre (rad/s, `> 0`).
//!
//! **Convention** : SI ; puissances en watts ; couple en N·m ; vitesse
//! angulaire en rad/s. **Limite honnête** : simple **bilan de puissance en
//! régime permanent** ; le rendement et les puissances sont **fournis par
//! l'appelant** (issus d'essais ou de la plaque signalétique), aucune valeur
//! « par défaut » n'est inventée. Ce module donne un **indicateur global** et
//! ne **décompose pas** les pertes (fer, cuivre, mécaniques/ventilation) —
//! `P_loss` en est la somme.

/// Rendement global du moteur `η = P_out / P_in`.
///
/// Panique si `input_power <= 0`, si `output_power < 0` ou si
/// `output_power > input_power` (rendement physiquement borné par 1).
pub fn motor_efficiency(output_power: f64, input_power: f64) -> f64 {
    assert!(input_power > 0.0, "P_in > 0 requis");
    assert!(output_power >= 0.0, "P_out ≥ 0 requis");
    assert!(
        output_power <= input_power,
        "P_out ≤ P_in requis (rendement ≤ 1)"
    );
    output_power / input_power
}

/// Puissance électrique/hydraulique à absorber pour une sortie visée
/// `P_in = P_out / η`.
///
/// Panique si `output_power < 0` ou si `efficiency` n'est pas dans `]0, 1]`.
pub fn motor_input_power(output_power: f64, efficiency: f64) -> f64 {
    assert!(output_power >= 0.0, "P_out ≥ 0 requis");
    assert!(efficiency > 0.0 && efficiency <= 1.0, "η ∈ ]0, 1] requis");
    output_power / efficiency
}

/// Pertes totales dissipées `P_loss = P_in − P_out` (somme non décomposée des
/// pertes fer, cuivre et mécaniques).
///
/// Panique si `input_power < 0`, si `output_power < 0` ou si
/// `output_power > input_power` (les pertes ne peuvent être négatives).
pub fn motor_losses(input_power: f64, output_power: f64) -> f64 {
    assert!(input_power >= 0.0, "P_in ≥ 0 requis");
    assert!(output_power >= 0.0, "P_out ≥ 0 requis");
    assert!(
        output_power <= input_power,
        "P_out ≤ P_in requis (pertes ≥ 0)"
    );
    input_power - output_power
}

/// Couple mécanique développé à l'arbre `T = P_out / ω`.
///
/// Panique si `output_power < 0` ou si `angular_speed_rad <= 0`.
pub fn motor_output_torque(output_power: f64, angular_speed_rad: f64) -> f64 {
    assert!(output_power >= 0.0, "P_out ≥ 0 requis");
    assert!(angular_speed_rad > 0.0, "ω > 0 requis");
    output_power / angular_speed_rad
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn efficiency_and_input_power_are_reciprocal() {
        // Réciprocité : retrouver P_in à partir du rendement qu'il produit.
        let p_out = 9_000.0_f64;
        let p_in = 10_000.0_f64;
        let eta = motor_efficiency(p_out, p_in);
        assert_relative_eq!(motor_input_power(p_out, eta), p_in, epsilon = 1e-9);
    }

    #[test]
    fn output_plus_losses_conserves_power() {
        // Conservation : P_out + P_loss = P_in quel que soit le rendement.
        let p_in = 7_500.0_f64;
        let p_out = 6_900.0_f64;
        assert_relative_eq!(p_out + motor_losses(p_in, p_out), p_in, epsilon = 1e-9);
    }

    #[test]
    fn lossless_motor_has_unit_efficiency() {
        // Cas limite : moteur idéal sans pertes → η = 1, P_loss = 0.
        let p = 3_200.0_f64;
        assert_relative_eq!(motor_efficiency(p, p), 1.0, epsilon = 1e-15);
        assert_relative_eq!(motor_losses(p, p), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn realistic_motor_case() {
        // Cas chiffré : moteur absorbant 10 kW pour 9 kW utiles.
        let p_in = 10_000.0_f64;
        let p_out = 9_000.0_f64;
        assert_relative_eq!(motor_efficiency(p_out, p_in), 0.9, epsilon = 1e-12);
        assert_relative_eq!(motor_losses(p_in, p_out), 1_000.0, epsilon = 1e-9);
        // Pour 9 kW utiles à η = 0.9, il faut bien absorber 10 kW.
        assert_relative_eq!(motor_input_power(p_out, 0.9), 10_000.0, epsilon = 1e-9);
    }

    #[test]
    fn output_torque_scales_inversely_with_speed() {
        // Cas chiffré + proportionnalité : 5 kW à 100 rad/s → 50 N·m ;
        // doubler ω à puissance constante divise le couple par deux.
        let p_out = 5_000.0_f64;
        assert_relative_eq!(motor_output_torque(p_out, 100.0), 50.0, epsilon = 1e-12);
        assert_relative_eq!(motor_output_torque(p_out, 200.0), 25.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "η ∈ ]0, 1] requis")]
    fn zero_efficiency_panics() {
        motor_input_power(1_000.0, 0.0);
    }
}
