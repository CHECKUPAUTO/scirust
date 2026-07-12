//! Boucle de **contre-réaction** — gain en boucle fermée, sensibilité et erreur
//! statique d'un asservissement.
//!
//! ```text
//! retour unitaire   T = G/(1 + G)
//! retour quelconque T = G/(1 + G·H)
//! sensibilité       S = 1/(1 + G)                (S + T = 1 en retour unitaire)
//! erreur statique   e_∞ = 1/(1 + Kp)             (échelon, système de type 0)
//! ```
//!
//! `G` gain de la chaîne directe (boucle ouverte), `H` gain de la chaîne de
//! retour, `T` gain en boucle fermée (fonction de transfert complémentaire), `S`
//! sensibilité (effet d'une variation de `G` sur `T`), `Kp` constante d'erreur de
//! position (gain statique de boucle ouverte), `e_∞` erreur statique en réponse à
//! un échelon.
//!
//! **Convention** : gains **statiques** sans dimension (valeurs réelles à basse
//! fréquence). **Limite honnête** : raisonnement sur les **gains continus**
//! (DC), pas sur les fonctions de transfert complètes ; l'erreur statique
//! `1/(1+Kp)` vaut pour un **échelon** sur un système de **type 0** (sans
//! intégrateur — sinon l'erreur est nulle). `G`, `H`, `Kp` sont fournis par
//! l'appelant.

/// Gain en boucle fermée à **retour unitaire** `T = G/(1 + G)`.
///
/// Panique si `1 + G == 0` (pôle en boucle fermée).
pub fn closed_loop_gain(open_loop_gain: f64) -> f64 {
    let denom = 1.0 + open_loop_gain;
    assert!(denom != 0.0, "1 + G nul : la boucle fermée est singulière");
    open_loop_gain / denom
}

/// Gain en boucle fermée à **retour quelconque** `T = G/(1 + G·H)`.
///
/// Panique si `1 + G·H == 0`.
pub fn closed_loop_gain_with_feedback(forward_gain: f64, feedback_gain: f64) -> f64 {
    let denom = 1.0 + forward_gain * feedback_gain;
    assert!(
        denom != 0.0,
        "1 + G·H nul : la boucle fermée est singulière"
    );
    forward_gain / denom
}

/// Sensibilité `S = 1/(1 + G)`.
///
/// Panique si `1 + G == 0`.
pub fn sensitivity(open_loop_gain: f64) -> f64 {
    let denom = 1.0 + open_loop_gain;
    assert!(denom != 0.0, "1 + G nul : sensibilité infinie");
    1.0 / denom
}

/// Erreur statique à un échelon (système de type 0) `e_∞ = 1/(1 + Kp)`.
///
/// Panique si `1 + Kp == 0`.
pub fn steady_state_error_step(position_error_constant: f64) -> f64 {
    let denom = 1.0 + position_error_constant;
    assert!(denom != 0.0, "1 + Kp nul");
    1.0 / denom
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn high_gain_pushes_closed_loop_to_unity() {
        // G ≫ 1 → T → 1 (bon suivi de consigne).
        assert!(closed_loop_gain(1000.0) > 0.999);
        assert_relative_eq!(closed_loop_gain(1.0), 0.5, epsilon = 1e-12);
    }

    #[test]
    fn sensitivity_and_closed_loop_sum_to_one() {
        // Retour unitaire : S + T = 1.
        let g = 9.0;
        assert_relative_eq!(sensitivity(g) + closed_loop_gain(g), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn feedback_gain_sets_closed_loop_level() {
        // T = G/(1+G·H). G=100, H=0,1 → 100/11 ≈ 9,09 (≈ 1/H pour G·H ≫ 1).
        let t = closed_loop_gain_with_feedback(100.0, 0.1);
        assert_relative_eq!(t, 100.0 / 11.0, epsilon = 1e-9);
        assert!(t > 9.0 && t < 9.1);
    }

    #[test]
    fn steady_state_error_shrinks_with_gain() {
        // Kp grand → erreur statique faible. Kp=99 → e=1 %.
        assert_relative_eq!(steady_state_error_step(99.0), 0.01, epsilon = 1e-12);
        assert!(steady_state_error_step(999.0) < steady_state_error_step(99.0));
    }

    #[test]
    #[should_panic(expected = "singulière")]
    fn unity_negative_gain_panics() {
        closed_loop_gain(-1.0);
    }
}
