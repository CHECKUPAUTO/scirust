//! **Équivalents de Thévenin et Norton** — conversions entre les deux modèles,
//! courant dans une charge, puissance maximale transférée et rendement du
//! transfert, pour un réseau linéaire **résistif**.
//!
//! ```text
//! Thévenin → Norton  I_n = V_th / R_th
//! Norton → Thévenin  V_th = I_n · R_n
//! courant de charge  I_L  = V_th / (R_th + R_L)
//! puissance max      P_max = V_th² / (4 · R_th)   (charge adaptée R_L = R_th)
//! rendement          η    = R_L / (R_L + R_th)
//! ```
//!
//! `V_th` tension de la source de Thévenin (V), `R_th` résistance interne de
//! Thévenin (Ω), `I_n` courant de la source de Norton (A), `R_n` résistance
//! interne de Norton (Ω, avec `R_n = R_th`), `R_L` résistance de charge (Ω),
//! `I_L` courant circulant dans la charge (A), `P_max` puissance active maximale
//! transférable à la charge adaptée (W), `η` rendement du transfert de puissance
//! (sans dimension, dans `[0, 1[`).
//!
//! **Convention** : SI ; tensions en V, courants en A, résistances en Ω,
//! puissances en W. Types `f64`. **Limite honnête** : réseau linéaire
//! **résistif** (régime continu, ou grandeurs en module en régime sinusoïdal
//! permanent lorsque les réactances sont négligeables), avec `R_n = R_th`
//! (résistances internes des deux équivalents **égales**). Le point de transfert
//! maximal de puissance (`R_L = R_th`) n'a qu'un **rendement de 50 %**, la moitié
//! de la puissance étant dissipée dans `R_th`. Les grandeurs de l'équivalent
//! (`V_th`, `R_th`, `I_n`, `R_n`, `R_L`) sont **fournies par l'appelant** (issues
//! de l'analyse ou de la mesure du réseau) — aucune valeur n'est inventée.

/// Courant de la source de Norton équivalente à un dipôle de Thévenin :
/// `I_n = V_th / R_th` (A). C'est le courant de court-circuit du dipôle.
///
/// Panique si `thevenin_resistance <= 0` (division par zéro).
pub fn thev_norton_current(thevenin_voltage: f64, thevenin_resistance: f64) -> f64 {
    assert!(
        thevenin_resistance > 0.0,
        "la résistance de Thévenin R_th doit être strictement positive"
    );
    thevenin_voltage / thevenin_resistance
}

/// Tension de la source de Thévenin équivalente à un dipôle de Norton :
/// `V_th = I_n · R_n` (V). C'est la tension à vide du dipôle.
///
/// Panique si `norton_resistance < 0` (résistance interne physiquement ≥ 0).
pub fn thev_thevenin_voltage(norton_current: f64, norton_resistance: f64) -> f64 {
    assert!(
        norton_resistance >= 0.0,
        "la résistance de Norton R_n doit être ≥ 0"
    );
    norton_current * norton_resistance
}

/// Courant circulant dans une charge `R_L` alimentée par le dipôle de Thévenin :
/// `I_L = V_th / (R_th + R_L)` (A).
///
/// Panique si `thevenin_resistance < 0`, si `load_resistance < 0`, ou si la somme
/// `thevenin_resistance + load_resistance <= 0` (division par zéro).
pub fn thev_load_current(
    thevenin_voltage: f64,
    thevenin_resistance: f64,
    load_resistance: f64,
) -> f64 {
    assert!(
        thevenin_resistance >= 0.0,
        "la résistance de Thévenin R_th doit être ≥ 0"
    );
    assert!(
        load_resistance >= 0.0,
        "la résistance de charge R_L doit être ≥ 0"
    );
    assert!(
        thevenin_resistance + load_resistance > 0.0,
        "la résistance totale R_th + R_L doit être strictement positive"
    );
    thevenin_voltage / (thevenin_resistance + load_resistance)
}

/// Puissance active maximale transférable à une charge **adaptée** `R_L = R_th` :
/// `P_max = V_th² / (4 · R_th)` (W). À l'adaptation, le rendement n'est que de
/// 50 % (autant de puissance dissipée dans `R_th` que dans la charge).
///
/// Panique si `thevenin_resistance <= 0` (division par zéro).
pub fn thev_maximum_power_transfer(thevenin_voltage: f64, thevenin_resistance: f64) -> f64 {
    assert!(
        thevenin_resistance > 0.0,
        "la résistance de Thévenin R_th doit être strictement positive"
    );
    thevenin_voltage * thevenin_voltage / (4.0 * thevenin_resistance)
}

/// Rendement du transfert de puissance vers la charge (fraction de la puissance
/// totale reçue par `R_L`) : `η = R_L / (R_L + R_th)` (sans dimension).
///
/// Panique si `load_resistance < 0`, si `thevenin_resistance < 0`, ou si la somme
/// `load_resistance + thevenin_resistance <= 0` (division par zéro).
pub fn thev_power_transfer_efficiency(load_resistance: f64, thevenin_resistance: f64) -> f64 {
    assert!(
        load_resistance >= 0.0,
        "la résistance de charge R_L doit être ≥ 0"
    );
    assert!(
        thevenin_resistance >= 0.0,
        "la résistance de Thévenin R_th doit être ≥ 0"
    );
    assert!(
        load_resistance + thevenin_resistance > 0.0,
        "la résistance totale R_L + R_th doit être strictement positive"
    );
    load_resistance / (load_resistance + thevenin_resistance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn thevenin_norton_round_trip() {
        // Réciprocité : convertir Thévenin → Norton puis Norton → Thévenin
        // (avec R_n = R_th) restitue la tension de départ.
        let v_th = 24.0_f64;
        let r_th = 8.0_f64;
        let i_n = thev_norton_current(v_th, r_th);
        let v_back = thev_thevenin_voltage(i_n, r_th);
        assert_relative_eq!(v_back, v_th, epsilon = 1e-9);
    }

    #[test]
    fn short_circuit_current_matches_norton() {
        // Cas limite : sous R_L = 0, le courant de charge est le courant de
        // court-circuit, égal au courant de Norton I_n = V_th/R_th.
        let v_th = 12.0_f64;
        let r_th = 4.0_f64;
        let i_sc = thev_load_current(v_th, r_th, 0.0);
        assert_relative_eq!(i_sc, thev_norton_current(v_th, r_th), epsilon = 1e-12);
    }

    #[test]
    fn matched_load_delivers_maximum_power() {
        // Cas chiffré réaliste : V_th = 20 V, R_th = 5 Ω, charge adaptée R_L = 5 Ω.
        //   I_L = 20 / (5 + 5) = 2 A
        //   P_L = I_L² · R_L = 4 · 5 = 20 W
        //   P_max = V_th² / (4·R_th) = 400 / 20 = 20 W
        // Les deux calculs concordent : la puissance sur charge adaptée est bien
        // le maximum théorique.
        let v_th = 20.0_f64;
        let r_th = 5.0_f64;
        let r_l = 5.0_f64;
        let i_l = thev_load_current(v_th, r_th, r_l);
        assert_relative_eq!(i_l, 2.0, epsilon = 1e-9);
        let p_load = i_l * i_l * r_l;
        assert_relative_eq!(p_load, 20.0, epsilon = 1e-9);
        let p_max = thev_maximum_power_transfer(v_th, r_th);
        assert_relative_eq!(p_max, 20.0, epsilon = 1e-9);
        assert_relative_eq!(p_load, p_max, epsilon = 1e-9);
    }

    #[test]
    fn efficiency_is_fifty_percent_at_match() {
        // Identité : à l'adaptation R_L = R_th, le rendement vaut exactement 50 %.
        let r_th = 15.0_f64;
        let eta = thev_power_transfer_efficiency(r_th, r_th);
        assert_relative_eq!(eta, 0.5, epsilon = 1e-12);
    }

    #[test]
    fn efficiency_tends_to_one_for_large_load() {
        // Cas limite : quand R_L ≫ R_th, le rendement tend vers 1 (mais la
        // puissance transférée diminue). Ici R_L = 999·R_th → η = 999/1000.
        let r_th = 2.0_f64;
        let r_l = 999.0 * r_th;
        let eta = thev_power_transfer_efficiency(r_l, r_th);
        assert_relative_eq!(eta, 0.999, epsilon = 1e-9);
    }

    #[test]
    fn power_scales_with_voltage_squared() {
        // Proportionnalité : à R_th fixée, P_max ∝ V_th², donc doubler V_th
        // quadruple P_max.
        let r_th = 10.0_f64;
        let p1 = thev_maximum_power_transfer(6.0, r_th);
        let p2 = thev_maximum_power_transfer(12.0, r_th);
        assert_relative_eq!(p2 / p1, 4.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "la résistance de Thévenin R_th doit être strictement positive")]
    fn zero_thevenin_resistance_panics() {
        thev_norton_current(24.0, 0.0);
    }
}
