//! **Système triphasé équilibré (topologie étoile/triangle)** — relations entre
//! grandeurs de phase et grandeurs de ligne et puissance active à partir des
//! grandeurs efficaces de ligne.
//!
//! ```text
//! étoile   : tension de ligne     U_L = √3 · V_ph
//! étoile   : tension de phase     V_ph = U_L / √3
//! triangle : courant de ligne     I_L = √3 · I_ph
//! triangle : courant de phase     I_ph = I_L / √3
//! puissance active (ligne)        P = √3 · U_L · I_L · cos φ
//! ```
//!
//! `V_ph` valeur efficace de la tension de phase (V), `U_L` valeur efficace de la
//! tension de ligne / composée (V), `I_ph` valeur efficace du courant de phase
//! (A), `I_L` valeur efficace du courant de ligne (A), `cos φ` facteur de
//! puissance (sans dimension, `∈ [0, 1]`), `P` puissance active totale du système
//! triphasé (W). En couplage **étoile** on a `I_L = I_ph` et `U_L = √3 · V_ph` ;
//! en couplage **triangle** on a `U_L = U_ph` et `I_L = √3 · I_ph`.
//!
//! **Convention** : SI ; tensions en V, courants en A, puissances en W,
//! déphasage `φ` en **radians** lorsqu'il intervient dans une fonction
//! trigonométrique. **Limite honnête** : système triphasé **équilibré** et
//! **sinusoïdal permanent** ; les grandeurs **efficaces** de ligne et de phase
//! ainsi que le facteur de puissance sont **fournis par l'appelant** (mesures
//! réseau, plaque signalétique de la charge, couplage réel des enroulements) —
//! aucune valeur « par défaut » de réseau, de composant ou de matériau n'est
//! inventée.

/// Tension de ligne en couplage **étoile** `U_L = √3 · V_ph` (V).
///
/// Panique si `phase_voltage < 0`.
pub fn tps_line_voltage_star(phase_voltage: f64) -> f64 {
    assert!(
        phase_voltage >= 0.0,
        "la tension de phase V_ph doit être ≥ 0"
    );
    3.0_f64.sqrt() * phase_voltage
}

/// Tension de phase en couplage **étoile** `V_ph = U_L / √3` (V).
///
/// Panique si `line_voltage < 0`.
pub fn tps_phase_voltage_star(line_voltage: f64) -> f64 {
    assert!(line_voltage >= 0.0, "la tension de ligne U_L doit être ≥ 0");
    line_voltage / 3.0_f64.sqrt()
}

/// Courant de ligne en couplage **triangle** `I_L = √3 · I_ph` (A).
///
/// Panique si `phase_current < 0`.
pub fn tps_line_current_delta(phase_current: f64) -> f64 {
    assert!(
        phase_current >= 0.0,
        "le courant de phase I_ph doit être ≥ 0"
    );
    3.0_f64.sqrt() * phase_current
}

/// Courant de phase en couplage **triangle** `I_ph = I_L / √3` (A).
///
/// Panique si `line_current < 0`.
pub fn tps_phase_current_delta(line_current: f64) -> f64 {
    assert!(line_current >= 0.0, "le courant de ligne I_L doit être ≥ 0");
    line_current / 3.0_f64.sqrt()
}

/// Puissance active totale du système triphasé équilibré à partir des grandeurs
/// de **ligne** `P = √3 · U_L · I_L · cos φ` (W).
///
/// Panique si `line_voltage < 0`, si `line_current < 0` ou si `power_factor`
/// n'est pas dans `[0, 1]`.
pub fn tps_balanced_active_power(line_voltage: f64, line_current: f64, power_factor: f64) -> f64 {
    assert!(line_voltage >= 0.0, "la tension de ligne U_L doit être ≥ 0");
    assert!(line_current >= 0.0, "le courant de ligne I_L doit être ≥ 0");
    assert!(
        (0.0..=1.0).contains(&power_factor),
        "le facteur de puissance cos φ doit être dans [0, 1]"
    );
    3.0_f64.sqrt() * line_voltage * line_current * power_factor
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn star_voltage_round_trip() {
        // Réciprocité étoile : V_ph → U_L → V_ph retrouve la tension de phase.
        let v_ph = 230.0_f64;
        let u_l = tps_line_voltage_star(v_ph);
        assert_relative_eq!(tps_phase_voltage_star(u_l), v_ph, epsilon = 1e-12);
    }

    #[test]
    fn delta_current_round_trip() {
        // Réciprocité triangle : I_ph → I_L → I_ph retrouve le courant de phase.
        let i_ph = 12.5_f64;
        let i_l = tps_line_current_delta(i_ph);
        assert_relative_eq!(tps_phase_current_delta(i_l), i_ph, epsilon = 1e-12);
    }

    #[test]
    fn star_line_to_phase_ratio_is_sqrt3() {
        // Le rapport U_L / V_ph vaut √3 quelle que soit la tension de phase.
        let v_ph = 133.0_f64;
        let ratio = tps_line_voltage_star(v_ph) / v_ph;
        assert_relative_eq!(ratio, 3.0_f64.sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn active_power_via_phase_quantities() {
        // Identité : en étoile P = √3·U_L·I_L·cos φ = 3·V_ph·I_ph·cos φ, car
        // U_L = √3·V_ph et I_L = I_ph.
        let v_ph = 230.0_f64;
        let i_ph = 8.0_f64;
        let pf = 0.85_f64;
        let u_l = tps_line_voltage_star(v_ph);
        let i_l = i_ph; // étoile : I_L = I_ph
        let p_line = tps_balanced_active_power(u_l, i_l, pf);
        let p_phase = 3.0 * v_ph * i_ph * pf;
        assert_relative_eq!(p_line, p_phase, epsilon = 1e-9);
    }

    #[test]
    fn power_scales_linearly_with_current() {
        // Proportionnalité : doubler le courant de ligne double la puissance.
        let p1 = tps_balanced_active_power(400.0, 10.0, 0.8);
        let p2 = tps_balanced_active_power(400.0, 20.0, 0.8);
        assert_relative_eq!(p2 / p1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn realistic_400v_10a_case() {
        // Cas chiffré, U_L = 400 V, I_L = 10 A, cos φ = 0,8 :
        //   P = √3·400·10·0,8 = 3200·√3 ≈ 5542,5626 W
        let p = tps_balanced_active_power(400.0, 10.0, 0.8);
        assert_relative_eq!(p, 3200.0_f64 * 3.0_f64.sqrt(), epsilon = 1e-6);
        assert_relative_eq!(p, 5542.562584, epsilon = 1e-3);
    }

    #[test]
    #[should_panic(expected = "le facteur de puissance cos φ doit être dans [0, 1]")]
    fn power_factor_above_one_panics() {
        tps_balanced_active_power(400.0, 10.0, 1.5);
    }
}
