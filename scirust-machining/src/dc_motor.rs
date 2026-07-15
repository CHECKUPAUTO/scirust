//! **Moteur à courant continu** idéal — force contre-électromotrice, couple,
//! tension aux bornes et vitesse d'un moteur à excitation constante.
//!
//! ```text
//! f.c.é.m.        E = Ke·ω                         (Ke = Kt en SI)
//! couple          C = Kt·Ia
//! tension bornes  V = E + Ia·Ra
//! vitesse         ω = (V − Ia·Ra) / Kt
//! ```
//!
//! `Ke` constante de f.c.é.m. (V·s/rad), `Kt` constante de couple (N·m/A) —
//! **numériquement égales** en unités SI, `ω` vitesse angulaire (rad/s), `E`
//! f.c.é.m. induite (V), `Ia` courant d'induit (A), `Ra` résistance d'induit
//! (Ω), `V` tension aux bornes (V), `C` couple électromagnétique (N·m).
//!
//! **Convention** : SI cohérent. **Limite honnête** : machine **idéale
//! linéaire** à flux **constant** (`Kt = Ke`) ; on néglige la réaction d'induit,
//! la saturation magnétique, l'inductance et les pertes fer/frottement. Toutes
//! les constantes physiques (constante de couple, résistance d'induit) sont
//! **fournies par l'appelant** — aucune valeur « par défaut » n'est inventée.

/// Force contre-électromotrice `E = Ke·ω` (V).
///
/// `torque_constant` en SI vaut `Kt = Ke` (N·m/A = V·s/rad).
/// Panique si `torque_constant <= 0`.
pub fn dc_back_emf(torque_constant: f64, angular_speed_rad: f64) -> f64 {
    assert!(
        torque_constant > 0.0,
        "la constante de couple/f.c.é.m. doit être strictement positive"
    );
    torque_constant * angular_speed_rad
}

/// Couple électromagnétique `C = Kt·Ia` (N·m).
///
/// Panique si `torque_constant <= 0`.
pub fn dc_torque(torque_constant: f64, armature_current: f64) -> f64 {
    assert!(
        torque_constant > 0.0,
        "la constante de couple doit être strictement positive"
    );
    torque_constant * armature_current
}

/// Tension aux bornes `V = E + Ia·Ra` (V).
///
/// Panique si `armature_resistance < 0`.
pub fn dc_terminal_voltage(back_emf: f64, armature_current: f64, armature_resistance: f64) -> f64 {
    assert!(
        armature_resistance >= 0.0,
        "la résistance d'induit doit être positive"
    );
    back_emf + armature_current * armature_resistance
}

/// Vitesse angulaire `ω = (V − Ia·Ra) / Kt` (rad/s).
///
/// Panique si `torque_constant <= 0` ou `armature_resistance < 0`.
pub fn dc_speed_rad(
    terminal_voltage: f64,
    armature_current: f64,
    armature_resistance: f64,
    torque_constant: f64,
) -> f64 {
    assert!(
        torque_constant > 0.0,
        "la constante de couple/f.c.é.m. doit être strictement positive"
    );
    assert!(
        armature_resistance >= 0.0,
        "la résistance d'induit doit être positive"
    );
    (terminal_voltage - armature_current * armature_resistance) / torque_constant
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Machine de référence : Kt = 0,05 N·m/A, Ra = 0,5 Ω.
    const KT: f64 = 0.05;
    const RA: f64 = 0.5;

    #[test]
    fn back_emf_proportional_to_speed() {
        // E = Ke·ω : à 100 rad/s → 5 V ; à vitesse double → f.c.é.m. double.
        assert_relative_eq!(dc_back_emf(KT, 100.0), 5.0, epsilon = 1e-12);
        assert_relative_eq!(
            dc_back_emf(KT, 200.0),
            2.0 * dc_back_emf(KT, 100.0),
            epsilon = 1e-12
        );
    }

    #[test]
    fn torque_proportional_to_current() {
        // C = Kt·Ia : 10 A → 0,5 N·m.
        assert_relative_eq!(dc_torque(KT, 10.0), 0.5, epsilon = 1e-12);
        assert_relative_eq!(dc_torque(KT, 0.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn speed_inverts_back_emf_and_terminal_voltage() {
        // Réciprocité : à vide de résistance nulle, ω = V/Kt = E/Ke.
        // Cas chargé : ω = 100 → E = 5 V ; Ia = 10 A ; V = 5 + 10·0,5 = 10 V.
        let omega = 100.0_f64;
        let ia = 10.0_f64;
        let e = dc_back_emf(KT, omega);
        let v = dc_terminal_voltage(e, ia, RA);
        assert_relative_eq!(v, 10.0, epsilon = 1e-12);
        // La vitesse reconstruite depuis (V, Ia, Ra, Kt) retrouve ω de départ.
        assert_relative_eq!(dc_speed_rad(v, ia, RA, KT), omega, epsilon = 1e-9);
    }

    #[test]
    fn stall_speed_is_zero() {
        // Rotor bloqué : V = Ia·Ra (E = 0) → ω = 0.
        let ia = 4.0_f64;
        let v = dc_terminal_voltage(0.0, ia, RA);
        assert_relative_eq!(v, 2.0, epsilon = 1e-12);
        assert_relative_eq!(dc_speed_rad(v, ia, RA, KT), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn electrical_power_matches_emf_plus_losses() {
        // Bilan : V·Ia = E·Ia + Ra·Ia² ; E·Ia = C·ω (puissance méca idéale).
        let omega = 100.0_f64;
        let ia = 10.0_f64;
        let e = dc_back_emf(KT, omega);
        let v = dc_terminal_voltage(e, ia, RA);
        let p_elec = v * ia;
        let p_mech = dc_torque(KT, ia) * omega;
        let p_joule = RA * ia * ia;
        assert_relative_eq!(p_elec, p_mech + p_joule, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "constante de couple")]
    fn zero_constant_panics() {
        dc_torque(0.0, 10.0);
    }
}
