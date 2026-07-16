//! **Machine à courant continu** (excitation séparée / shunt) — module de la
//! force contre-électromotrice, du couple électromagnétique, de la vitesse d'un
//! moteur et du rendement d'une machine à courant continu.
//!
//! ```text
//! f.c.é.m.        E   = k·φ·ω
//! couple          T   = k·φ·Ia
//! vitesse moteur  ω   = (U − Ia·Ra) / (k·φ)
//! rendement       η   = Pout / (Pout + P_a + P_f + P_m)
//! ```
//!
//! `k` constante de la machine (sans dimension, liée au bobinage), `φ` flux par
//! pôle (Wb), `k·φ` constante machine « k·φ » (V·s/rad, soit N·m/A), `ω`
//! vitesse angulaire du rotor (rad/s), `E` force contre-électromotrice induite
//! (V), `Ia` courant d'induit (A), `T` couple électromagnétique (N·m), `U`
//! tension aux bornes de l'induit (V), `Ra` résistance d'induit (Ω), `Pout`
//! puissance utile de sortie (W), `P_a` pertes Joule d'induit (W), `P_f` pertes
//! d'excitation / champ (W), `P_m` pertes mécaniques et fer (W), `η` rendement
//! (sans dimension).
//!
//! **Convention** : SI ; tensions et f.c.é.m. en V, courants en A, résistances
//! en Ω, flux en Wb, vitesses angulaires en **rad/s**, couples en N·m,
//! puissances et pertes en W ; les angles et vitesses sont en **radians**.
//! **Limite honnête** : machine à courant continu idéalisée, **réaction
//! d'induit et saturation magnétique NÉGLIGÉES** ; en **excitation séparée** le
//! flux `φ` est supposé **constant**. La constante machine `k·φ` (ou le couple
//! `(k, φ)`), les résistances, les tensions/courants réseau et les **pertes**
//! sont **FOURNIS par l'appelant** (essais, plaque signalétique, mesures) —
//! aucune valeur n'est inventée. Distinct d'un module `dc_motor` basique d'une
//! autre crate.

/// Force contre-électromotrice induite `E = k·φ·ω` (V).
///
/// Panique si `machine_constant <= 0`, `flux <= 0` ou `angular_speed < 0`.
pub fn dcm_back_emf(machine_constant: f64, flux: f64, angular_speed: f64) -> f64 {
    assert!(
        machine_constant > 0.0,
        "la constante machine k doit être strictement positive"
    );
    assert!(flux > 0.0, "le flux φ doit être strictement positif");
    assert!(angular_speed >= 0.0, "la vitesse angulaire ω doit être ≥ 0");
    machine_constant * flux * angular_speed
}

/// Couple électromagnétique `T = k·φ·Ia` (N·m).
///
/// Panique si `machine_constant <= 0` ou `flux <= 0`.
pub fn dcm_torque(machine_constant: f64, flux: f64, armature_current: f64) -> f64 {
    assert!(
        machine_constant > 0.0,
        "la constante machine k doit être strictement positive"
    );
    assert!(flux > 0.0, "le flux φ doit être strictement positif");
    machine_constant * flux * armature_current
}

/// Vitesse angulaire d'un moteur `ω = (U − Ia·Ra) / (k·φ)` (rad/s).
///
/// Panique si `armature_resistance < 0`, `machine_constant <= 0` ou
/// `flux <= 0` (division par zéro exclue par `k·φ > 0`).
pub fn dcm_speed(
    terminal_voltage: f64,
    armature_current: f64,
    armature_resistance: f64,
    machine_constant: f64,
    flux: f64,
) -> f64 {
    assert!(
        armature_resistance >= 0.0,
        "la résistance d'induit Ra doit être ≥ 0"
    );
    assert!(
        machine_constant > 0.0,
        "la constante machine k doit être strictement positive"
    );
    assert!(flux > 0.0, "le flux φ doit être strictement positif");
    (terminal_voltage - armature_current * armature_resistance) / (machine_constant * flux)
}

/// Rendement `η = Pout / (Pout + P_a + P_f + P_m)` (sans dimension).
///
/// Panique si `output_power <= 0`, si une perte est `< 0`, ou si la puissance
/// totale absorbée est nulle.
pub fn dcm_efficiency(
    output_power: f64,
    armature_loss: f64,
    field_loss: f64,
    mechanical_loss: f64,
) -> f64 {
    assert!(
        output_power > 0.0,
        "la puissance utile Pout doit être strictement positive"
    );
    assert!(
        armature_loss >= 0.0,
        "les pertes d'induit P_a doivent être ≥ 0"
    );
    assert!(
        field_loss >= 0.0,
        "les pertes d'excitation P_f doivent être ≥ 0"
    );
    assert!(
        mechanical_loss >= 0.0,
        "les pertes mécaniques P_m doivent être ≥ 0"
    );
    let input_power = output_power + armature_loss + field_loss + mechanical_loss;
    assert!(
        input_power > 0.0,
        "la puissance absorbée totale doit être strictement positive"
    );
    output_power / input_power
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn emf_and_torque_share_machine_constant() {
        // Identité structurelle : E/ω = T/Ia = k·φ. Les deux relations
        // partagent la même constante machine.
        let k = 2.0;
        let flux = 0.5;
        let omega = 100.0;
        let ia = 20.0;
        let e = dcm_back_emf(k, flux, omega);
        let t = dcm_torque(k, flux, ia);
        assert_relative_eq!(e / omega, k * flux, epsilon = 1e-12);
        assert_relative_eq!(t / ia, k * flux, epsilon = 1e-12);
        assert_relative_eq!(e / omega, t / ia, epsilon = 1e-12);
    }

    #[test]
    fn emf_is_zero_at_standstill() {
        // Cas limite : à l'arrêt (ω = 0) la f.c.é.m. est nulle.
        assert_relative_eq!(dcm_back_emf(2.5, 0.4, 0.0), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn speed_inverts_back_emf() {
        // Réciprocité : si U = E + Ia·Ra avec E = k·φ·ω, alors dcm_speed
        // restitue exactement la vitesse ω de départ.
        let k = 1.2;
        let flux = 0.8;
        let omega = 157.0;
        let ia = 15.0;
        let ra = 0.6;
        let e = dcm_back_emf(k, flux, omega);
        let u = e + ia * ra;
        let omega_back = dcm_speed(u, ia, ra, k, flux);
        assert_relative_eq!(omega_back, omega, epsilon = 1e-9);
    }

    #[test]
    fn realistic_separately_excited_motor() {
        // Cas chiffré réaliste (excitation séparée, flux constant), avec la
        // constante machine k·φ = 2·0,5 = 1,0 V·s/rad :
        //   E = k·φ·ω = 1,0·210 = 210 V
        //   T = k·φ·Ia = 1,0·20 = 20 N·m
        //   ω = (U − Ia·Ra)/(k·φ) = (220 − 20·0,5)/1,0 = (220 − 10)/1,0 = 210 rad/s
        let k = 2.0;
        let flux = 0.5;
        assert_relative_eq!(dcm_back_emf(k, flux, 210.0), 210.0, epsilon = 1e-9);
        assert_relative_eq!(dcm_torque(k, flux, 20.0), 20.0, epsilon = 1e-9);
        assert_relative_eq!(dcm_speed(220.0, 20.0, 0.5, k, flux), 210.0, epsilon = 1e-9);
    }

    #[test]
    fn realistic_efficiency_case() {
        // Cas chiffré réaliste : Pout = 1000 W ; pertes fournies P_a = 200 W
        // (= Ia²·Ra = 20²·0,5), P_f = 50 W, P_m = 50 W.
        //   Pin = 1000 + 200 + 50 + 50 = 1300 W
        //   η   = 1000/1300 = 0,769 230 769…
        let eta = dcm_efficiency(1000.0, 200.0, 50.0, 50.0);
        assert_relative_eq!(eta, 1000.0 / 1300.0, epsilon = 1e-12);
        assert_relative_eq!(eta, 0.769_230_769_230_769, epsilon = 1e-6);
    }

    #[test]
    fn lossless_efficiency_is_unity() {
        // Cas limite : sans pertes le rendement vaut exactement 1.
        assert_relative_eq!(dcm_efficiency(500.0, 0.0, 0.0, 0.0), 1.0, epsilon = 1e-15);
    }

    #[test]
    #[should_panic(expected = "le flux φ doit être strictement positif")]
    fn zero_flux_torque_panics() {
        dcm_torque(2.0, 0.0, 10.0);
    }
}
