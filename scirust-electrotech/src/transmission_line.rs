//! **Ligne électrique courte (modèle à paramètres localisés)** — chute de
//! tension, régulation, pertes Joule triphasées, rendement et impédance
//! caractéristique d'une ligne courte en régime permanent équilibré.
//!
//! ```text
//! tension d'entrée (approx. ligne courte)
//!     V_s = V_r + I·(R·cos φ + X·sin φ),  sin φ = √(1 − cos²φ)
//! régulation de tension          reg = (V_s − V_r) / V_r
//! pertes Joule triphasées        P_loss = 3·I²·R_ph
//! rendement de transport         η = P_r / (P_r + P_loss)
//! impédance caractéristique      Z_c = √(L' / C')
//! ```
//!
//! `V_s` valeur efficace de la tension en tête de ligne / d'entrée (V), `V_r`
//! valeur efficace de la tension en bout de ligne / de réception (V), `I` valeur
//! efficace du courant de ligne (A), `R` résistance de ligne (Ω), `X` réactance
//! de ligne (Ω), `cos φ` facteur de puissance de la charge (sans dimension,
//! `∈ [0, 1]`), `sin φ` facteur réactif associé, `reg` régulation de tension
//! (sans dimension, en p.u.), `R_ph` résistance par phase (Ω), `P_loss` pertes
//! Joule totales des trois phases (W), `P_r` puissance active reçue par la
//! charge (W), `η` rendement de transport (sans dimension), `L'` inductance
//! linéique (H/m), `C'` capacité linéique (F/m), `Z_c` impédance
//! caractéristique / d'onde (Ω).
//!
//! **Convention** : SI ; tensions en V, courants en A, puissances en W,
//! impédances en Ω, inductance/capacité linéiques en H/m et F/m, déphasage `φ`
//! en **radians** (le facteur de puissance `cos φ` est fourni directement).
//! **Limite honnête** : modèle de **ligne courte** (`< ~80 km`) — la capacité
//! transversale est **négligée** dans le calcul de tension et de pertes, et
//! l'approximation de chute de tension **projette** la chute sur l'axe de la
//! tension de réception (`V_s ≈ V_r + I·(R·cos φ + X·sin φ)`) ; l'impédance
//! caractéristique `Z_c` est donnée **pour information**. Régime **permanent
//! équilibré**. Les grandeurs de réseau et de composant (`V_r`, `I`, `R`, `X`,
//! `cos φ`, `R_ph`, `L'`, `C'`) sont **fournies par l'appelant** d'après une
//! fiche de câble, un plan de réseau ou une mesure — aucune valeur « par
//! défaut » de ligne ou de matériau n'est inventée.

/// Tension d'entrée d'une **ligne courte** par l'approximation de chute de
/// tension `V_s = V_r + I·(R·cos φ + X·sin φ)` avec `sin φ = √(1 − cos²φ)` (V).
///
/// Panique si `receiving_voltage < 0`, si `current < 0`, si `resistance < 0`,
/// si `reactance < 0` ou si `power_factor` n'est pas dans `[0, 1]`.
pub fn line_sending_voltage_short(
    receiving_voltage: f64,
    current: f64,
    resistance: f64,
    reactance: f64,
    power_factor: f64,
) -> f64 {
    assert!(
        receiving_voltage >= 0.0,
        "la tension de réception V_r doit être ≥ 0"
    );
    assert!(current >= 0.0, "le courant de ligne I doit être ≥ 0");
    assert!(resistance >= 0.0, "la résistance de ligne R doit être ≥ 0");
    assert!(reactance >= 0.0, "la réactance de ligne X doit être ≥ 0");
    assert!(
        (0.0..=1.0).contains(&power_factor),
        "le facteur de puissance cos φ doit être dans [0, 1]"
    );
    let sin_phi = (1.0 - power_factor * power_factor).sqrt();
    receiving_voltage + current * (resistance * power_factor + reactance * sin_phi)
}

/// Régulation de tension d'une ligne `reg = (V_s − V_r) / V_r` (sans dimension,
/// en p.u.).
///
/// Panique si `receiving_voltage <= 0` ou si `sending_voltage < 0`.
pub fn line_voltage_regulation(sending_voltage: f64, receiving_voltage: f64) -> f64 {
    assert!(
        receiving_voltage > 0.0,
        "la tension de réception V_r doit être > 0"
    );
    assert!(
        sending_voltage >= 0.0,
        "la tension d'entrée V_s doit être ≥ 0"
    );
    (sending_voltage - receiving_voltage) / receiving_voltage
}

/// Pertes Joule totales d'une ligne **triphasée** `P_loss = 3·I²·R_ph` (W).
///
/// Panique si `current < 0` ou si `resistance_per_phase < 0`.
pub fn line_loss_three_phase(current: f64, resistance_per_phase: f64) -> f64 {
    assert!(current >= 0.0, "le courant de ligne I doit être ≥ 0");
    assert!(
        resistance_per_phase >= 0.0,
        "la résistance par phase R_ph doit être ≥ 0"
    );
    3.0 * current * current * resistance_per_phase
}

/// Rendement de transport d'une ligne `η = P_r / (P_r + P_loss)` (sans
/// dimension).
///
/// Panique si `receiving_power < 0`, si `line_loss < 0` ou si la puissance
/// d'entrée `P_r + P_loss` est nulle.
pub fn line_efficiency(receiving_power: f64, line_loss: f64) -> f64 {
    assert!(
        receiving_power >= 0.0,
        "la puissance reçue P_r doit être ≥ 0"
    );
    assert!(line_loss >= 0.0, "les pertes P_loss doivent être ≥ 0");
    let input_power = receiving_power + line_loss;
    assert!(
        input_power > 0.0,
        "la puissance d'entrée P_r + P_loss doit être > 0"
    );
    receiving_power / input_power
}

/// Impédance caractéristique (d'onde) d'une ligne `Z_c = √(L' / C')` (Ω),
/// donnée pour information dans le cadre du modèle de ligne courte.
///
/// Panique si `inductance_per_length < 0` ou si `capacitance_per_length <= 0`.
pub fn line_surge_impedance(inductance_per_length: f64, capacitance_per_length: f64) -> f64 {
    assert!(
        inductance_per_length >= 0.0,
        "l'inductance linéique L' doit être ≥ 0"
    );
    assert!(
        capacitance_per_length > 0.0,
        "la capacité linéique C' doit être > 0"
    );
    (inductance_per_length / capacitance_per_length).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn sending_voltage_unity_power_factor_drops_reactance() {
        // À cos φ = 1, sin φ = 0 : V_s = V_r + I·R (la réactance ne contribue pas).
        let v_r = 230.0_f64;
        let i = 50.0_f64;
        let r = 0.1_f64;
        let x = 0.3_f64;
        let v_s = line_sending_voltage_short(v_r, i, r, x, 1.0);
        assert_relative_eq!(v_s, v_r + i * r, epsilon = 1e-12);
    }

    #[test]
    fn sending_voltage_realistic_case() {
        // Cas chiffré : V_r = 230 V, I = 100 A, R = 0,1 Ω, X = 0,2 Ω, cos φ = 0,8.
        //   sin φ = √(1 − 0,64) = √0,36 = 0,6
        //   V_s = 230 + 100·(0,1·0,8 + 0,2·0,6)
        //       = 230 + 100·(0,08 + 0,12) = 230 + 100·0,2 = 250 V
        let v_s = line_sending_voltage_short(230.0, 100.0, 0.1, 0.2, 0.8);
        assert_relative_eq!(v_s, 250.0, epsilon = 1e-9);
    }

    #[test]
    fn regulation_matches_definition_on_realistic_case() {
        // Avec le cas ci-dessus, V_s = 250 V et V_r = 230 V :
        //   reg = (250 − 230) / 230 = 20 / 230 ≈ 0,086956521739
        let v_r = 230.0_f64;
        let v_s = line_sending_voltage_short(v_r, 100.0, 0.1, 0.2, 0.8);
        let reg = line_voltage_regulation(v_s, v_r);
        assert_relative_eq!(reg, 20.0 / 230.0, epsilon = 1e-12);
        assert_relative_eq!(reg, 0.086_956_521_739, epsilon = 1e-9);
    }

    #[test]
    fn loss_scales_with_current_squared() {
        // P_loss ∝ I² : doubler le courant quadruple les pertes Joule.
        let p1 = line_loss_three_phase(100.0, 0.1);
        let p2 = line_loss_three_phase(200.0, 0.1);
        assert_relative_eq!(p1, 3000.0, epsilon = 1e-9); // 3·100²·0,1 = 3000 W
        assert_relative_eq!(p2 / p1, 4.0, epsilon = 1e-12);
    }

    #[test]
    fn efficiency_complements_loss_fraction() {
        // Identité : η + P_loss/(P_r + P_loss) = 1 ; sans perte η = 1.
        let p_r = 120_000.0_f64;
        let loss = 3000.0_f64;
        let eta = line_efficiency(p_r, loss);
        let loss_fraction = loss / (p_r + loss);
        assert_relative_eq!(eta + loss_fraction, 1.0, epsilon = 1e-12);
        assert_relative_eq!(eta, 120_000.0 / 123_000.0, epsilon = 1e-12);
        assert_relative_eq!(line_efficiency(p_r, 0.0), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn surge_impedance_squared_is_l_over_c() {
        // Z_c² = L'/C' ; avec L' = 1e−3 H/m et C' = 1e−8 F/m :
        //   Z_c = √(1e−3 / 1e−8) = √1e5 ≈ 316,227766 Ω
        let l = 1.0e-3_f64;
        let c = 1.0e-8_f64;
        let zc = line_surge_impedance(l, c);
        assert_relative_eq!(zc * zc, l / c, epsilon = 1e-6);
        assert_relative_eq!(zc, 1.0e5_f64.sqrt(), epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "le facteur de puissance cos φ doit être dans [0, 1]")]
    fn sending_voltage_rejects_power_factor_above_one() {
        line_sending_voltage_short(230.0, 100.0, 0.1, 0.2, 1.5);
    }
}
