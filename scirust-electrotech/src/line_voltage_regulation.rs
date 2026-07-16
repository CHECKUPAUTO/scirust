//! **Régulation de tension d'un départ (feeder)** — module de calcul de la
//! chute de tension approchée le long d'un départ radial, de la régulation en
//! pourcentage entre extrémités, de la tension à l'arrivée et de l'élévation de
//! Ferranti d'une ligne longue à vide.
//!
//! ```text
//! chute de tension approchée   ΔV = I·(R·cosφ + X·sinφ),  sinφ = √(1 − cos²φ)
//! régulation en pourcentage    reg% = 100·(V_s − V_r) / V_r
//! tension à l'arrivée          V_r = V_s − ΔV
//! élévation de Ferranti        ΔV_F = V_r·X_l·B_l·ℓ² / 2
//! ```
//!
//! `ΔV` chute de tension en projection sur l'axe de la tension (V), `I` courant
//! de ligne (A), `R` résistance du départ (Ω), `X` réactance du départ (Ω),
//! `cosφ` facteur de puissance de la charge (sans dimension, ∈ [0, 1]), `sinφ`
//! facteur de puissance réactif (sans dimension), `reg%` régulation en
//! pourcentage (%), `V_s` tension au départ / à l'émission (V), `V_r` tension à
//! l'arrivée / à la réception (V), `ΔV_F` élévation de Ferranti à vide (V),
//! `X_l` réactance linéique de ligne (Ω/km), `B_l` susceptance linéique de
//! ligne (S/km), `ℓ` longueur de la ligne (km).
//!
//! **Convention** : SI (V, A, Ω), angles en radians ; la charge est supposée
//! **inductive** (chute de tension classique, projection sur l'axe de la
//! tension à l'arrivée). **Limite honnête** : départ **radial** en régime
//! permanent équilibré ; la **résistance**, la **réactance** de ligne et le
//! **facteur de puissance** de la charge sont **fournis par l'appelant** (relevé
//! de câble, catalogue constructeur, mesure). La chute de tension est
//! l'**approximation classique** ΔV ≈ I·(R·cosφ + X·sinφ), valable tant que la
//! chute reste faible devant la tension et que le terme en quadrature est
//! négligeable ; elle n'inclut pas le terme de second ordre exact. L'**effet de
//! Ferranti** (élévation de tension à vide) n'apparaît que sur les lignes
//! **longues et capacitives** : ici on retient l'**approximation** ΔV_F ≈
//! V_r·X_l·B_l·ℓ²/2 à partir des paramètres linéiques **fournis** par
//! l'appelant. Aucune valeur « typique » de réseau n'est inventée.

/// Chute de tension approchée d'un départ radial `ΔV = current·(resistance·cosφ
/// + reactance·sinφ)` (V), projection de la chute sur l'axe de la tension avec
/// `sinφ = √(1 − power_factor²)` (charge inductive).
///
/// `power_factor` est le facteur de puissance `cosφ` de la charge (sans
/// dimension, ∈ [0, 1]).
///
/// Panique si `current < 0`, si `resistance < 0`, si `reactance < 0` ou si
/// `power_factor` n'est pas dans `[0, 1]` (racine d'un nombre négatif ou
/// grandeurs non physiques).
pub fn linreg_voltage_drop(
    current: f64,
    resistance: f64,
    reactance: f64,
    power_factor: f64,
) -> f64 {
    assert!(current >= 0.0, "le courant current doit être ≥ 0");
    assert!(resistance >= 0.0, "la résistance resistance doit être ≥ 0");
    assert!(reactance >= 0.0, "la réactance reactance doit être ≥ 0");
    assert!(
        (0.0..=1.0).contains(&power_factor),
        "le facteur de puissance power_factor doit être dans [0, 1]"
    );
    let sin_phi = (1.0 - power_factor * power_factor).sqrt();
    current * (resistance * power_factor + reactance * sin_phi)
}

/// Régulation de tension en pourcentage `reg% = 100·(sending_voltage −
/// receiving_voltage) / receiving_voltage` (%), écart relatif entre la tension
/// au départ et celle à l'arrivée, rapporté à l'arrivée.
///
/// Panique si `sending_voltage <= 0` ou si `receiving_voltage <= 0` (division
/// par zéro ou tensions non physiques).
pub fn linreg_percent_regulation(sending_voltage: f64, receiving_voltage: f64) -> f64 {
    assert!(
        sending_voltage > 0.0,
        "la tension au départ sending_voltage doit être strictement positive"
    );
    assert!(
        receiving_voltage > 0.0,
        "la tension à l'arrivée receiving_voltage doit être strictement positive"
    );
    100.0 * (sending_voltage - receiving_voltage) / receiving_voltage
}

/// Élévation de Ferranti à vide `ΔV_F = receiving_voltage·line_reactance·
/// line_susceptance·length² / 2` (V), approximation de la surtension à
/// l'arrivée d'une ligne longue capacitive fonctionnant à vide.
///
/// `line_reactance` est la réactance linéique (Ω/km), `line_susceptance` la
/// susceptance linéique (S/km) et `length` la longueur de ligne (km).
///
/// Panique si `receiving_voltage <= 0`, si `line_reactance < 0`, si
/// `line_susceptance < 0` ou si `length < 0` (grandeurs non physiques).
pub fn linreg_ferranti_rise(
    receiving_voltage: f64,
    line_reactance: f64,
    line_susceptance: f64,
    length: f64,
) -> f64 {
    assert!(
        receiving_voltage > 0.0,
        "la tension à l'arrivée receiving_voltage doit être strictement positive"
    );
    assert!(
        line_reactance >= 0.0,
        "la réactance linéique line_reactance doit être ≥ 0"
    );
    assert!(
        line_susceptance >= 0.0,
        "la susceptance linéique line_susceptance doit être ≥ 0"
    );
    assert!(length >= 0.0, "la longueur length doit être ≥ 0");
    receiving_voltage * line_reactance * line_susceptance * length * length / 2.0
}

/// Tension à l'arrivée `V_r = sending_voltage − voltage_drop` (V), tension au
/// départ diminuée de la chute de tension du départ.
///
/// Panique si `sending_voltage <= 0` ou si `voltage_drop < 0` (grandeurs non
/// physiques).
pub fn linreg_receiving_voltage(sending_voltage: f64, voltage_drop: f64) -> f64 {
    assert!(
        sending_voltage > 0.0,
        "la tension au départ sending_voltage doit être strictement positive"
    );
    assert!(
        voltage_drop >= 0.0,
        "la chute de tension voltage_drop doit être ≥ 0"
    );
    sending_voltage - voltage_drop
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn voltage_drop_numeric_case() {
        // Cas chiffré : I = 100 A, R = 0,5 Ω, X = 0,3 Ω, cosφ = 0,8.
        //   sinφ = √(1 − 0,64) = √0,36 = 0,6.
        //   ΔV = 100·(0,5·0,8 + 0,3·0,6) = 100·(0,40 + 0,18) = 100·0,58 = 58 V.
        let dv = linreg_voltage_drop(100.0_f64, 0.5_f64, 0.3_f64, 0.8_f64);
        assert_relative_eq!(dv, 58.0, epsilon = 1e-3);
    }

    #[test]
    fn voltage_drop_power_factor_limits() {
        // Limite cosφ = 1 (charge purement résistive) : sinφ = 0, la chute se
        // réduit au terme résistif ΔV = I·R.
        let i = 120.0_f64;
        let r = 0.25_f64;
        let x = 0.40_f64;
        assert_relative_eq!(linreg_voltage_drop(i, r, x, 1.0), i * r, epsilon = 1e-9);
        // Limite cosφ = 0 (charge purement inductive) : sinφ = 1, la chute se
        // réduit au terme réactif ΔV = I·X.
        assert_relative_eq!(linreg_voltage_drop(i, r, x, 0.0), i * x, epsilon = 1e-9);
    }

    #[test]
    fn drop_and_receiving_voltage_reciprocity() {
        // Réciprocité : partir de V_s et ΔV, calculer V_r, puis vérifier que la
        // chute reconstituée V_s − V_r redonne ΔV.
        //   V_s = 240 V, ΔV = 58 V → V_r = 240 − 58 = 182 V.
        let v_s = 240.0_f64;
        let dv = 58.0_f64;
        let v_r = linreg_receiving_voltage(v_s, dv);
        assert_relative_eq!(v_r, 182.0, epsilon = 1e-9);
        assert_relative_eq!(v_s - v_r, dv, epsilon = 1e-9);
    }

    #[test]
    fn percent_regulation_numeric_and_zero() {
        // Cas chiffré : V_s = 231 V, V_r = 220 V.
        //   reg% = 100·(231 − 220)/220 = 100·11/220 = 1100/220 = 5 %.
        assert_relative_eq!(
            linreg_percent_regulation(231.0_f64, 220.0_f64),
            5.0,
            epsilon = 1e-9
        );
        // Cas limite : tensions égales → régulation nulle.
        assert_relative_eq!(
            linreg_percent_regulation(220.0_f64, 220.0_f64),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn ferranti_numeric_and_length_squared_scaling() {
        // Cas chiffré : V_r = 100 000 V, X_l = 0,4 Ω/km, B_l = 3·10⁻⁶ S/km,
        //   ℓ = 100 km.
        //   ΔV_F = 100000·0,4·3e-6·100²/2 = 100000·0,4 = 40000 ;
        //          40000·3e-6 = 0,12 ; 0,12·10000 = 1200 ; 1200/2 = 600 V.
        let rise = linreg_ferranti_rise(100_000.0_f64, 0.4_f64, 3.0e-6_f64, 100.0_f64);
        assert_relative_eq!(rise, 600.0, epsilon = 1e-3);
        // Proportionnalité en ℓ² : doubler la longueur quadruple l'élévation.
        let rise_double = linreg_ferranti_rise(100_000.0_f64, 0.4_f64, 3.0e-6_f64, 200.0_f64);
        assert_relative_eq!(rise_double, 4.0 * rise, epsilon = 1e-6);
        // Susceptance nulle (ligne sans effet capacitif) → aucune élévation.
        assert_relative_eq!(
            linreg_ferranti_rise(100_000.0_f64, 0.4_f64, 0.0_f64, 100.0_f64),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic(
        expected = "la tension à l'arrivée receiving_voltage doit être strictement positive"
    )]
    fn zero_receiving_voltage_panics() {
        linreg_percent_regulation(230.0_f64, 0.0_f64);
    }
}
