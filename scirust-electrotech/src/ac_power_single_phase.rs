//! **Puissances en régime sinusoïdal monophasé** — puissances apparente, active
//! et réactive, facteur de puissance et module d'impédance à partir des valeurs
//! efficaces de tension et de courant.
//!
//! ```text
//! puissance apparente     S = V_rms · I_rms
//! puissance active        P = V_rms · I_rms · cos φ
//! puissance réactive      Q = V_rms · I_rms · √(1 − cos²φ)
//! facteur de puissance    cos φ = P / S
//! module d'impédance      |Z| = V_rms / I_rms
//! ```
//!
//! `V_rms` valeur efficace de la tension (V), `I_rms` valeur efficace du courant
//! (A), `S` puissance apparente (VA), `P` puissance active (W), `Q` puissance
//! réactive (var), `cos φ` facteur de puissance (sans dimension, `∈ [0, 1]`),
//! `|Z|` module de l'impédance de charge (Ω). Le triangle des puissances vérifie
//! `S² = P² + Q²`.
//!
//! **Convention** : SI ; tensions en V, courants en A, puissances en W/var/VA,
//! impédances en Ω, déphasage `φ` en **radians** lorsqu'il intervient dans une
//! fonction trigonométrique. **Limite honnête** : régime **sinusoïdal permanent
//! monophasé** ; le facteur de puissance est ici `cos φ`, cosinus du **déphasage
//! tension–courant** de fondamentaux **sans harmoniques** — en présence
//! d'harmoniques, le facteur de puissance (`P/S`) diffère du `cos φ` du
//! fondamental et il faut les distinguer. Les valeurs **efficaces** de tension et
//! de courant ainsi que le facteur de puissance sont **fournis par l'appelant**
//! (mesures réseau, caractéristiques de charge) — aucune valeur « par défaut »
//! n'est inventée.

/// Puissance apparente `S = V_rms · I_rms` (VA).
///
/// Panique si `voltage_rms < 0` ou si `current_rms < 0`.
pub fn ac_apparent_power(voltage_rms: f64, current_rms: f64) -> f64 {
    assert!(
        voltage_rms >= 0.0,
        "la tension efficace V_rms doit être ≥ 0"
    );
    assert!(
        current_rms >= 0.0,
        "le courant efficace I_rms doit être ≥ 0"
    );
    voltage_rms * current_rms
}

/// Puissance active `P = V_rms · I_rms · cos φ` (W).
///
/// Panique si `voltage_rms < 0`, si `current_rms < 0` ou si `power_factor`
/// n'est pas dans `[0, 1]`.
pub fn ac_active_power(voltage_rms: f64, current_rms: f64, power_factor: f64) -> f64 {
    assert!(
        voltage_rms >= 0.0,
        "la tension efficace V_rms doit être ≥ 0"
    );
    assert!(
        current_rms >= 0.0,
        "le courant efficace I_rms doit être ≥ 0"
    );
    assert!(
        (0.0..=1.0).contains(&power_factor),
        "le facteur de puissance cos φ doit être dans [0, 1]"
    );
    voltage_rms * current_rms * power_factor
}

/// Puissance réactive `Q = V_rms · I_rms · √(1 − cos²φ)` (var).
///
/// Panique si `voltage_rms < 0`, si `current_rms < 0` ou si `power_factor`
/// n'est pas dans `[0, 1]`.
pub fn ac_reactive_power(voltage_rms: f64, current_rms: f64, power_factor: f64) -> f64 {
    assert!(
        voltage_rms >= 0.0,
        "la tension efficace V_rms doit être ≥ 0"
    );
    assert!(
        current_rms >= 0.0,
        "le courant efficace I_rms doit être ≥ 0"
    );
    assert!(
        (0.0..=1.0).contains(&power_factor),
        "le facteur de puissance cos φ doit être dans [0, 1]"
    );
    voltage_rms * current_rms * (1.0 - power_factor * power_factor).sqrt()
}

/// Facteur de puissance `cos φ = P / S` (sans dimension), déduit des puissances
/// active et apparente.
///
/// Panique si `apparent_power <= 0`, si `active_power < 0` ou si
/// `active_power > apparent_power` (facteur hors de `[0, 1]`, physiquement
/// exclu).
pub fn ac_power_factor_from_powers(active_power: f64, apparent_power: f64) -> f64 {
    assert!(
        apparent_power > 0.0,
        "la puissance apparente S doit être strictement positive"
    );
    assert!(active_power >= 0.0, "la puissance active P doit être ≥ 0");
    assert!(
        active_power <= apparent_power,
        "P ≤ S requis (facteur de puissance ≤ 1)"
    );
    active_power / apparent_power
}

/// Module de l'impédance de charge `|Z| = V_rms / I_rms` (Ω).
///
/// Panique si `voltage_rms < 0` ou si `current_rms <= 0` (division par zéro).
pub fn ac_impedance_magnitude(voltage_rms: f64, current_rms: f64) -> f64 {
    assert!(
        voltage_rms >= 0.0,
        "la tension efficace V_rms doit être ≥ 0"
    );
    assert!(
        current_rms > 0.0,
        "le courant efficace I_rms doit être strictement positif"
    );
    voltage_rms / current_rms
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn power_triangle_identity() {
        // Identité du triangle des puissances : S² = P² + Q², quel que soit
        // le facteur de puissance dans [0, 1].
        let v = 230.0_f64;
        let i = 5.0_f64;
        let pf = 0.8_f64;
        let s = ac_apparent_power(v, i);
        let p = ac_active_power(v, i, pf);
        let q = ac_reactive_power(v, i, pf);
        assert_relative_eq!(p * p + q * q, s * s, epsilon = 1e-9);
    }

    #[test]
    fn power_factor_round_trip() {
        // Réciprocité : cos φ = P/S retrouve le facteur de puissance injecté.
        let v = 400.0_f64;
        let i = 12.5_f64;
        let pf = 0.9_f64;
        let s = ac_apparent_power(v, i);
        let p = ac_active_power(v, i, pf);
        assert_relative_eq!(ac_power_factor_from_powers(p, s), pf, epsilon = 1e-12);
    }

    #[test]
    fn apparent_power_scales_linearly() {
        // Proportionnalité : doubler le courant double la puissance apparente.
        let s1 = ac_apparent_power(230.0, 4.0);
        let s2 = ac_apparent_power(230.0, 8.0);
        assert_relative_eq!(s2 / s1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn unity_power_factor_gives_zero_reactive() {
        // Cas limite : à cos φ = 1 la charge est résistive pure, donc P = S
        // et Q = 0.
        let v = 120.0_f64;
        let i = 10.0_f64;
        let s = ac_apparent_power(v, i);
        assert_relative_eq!(ac_active_power(v, i, 1.0), s, epsilon = 1e-12);
        assert_relative_eq!(ac_reactive_power(v, i, 1.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn realistic_230v_5a_case() {
        // Cas chiffré, V_rms = 230 V, I_rms = 5 A, cos φ = 0,8 :
        //   S   = 230·5           = 1150 VA
        //   P   = 1150·0,8        =  920 W
        //   Q   = 1150·√(1−0,64)  = 1150·0,6 = 690 var
        //   |Z| = 230/5           =   46 Ω
        let v = 230.0_f64;
        let i = 5.0_f64;
        let pf = 0.8_f64;
        assert_relative_eq!(ac_apparent_power(v, i), 1150.0, epsilon = 1e-6);
        assert_relative_eq!(ac_active_power(v, i, pf), 920.0, epsilon = 1e-6);
        assert_relative_eq!(ac_reactive_power(v, i, pf), 690.0, epsilon = 1e-6);
        assert_relative_eq!(ac_impedance_magnitude(v, i), 46.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "le facteur de puissance cos φ doit être dans [0, 1]")]
    fn power_factor_above_one_panics() {
        ac_active_power(230.0, 5.0, 1.2);
    }
}
