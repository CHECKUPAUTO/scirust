//! **Protection thermique des moteurs** — dimensionnement des courants
//! caractéristiques d'un moteur triphasé et **temps de déclenchement** d'une
//! protection à **image thermique** (modèle I²t du premier ordre). On calcule
//! le courant nominal (courant de pleine charge) à partir de la plaque
//! signalétique, le courant de rotor bloqué au démarrage, le courant admissible
//! avec facteur de service, et le temps mis par le relais à image thermique
//! pour déclencher sous une surcharge donnée.
//!
//! ```text
//! courant nominal        I_fl = P / (sqrt(3) · U · cos φ · η)
//! courant rotor bloqué   I_lr = I_fl · k_lr
//! temps de déclenchement t    = τ · ln( r² / (r² − 1) ),  r = I / I_trip
//! courant facteur service I_sf = I_fl · SF
//! ```
//!
//! `I_fl` courant de pleine charge (A), `P` puissance mécanique nominale (W),
//! `U` tension composée entre phases (V), `cos φ` facteur de puissance (sans
//! dimension), `η` rendement (sans dimension, 0 < η ≤ 1), `I_lr` courant de
//! rotor bloqué (A), `k_lr` ratio de rotor bloqué (sans dimension, typiquement
//! 6 à 8), `t` temps de déclenchement (s), `τ` constante de temps thermique du
//! moteur (s), `I` courant traversant (A), `I_trip` courant de réglage du
//! relais (A), `r` surcharge relative (sans dimension), `I_sf` courant
//! admissible (A), `SF` facteur de service (sans dimension, ≥ 1).
//!
//! **Convention** : SI ; puissances en W, tensions en V, courants en A, temps
//! et constantes de temps en s ; facteur de puissance, rendement, ratios et
//! facteurs de service sans dimension. Le logarithme est le logarithme népérien.
//!
//! **Limite honnête** : protection **thermique à image thermique** (modèle I²t,
//! premier ordre). Le courant nominal est calculé depuis la **plaque
//! signalétique** — puissance `P`, tension `U`, facteur de puissance `cos φ` et
//! rendement `η` sont **FOURNIS par l'appelant**. Le **ratio de rotor bloqué**
//! `k_lr` et la **constante de temps thermique** `τ` sont **FOURNIS par le
//! constructeur** (catalogue, essais). Le temps de déclenchement suit un
//! **modèle thermique du premier ordre** valable pour une surcharge soutenue
//! avec `I > I_trip` (au-delà du courant de réglage) : à la limite `I = I_trip`
//! le temps tend vers l'infini. Ce module ne modélise **pas** l'échauffement
//! réel des enroulements (répartition thermique, refroidissement forcé) ni
//! l'accumulation thermique des **démarrages répétés**.

/// Courant de pleine charge d'un moteur triphasé `I_fl = P / (sqrt(3) · U · cos φ · η)` (A).
///
/// `P` est la puissance mécanique **utile** (à l'arbre), le rendement `η`
/// ramenant à la puissance électrique absorbée.
///
/// Panique si `rated_power < 0`, si `voltage <= 0`, si `power_factor` n'est pas
/// dans ]0 ; 1] ou si `efficiency` n'est pas dans ]0 ; 1] (division par zéro et
/// valeurs non physiques exclues).
pub fn motprot_full_load_current(
    rated_power: f64,
    voltage: f64,
    power_factor: f64,
    efficiency: f64,
) -> f64 {
    assert!(rated_power >= 0.0, "la puissance nominale P doit être ≥ 0");
    assert!(
        voltage > 0.0,
        "la tension composée U doit être strictement positive"
    );
    assert!(
        power_factor > 0.0 && power_factor <= 1.0,
        "le facteur de puissance cos φ doit être dans ]0 ; 1]"
    );
    assert!(
        efficiency > 0.0 && efficiency <= 1.0,
        "le rendement η doit être dans ]0 ; 1]"
    );
    rated_power / (3.0_f64.sqrt() * voltage * power_factor * efficiency)
}

/// Courant de rotor bloqué `I_lr = I_fl · k_lr` (A).
///
/// Courant absorbé au démarrage rotor à l'arrêt ; le ratio `k_lr`
/// (typiquement 6 à 8) est **fourni par le constructeur**.
///
/// Panique si `full_load_current < 0` ou si `locked_rotor_ratio < 1`
/// (le courant de rotor bloqué n'est jamais inférieur au courant nominal).
pub fn motprot_locked_rotor_current(full_load_current: f64, locked_rotor_ratio: f64) -> f64 {
    assert!(
        full_load_current >= 0.0,
        "le courant de pleine charge I_fl doit être ≥ 0"
    );
    assert!(
        locked_rotor_ratio >= 1.0,
        "le ratio de rotor bloqué k_lr doit être ≥ 1"
    );
    full_load_current * locked_rotor_ratio
}

/// Temps de déclenchement d'une protection à image thermique
/// `t = τ · ln( r² / (r² − 1) )` avec `r = I / I_trip` (s).
///
/// Modèle thermique I²t du premier ordre : plus la surcharge `r` est forte,
/// plus le temps de déclenchement est court. À la limite `I = I_trip` (`r = 1`)
/// le temps diverge ; le modèle exige donc une surcharge stricte `I > I_trip`.
///
/// Panique si `trip_current <= 0`, si `thermal_time_constant <= 0` ou si
/// `current <= trip_current` (surcharge stricte requise, `r > 1`).
pub fn motprot_overload_trip_time(
    current: f64,
    trip_current: f64,
    thermal_time_constant: f64,
) -> f64 {
    assert!(
        trip_current > 0.0,
        "le courant de réglage I_trip doit être strictement positif"
    );
    assert!(
        thermal_time_constant > 0.0,
        "la constante de temps thermique τ doit être strictement positive"
    );
    assert!(
        current > trip_current,
        "le courant I doit être strictement supérieur au courant de réglage I_trip"
    );
    let ratio = current / trip_current;
    let ratio_squared = ratio * ratio;
    let argument: f64 = ratio_squared / (ratio_squared - 1.0);
    thermal_time_constant * argument.ln()
}

/// Courant admissible avec facteur de service `I_sf = I_fl · SF` (A).
///
/// Le facteur de service `SF` (≥ 1) quantifie la surcharge continue admissible
/// du moteur ; il est **fourni par le constructeur** (plaque signalétique).
///
/// Panique si `full_load_current < 0` ou si `service_factor < 1`
/// (un facteur de service est par définition ≥ 1).
pub fn motprot_service_factor_current(full_load_current: f64, service_factor: f64) -> f64 {
    assert!(
        full_load_current >= 0.0,
        "le courant de pleine charge I_fl doit être ≥ 0"
    );
    assert!(
        service_factor >= 1.0,
        "le facteur de service SF doit être ≥ 1"
    );
    full_load_current * service_factor
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn full_load_current_reconstructs_rated_power() {
        // Identité de définition : P = sqrt(3) · U · I_fl · cos φ · η. Le courant
        // calculé, réinjecté, restitue exactement la puissance de plaque.
        let p = 15_000.0;
        let u = 400.0;
        let cos_phi = 0.86;
        let eta = 0.91;
        let i_fl = motprot_full_load_current(p, u, cos_phi, eta);
        let p_back = 3.0_f64.sqrt() * u * i_fl * cos_phi * eta;
        assert_relative_eq!(p_back, p, epsilon = 1e-6);
    }

    #[test]
    fn locked_rotor_and_service_factor_scale_linearly() {
        // Les deux courants dérivés sont proportionnels au courant nominal :
        // doubler I_fl double I_lr et I_sf.
        let i1 = motprot_locked_rotor_current(10.0, 6.5);
        let i2 = motprot_locked_rotor_current(20.0, 6.5);
        assert_relative_eq!(i2, 2.0 * i1, epsilon = 1e-12);

        let s1 = motprot_service_factor_current(10.0, 1.15);
        let s2 = motprot_service_factor_current(20.0, 1.15);
        assert_relative_eq!(s2, 2.0 * s1, epsilon = 1e-12);
    }

    #[test]
    fn locked_rotor_current_matches_direct_product() {
        // Cas chiffré : I_fl = 27 A, k_lr = 7 ⇒ I_lr = 189 A.
        assert_relative_eq!(
            motprot_locked_rotor_current(27.0, 7.0),
            189.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn trip_time_equals_tau_ln2_at_ratio_squared_two() {
        // Identité remarquable : pour r² = 2 (soit I = I_trip·√2), l'argument
        // du logarithme vaut 2/(2−1) = 2, donc t = τ·ln 2.
        let tau = 300.0;
        let trip = 10.0;
        let current = trip * core::f64::consts::SQRT_2; // r² = 2
        let t = motprot_overload_trip_time(current, trip, tau);
        assert_relative_eq!(t, tau * core::f64::consts::LN_2, epsilon = 1e-3);
    }

    #[test]
    fn trip_time_decreases_with_increasing_overload() {
        // Monotonie physique : une surcharge plus forte déclenche plus vite.
        let tau = 120.0;
        let trip = 5.0;
        let t_low = motprot_overload_trip_time(6.0, trip, tau);
        let t_high = motprot_overload_trip_time(12.0, trip, tau);
        assert!(t_high < t_low);
    }

    #[test]
    fn trip_time_scales_with_thermal_time_constant() {
        // Proportionnalité en τ : à surcharge identique, doubler la constante de
        // temps thermique double le temps de déclenchement.
        let t1 = motprot_overload_trip_time(8.0, 5.0, 100.0);
        let t2 = motprot_overload_trip_time(8.0, 5.0, 200.0);
        assert_relative_eq!(t2, 2.0 * t1, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(
        expected = "le courant I doit être strictement supérieur au courant de réglage I_trip"
    )]
    fn trip_time_panics_without_overload() {
        // r = 1 : pas de surcharge stricte ⇒ le modèle diverge, on refuse.
        motprot_overload_trip_time(10.0, 10.0, 300.0);
    }
}
