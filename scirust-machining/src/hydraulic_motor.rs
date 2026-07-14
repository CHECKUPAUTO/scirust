//! **Moteur hydraulique volumétrique** — couple, vitesse de rotation et puissance
//! utile d'un moteur volumétrique en régime établi, à partir de la cylindrée, de
//! la chute de pression, du débit et des rendements.
//!
//! ```text
//! couple utile     T   = (Vd·ΔP / (2·π)) · η_m
//! vitesse          n   = Q·η_v / Vd
//! puissance hydr.  P_h = Q·ΔP
//! puissance utile  P_u = T·ω
//! rendement global η   = η_m·η_v
//! ```
//!
//! `Vd` cylindrée par tour (m³/tr), `ΔP` chute de pression aux orifices (Pa),
//! `η_m` rendement mécanique (sans unité, 0 < η_m ≤ 1), `T` couple utile sur
//! l'arbre (N·m), `Q` débit d'alimentation (m³/s), `η_v` rendement volumétrique
//! (sans unité, 0 < η_v ≤ 1), `n` vitesse de rotation (**tr/s**), `ω` vitesse
//! angulaire (rad/s = 2·π·n), `P_h` puissance hydraulique fournie (W),
//! `P_u` puissance mécanique utile (W), `η` rendement global (sans unité).
//!
//! **Convention** : unités SI cohérentes. La cylindrée est exprimée **par tour**
//! (m³/tr) ; la vitesse [`hydromotor_speed`] est donc en **tr/s** et doit être
//! convertie en rad/s (multiplier par `2·π`) avant d'alimenter
//! [`hydromotor_output_power`], qui attend une vitesse angulaire.
//!
//! **Limite honnête** : modèle de **régime établi** (pas de transitoire, pas
//! d'inertie du rotor ni de compressibilité du fluide). Les rendements mécanique
//! et volumétrique sont supposés **constants** au point de fonctionnement et sont
//! des données **fournies par l'appelant** (elles dépendent de la pression, de la
//! vitesse et de la viscosité) ; aucune valeur « par défaut » n'est supposée. Les
//! fuites internes sont entièrement portées par `η_v` et les pertes par frottement
//! par `η_m`.

use core::f64::consts::PI;

/// Couple utile sur l'arbre `T = (Vd·ΔP / (2·π))·η_m` (moteur volumétrique,
/// cylindrée par tour, régime établi).
///
/// Panique si `displacement <= 0`, `pressure_drop < 0`, ou si
/// `mechanical_efficiency` n'est pas dans `]0, 1]`.
pub fn hydromotor_torque(displacement: f64, pressure_drop: f64, mechanical_efficiency: f64) -> f64 {
    assert!(
        displacement > 0.0,
        "la cylindrée Vd doit être strictement positive"
    );
    assert!(
        pressure_drop >= 0.0,
        "la chute de pression ΔP ne peut pas être négative"
    );
    assert!(
        mechanical_efficiency > 0.0 && mechanical_efficiency <= 1.0,
        "le rendement mécanique η_m doit être dans ]0, 1]"
    );
    (displacement * pressure_drop / (2.0 * PI)) * mechanical_efficiency
}

/// Vitesse de rotation `n = Q·η_v / Vd`, exprimée en **tr/s** (cylindrée par
/// tour, régime établi).
///
/// Panique si `displacement <= 0`, `flow_rate < 0`, ou si
/// `volumetric_efficiency` n'est pas dans `]0, 1]`.
pub fn hydromotor_speed(flow_rate: f64, displacement: f64, volumetric_efficiency: f64) -> f64 {
    assert!(
        displacement > 0.0,
        "la cylindrée Vd doit être strictement positive"
    );
    assert!(flow_rate >= 0.0, "le débit Q ne peut pas être négatif");
    assert!(
        volumetric_efficiency > 0.0 && volumetric_efficiency <= 1.0,
        "le rendement volumétrique η_v doit être dans ]0, 1]"
    );
    flow_rate * volumetric_efficiency / displacement
}

/// Puissance mécanique utile `P_u = T·ω` (couple × vitesse **angulaire**).
///
/// La vitesse est attendue en rad/s : convertir une vitesse en tr/s issue de
/// [`hydromotor_speed`] en la multipliant par `2·π`.
///
/// Panique si `torque < 0` ou `angular_speed < 0`.
pub fn hydromotor_output_power(torque: f64, angular_speed: f64) -> f64 {
    assert!(torque >= 0.0, "le couple T ne peut pas être négatif");
    assert!(
        angular_speed >= 0.0,
        "la vitesse angulaire ω ne peut pas être négative"
    );
    torque * angular_speed
}

/// Puissance hydraulique fournie au moteur `P_h = Q·ΔP` (avant pertes).
///
/// Panique si `flow_rate < 0` ou `pressure_drop < 0`.
pub fn hydromotor_hydraulic_power(flow_rate: f64, pressure_drop: f64) -> f64 {
    assert!(flow_rate >= 0.0, "le débit Q ne peut pas être négatif");
    assert!(
        pressure_drop >= 0.0,
        "la chute de pression ΔP ne peut pas être négative"
    );
    flow_rate * pressure_drop
}

/// Rendement global `η = η_m·η_v` (produit des rendements mécanique et
/// volumétrique).
///
/// Panique si l'un des rendements n'est pas dans `]0, 1]`.
pub fn hydromotor_overall_efficiency(
    mechanical_efficiency: f64,
    volumetric_efficiency: f64,
) -> f64 {
    assert!(
        mechanical_efficiency > 0.0 && mechanical_efficiency <= 1.0,
        "le rendement mécanique η_m doit être dans ]0, 1]"
    );
    assert!(
        volumetric_efficiency > 0.0 && volumetric_efficiency <= 1.0,
        "le rendement volumétrique η_v doit être dans ]0, 1]"
    );
    mechanical_efficiency * volumetric_efficiency
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn output_power_equals_hydraulic_power_times_overall_efficiency() {
        // Identité de conservation d'énergie : P_u = T·ω = ΔP·Q·η_m·η_v
        //                                          = P_h·η_global.
        let (vd, dp, q) = (50e-6, 200e5, 8.0e-4);
        let (eta_m, eta_v) = (0.92, 0.95);
        let t = hydromotor_torque(vd, dp, eta_m);
        let n = hydromotor_speed(q, vd, eta_v); // tr/s
        let omega = 2.0 * PI * n; // rad/s
        let p_u = hydromotor_output_power(t, omega);
        let p_h = hydromotor_hydraulic_power(q, dp);
        let eta = hydromotor_overall_efficiency(eta_m, eta_v);
        assert_relative_eq!(p_u, p_h * eta, epsilon = 1e-9);
    }

    #[test]
    fn torque_is_proportional_to_pressure_drop() {
        // T ∝ ΔP à cylindrée et rendement fixés : doubler ΔP double le couple.
        let (vd, eta_m) = (63e-6, 0.9);
        let t1 = hydromotor_torque(vd, 100e5, eta_m);
        let t2 = hydromotor_torque(vd, 200e5, eta_m);
        assert_relative_eq!(t2, 2.0 * t1, epsilon = 1e-12);
    }

    #[test]
    fn speed_is_inversely_proportional_to_displacement() {
        // n ∝ 1/Vd : à débit et rendement fixés, n·Vd est constant.
        let (q, eta_v) = (1.0e-3, 0.96);
        let n1 = hydromotor_speed(q, 40e-6, eta_v);
        let n2 = hydromotor_speed(q, 80e-6, eta_v);
        assert_relative_eq!(n1 * 40e-6, n2 * 80e-6, epsilon = 1e-15);
    }

    #[test]
    fn ideal_efficiencies_recover_theoretical_torque_and_speed() {
        // Rendements unitaires : couple théorique Vd·ΔP/(2π) et vitesse Q/Vd.
        let (vd, dp, q) = (32e-6, 150e5, 6.0e-4);
        assert_relative_eq!(
            hydromotor_torque(vd, dp, 1.0),
            vd * dp / (2.0 * PI),
            epsilon = 1e-12
        );
        assert_relative_eq!(hydromotor_speed(q, vd, 1.0), q / vd, epsilon = 1e-12);
    }

    #[test]
    fn overall_efficiency_bounds_output_below_hydraulic_input() {
        // Second principe pratique : P_u ≤ P_h puisque 0 < η_m·η_v ≤ 1.
        let (vd, dp, q) = (45e-6, 250e5, 9.0e-4);
        let (eta_m, eta_v) = (0.88, 0.93);
        let t = hydromotor_torque(vd, dp, eta_m);
        let omega = 2.0 * PI * hydromotor_speed(q, vd, eta_v);
        let p_u = hydromotor_output_power(t, omega);
        let p_h = hydromotor_hydraulic_power(q, dp);
        assert!(p_u < p_h);
        assert!(p_u > 0.0);
    }

    #[test]
    fn realistic_torque_case() {
        // Cylindrée 100 cm³/tr, ΔP = 210 bar, η_m = 0,90 :
        // T = 100e-6·210e5/(2π)·0,90 N·m.
        let t = hydromotor_torque(100e-6, 210e5, 0.90);
        let expected = 100e-6 * 210e5 / (2.0 * PI) * 0.90;
        assert_relative_eq!(t, expected, epsilon = 1e-9);
        // ≈ 300,7 N·m sur l'arbre.
        assert!(t > 300.0 && t < 301.0);
    }

    #[test]
    #[should_panic(expected = "le rendement mécanique η_m doit être dans ]0, 1]")]
    fn efficiency_above_one_panics() {
        hydromotor_torque(50e-6, 200e5, 1.2);
    }
}
