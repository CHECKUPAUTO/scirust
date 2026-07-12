//! Coup de bélier — célérité de l'onde de pression, surpression de **Joukowsky**
//! lors d'une fermeture rapide et durée critique de manœuvre.
//!
//! ```text
//! célérité (paroi rigide)  a = √(K/ρ)
//! célérité (Korteweg)      a = √( (K/ρ) / (1 + K·D/(E·e)) )
//! surpression (Joukowsky)  ΔP = ρ·a·Δv       (fermeture rapide)
//! durée critique           tc = 2·L/a        (aller-retour de l'onde)
//! ```
//!
//! `K` module de compressibilité du fluide (Pa), `ρ` masse volumique (kg/m³), `D`
//! diamètre de conduite (m), `e` épaisseur de paroi (m), `E` module de la paroi
//! (Pa), `a` célérité de l'onde (m/s), `Δv` variation de vitesse (m/s), `L`
//! longueur de conduite (m). Une fermeture plus rapide que `tc` produit la
//! surpression maximale de Joukowsky.
//!
//! **Convention** : SI cohérent. **Limite honnête** : théorie **élastique** de la
//! colonne d'eau (fermeture instantanée, conduite à paroi mince) ; les fermetures
//! lentes (`t > tc`) réduisent la surpression et relèvent d'un calcul transitoire
//! que ce module ne fait pas.

/// Célérité de l'onde en conduite **rigide** `a = √(K/ρ)` (m/s).
///
/// Panique si `rho <= 0` ou `bulk_modulus < 0`.
pub fn wave_speed_rigid(bulk_modulus: f64, rho: f64) -> f64 {
    assert!(rho > 0.0 && bulk_modulus >= 0.0, "ρ > 0 et K ≥ 0 requis");
    (bulk_modulus / rho).sqrt()
}

/// Célérité de l'onde en conduite **élastique** (Korteweg)
/// `a = √( (K/ρ)/(1 + K·D/(E·e)) )` (m/s).
///
/// Panique si `rho <= 0`, `E·e <= 0`.
pub fn wave_speed_elastic(
    bulk_modulus: f64,
    rho: f64,
    pipe_diameter: f64,
    wall_thickness: f64,
    pipe_modulus: f64,
) -> f64 {
    assert!(
        rho > 0.0,
        "la masse volumique doit être strictement positive"
    );
    let ee = pipe_modulus * wall_thickness;
    assert!(ee > 0.0, "E·e doit être strictement positif");
    (bulk_modulus / rho / (1.0 + bulk_modulus * pipe_diameter / ee)).sqrt()
}

/// Surpression de Joukowsky `ΔP = ρ·a·Δv` (Pa).
pub fn joukowsky_surge(rho: f64, wave_speed: f64, velocity_change: f64) -> f64 {
    rho * wave_speed * velocity_change
}

/// Durée critique de manœuvre `tc = 2·L/a` (s).
///
/// Panique si `wave_speed <= 0`.
pub fn critical_time(length: f64, wave_speed: f64) -> f64 {
    assert!(
        wave_speed > 0.0,
        "la célérité doit être strictement positive"
    );
    2.0 * length / wave_speed
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn rigid_wave_speed_of_water() {
        // K=2,2 GPa, ρ=1000 → a = √(2,2e6) ≈ 1483 m/s.
        let a = wave_speed_rigid(2.2e9, 1000.0);
        assert_relative_eq!(a, (2.2e9f64 / 1000.0).sqrt(), epsilon = 1e-6);
        assert!(a > 1400.0 && a < 1500.0);
    }

    #[test]
    fn elastic_pipe_slows_the_wave() {
        // La flexibilité de la paroi abaisse la célérité sous la valeur rigide.
        let rigid = wave_speed_rigid(2.2e9, 1000.0);
        let elastic = wave_speed_elastic(2.2e9, 1000.0, 0.3, 0.005, 200e9);
        assert!(elastic < rigid);
    }

    #[test]
    fn joukowsky_surge_scales_with_velocity_change() {
        // ρ=1000, a=1000, Δv=2 → ΔP = 2e6 Pa = 20 bar.
        assert_relative_eq!(joukowsky_surge(1000.0, 1000.0, 2.0), 2e6, epsilon = 1e-3);
    }

    #[test]
    fn critical_time_is_wave_round_trip() {
        // L=500 m, a=1000 m/s → tc = 1 s.
        assert_relative_eq!(critical_time(500.0, 1000.0), 1.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "célérité")]
    fn zero_wave_speed_critical_time_panics() {
        critical_time(500.0, 0.0);
    }
}
