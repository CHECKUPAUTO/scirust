//! Vitesses critiques des arbres tournants (fouettement/whirling) — vitesse
//! critique d'un disque sur arbre, formule de **Rankine** (flèche statique) et
//! combinaison de **Dunkerley** pour plusieurs masses.
//!
//! ```text
//! critique (1 masse)   ωc = √(k/m) = √(g/δ_st)     Nc = 60·ωc/(2π)
//! Dunkerley            1/ωc² = 1/ω1² + 1/ω2² + …    (borne inférieure)
//! ```
//!
//! `k` raideur transversale de l'arbre (N/m), `m` masse du disque (kg), `δ_st`
//! flèche statique sous le poids (m), `g` pesanteur (m/s²). La vitesse critique
//! coïncide numériquement avec la pulsation propre de flexion : à cette vitesse
//! la déformée du rotor devient instable.
//!
//! **Convention** : SI cohérent ; `ω` en rad/s, `N` en tr/min. **Limite honnête** :
//! rotor idéal (arbre sans masse ou masses concentrées), amortissement négligé,
//! premier mode ; Dunkerley donne une **borne inférieure** de la première
//! critique — pas la valeur exacte d'un système couplé. `g` est fourni par
//! l'appelant.

use core::f64::consts::PI;

/// Vitesse critique `ωc = √(k/m)` (rad/s), raideur transversale `k`, masse `m`.
///
/// Panique si `m <= 0` ou `k < 0`.
pub fn critical_speed_rad(stiffness_n_m: f64, mass_kg: f64) -> f64 {
    assert!(
        mass_kg > 0.0 && stiffness_n_m >= 0.0,
        "m > 0 et k ≥ 0 requis"
    );
    (stiffness_n_m / mass_kg).sqrt()
}

/// Vitesse critique par la **flèche statique** `ωc = √(g/δ_st)` (rad/s).
///
/// Panique si `static_deflection <= 0`.
pub fn critical_speed_from_deflection_rad(g_m_s2: f64, static_deflection_m: f64) -> f64 {
    assert!(
        static_deflection_m > 0.0,
        "la flèche statique doit être strictement positive"
    );
    (g_m_s2 / static_deflection_m).sqrt()
}

/// Conversion `ω` (rad/s) → vitesse critique en tr/min `Nc = 60·ω/(2π)`.
pub fn rad_to_rpm(omega_rad_s: f64) -> f64 {
    60.0 * omega_rad_s / (2.0 * PI)
}

/// Combinaison de **Dunkerley** : `1/ωc² = Σ 1/ωi²` → renvoie `ωc` (rad/s),
/// borne inférieure de la première vitesse critique d'un rotor à plusieurs masses.
///
/// Panique si la liste est vide ou contient une vitesse `≤ 0`.
pub fn dunkerley_critical_speed_rad(component_speeds_rad_s: &[f64]) -> f64 {
    assert!(
        !component_speeds_rad_s.is_empty(),
        "au moins une vitesse critique partielle est requise"
    );
    let sum_inv_sq: f64 = component_speeds_rad_s
        .iter()
        .map(|&w| {
            assert!(
                w > 0.0,
                "chaque vitesse partielle doit être strictement positive"
            );
            1.0 / (w * w)
        })
        .sum();
    (1.0 / sum_inv_sq).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn critical_speed_of_a_disk_on_shaft() {
        // k=1e6 N/m, m=10 kg → ωc = √(1e5) ≈ 316,2 rad/s.
        assert_relative_eq!(
            critical_speed_rad(1e6, 10.0),
            (1e5f64).sqrt(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn deflection_and_stiffness_agree() {
        // δ_st = m·g/k. Avec g=9,81, m=10, k=1e6 → δ=9,81e-5 m.
        // √(g/δ) doit égaler √(k/m).
        let (g, m, k) = (9.81, 10.0, 1e6);
        let delta = m * g / k;
        assert_relative_eq!(
            critical_speed_from_deflection_rad(g, delta),
            critical_speed_rad(k, m),
            epsilon = 1e-6
        );
    }

    #[test]
    fn rpm_conversion() {
        // ω = 2π rad/s → 60 tr/min.
        assert_relative_eq!(rad_to_rpm(2.0 * PI), 60.0, epsilon = 1e-12);
    }

    #[test]
    fn dunkerley_is_below_each_component() {
        // ω1=300, ω2=400 → 1/ωc² = 1/90000 + 1/160000 → ωc = 240 rad/s.
        let wc = dunkerley_critical_speed_rad(&[300.0, 400.0]);
        assert_relative_eq!(wc, 240.0, epsilon = 1e-6);
        // La combinaison abaisse la critique sous la plus faible composante.
        assert!(wc < 300.0);
    }

    #[test]
    fn single_component_dunkerley_is_identity() {
        assert_relative_eq!(
            dunkerley_critical_speed_rad(&[250.0]),
            250.0,
            epsilon = 1e-9
        );
    }

    #[test]
    #[should_panic(expected = "flèche statique")]
    fn zero_deflection_panics() {
        critical_speed_from_deflection_rad(9.81, 0.0);
    }
}
