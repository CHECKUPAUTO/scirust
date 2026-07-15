//! Écoulement isentropique compressible d'un gaz parfait — rapports d'arrêt (stagnation).
//!
//! ```text
//! rapport de température   T0/T   = 1 + (γ-1)/2·M²
//! rapport de pression      p0/p   = (1 + (γ-1)/2·M²)^(γ/(γ-1))
//! rapport de masse volum.  ρ0/ρ   = (1 + (γ-1)/2·M²)^(1/(γ-1))
//! rapport de section       A/A*   = (1/M)·((2/(γ+1))·(1+(γ-1)/2·M²))^((γ+1)/(2·(γ-1)))
//! ```
//!
//! `M` nombre de Mach local (sans dimension, M ≥ 0), `γ` rapport des chaleurs
//! spécifiques (sans dimension, γ > 1), `T0`/`T` température d'arrêt et statique
//! (K), `p0`/`p` pression d'arrêt et statique (Pa), `ρ0`/`ρ` masse volumique
//! d'arrêt et statique (kg·m⁻³), `A` section locale (m²), `A*` section du col
//! sonique (M = 1) (m²).
//!
//! **Convention** : SI cohérent ; les rapports renvoyés sont ≥ 1 (propriétés
//! d'arrêt locales, sans dimension). Complète [`crate::choked_flow`] (blocage
//! sonique). **Limite honnête** : gaz parfait, écoulement isentropique 1D
//! adiabatique réversible ; le rapport des chaleurs spécifiques `γ` et les
//! conditions d'écoulement (`M`) sont **fournis par l'appelant** — aucune valeur
//! de fluide, de matériau ou de procédé n'est supposée par défaut.

/// Rapport de température d'arrêt `T0/T = 1 + (γ-1)/2·M²` (sans dimension).
///
/// Panique si `mach < 0` ou `gamma <= 1`.
pub fn isentropic_temperature_ratio(mach: f64, gamma: f64) -> f64 {
    assert!(mach >= 0.0, "le nombre de Mach doit être positif ou nul");
    assert!(
        gamma > 1.0,
        "le rapport des chaleurs spécifiques doit être strictement supérieur à 1"
    );
    1.0 + (gamma - 1.0) / 2.0 * mach * mach
}

/// Rapport de pression d'arrêt `p0/p = (1 + (γ-1)/2·M²)^(γ/(γ-1))` (sans dimension).
///
/// Panique si `mach < 0` ou `gamma <= 1`.
pub fn isentropic_pressure_ratio(mach: f64, gamma: f64) -> f64 {
    assert!(mach >= 0.0, "le nombre de Mach doit être positif ou nul");
    assert!(
        gamma > 1.0,
        "le rapport des chaleurs spécifiques doit être strictement supérieur à 1"
    );
    isentropic_temperature_ratio(mach, gamma).powf(gamma / (gamma - 1.0))
}

/// Rapport de masse volumique d'arrêt `ρ0/ρ = (1 + (γ-1)/2·M²)^(1/(γ-1))` (sans dimension).
///
/// Panique si `mach < 0` ou `gamma <= 1`.
pub fn isentropic_density_ratio(mach: f64, gamma: f64) -> f64 {
    assert!(mach >= 0.0, "le nombre de Mach doit être positif ou nul");
    assert!(
        gamma > 1.0,
        "le rapport des chaleurs spécifiques doit être strictement supérieur à 1"
    );
    isentropic_temperature_ratio(mach, gamma).powf(1.0 / (gamma - 1.0))
}

/// Rapport de section
/// `A/A* = (1/M)·((2/(γ+1))·(1+(γ-1)/2·M²))^((γ+1)/(2·(γ-1)))` (sans dimension),
/// où `A*` est la section du col sonique.
///
/// Panique si `mach <= 0` ou `gamma <= 1`.
pub fn isentropic_area_ratio(mach: f64, gamma: f64) -> f64 {
    assert!(
        mach > 0.0,
        "le nombre de Mach doit être strictement positif (division par M)"
    );
    assert!(
        gamma > 1.0,
        "le rapport des chaleurs spécifiques doit être strictement supérieur à 1"
    );
    let exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0));
    let base = 2.0 / (gamma + 1.0) * isentropic_temperature_ratio(mach, gamma);
    base.powf(exponent) / mach
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn temperature_ratio_at_rest_is_unity() {
        // À M = 0, l'écoulement est au repos : T0/T = 1 quel que soit γ.
        assert_relative_eq!(isentropic_temperature_ratio(0.0, 1.4), 1.0, epsilon = 1e-12);
        assert_relative_eq!(isentropic_temperature_ratio(0.0, 1.3), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn pressure_ratio_is_temperature_ratio_to_gamma_exponent() {
        // Identité isentropique : p0/p = (T0/T)^(γ/(γ-1)).
        let (mach, gamma) = (1.7, 1.4);
        let t = isentropic_temperature_ratio(mach, gamma);
        assert_relative_eq!(
            isentropic_pressure_ratio(mach, gamma),
            t.powf(gamma / (gamma - 1.0)),
            epsilon = 1e-12
        );
    }

    #[test]
    fn pressure_ratio_equals_density_times_temperature_ratio() {
        // Loi des gaz parfaits : p0/p = (ρ0/ρ)·(T0/T).
        let (mach, gamma) = (2.5, 1.33);
        assert_relative_eq!(
            isentropic_pressure_ratio(mach, gamma),
            isentropic_density_ratio(mach, gamma) * isentropic_temperature_ratio(mach, gamma),
            epsilon = 1e-10
        );
    }

    #[test]
    fn area_ratio_is_unity_at_sonic_throat() {
        // À M = 1, la section vaut celle du col : A/A* = 1 quel que soit γ.
        assert_relative_eq!(isentropic_area_ratio(1.0, 1.4), 1.0, epsilon = 1e-12);
        assert_relative_eq!(isentropic_area_ratio(1.0, 1.2), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn realistic_case_mach_two_air() {
        // Air (γ = 1,4) à M = 2 :
        //   T0/T = 1 + 0,2·4 = 1,8
        //   A/A* = (1/2)·((2/2,4)·1,8)^3 = 0,5·1,5^3 = 0,5·3,375 = 1,6875
        assert_relative_eq!(isentropic_temperature_ratio(2.0, 1.4), 1.8, epsilon = 1e-12);
        assert_relative_eq!(isentropic_area_ratio(2.0, 1.4), 1.6875, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "strictement supérieur à 1")]
    fn invalid_gamma_panics() {
        let _ = isentropic_pressure_ratio(1.5, 1.0);
    }
}
