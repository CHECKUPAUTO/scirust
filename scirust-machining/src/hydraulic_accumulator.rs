//! **Accumulateur hydropneumatique** — détente isotherme du gaz (loi de Boyle) :
//! volume de gaz sous pression de service, volume de fluide utile et
//! dimensionnement du volume de gaz de pré-charge.
//!
//! ```text
//! Boyle isotherme   P₀·V₀ = P·V
//! volume de gaz     V(P) = P₀·V₀ / P
//! fluide utile      ΔV = P₀·V₀·(1/P_min − 1/P_max)
//! volume de gaz     V₀ = ΔV / (P₀·(1/P_min − 1/P_max))
//! ```
//!
//! `P₀` pression de pré-charge (Pa, **absolue**), `V₀` volume de gaz à la
//! pré-charge (m³), `P` pression de service (Pa, absolue), `V(P)` volume de gaz
//! comprimé (m³), `P_min`/`P_max` pressions de service basse/haute (Pa, absolues)
//! avec `P₀ ≤ P_min < P_max`, `ΔV` volume de fluide utile échangé (m³).
//!
//! **Convention** : pressions **absolues** (ajouter la pression atmosphérique aux
//! pressions relatives), volumes en m³, SI.
//! **Limite honnête** : gaz **parfait** et détente **isotherme** (loi de Boyle),
//! qui constitue la **borne haute** du fluide utile ; une détente rapide est
//! quasi **adiabatique** et fournit **moins** de fluide utile. Aucun volume mort,
//! aucune perte, aucun effet de solubilité du gaz. Les pressions de pré-charge et
//! de service sont des données de l'installation fournies par l'appelant ; aucune
//! valeur « par défaut » n'est supposée.

/// Volume de gaz sous une pression de service donnée `V(P) = P₀·V₀ / P`
/// (loi de Boyle isotherme, pressions **absolues**).
///
/// Panique si un paramètre `<= 0`.
pub fn gas_volume_at_pressure(
    precharge_pressure: f64,
    gas_volume_precharge: f64,
    working_pressure: f64,
) -> f64 {
    assert!(
        precharge_pressure > 0.0 && gas_volume_precharge > 0.0 && working_pressure > 0.0,
        "P₀, V₀ et P (absolus) doivent être strictement positifs"
    );
    precharge_pressure * gas_volume_precharge / working_pressure
}

/// Volume de fluide **utile** échangé entre `P_min` et `P_max`
/// `ΔV = P₀·V₀·(1/P_min − 1/P_max)` (détente isotherme, pressions absolues).
///
/// Panique si un paramètre `<= 0` ou si `min_pressure >= max_pressure`.
pub fn usable_fluid_volume(
    precharge_pressure: f64,
    gas_volume_precharge: f64,
    min_pressure: f64,
    max_pressure: f64,
) -> f64 {
    assert!(
        precharge_pressure > 0.0
            && gas_volume_precharge > 0.0
            && min_pressure > 0.0
            && max_pressure > 0.0,
        "P₀, V₀, P_min et P_max (absolus) doivent être strictement positifs"
    );
    assert!(
        min_pressure < max_pressure,
        "P_min doit être strictement inférieure à P_max"
    );
    precharge_pressure * gas_volume_precharge * (1.0 / min_pressure - 1.0 / max_pressure)
}

/// Volume de gaz de pré-charge requis pour fournir un fluide utile donné
/// `V₀ = ΔV / (P₀·(1/P_min − 1/P_max))` (réciproque isotherme de
/// [`usable_fluid_volume`]).
///
/// Panique si un paramètre `<= 0` ou si `min_pressure >= max_pressure`.
pub fn required_gas_volume(
    usable_volume: f64,
    precharge_pressure: f64,
    min_pressure: f64,
    max_pressure: f64,
) -> f64 {
    assert!(
        usable_volume > 0.0 && precharge_pressure > 0.0 && min_pressure > 0.0 && max_pressure > 0.0,
        "ΔV, P₀, P_min et P_max (absolus) doivent être strictement positifs"
    );
    assert!(
        min_pressure < max_pressure,
        "P_min doit être strictement inférieure à P_max"
    );
    usable_volume / (precharge_pressure * (1.0 / min_pressure - 1.0 / max_pressure))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn boyle_law_conserves_pv_product() {
        // P·V(P) doit rester égal à P₀·V₀ (loi de Boyle isotherme).
        let (p0, v0) = (100e5, 10e-3);
        let v = gas_volume_at_pressure(p0, v0, 250e5);
        assert_relative_eq!(250e5 * v, p0 * v0, epsilon = 1e-9);
    }

    #[test]
    fn gas_volume_unchanged_at_precharge() {
        // À P = P₀, le gaz occupe tout le volume de pré-charge.
        let (p0, v0) = (120e5, 5e-3);
        assert_relative_eq!(gas_volume_at_pressure(p0, v0, p0), v0, epsilon = 1e-15);
    }

    #[test]
    fn usable_volume_is_difference_of_gas_volumes() {
        // ΔV = V(P_min) − V(P_max) : le fluide utile est le gaz chassé/repris.
        let (p0, v0, pmin, pmax) = (100e5, 8e-3, 150e5, 300e5);
        let du = usable_fluid_volume(p0, v0, pmin, pmax);
        let diff = gas_volume_at_pressure(p0, v0, pmin) - gas_volume_at_pressure(p0, v0, pmax);
        assert_relative_eq!(du, diff, epsilon = 1e-12);
    }

    #[test]
    fn required_gas_volume_inverts_usable_volume() {
        // Réciprocité : dimensionner puis évaluer redonne le fluide utile visé.
        let (p0, pmin, pmax) = (90e5, 130e5, 280e5);
        let target = 2.0e-3;
        let v0 = required_gas_volume(target, p0, pmin, pmax);
        assert_relative_eq!(
            usable_fluid_volume(p0, v0, pmin, pmax),
            target,
            epsilon = 1e-12
        );
    }

    #[test]
    fn usable_volume_proportional_to_gas_volume() {
        // ΔV est linéaire en V₀ : doubler V₀ double le fluide utile.
        let (p0, pmin, pmax) = (100e5, 160e5, 320e5);
        let du1 = usable_fluid_volume(p0, 4e-3, pmin, pmax);
        let du2 = usable_fluid_volume(p0, 8e-3, pmin, pmax);
        assert_relative_eq!(du2, 2.0 * du1, epsilon = 1e-12);
    }

    #[test]
    fn realistic_usable_volume_case() {
        // Accumulateur 10 L pré-chargé à 100 bar (abs), service 150→300 bar abs :
        // ΔV = 100e5·10e-3·(1/150e5 − 1/300e5) = 100000·(1/150e5 − 1/300e5) m³.
        let du = usable_fluid_volume(100e5, 10e-3, 150e5, 300e5);
        let expected = 100e5 * 10e-3 * (1.0 / 150e5 - 1.0 / 300e5);
        assert_relative_eq!(du, expected, epsilon = 1e-12);
        // ≈ 3,33 L d'huile disponibles.
        assert!(du > 3.3e-3 && du < 3.4e-3);
    }

    #[test]
    #[should_panic(expected = "P_min doit être strictement inférieure à P_max")]
    fn inverted_pressures_panic() {
        usable_fluid_volume(100e5, 10e-3, 300e5, 150e5);
    }
}
