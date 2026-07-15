//! **Coût de l'air comprimé et fuites** — puissance spécifique du compresseur,
//! débit d'une fuite par orifice, coût annuel d'une fuite et énergie annuelle
//! consommée.
//!
//! ```text
//! débit de fuite        Q     = Cd·A·√(2·Δp/ρ)              (orifice, incompressible)
//! puissance spécifique  e_s   = P / (FAD·60)                (kWh·m⁻³)
//! coût annuel d'une fuite  C  = Q·e_s·c·h
//! énergie annuelle      E     = P·h·k
//! ```
//!
//! `Cd` coefficient de décharge de l'orifice (sans dimension), `A` section de
//! l'orifice (m²), `Δp` pression relative amont−aval (Pa), `ρ` masse volumique de
//! l'air (kg·m⁻³), `Q` débit volumique de la fuite (m³·s⁻¹ en sortie de l'orifice,
//! ou m³·h⁻¹ pour le bilan de coût), `P` puissance électrique du compresseur (kW),
//! `FAD` débit d'air libre du compresseur (m³·min⁻¹), `e_s` puissance spécifique
//! (kWh·m⁻³, énergie électrique par m³ d'air libre), `c` prix de l'énergie
//! (€·kWh⁻¹), `h` heures de fonctionnement (h), `k` facteur de charge
//! (sans dimension, 0..1), `E` énergie annuelle (kWh).
//!
//! **Convention** : SI pour la physique de l'orifice (m², Pa, kg·m⁻³, m³·s⁻¹) ;
//! bilan économique cohérent en débit m³·h⁻¹, heures h, prix €·kWh⁻¹. Le débit de
//! fuite issu de [`compressed_air_leak_flow`] est en m³·s⁻¹ ; convertissez-le en
//! m³·h⁻¹ (×3600) avant de le passer à [`compressed_air_leak_cost`].
//!
//! **Limite honnête** : modèle d'orifice **incompressible** (borne indicative pour
//! l'air comprimé, néglige la compressibilité et l'écoulement critique/sonique) ;
//! bilan économique linéaire. La masse volumique `ρ`, le coefficient de décharge
//! `Cd`, la puissance spécifique du compresseur `e_s`, le prix de l'énergie `c`
//! et le facteur de charge `k` sont des données de procédé FOURNIES par l'appelant
//! ; aucune valeur « par défaut » n'est inventée. On néglige la variation de
//! rendement du compresseur en charge partielle (facteur de charge appliqué
//! linéairement).

/// Débit volumique d'une fuite par orifice `Q = Cd·A·√(2·Δp/ρ)` (m³·s⁻¹).
///
/// Modèle d'orifice incompressible : `orifice_area` en m², `supply_pressure` la
/// pression relative motrice amont−aval en Pa, `discharge_coefficient` = `Cd`
/// (0 < Cd ≤ 1), `air_density` = `ρ` en kg·m⁻³.
///
/// Panique si `orifice_area <= 0`, `supply_pressure <= 0`, `air_density <= 0`,
/// ou si `discharge_coefficient` hors de `]0, 1]`.
pub fn compressed_air_leak_flow(
    orifice_area: f64,
    supply_pressure: f64,
    discharge_coefficient: f64,
    air_density: f64,
) -> f64 {
    assert!(
        orifice_area > 0.0,
        "la section de l'orifice doit être strictement positive"
    );
    assert!(
        supply_pressure > 0.0,
        "la pression motrice doit être strictement positive"
    );
    assert!(
        air_density > 0.0,
        "la masse volumique de l'air doit être strictement positive"
    );
    assert!(
        discharge_coefficient > 0.0 && discharge_coefficient <= 1.0,
        "le coefficient de décharge doit être dans ]0, 1]"
    );
    discharge_coefficient * orifice_area * (2.0 * supply_pressure / air_density).sqrt()
}

/// Coût annuel d'une fuite `C = Q·e_s·c·h`.
///
/// `leak_volume_flow` = `Q` débit de la fuite en **m³·h⁻¹** (cohérent avec les
/// heures), `specific_power` = `e_s` en kWh·m⁻³, `energy_price` = `c` en €·kWh⁻¹,
/// `operating_hours` = `h` en h. Résultat en unité monétaire de `energy_price`.
///
/// Panique si un argument est `< 0`.
pub fn compressed_air_leak_cost(
    leak_volume_flow: f64,
    specific_power: f64,
    energy_price: f64,
    operating_hours: f64,
) -> f64 {
    assert!(
        leak_volume_flow >= 0.0,
        "le débit de fuite doit être positif ou nul"
    );
    assert!(
        specific_power >= 0.0,
        "la puissance spécifique doit être positive ou nulle"
    );
    assert!(
        energy_price >= 0.0,
        "le prix de l'énergie doit être positif ou nul"
    );
    assert!(
        operating_hours >= 0.0,
        "les heures de fonctionnement doivent être positives ou nulles"
    );
    leak_volume_flow * specific_power * energy_price * operating_hours
}

/// Puissance spécifique du compresseur `e_s = P / (FAD·60)` (kWh·m⁻³).
///
/// Énergie électrique consommée par m³ d'air libre produit. `compressor_power_kw`
/// = `P` en kW, `free_air_delivery_m3min` = `FAD` en m³·min⁻¹. Le facteur 60
/// convertit le débit m³·min⁻¹ en m³·h⁻¹, d'où kW/(m³·h⁻¹) = kWh·m⁻³.
///
/// Panique si `compressor_power_kw < 0` ou `free_air_delivery_m3min <= 0`.
pub fn compressed_air_specific_power(
    compressor_power_kw: f64,
    free_air_delivery_m3min: f64,
) -> f64 {
    assert!(
        compressor_power_kw >= 0.0,
        "la puissance du compresseur doit être positive ou nulle"
    );
    assert!(
        free_air_delivery_m3min > 0.0,
        "le débit d'air libre doit être strictement positif"
    );
    compressor_power_kw / (free_air_delivery_m3min * 60.0)
}

/// Énergie annuelle consommée `E = P·h·k` (kWh).
///
/// `power_kw` = `P` puissance nominale en kW, `operating_hours` = `h` en h,
/// `load_factor` = `k` facteur de charge moyen (0..1, sans dimension).
///
/// Panique si `power_kw < 0`, `operating_hours < 0`, ou `load_factor` hors de
/// `[0, 1]`.
pub fn compressed_air_annual_energy(power_kw: f64, operating_hours: f64, load_factor: f64) -> f64 {
    assert!(power_kw >= 0.0, "la puissance doit être positive ou nulle");
    assert!(
        operating_hours >= 0.0,
        "les heures de fonctionnement doivent être positives ou nulles"
    );
    assert!(
        (0.0..=1.0).contains(&load_factor),
        "le facteur de charge doit être dans [0, 1]"
    );
    power_kw * operating_hours * load_factor
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn leak_flow_realistic_case() {
        // Cd = 0,6 ; A = 1·10⁻⁴ m² ; Δp = 2·10⁵ Pa ; ρ = 1,6 kg·m⁻³.
        // √(2·2·10⁵/1,6) = √250000 = 500 m·s⁻¹.
        // Q = 0,6·10⁻⁴·500 = 0,03 m³·s⁻¹.
        let q = compressed_air_leak_flow(1e-4, 2e5, 0.6, 1.6);
        assert_relative_eq!(q, 0.03, epsilon = 1e-12);
    }

    #[test]
    fn leak_flow_scales_as_sqrt_pressure() {
        // Q ∝ √Δp : quadrupler la pression motrice double le débit.
        let q1 = compressed_air_leak_flow(1e-4, 2e5, 0.6, 1.6);
        let q4 = compressed_air_leak_flow(1e-4, 8e5, 0.6, 1.6);
        assert_relative_eq!(q4, 2.0 * q1, epsilon = 1e-12);
    }

    #[test]
    fn specific_power_realistic_case() {
        // P = 90 kW ; FAD = 10 m³·min⁻¹ → e_s = 90/(10·60) = 0,15 kWh·m⁻³.
        let es = compressed_air_specific_power(90.0, 10.0);
        assert_relative_eq!(es, 0.15, epsilon = 1e-12);
    }

    #[test]
    fn specific_power_inversely_proportional_to_fad() {
        // e_s ∝ 1/FAD : doubler le débit d'air libre divise la puissance spécifique par deux.
        let es1 = compressed_air_specific_power(90.0, 10.0);
        let es2 = compressed_air_specific_power(90.0, 20.0);
        assert_relative_eq!(es2, es1 / 2.0, epsilon = 1e-12);
    }

    #[test]
    fn annual_energy_realistic_case_and_load_proportionality() {
        // P = 90 kW ; h = 8000 h ; k = 0,7 → E = 90·8000·0,7 = 504000 kWh.
        let e = compressed_air_annual_energy(90.0, 8000.0, 0.7);
        assert_relative_eq!(e, 504_000.0, epsilon = 1e-6);
        // E ∝ k : pleine charge (k = 1) redonne P·h.
        let e_full = compressed_air_annual_energy(90.0, 8000.0, 1.0);
        assert_relative_eq!(e, e_full * 0.7, epsilon = 1e-6);
    }

    #[test]
    fn leak_cost_realistic_case_and_reciprocity() {
        // Q = 10 m³·h⁻¹ ; e_s = 0,12 kWh·m⁻³ ; c = 0,15 €·kWh⁻¹ ; h = 8000 h.
        // C = 10·0,12·0,15·8000 = 1440 €.
        let c = compressed_air_leak_cost(10.0, 0.12, 0.15, 8000.0);
        assert_relative_eq!(c, 1440.0, epsilon = 1e-9);
        // Réciprocité : C/h redonne le coût horaire Q·e_s·c.
        assert_relative_eq!(c / 8000.0, 10.0 * 0.12 * 0.15, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "coefficient de décharge doit être dans ]0, 1]")]
    fn discharge_coefficient_above_one_panics() {
        compressed_air_leak_flow(1e-4, 2e5, 1.2, 1.6);
    }
}
