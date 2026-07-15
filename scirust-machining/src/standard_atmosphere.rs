//! Atmosphère standard — modèle **barométrique de la troposphère (ISA)** à
//! gradient de température linéaire, air assimilé à un gaz parfait.
//!
//! ```text
//! température      T(h) = T0 − lapse·h
//! pression         p(h) = p0·(1 − lapse·h/T0)^(g·M/(R·lapse))
//! masse volumique  ρ    = p·M/(R·T)
//! ```
//!
//! `h` altitude au-dessus du niveau de la mer [m], `T0`/`p0` température [K] et
//! pression [Pa] au sol, `lapse` gradient thermique vertical [K/m], `T`/`p`
//! température [K] et pression [Pa] locales, `g` accélération de la pesanteur
//! [m/s²], `M` masse molaire de l'air [kg/mol], `R` constante des gaz parfaits
//! [J/(mol·K)], `ρ` masse volumique [kg/m³]. Toutes les grandeurs sont en
//! **unités SI**.
//!
//! **Limite honnête** : ce modèle décrit la **troposphère** (gradient linéaire,
//! valable jusqu'à ~11 km, la tropopause) et suppose l'air **parfait**. Les
//! conditions au sol (`p0`, `T0`), le gradient `lapse`, la pesanteur `g`, la
//! masse molaire `M` et la constante `R` sont des **paramètres fournis par
//! l'appelant** : les constantes `ISA_*` exposées ci-dessous ne sont qu'un jeu
//! de référence **documenté** (atmosphère type de l'OACI), jamais imposé.

/// Pression standard au niveau de la mer (référence ISA) [Pa].
pub const ISA_SEA_LEVEL_PRESSURE: f64 = 101325.0;
/// Température standard au niveau de la mer (référence ISA) [K] (15 °C).
pub const ISA_SEA_LEVEL_TEMPERATURE: f64 = 288.15;
/// Gradient thermique vertical de la troposphère (référence ISA) [K/m].
pub const ISA_LAPSE_RATE: f64 = 0.0065;
/// Accélération de la pesanteur standard [m/s²].
pub const ISA_GRAVITY: f64 = 9.80665;
/// Masse molaire de l'air sec (référence ISA) [kg/mol].
pub const ISA_MOLAR_MASS_AIR: f64 = 0.0289644;
/// Constante universelle des gaz parfaits [J/(mol·K)].
pub const ISA_GAS_CONSTANT: f64 = 8.314462618;
/// Altitude de la tropopause, limite de validité du modèle [m].
pub const ISA_TROPOPAUSE_ALTITUDE: f64 = 11000.0;

/// Température locale `T(h) = T0 − lapse·h` [K].
///
/// `sea_level_temperature` [K], `lapse_rate` [K/m], `altitude` [m].
///
/// Panique si `sea_level_temperature <= 0`, ou si `lapse_rate` ou `altitude`
/// n'est pas fini.
pub fn isa_temperature(sea_level_temperature: f64, lapse_rate: f64, altitude: f64) -> f64 {
    assert!(
        sea_level_temperature > 0.0,
        "la température au sol doit être strictement positive (K)"
    );
    assert!(lapse_rate.is_finite(), "le gradient doit être fini (K/m)");
    assert!(altitude.is_finite(), "l'altitude doit être finie (m)");
    sea_level_temperature - lapse_rate * altitude
}

/// Pression locale `p(h) = p0·(1 − lapse·h/T0)^(g·M/(R·lapse))` [Pa].
///
/// `sea_level_pressure` [Pa], `altitude` [m], `lapse_rate` [K/m],
/// `sea_level_temperature` [K], `gravity` [m/s²], `molar_mass` [kg/mol],
/// `gas_constant` [J/(mol·K)].
///
/// Panique si l'un des paramètres physiques (`sea_level_pressure`,
/// `sea_level_temperature`, `lapse_rate`, `gravity`, `molar_mass`,
/// `gas_constant`) n'est pas strictement positif, ou si la base
/// `1 − lapse·h/T0` n'est pas strictement positive (altitude au-delà du domaine
/// du modèle).
pub fn isa_pressure(
    sea_level_pressure: f64,
    altitude: f64,
    lapse_rate: f64,
    sea_level_temperature: f64,
    gravity: f64,
    molar_mass: f64,
    gas_constant: f64,
) -> f64 {
    assert!(
        sea_level_pressure > 0.0,
        "la pression au sol doit être strictement positive (Pa)"
    );
    assert!(
        sea_level_temperature > 0.0,
        "la température au sol doit être strictement positive (K)"
    );
    assert!(
        lapse_rate > 0.0,
        "le gradient thermique doit être strictement positif (K/m)"
    );
    assert!(
        gravity > 0.0,
        "la pesanteur doit être strictement positive (m/s²)"
    );
    assert!(
        molar_mass > 0.0,
        "la masse molaire doit être strictement positive (kg/mol)"
    );
    assert!(
        gas_constant > 0.0,
        "la constante des gaz doit être strictement positive (J/(mol·K))"
    );
    let base = 1.0 - lapse_rate * altitude / sea_level_temperature;
    assert!(
        base > 0.0,
        "la base (1 − lapse·h/T0) doit être strictement positive (altitude hors domaine)"
    );
    let exponent = gravity * molar_mass / (gas_constant * lapse_rate);
    sea_level_pressure * base.powf(exponent)
}

/// Masse volumique de l'air parfait `ρ = p·M/(R·T)` [kg/m³].
///
/// `pressure` [Pa], `temperature` [K], `molar_mass` [kg/mol],
/// `gas_constant` [J/(mol·K)].
///
/// Panique si `pressure < 0`, `temperature <= 0`, `molar_mass <= 0` ou
/// `gas_constant <= 0`.
pub fn isa_density(pressure: f64, temperature: f64, molar_mass: f64, gas_constant: f64) -> f64 {
    assert!(
        pressure >= 0.0,
        "la pression doit être positive ou nulle (Pa)"
    );
    assert!(
        temperature > 0.0,
        "la température doit être strictement positive (K)"
    );
    assert!(
        molar_mass > 0.0,
        "la masse molaire doit être strictement positive (kg/mol)"
    );
    assert!(
        gas_constant > 0.0,
        "la constante des gaz doit être strictement positive (J/(mol·K))"
    );
    pressure * molar_mass / (gas_constant * temperature)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn temperature_at_sea_level_equals_ground() {
        // À h = 0, T(0) = T0 quel que soit le gradient.
        assert_relative_eq!(
            isa_temperature(ISA_SEA_LEVEL_TEMPERATURE, ISA_LAPSE_RATE, 0.0),
            ISA_SEA_LEVEL_TEMPERATURE,
            epsilon = 1e-12
        );
    }

    #[test]
    fn temperature_at_tropopause_is_216_65_k() {
        // Cas chiffré : T(11 km) = 288,15 − 0,0065·11000 = 288,15 − 71,5 = 216,65 K.
        assert_relative_eq!(
            isa_temperature(
                ISA_SEA_LEVEL_TEMPERATURE,
                ISA_LAPSE_RATE,
                ISA_TROPOPAUSE_ALTITUDE
            ),
            216.65,
            epsilon = 1e-9
        );
    }

    #[test]
    fn pressure_at_sea_level_equals_ground() {
        // À h = 0, la base vaut 1 donc p(0) = p0 exactement.
        assert_relative_eq!(
            isa_pressure(
                ISA_SEA_LEVEL_PRESSURE,
                0.0,
                ISA_LAPSE_RATE,
                ISA_SEA_LEVEL_TEMPERATURE,
                ISA_GRAVITY,
                ISA_MOLAR_MASS_AIR,
                ISA_GAS_CONSTANT
            ),
            ISA_SEA_LEVEL_PRESSURE,
            epsilon = 1e-6
        );
    }

    #[test]
    fn pressure_at_tropopause_matches_isa_reference() {
        // Cas chiffré : au sommet de la troposphère (11 km) la pression ISA
        // vaut environ 22 632 Pa (exposant g·M/(R·lapse) ≈ 5,2558).
        let p = isa_pressure(
            ISA_SEA_LEVEL_PRESSURE,
            ISA_TROPOPAUSE_ALTITUDE,
            ISA_LAPSE_RATE,
            ISA_SEA_LEVEL_TEMPERATURE,
            ISA_GRAVITY,
            ISA_MOLAR_MASS_AIR,
            ISA_GAS_CONSTANT,
        );
        assert_relative_eq!(p, 22632.0, epsilon = 2.0);
    }

    #[test]
    fn density_at_sea_level_is_1_225() {
        // Cas chiffré : ρ0 = p0·M/(R·T0) = 101325·0,0289644/(8,314462618·288,15)
        // ≈ 1,225 kg/m³ (masse volumique type ISA au niveau de la mer).
        let rho = isa_density(
            ISA_SEA_LEVEL_PRESSURE,
            ISA_SEA_LEVEL_TEMPERATURE,
            ISA_MOLAR_MASS_AIR,
            ISA_GAS_CONSTANT,
        );
        assert_relative_eq!(rho, 1.225, epsilon = 1e-3);
    }

    #[test]
    fn density_is_proportional_to_pressure() {
        // ρ = p·M/(R·T) est linéaire en p : doubler p double ρ.
        let rho1 = isa_density(50000.0, 250.0, ISA_MOLAR_MASS_AIR, ISA_GAS_CONSTANT);
        let rho2 = isa_density(100000.0, 250.0, ISA_MOLAR_MASS_AIR, ISA_GAS_CONSTANT);
        assert_relative_eq!(rho2, 2.0 * rho1, epsilon = 1e-12);
    }

    #[test]
    fn pressure_decreases_with_altitude() {
        // La pression barométrique décroît strictement avec l'altitude.
        let low = isa_pressure(
            ISA_SEA_LEVEL_PRESSURE,
            1000.0,
            ISA_LAPSE_RATE,
            ISA_SEA_LEVEL_TEMPERATURE,
            ISA_GRAVITY,
            ISA_MOLAR_MASS_AIR,
            ISA_GAS_CONSTANT,
        );
        let high = isa_pressure(
            ISA_SEA_LEVEL_PRESSURE,
            5000.0,
            ISA_LAPSE_RATE,
            ISA_SEA_LEVEL_TEMPERATURE,
            ISA_GRAVITY,
            ISA_MOLAR_MASS_AIR,
            ISA_GAS_CONSTANT,
        );
        assert!(high < low && low < ISA_SEA_LEVEL_PRESSURE);
    }

    #[test]
    #[should_panic(expected = "strictement positive")]
    fn negative_ground_temperature_panics() {
        isa_temperature(-5.0, ISA_LAPSE_RATE, 100.0);
    }
}
