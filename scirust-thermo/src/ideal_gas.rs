//! Calorically perfect ideal gas: equation of state, quasi-static
//! process work/heat, and entropy changes.
//!
//! All quantities are **specific** (per unit mass, SI): pressures in Pa,
//! temperatures in K, specific volumes in m³/kg, energies in J/kg.
//! Work is the work done **by** the gas (positive on expansion).

use crate::error::{ThermoError, in_range, positive};

/// A calorically perfect ideal gas, defined by its specific gas constant
/// `R` \[J/(kg·K)\] and heat-capacity ratio `γ = cp/cv`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IdealGas {
    r: f64,
    gamma: f64,
}

impl IdealGas {
    /// Build a gas from its specific gas constant `R > 0` \[J/(kg·K)\]
    /// and heat-capacity ratio `γ ∈ (1, 2]`.
    pub fn new(r: f64, gamma: f64) -> Result<Self, ThermoError> {
        positive("r", r)?;
        in_range("gamma", gamma, 1.0 + 1e-12, 2.0)?;
        Ok(Self { r, gamma })
    }

    /// Dry air as a perfect gas: `R = 287.052874 J/(kg·K)`, `γ = 1.4`.
    pub fn air() -> Self {
        Self {
            r: 287.052_874,
            gamma: 1.4,
        }
    }

    /// Specific gas constant `R` \[J/(kg·K)\].
    pub fn r(&self) -> f64 {
        self.r
    }

    /// Heat-capacity ratio `γ`.
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    /// Isobaric specific heat `cp = γ R/(γ−1)` \[J/(kg·K)\].
    pub fn cp(&self) -> f64 {
        self.gamma * self.r / (self.gamma - 1.0)
    }

    /// Isochoric specific heat `cv = R/(γ−1)` \[J/(kg·K)\].
    pub fn cv(&self) -> f64 {
        self.r / (self.gamma - 1.0)
    }

    /// Pressure from density and temperature, `p = ρ R T` \[Pa\].
    pub fn pressure(&self, density: f64, temperature: f64) -> Result<f64, ThermoError> {
        positive("density", density)?;
        positive("temperature", temperature)?;
        Ok(density * self.r * temperature)
    }

    /// Density from pressure and temperature, `ρ = p/(R T)` \[kg/m³\].
    pub fn density(&self, pressure: f64, temperature: f64) -> Result<f64, ThermoError> {
        positive("pressure", pressure)?;
        positive("temperature", temperature)?;
        Ok(pressure / (self.r * temperature))
    }

    /// Temperature from pressure and density, `T = p/(ρ R)` \[K\].
    pub fn temperature(&self, pressure: f64, density: f64) -> Result<f64, ThermoError> {
        positive("pressure", pressure)?;
        positive("density", density)?;
        Ok(pressure / (density * self.r))
    }

    /// Specific work done by the gas in a reversible **isothermal**
    /// process at temperature `T` from `v1` to `v2`:
    /// `w = R T ln(v₂/v₁)` \[J/kg\] (equals the heat received).
    pub fn isothermal_work(&self, temperature: f64, v1: f64, v2: f64) -> Result<f64, ThermoError> {
        positive("temperature", temperature)?;
        positive("v1", v1)?;
        positive("v2", v2)?;
        Ok(self.r * temperature * (v2 / v1).ln())
    }

    /// Specific work done by the gas in an **isobaric** process,
    /// `w = p (v₂ − v₁)` \[J/kg\].
    pub fn isobaric_work(&self, pressure: f64, v1: f64, v2: f64) -> Result<f64, ThermoError> {
        positive("pressure", pressure)?;
        positive("v1", v1)?;
        positive("v2", v2)?;
        Ok(pressure * (v2 - v1))
    }

    /// Specific heat received in an **isobaric** process between
    /// temperatures `T₁` and `T₂`: `q = cp (T₂ − T₁)` \[J/kg\].
    pub fn isobaric_heat(&self, t1: f64, t2: f64) -> Result<f64, ThermoError> {
        positive("t1", t1)?;
        positive("t2", t2)?;
        Ok(self.cp() * (t2 - t1))
    }

    /// Specific heat received in an **isochoric** process,
    /// `q = cv (T₂ − T₁)` \[J/kg\] (no work is done).
    pub fn isochoric_heat(&self, t1: f64, t2: f64) -> Result<f64, ThermoError> {
        positive("t1", t1)?;
        positive("t2", t2)?;
        Ok(self.cv() * (t2 - t1))
    }

    /// Specific work done by the gas in a reversible **adiabatic**
    /// (isentropic) process between temperatures `T₁` and `T₂`:
    /// `w = cv (T₁ − T₂)` \[J/kg\].
    pub fn adiabatic_work(&self, t1: f64, t2: f64) -> Result<f64, ThermoError> {
        positive("t1", t1)?;
        positive("t2", t2)?;
        Ok(self.cv() * (t1 - t2))
    }

    /// Final temperature of a reversible adiabatic process from `T₁`
    /// with volume ratio `v₁/v₂`: `T₂ = T₁ (v₁/v₂)^{γ−1}` \[K\].
    pub fn adiabatic_temperature(&self, t1: f64, v1: f64, v2: f64) -> Result<f64, ThermoError> {
        positive("t1", t1)?;
        positive("v1", v1)?;
        positive("v2", v2)?;
        Ok(t1 * (v1 / v2).powf(self.gamma - 1.0))
    }

    /// Final pressure of a reversible adiabatic process from `p₁` with
    /// volume ratio `v₁/v₂`: `p₂ = p₁ (v₁/v₂)^γ` \[Pa\].
    pub fn adiabatic_pressure(&self, p1: f64, v1: f64, v2: f64) -> Result<f64, ThermoError> {
        positive("p1", p1)?;
        positive("v1", v1)?;
        positive("v2", v2)?;
        Ok(p1 * (v1 / v2).powf(self.gamma))
    }

    /// Specific work done by the gas in a **polytropic** process
    /// `p vⁿ = const` with exponent `n ≠ 1`:
    /// `w = (p₁v₁ − p₂v₂)/(n − 1)` \[J/kg\].
    pub fn polytropic_work(
        &self,
        n: f64,
        p1: f64,
        v1: f64,
        p2: f64,
        v2: f64,
    ) -> Result<f64, ThermoError> {
        crate::error::finite("n", n)?;
        if (n - 1.0).abs() < 1e-12
        {
            return Err(ThermoError::OutOfRange {
                name: "n",
                value: n,
                min: 1.0,
                max: 1.0,
            });
        }
        positive("p1", p1)?;
        positive("v1", v1)?;
        positive("p2", p2)?;
        positive("v2", v2)?;
        Ok((p1 * v1 - p2 * v2) / (n - 1.0))
    }

    /// Specific-entropy change between states `(T₁, p₁)` and `(T₂, p₂)`:
    /// `Δs = cp ln(T₂/T₁) − R ln(p₂/p₁)` \[J/(kg·K)\].
    pub fn entropy_change(&self, t1: f64, p1: f64, t2: f64, p2: f64) -> Result<f64, ThermoError> {
        positive("t1", t1)?;
        positive("p1", p1)?;
        positive("t2", t2)?;
        positive("p2", p2)?;
        Ok(self.cp() * (t2 / t1).ln() - self.r * (p2 / p1).ln())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn air_heat_capacities() {
        // γR/(γ−1) = 1.4·287.052874/0.4 = 1004.685; cv = 717.632.
        let air = IdealGas::air();
        assert!((air.cp() - 1004.685).abs() < 1e-2);
        assert!((air.cv() - 717.632).abs() < 1e-2);
        assert!((air.cp() - air.cv() - air.r()).abs() < 1e-9); // Mayer
    }

    #[test]
    fn isa_sea_level_density() {
        // 101325 Pa, 288.15 K → ρ = 1.2250 kg/m³ (ISA standard value).
        let rho = IdealGas::air().density(101_325.0, 288.15).unwrap();
        assert!((rho - 1.225).abs() < 5e-4, "rho = {rho}");
    }

    #[test]
    fn state_equation_roundtrip() {
        let air = IdealGas::air();
        let rho = air.density(2.0e5, 350.0).unwrap();
        assert!((air.pressure(rho, 350.0).unwrap() - 2.0e5).abs() < 1e-6);
        assert!((air.temperature(2.0e5, rho).unwrap() - 350.0).abs() < 1e-9);
    }

    #[test]
    fn adiabatic_compression_8_to_1() {
        // T₂ = 300·8^0.4 = 689.22 K; p grows by 8^1.4 = 18.379.
        let air = IdealGas::air();
        let t2 = air.adiabatic_temperature(300.0, 8.0, 1.0).unwrap();
        assert!((t2 - 689.219).abs() < 5e-3, "T2 = {t2}");
        let p2 = air.adiabatic_pressure(1.0e5, 8.0, 1.0).unwrap();
        assert!((p2 - 18.379_2e5).abs() / p2 < 1e-4, "p2 = {p2}");
        // Isentropic: entropy change must vanish.
        let ds = air.entropy_change(300.0, 1.0e5, t2, p2).unwrap();
        assert!(ds.abs() < 1e-9, "ds = {ds}");
    }

    #[test]
    fn isothermal_work_equals_heat_and_entropy_matches() {
        // Isothermal expansion doubles the volume at 300 K:
        // w = q = RT ln 2; Δs = q/T = R ln 2.
        let air = IdealGas::air();
        let w = air.isothermal_work(300.0, 1.0, 2.0).unwrap();
        assert!((w - 287.052_874 * 300.0 * 2.0f64.ln()).abs() < 1e-9);
        let ds = air.entropy_change(300.0, 2.0e5, 300.0, 1.0e5).unwrap();
        assert!((ds - w / 300.0).abs() < 1e-9);
    }

    #[test]
    fn first_law_isobaric() {
        // Isobaric: q = w + Δu, i.e. cpΔT = pΔv + cvΔT.
        let air = IdealGas::air();
        let p = 1.0e5;
        let (t1, t2) = (300.0, 400.0);
        let v1 = air.r() * t1 / p;
        let v2 = air.r() * t2 / p;
        let q = air.isobaric_heat(t1, t2).unwrap();
        let w = air.isobaric_work(p, v1, v2).unwrap();
        let du = air.cv() * (t2 - t1);
        assert!((q - (w + du)).abs() < 1e-9);
    }

    #[test]
    fn polytropic_reduces_to_adiabatic() {
        // With n = γ the polytropic work equals the adiabatic work.
        let air = IdealGas::air();
        let (p1, t1) = (1.0e5, 300.0);
        let v1 = air.r() * t1 / p1;
        let v2 = v1 / 5.0;
        let p2 = air.adiabatic_pressure(p1, v1, v2).unwrap();
        let t2 = air.adiabatic_temperature(t1, v1, v2).unwrap();
        let w_poly = air.polytropic_work(air.gamma(), p1, v1, p2, v2).unwrap();
        let w_adia = air.adiabatic_work(t1, t2).unwrap();
        assert!((w_poly - w_adia).abs() / w_adia.abs() < 1e-12);
    }

    #[test]
    fn rejects_bad_gas_and_isothermal_exponent() {
        assert!(IdealGas::new(-287.0, 1.4).is_err());
        assert!(IdealGas::new(287.0, 1.0).is_err());
        let air = IdealGas::air();
        assert!(air.polytropic_work(1.0, 1e5, 1.0, 1e5, 1.0).is_err());
        assert!(air.density(0.0, 300.0).is_err());
    }
}
