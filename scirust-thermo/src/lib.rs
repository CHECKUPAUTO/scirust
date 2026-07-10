//! # `scirust-thermo` — deterministic engineering thermodynamics
//!
//! Pure-Rust, dependency-free, `#![forbid(unsafe_code)]` building blocks
//! for engineering thermodynamics and heat transfer. Every function
//! validates its inputs and returns a [`ThermoError`] instead of
//! panicking or propagating NaN.
//!
//! ## Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`ideal_gas`] | perfect-gas state equation, process work/heat (isothermal, isobaric, isochoric, adiabatic, polytropic), entropy changes |
//! | [`cycles`] | Carnot (efficiency & COPs), Otto, Diesel, Brayton air-standard cycles |
//! | [`heat_transfer`] | conduction resistances (plane, cylindrical), convection, radiation, LMTD, effectiveness-NTU, Dittus–Boelter |
//! | [`psychro`] | moist air: Hyland–Wexler saturation pressure, humidity ratio, dew point, enthalpy, specific volume |
//! | [`steam`] | water saturation line, IAPWS-IF97 region 4 (`p_sat(T)`, `T_sat(p)`) |
//!
//! ## Guarantees
//!
//! - **Deterministic**: no global state, no RNG; the only iterative
//!   solver (dew point) is a fixed bisection — identical inputs give
//!   identical outputs on every platform.
//! - **Validated**: non-finite, non-positive or out-of-domain arguments
//!   are rejected with a typed [`ThermoError`]; correlation validity
//!   ranges are enforced, not just documented.
//! - **Oracle-tested**: results are pinned to published values (IF97
//!   verification tables 35/36, ASHRAE psychrometric tables, ISA air
//!   properties, classic cycle efficiencies, Incropera NTU tables, …).
//!
//! ## Example
//!
//! ```
//! use scirust_thermo::{cycles, ideal_gas::IdealGas, steam};
//!
//! // Otto cycle with compression ratio 9 on air.
//! let air = IdealGas::air();
//! let eta = cycles::otto_efficiency(9.0, air.gamma()).unwrap();
//! assert!(eta > 0.55 && eta < 0.60);
//!
//! // Water boils just below 100 °C at one atmosphere.
//! let t_boil = steam::saturation_temperature(101_325.0).unwrap();
//! assert!((t_boil - 373.12).abs() < 0.1);
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod error;

pub mod cycles;
pub mod heat_transfer;
pub mod ideal_gas;
pub mod psychro;
pub mod steam;

pub use error::ThermoError;

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------- //
    //  Cross-module consistency checks.                                //
    // ---------------------------------------------------------------- //

    #[test]
    fn carnot_bounds_every_real_cycle() {
        // An Otto cycle between the same temperature extremes can never
        // beat Carnot: T₁ = 300 K, r = 8 → T₂ = 689.2 K after compression;
        // take T_max = 2000 K at combustion.
        let air = ideal_gas::IdealGas::air();
        let eta_otto = cycles::otto_efficiency(8.0, air.gamma()).unwrap();
        let eta_carnot = cycles::carnot_efficiency(2000.0, 300.0).unwrap();
        assert!(eta_otto < eta_carnot);
    }

    #[test]
    fn steam_and_psychro_saturation_agree_at_ambient() {
        // Two independent formulations (IF97 vs Hyland-Wexler) of the
        // same physical curve must agree closely in the overlap.
        for &t in &[273.15, 293.15, 313.15, 353.15, 373.15]
        {
            let p_if97 = steam::saturation_pressure(t).unwrap();
            let p_hw = psychro::saturation_pressure(t).unwrap();
            assert!(
                (p_if97 - p_hw).abs() / p_if97 < 5e-3,
                "disagreement at {t} K: {p_if97} vs {p_hw}"
            );
        }
    }

    #[test]
    fn entropy_of_carnot_cycle_closes() {
        // Reversible cycle: net entropy change of the working gas is zero.
        // Isothermal expansion at T_h, adiabatic to T_c, isothermal
        // compression at T_c, adiabatic back — track (T, p) around.
        let air = ideal_gas::IdealGas::air();
        let (th, tc) = (600.0_f64, 300.0_f64);
        let p1 = 10.0e5;
        let p2 = 5.0e5; // isothermal expansion at T_h
        let pr = (tc / th).powf(air.gamma() / (air.gamma() - 1.0));
        let p3 = p2 * pr; // adiabatic T_h → T_c
        let p4 = p1 * pr; // where the second adiabat must land
        let ds = air.entropy_change(th, p1, th, p2).unwrap()
            + air.entropy_change(th, p2, tc, p3).unwrap()
            + air.entropy_change(tc, p3, tc, p4).unwrap()
            + air.entropy_change(tc, p4, th, p1).unwrap();
        assert!(ds.abs() < 1e-10, "ds = {ds}");
    }
}
