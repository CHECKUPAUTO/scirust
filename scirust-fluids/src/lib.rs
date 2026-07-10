//! # `scirust-fluids` — deterministic fluid mechanics
//!
//! Pure-Rust, dependency-free, `#![forbid(unsafe_code)]` building blocks
//! for engineering fluid mechanics. Every function validates its inputs
//! and returns a [`FluidsError`] instead of panicking or propagating NaN.
//!
//! ## Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`dimensionless`] | Reynolds, Prandtl, Mach, Froude, Weber, Péclet, Strouhal, Nusselt |
//! | [`pipe`] | Darcy friction factors (laminar, Colebrook–White, Haaland, Swamee–Jain), Darcy–Weisbach losses, minor losses, hydraulic diameter |
//! | [`bernoulli`] | dynamic/stagnation pressure, Pitot, Torricelli, Bernoulli between stations, Venturi & orifice metering |
//! | [`external`] | Stokes drag, standard sphere drag curve (Clift–Gauvin), terminal settling velocity |
//! | [`boundary_layer`] | Blasius laminar & 1/7-power turbulent flat-plate results |
//! | [`compressible`] | speed of sound, isentropic ratios, `A/A*`, normal-shock jump relations |
//! | [`channel`] | Manning's equation, critical & normal depth, specific energy, hydraulic jump |
//! | [`network`] | looped pipe networks solved by the Hardy Cross method |
//!
//! ## Guarantees
//!
//! - **Deterministic**: no global state, no RNG; the iterative solvers
//!   (Colebrook–White, terminal velocity, normal depth) use fixed,
//!   input-independent algorithms — identical inputs give identical
//!   outputs on every platform.
//! - **Validated**: non-finite, non-positive or out-of-domain arguments
//!   are rejected with a typed [`FluidsError`].
//! - **Oracle-tested**: results are pinned to published values (Moody
//!   chart, NACA Report 1135 shock tables, Blasius constants, standard
//!   drag curve, Bélanger equation, …).
//!
//! ## Example
//!
//! ```
//! use scirust_fluids::{dimensionless, pipe};
//!
//! // Water at 20 °C in a 50 mm pipe at 2 m/s.
//! let re = dimensionless::reynolds(998.0, 2.0, 0.05, 1.002e-3).unwrap();
//! assert!(re > 4000.0); // turbulent
//!
//! // Colebrook–White friction factor for commercial steel (ε = 45 µm).
//! let f = pipe::friction_factor(re, 45e-6 / 0.05).unwrap();
//!
//! // Pressure drop over 25 m of pipe.
//! let dp = pipe::darcy_pressure_drop(f, 25.0, 0.05, 998.0, 2.0).unwrap();
//! assert!(dp > 0.0);
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod error;

pub mod bernoulli;
pub mod boundary_layer;
pub mod channel;
pub mod compressible;
pub mod dimensionless;
pub mod external;
pub mod network;
pub mod pipe;

pub use error::FluidsError;

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------- //
    //  Cross-module consistency checks.                                //
    // ---------------------------------------------------------------- //

    #[test]
    fn pitot_and_isentropic_agree_at_low_mach() {
        // At M → 0 the compressible stagnation pressure tends to the
        // incompressible p + ρV²/2. Check at M = 0.1 (< 0.5 % apart).
        let (gamma, r, t, p) = (1.4, 287.052_874, 288.15, 101_325.0);
        let a = compressible::speed_of_sound(gamma, r, t).unwrap();
        let v = 0.1 * a;
        let rho = p / (r * t);
        let p0_incomp = bernoulli::stagnation_pressure(p, rho, v).unwrap();
        let p0_comp = p * compressible::isentropic_pressure_ratio(0.1, gamma).unwrap();
        assert!((p0_comp - p0_incomp).abs() / (p0_comp - p) < 5e-3);
    }

    #[test]
    fn friction_blends_into_moody_regimes() {
        // Laminar and turbulent branches of the dispatcher match the
        // dedicated functions.
        let f_lam = pipe::friction_factor(1500.0, 1e-4).unwrap();
        assert!((f_lam - 64.0 / 1500.0).abs() < 1e-15);
        let f_turb = pipe::friction_factor(1e6, 1e-4).unwrap();
        assert!((f_turb - pipe::friction_colebrook(1e6, 1e-4).unwrap()).abs() < 1e-15);
    }

    #[test]
    fn froude_of_critical_flow_is_one() {
        let g = 9.81;
        let yc = channel::critical_depth_rectangular(3.0, g).unwrap();
        let fr = dimensionless::froude(3.0 / yc, g, yc).unwrap();
        assert!((fr - 1.0).abs() < 1e-12);
    }
}
