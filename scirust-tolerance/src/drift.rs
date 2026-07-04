//! Short-term vs long-term capability: the process-drift (1.5σ) correction.
//!
//! A capability estimated from a short run captures only the **within**
//! (short-term) dispersion `σ_st`. Over weeks the mean wanders — tool wear,
//! batch-to-batch material, temperature — so the **long-term** spread is larger.
//! Modelling the mean as drifting uniformly over `±d` about its nominal adds a
//! between-time variance `Var(U[−d, d]) = d²/3`, independent of the within
//! noise, so
//!
//! ```text
//! σ_lt = √(σ_st² + d²/3) ,   I_lt = √(δ² + σ_st² + d²/3) .
//! ```
//!
//! The Motorola "6σ" convention instead books a fixed **1.5σ shift** of the
//! capability `Z`: `Z_lt = Z_st − 1.5`, equivalently `Ppk = Cpk − 0.5` (a shift
//! of `1.5` in `Z = 3·Cpk` is `0.5` in `Cpk`). Both views live here:
//! [`long_term_sigma`] / [`long_term_inertia`] from an explicit drift amplitude,
//! and [`shifted_z`] / [`cpk_to_ppk`] for the fixed-shift rule of thumb.
//!
//! The inertia `I = √(δ²+σ²)` is itself indifferent to *how* the dispersion
//! arises — drift simply enlarges `σ`; what this module adds is the bookkeeping
//! between the two horizons.

use crate::capability::nonconformity_ppm;
use crate::inertia::Inertia;

/// The classical Motorola long-term drift of the capability `Z`, `1.5` sigma.
pub const CLASSIC_SHIFT: f64 = 1.5;

/// Long-term dispersion `σ_lt = √(σ_st² + d²/3)` from the short-term (within)
/// dispersion `sigma_st` and a uniform mean-drift half-width `drift_halfwidth`
/// (`d`). With `d = 0` it returns `σ_st` unchanged.
pub fn long_term_sigma(sigma_st: f64, drift_halfwidth: f64) -> f64 {
    let d = drift_halfwidth.abs();
    (sigma_st * sigma_st + d * d / 3.0).sqrt()
}

/// Long-term inertia from a short-term [`Inertia`] and a uniform mean-drift
/// half-width `d`: the off-centering `δ` is kept and the dispersion inflated to
/// `√(σ_st² + d²/3)`. With `d = 0` it returns the input inertia.
pub fn long_term_inertia(short_term: &Inertia, drift_halfwidth: f64) -> Inertia {
    Inertia::new(
        short_term.off_centering,
        long_term_sigma(short_term.sigma, drift_halfwidth),
    )
}

/// Apply a capability-`Z` drift: `Z_lt = Z_st − shift`. The Motorola convention
/// uses `shift = `[`CLASSIC_SHIFT`] `= 1.5`.
pub fn shifted_z(z_st: f64, shift: f64) -> f64 {
    z_st - shift
}

/// Convert a short-term `Cpk` to a long-term `Ppk` under a `Z`-shift of `shift`
/// sigma: `Ppk = Cpk − shift/3` (a `shift`-sigma move of `Z = 3·Cpk`). With
/// [`CLASSIC_SHIFT`] this is the familiar `Cpk − 0.5`.
pub fn cpk_to_ppk(cpk: f64, shift: f64) -> f64 {
    cpk - shift / 3.0
}

/// Inverse of [`cpk_to_ppk`]: recover the short-term `Cpk` from a long-term
/// `Ppk`, `Cpk = Ppk + shift/3`.
pub fn ppk_to_cpk(ppk: f64, shift: f64) -> f64 {
    ppk + shift / 3.0
}

/// Long-term non-conformity in ppm for a process centred at `mean` with
/// short-term dispersion `sigma_st` and uniform mean drift `±drift_halfwidth`,
/// against `[lsl, usl]`. Uses the normal approximation with the inflated
/// long-term dispersion `σ_lt` (the standard capability treatment). Reduces to
/// [`crate::capability::nonconformity_ppm`] when the drift is 0.
pub fn long_term_ppm(mean: f64, sigma_st: f64, drift_halfwidth: f64, lsl: f64, usl: f64) -> f64 {
    nonconformity_ppm(mean, long_term_sigma(sigma_st, drift_halfwidth), lsl, usl)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::nonconformity_ppm;
    use approx::assert_relative_eq;

    #[test]
    fn zero_drift_is_identity() {
        assert_relative_eq!(long_term_sigma(0.3, 0.0), 0.3, epsilon = 1e-12);
        let i = Inertia::new(0.05, 0.3);
        let lt = long_term_inertia(&i, 0.0);
        assert_relative_eq!(lt.sigma, 0.3, epsilon = 1e-12);
        assert_relative_eq!(lt.off_centering, 0.05, epsilon = 1e-12);
    }

    #[test]
    fn drift_adds_uniform_variance() {
        // σ_st = 0.3, d = 0.6 ⇒ σ_lt² = 0.09 + 0.36/3 = 0.09 + 0.12 = 0.21.
        assert_relative_eq!(long_term_sigma(0.3, 0.6), 0.21f64.sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn classic_shift_is_half_a_cpk() {
        assert_relative_eq!(cpk_to_ppk(1.5, CLASSIC_SHIFT), 1.0, epsilon = 1e-12);
        assert_relative_eq!(ppk_to_cpk(1.0, CLASSIC_SHIFT), 1.5, epsilon = 1e-12);
        // Round-trip.
        assert_relative_eq!(
            ppk_to_cpk(cpk_to_ppk(1.33, 1.5), 1.5),
            1.33,
            epsilon = 1e-12
        );
        assert_relative_eq!(shifted_z(4.5, CLASSIC_SHIFT), 3.0, epsilon = 1e-12);
    }

    #[test]
    fn long_term_ppm_exceeds_short_term() {
        let (mean, sd, lsl, usl) = (0.0, 1.0, -4.0, 4.0);
        let st = nonconformity_ppm(mean, sd, lsl, usl);
        let lt = long_term_ppm(mean, sd, 1.5, lsl, usl);
        assert!(lt > st, "long-term {lt} should exceed short-term {st}");
        // Zero drift reduces exactly.
        assert_relative_eq!(long_term_ppm(mean, sd, 0.0, lsl, usl), st, epsilon = 1e-9);
    }
}
