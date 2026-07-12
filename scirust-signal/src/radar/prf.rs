//! Pulse-repetition-frequency ambiguities and blind speeds.
//!
//! A pulse-Doppler radar samples the scene once per pulse, at the pulse
//! repetition frequency (PRF). That sampling sets two hard limits. In range, an
//! echo arriving later than one pulse interval is mistaken for a nearby echo of
//! the *next* pulse — the **unambiguous range** `R_ua = c/(2·PRF)`. In velocity,
//! the Doppler shift is sampled at the PRF, so it aliases beyond `±PRF/2` — the
//! **unambiguous velocity** `v_ua = λ·PRF/4`. The two pull in opposite
//! directions: raising the PRF widens the velocity window but shortens the range
//! window, and their product `R_ua·v_ua = cλ/8` is fixed — the pulse-Doppler
//! dilemma. On top of that, an MTI/pulse-Doppler filter nulls targets whose
//! Doppler falls on a multiple of the PRF: the **blind speeds**
//! `v_blind = n·λ·PRF/2`. This module gives those limits and folds a true range
//! or velocity into its measured (aliased) value. Dependency-free.

/// Speed of light `c` (m/s).
const C_LIGHT: f64 = 2.997_924_58e8;

/// The **unambiguous range** `R_ua = c/(2·PRF)` (m): the maximum range whose echo
/// still returns before the next pulse. `f64::INFINITY` for a non-positive PRF.
pub fn unambiguous_range(prf: f64) -> f64 {
    if prf <= 0.0
    {
        return f64::INFINITY;
    }
    C_LIGHT / (2.0 * prf)
}

/// The **unambiguous velocity** `v_ua = λ·PRF/4` (m/s): the maximum radial speed
/// (magnitude) whose Doppler is sampled without aliasing (the full unambiguous
/// span is `±v_ua`).
pub fn unambiguous_velocity(wavelength: f64, prf: f64) -> f64 {
    wavelength * prf.max(0.0) / 4.0
}

/// The maximum unambiguous Doppler frequency `PRF/2` (Hz) — the Nyquist limit of
/// pulse-to-pulse sampling.
pub fn max_doppler(prf: f64) -> f64 {
    prf.max(0.0) / 2.0
}

/// The `n`-th **blind speed** `v_blind = n·λ·PRF/2` (m/s): a radial speed whose
/// Doppler lands on `n·PRF`, so an MTI / pulse-Doppler canceller nulls it along
/// with the stationary clutter. `n = 0` is the clutter DC itself.
pub fn blind_speed(harmonic: usize, wavelength: f64, prf: f64) -> f64 {
    harmonic as f64 * wavelength * prf.max(0.0) / 2.0
}

/// The radial velocity corresponding to a Doppler shift `doppler` (Hz) at
/// wavelength `wavelength`: `v = λ·f_d/2`.
pub fn velocity_from_doppler(doppler: f64, wavelength: f64) -> f64 {
    wavelength * doppler / 2.0
}

/// Fold a `true_range` into the **apparent range** measured under range
/// ambiguity: `R mod R_ua`. A target beyond the unambiguous range appears closer.
pub fn fold_range(true_range: f64, prf: f64) -> f64 {
    let r_ua = unambiguous_range(prf);
    if !r_ua.is_finite() || r_ua <= 0.0
    {
        return true_range;
    }
    true_range.rem_euclid(r_ua)
}

/// Fold a `true_velocity` into the **apparent (aliased) velocity** in
/// `(−v_ua, v_ua]`: a target faster than the unambiguous velocity wraps to the
/// opposite sign, the Doppler-aliasing counterpart of range folding.
pub fn fold_velocity(true_velocity: f64, wavelength: f64, prf: f64) -> f64 {
    let span = wavelength * prf.max(0.0) / 2.0; // full span = 2·v_ua
    if span <= 0.0
    {
        return true_velocity;
    }
    (true_velocity + span / 2.0).rem_euclid(span) - span / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unambiguous_range_is_inverse_prf() {
        // R_ua = c/(2·PRF): a higher PRF shortens the unambiguous range.
        assert!((unambiguous_range(1000.0) - C_LIGHT / 2000.0).abs() < 1e-6);
        assert!(unambiguous_range(2000.0) < unambiguous_range(1000.0));
        assert!(unambiguous_range(0.0).is_infinite());
    }

    #[test]
    fn range_velocity_ambiguity_product_is_invariant() {
        // The pulse-Doppler dilemma: R_ua·v_ua = cλ/8, independent of the PRF.
        let wavelength = 0.03;
        let expect = C_LIGHT * wavelength / 8.0;
        for &prf in &[500.0, 2000.0, 8000.0]
        {
            let product = unambiguous_range(prf) * unambiguous_velocity(wavelength, prf);
            assert!(
                (product - expect).abs() / expect < 1e-12,
                "prf {prf}: {product} vs {expect}"
            );
        }
    }

    #[test]
    fn blind_speeds_are_evenly_spaced_multiples() {
        let (wavelength, prf) = (0.03, 4000.0);
        assert_eq!(blind_speed(0, wavelength, prf), 0.0); // clutter DC
        let first = blind_speed(1, wavelength, prf);
        // The first blind speed is twice the unambiguous velocity (Doppler = PRF).
        assert!((first - 2.0 * unambiguous_velocity(wavelength, prf)).abs() < 1e-9);
        // Blind speeds are uniformly spaced by λ·PRF/2.
        let spacing = wavelength * prf / 2.0;
        for n in 1..5
        {
            let d = blind_speed(n + 1, wavelength, prf) - blind_speed(n, wavelength, prf);
            assert!((d - spacing).abs() < 1e-9);
        }
    }

    #[test]
    fn max_doppler_and_velocity_from_doppler() {
        assert!((max_doppler(4000.0) - 2000.0).abs() < 1e-12);
        // At the Nyquist Doppler PRF/2, the velocity is exactly v_ua.
        let (wavelength, prf) = (0.03, 4000.0);
        let v = velocity_from_doppler(max_doppler(prf), wavelength);
        assert!((v - unambiguous_velocity(wavelength, prf)).abs() < 1e-9);
    }

    #[test]
    fn range_folding_wraps_beyond_the_unambiguous_range() {
        let prf = 10_000.0;
        let r_ua = unambiguous_range(prf); // 14 989.6 m
        // Inside the window, the range is unchanged.
        assert!((fold_range(0.4 * r_ua, prf) - 0.4 * r_ua).abs() < 1e-6);
        // One window beyond wraps back to the same apparent range.
        assert!((fold_range(r_ua + 0.4 * r_ua, prf) - 0.4 * r_ua).abs() < 1e-6);
        assert!((fold_range(2.0 * r_ua + 100.0, prf) - 100.0).abs() < 1e-6);
    }

    #[test]
    fn velocity_folding_aliases_beyond_the_unambiguous_velocity() {
        let (wavelength, prf) = (0.03, 4000.0);
        let v_ua = unambiguous_velocity(wavelength, prf); // half-span
        // Inside ±v_ua, unchanged.
        assert!((fold_velocity(0.5 * v_ua, wavelength, prf) - 0.5 * v_ua).abs() < 1e-9);
        // Just past +v_ua aliases near −v_ua.
        let folded = fold_velocity(v_ua + 0.1 * v_ua, wavelength, prf);
        assert!(
            (folded - (-v_ua + 0.1 * v_ua)).abs() < 1e-9,
            "folded {folded}"
        );
        // The result always lands within (−v_ua, v_ua].
        for &v in &[3.0 * v_ua, -2.5 * v_ua, 10.0 * v_ua]
        {
            let f = fold_velocity(v, wavelength, prf);
            assert!(f > -v_ua - 1e-9 && f <= v_ua + 1e-9, "{f} outside ±{v_ua}");
        }
    }
}
