//! Doppler beam sharpening (DBS): trading Doppler resolution for azimuth resolution.
//!
//! On a moving platform, each patch of ground returns at a Doppler shift set by
//! the angle between the line of sight and the velocity vector — the **cone
//! angle** `θ`. A stationary point returns `f_d = (2v/λ)·cos θ`, so points at
//! different azimuths *within the real antenna beam* sit at different Doppler
//! frequencies. Resolving Doppler to `1/T` over a dwell `T` therefore resolves
//! azimuth far finer than the physical beam: the sensitivity is the Doppler
//! gradient `|d f_d/dθ| = (2v/λ)·|sin θ|`, which vanishes at boresight (`θ = 0`,
//! looking along the velocity — no sharpening) and peaks broadside (`θ = 90°`).
//! The resulting cross-range angular resolution is `δθ = λ/(2·v·T·|sin θ|)`, and
//! the **sharpening ratio** — the real beamwidth divided by `δθ` — measures the
//! improvement over the real beam. Dependency-free.

/// Doppler shift (Hz) of a stationary point at cone angle `cone_angle` (rad):
/// `f_d = (2·velocity/λ)·cos θ`. Maximum `2·velocity/λ` along the velocity vector
/// (`θ = 0`), zero broadside (`θ = 90°`); even in the cone angle.
pub fn azimuth_doppler(velocity: f64, wavelength: f64, cone_angle: f64) -> f64 {
    (2.0 * velocity / wavelength) * cone_angle.cos()
}

/// The **Doppler gradient** `|d f_d/dθ| = (2·velocity/λ)·|sin θ|` (Hz/rad): how
/// fast a ground point's Doppler changes with cone angle — the angular
/// sensitivity DBS exploits. Zero at boresight (`θ = 0`), maximum broadside
/// (`θ = 90°`); always non-negative.
pub fn doppler_gradient(velocity: f64, wavelength: f64, cone_angle: f64) -> f64 {
    (2.0 * velocity / wavelength) * cone_angle.sin().abs()
}

/// The DBS **cross-range angular resolution** `δθ = λ/(2·v·T·|sin θ|)` (rad): the
/// azimuth resolution won by resolving Doppler to `1/T` over a dwell `dwell` (s).
/// Returns `f64::INFINITY` at boresight (`sin θ = 0`: no Doppler spread across
/// the beam, no sharpening) and for non-positive `velocity` or `dwell`; finer
/// (smaller) with longer dwell or higher velocity, minimised broadside.
pub fn dbs_azimuth_resolution(velocity: f64, wavelength: f64, dwell: f64, cone_angle: f64) -> f64 {
    let denom = 2.0 * velocity * dwell * cone_angle.sin().abs();
    if denom <= 0.0
    {
        return f64::INFINITY;
    }
    wavelength / denom
}

/// The **sharpening ratio** `real_beamwidth / δθ`: the DBS azimuth-resolution
/// improvement over the real antenna beam (`> 1` is sharpening). Greatest
/// broadside, and `0` at boresight — where the resolution degrades to infinity —
/// or for any degenerate input that makes the resolution non-finite.
pub fn sharpening_ratio(
    real_beamwidth: f64,
    velocity: f64,
    wavelength: f64,
    dwell: f64,
    cone_angle: f64,
) -> f64 {
    let resolution = dbs_azimuth_resolution(velocity, wavelength, dwell, cone_angle);
    if !resolution.is_finite() || resolution <= 0.0
    {
        return 0.0;
    }
    real_beamwidth / resolution
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::f64::consts::FRAC_PI_2;

    #[test]
    fn azimuth_doppler_closed_form() {
        let (velocity, wavelength) = (100.0_f64, 0.03);
        let peak = 2.0 * velocity / wavelength;
        // Along the velocity vector (θ = 0): the full 2v/λ.
        assert!((azimuth_doppler(velocity, wavelength, 0.0) - peak).abs() < 1e-9);
        // Broadside (θ = 90°): a stationary point has zero Doppler.
        assert!(azimuth_doppler(velocity, wavelength, FRAC_PI_2).abs() < 1e-6);
        // Even in the cone angle (cos is even).
        assert!(
            (azimuth_doppler(velocity, wavelength, 0.7)
                - azimuth_doppler(velocity, wavelength, -0.7))
            .abs()
                < 1e-12
        );
    }

    #[test]
    fn gradient_zero_at_boresight_and_max_broadside() {
        let (velocity, wavelength) = (120.0_f64, 0.02);
        let peak = 2.0 * velocity / wavelength;
        // Boresight: no angular sensitivity.
        assert!(doppler_gradient(velocity, wavelength, 0.0).abs() < 1e-9);
        // Broadside: full sensitivity 2v/λ.
        assert!((doppler_gradient(velocity, wavelength, FRAC_PI_2) - peak).abs() < 1e-6);
        // A magnitude — never negative.
        assert!(doppler_gradient(velocity, wavelength, -1.0) >= 0.0);
    }

    #[test]
    fn doppler_and_gradient_obey_pythagoras() {
        // f_d² + (df_d/dθ)² = (2v/λ)² for every cone angle.
        let (velocity, wavelength) = (85.0_f64, 0.035);
        let amp2 = (2.0 * velocity / wavelength).powi(2);
        for &theta in &[0.1, 0.6, 1.2, 2.4, -0.9]
        {
            let f = azimuth_doppler(velocity, wavelength, theta);
            let g = doppler_gradient(velocity, wavelength, theta);
            assert!((f * f + g * g - amp2).abs() / amp2 < 1e-12, "θ = {theta}");
        }
    }

    #[test]
    fn resolution_broadside_closed_form_and_infinite_at_boresight() {
        let (velocity, wavelength, dwell) = (100.0_f64, 0.03, 0.5);
        // Broadside sin θ = 1: δθ = λ/(2·v·T).
        let expect = wavelength / (2.0 * velocity * dwell);
        assert!(
            (dbs_azimuth_resolution(velocity, wavelength, dwell, FRAC_PI_2) - expect).abs() < 1e-12
        );
        // Boresight (sin θ = 0): infinite — no sharpening.
        assert!(dbs_azimuth_resolution(velocity, wavelength, dwell, 0.0).is_infinite());
    }

    #[test]
    fn resolution_scales_with_dwell_and_velocity() {
        let (velocity, wavelength, dwell) = (100.0_f64, 0.03, 0.4);
        let base = dbs_azimuth_resolution(velocity, wavelength, dwell, 1.0);
        // Twice the dwell halves the angular resolution (finer).
        let longer = dbs_azimuth_resolution(velocity, wavelength, 2.0 * dwell, 1.0);
        assert!((longer - base / 2.0).abs() < 1e-12);
        // Twice the velocity also halves it.
        let faster = dbs_azimuth_resolution(2.0 * velocity, wavelength, dwell, 1.0);
        assert!((faster - base / 2.0).abs() < 1e-12);
    }

    #[test]
    fn sharpening_ratio_beats_beam_broadside_and_vanishes_at_boresight() {
        let (velocity, wavelength, dwell) = (150.0_f64, 0.03, 1.0);
        let real_beamwidth = 0.05_f64; // ~3° real beam
        let res = dbs_azimuth_resolution(velocity, wavelength, dwell, FRAC_PI_2);
        let ratio = sharpening_ratio(real_beamwidth, velocity, wavelength, dwell, FRAC_PI_2);
        // Ratio is exactly real_beamwidth / δθ.
        assert!((ratio - real_beamwidth / res).abs() < 1e-12);
        assert!(ratio > 1.0, "expected sharpening, got {ratio}");
        // Boresight: no sharpening at all.
        assert_eq!(
            sharpening_ratio(real_beamwidth, velocity, wavelength, dwell, 0.0),
            0.0
        );
    }

    #[test]
    fn degenerate_inputs_are_safe() {
        let (wavelength, dwell) = (0.03_f64, 0.5);
        // Zero velocity or zero dwell: no Doppler spread → infinite resolution, zero ratio.
        assert!(dbs_azimuth_resolution(0.0, wavelength, dwell, FRAC_PI_2).is_infinite());
        assert!(dbs_azimuth_resolution(100.0, wavelength, 0.0, FRAC_PI_2).is_infinite());
        assert_eq!(
            sharpening_ratio(0.05, 0.0, wavelength, dwell, FRAC_PI_2),
            0.0
        );
        assert_eq!(
            sharpening_ratio(0.05, 100.0, wavelength, 0.0, FRAC_PI_2),
            0.0
        );
    }
}
