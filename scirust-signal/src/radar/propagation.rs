//! Two-ray (flat-earth) multipath and the pattern-propagation factor.
//!
//! Over a reflecting surface a radar sees each target twice: the direct ray and
//! a ray bounced off the ground. The two interfere, and the free-space field is
//! modulated by the **pattern-propagation factor** `F`. For a perfectly
//! reflecting surface (reflection coefficient `−1`) the path-length difference of
//! the direct and reflected rays is `Δ = 2·h_ant·h_target/R` (the
//! small-grazing-angle approximation), their phase difference is `φ = 2π·Δ/λ`,
//! and `F = 2·|sin(φ/2)| = 2·|sin(2π·h_ant·h_target/(λ·R))|`, oscillating between
//! `0` at the interference nulls and `2` at the lobe maxima. Received power
//! scales as `F⁴` (two-way), so the smooth `1/R⁴` law breaks into a lobed
//! pattern; beyond the last lobe — where `F ∝ 1/R` — the effective fall-off
//! steepens to `1/R⁸`. Dependency-free.

use std::f64::consts::PI;

/// The **path-length difference** `Δ = 2·h_ant·h_target/R` (m) between the
/// direct ray and the surface-reflected ray, in the small-grazing-angle
/// approximation. `0.0` for a non-positive `range`.
pub fn path_length_difference(h_ant: f64, h_target: f64, range: f64) -> f64 {
    if range <= 0.0
    {
        return 0.0;
    }
    2.0 * h_ant * h_target / range
}

/// The **phase difference** `φ = 2π·Δ/λ` (rad) between the direct and reflected
/// rays, where `Δ` is the [`path_length_difference`]. `0.0` for a non-positive
/// `range` or `wavelength`.
pub fn phase_difference(h_ant: f64, h_target: f64, range: f64, wavelength: f64) -> f64 {
    if range <= 0.0 || wavelength <= 0.0
    {
        return 0.0;
    }
    2.0 * PI * path_length_difference(h_ant, h_target, range) / wavelength
}

/// The **pattern-propagation factor** `F = 2·|sin(φ/2)|` (dimensionless), the
/// interference of the direct and reflected rays over a perfectly reflecting
/// surface. Equivalent to `2·|sin(2π·h_ant·h_target/(λ·R))|`, bounded to `[0, 2]`
/// — `0` at the nulls (`φ = 2π·k`), `2` at the maxima (`φ = π·(2k+1)`). Returns
/// `1.0` (free-space, unmodulated) for a non-positive `range` or `wavelength`.
pub fn propagation_factor(h_ant: f64, h_target: f64, range: f64, wavelength: f64) -> f64 {
    if range <= 0.0 || wavelength <= 0.0
    {
        return 1.0;
    }
    let phi = phase_difference(h_ant, h_target, range, wavelength);
    (2.0 * (phi / 2.0).sin().abs()).clamp(0.0, 2.0)
}

/// The **two-way power factor** `F⁴`, the multiplier multipath applies to the
/// free-space `1/R⁴` received power. `1.0` (free-space) for a non-positive
/// `range` or `wavelength`, since then `F = 1`.
pub fn power_factor(h_ant: f64, h_target: f64, range: f64, wavelength: f64) -> f64 {
    propagation_factor(h_ant, h_target, range, wavelength).powi(4)
}

/// The **first-null range** `R₁ = 2·h_ant·h_target/λ` (m) — the largest range at
/// which the direct and reflected rays cancel (`φ = 2π`); nulls of higher order
/// fall at `R₁/k`, and the first lobe maximum at `2·R₁` (`φ = π`). `0.0` for a
/// non-positive `wavelength`.
pub fn first_null_range(h_ant: f64, h_target: f64, wavelength: f64) -> f64 {
    if wavelength <= 0.0
    {
        return 0.0;
    }
    2.0 * h_ant * h_target / wavelength
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factor_is_bounded_and_matches_closed_form() {
        // F stays in [0, 2] over a range sweep, and equals the direct closed form
        // 2·|sin(2π·h_ant·h_target/(λ·R))|.
        let (h_ant, h_target, wavelength) = (12.0_f64, 300.0, 0.1);
        for k in 1..200
        {
            let range = 500.0 * k as f64;
            let f = propagation_factor(h_ant, h_target, range, wavelength);
            assert!((0.0..=2.0).contains(&f), "F {f} out of [0,2] at R {range}");
            let closed = 2.0
                * (2.0 * PI * h_ant * h_target / (wavelength * range))
                    .sin()
                    .abs();
            assert!((f - closed).abs() < 1e-12, "F {f} vs closed {closed}");
        }
    }

    #[test]
    fn maxima_at_odd_phase_multiples() {
        // φ = π (range = 2·R₁) is a lobe maximum: F = 2·|sin(π/2)| = 2.
        let (h_ant, h_target, wavelength) = (12.0_f64, 300.0, 0.1);
        let r1 = first_null_range(h_ant, h_target, wavelength);
        for k in 0..4
        {
            // φ = (2k+1)·π at range = 2·R₁/(2k+1).
            let range = 2.0 * r1 / (2 * k + 1) as f64;
            let f = propagation_factor(h_ant, h_target, range, wavelength);
            assert!(
                (f - 2.0).abs() < 1e-9,
                "F {f} at odd-multiple range {range}"
            );
        }
    }

    #[test]
    fn nulls_at_even_phase_multiples() {
        // φ = 2π·k (range = R₁/k) is an interference null: F ≈ 0.
        let (h_ant, h_target, wavelength) = (12.0_f64, 300.0, 0.1);
        let r1 = first_null_range(h_ant, h_target, wavelength);
        for k in 1..5
        {
            let range = r1 / k as f64;
            let f = propagation_factor(h_ant, h_target, range, wavelength);
            assert!(f < 1e-9, "F {f} should vanish at null range {range}");
        }
    }

    #[test]
    fn phase_is_two_pi_path_difference_over_wavelength() {
        // φ = 2π·Δ/λ, tying phase_difference to path_length_difference.
        let (h_ant, h_target, wavelength) = (8.0_f64, 150.0, 0.03);
        let range = 40_000.0;
        let delta = path_length_difference(h_ant, h_target, range);
        assert!((delta - 2.0 * h_ant * h_target / range).abs() < 1e-12);
        let phi = phase_difference(h_ant, h_target, range, wavelength);
        assert!((phi - 2.0 * PI * delta / wavelength).abs() < 1e-12);
    }

    #[test]
    fn first_null_range_matches_closed_form() {
        // R₁ = 2·h_ant·h_target/λ, and the phase there is exactly 2π (F ≈ 0).
        let (h_ant, h_target, wavelength) = (12.0_f64, 300.0, 0.1);
        let r1 = first_null_range(h_ant, h_target, wavelength);
        assert!((r1 - 2.0 * h_ant * h_target / wavelength).abs() < 1e-9);
        let phi = phase_difference(h_ant, h_target, r1, wavelength);
        assert!((phi - 2.0 * PI).abs() < 1e-12, "phase {phi} at R₁");
        assert!(propagation_factor(h_ant, h_target, r1, wavelength) < 1e-9);
    }

    #[test]
    fn power_factor_is_factor_to_the_fourth() {
        let (h_ant, h_target, wavelength) = (10.0_f64, 250.0, 0.05);
        for k in 1..50
        {
            let range = 1_000.0 * k as f64;
            let f = propagation_factor(h_ant, h_target, range, wavelength);
            let p = power_factor(h_ant, h_target, range, wavelength);
            assert!((p - f.powi(4)).abs() < 1e-12, "P {p} vs F⁴ {}", f.powi(4));
        }
    }

    #[test]
    fn far_field_adds_an_inverse_fourth_power_rolloff() {
        // Beyond the last lobe F ∝ 1/R, so the power factor drops ~16× per range
        // doubling — the extra 1/R⁴ that turns the free-space 1/R⁴ into 1/R⁸.
        let (h_ant, h_target, wavelength) = (1.0_f64, 1.0, 1.0);
        let r1 = first_null_range(h_ant, h_target, wavelength); // 2 m
        let range = 2_000.0 * r1; // deep in the last lobe
        let near = power_factor(h_ant, h_target, range, wavelength);
        let far = power_factor(h_ant, h_target, 2.0 * range, wavelength);
        assert!(
            (near / far - 16.0).abs() < 1e-2,
            "power ratio {}",
            near / far
        );
    }

    #[test]
    fn degenerate_inputs_return_unity() {
        // Non-positive range or wavelength ⇒ free-space (F = 1, F⁴ = 1).
        assert_eq!(propagation_factor(10.0, 100.0, 0.0, 0.1), 1.0);
        assert_eq!(propagation_factor(10.0, 100.0, -5.0, 0.1), 1.0);
        assert_eq!(propagation_factor(10.0, 100.0, 1_000.0, 0.0), 1.0);
        assert_eq!(power_factor(10.0, 100.0, 0.0, 0.1), 1.0);
        assert_eq!(first_null_range(10.0, 100.0, 0.0), 0.0);
        assert_eq!(path_length_difference(10.0, 100.0, 0.0), 0.0);
    }
}
