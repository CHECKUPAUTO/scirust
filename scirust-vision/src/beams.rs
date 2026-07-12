//! Ray optics and Gaussian-beam propagation — ABCD matrices and the q-parameter.
//!
//! The optical train ahead of a precision imager or laser (lenses, mirrors, free
//! space) is modelled by **ABCD ray-transfer matrices**: each element is a 2×2
//! matrix acting on a ray `(height y, slope θ)`, and a system is their product.
//! The same matrices propagate a **Gaussian beam** through the complex
//! *q-parameter* by the bilinear map `q' = (A·q + B)/(C·q + D)`, from which the
//! spot size and wavefront curvature at any plane follow. This is the core of
//! optical-train design for optronic systems (collimators, beam expanders, laser
//! cavities). Dependency-light: reuses [`scirust_signal::Complex`] for `q`.

use scirust_signal::Complex;

/// A 2×2 **ABCD ray-transfer matrix** `[[a, b], [c, d]]` acting on a paraxial
/// ray `(y, θ)` as `y' = a·y + b·θ`, `θ' = c·y + d·θ`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayMatrix {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
}

impl RayMatrix {
    /// The identity element (no change).
    pub fn identity() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
        }
    }

    /// Propagation through a distance `d` of free space (or uniform medium).
    pub fn free_space(d: f64) -> Self {
        Self {
            a: 1.0,
            b: d,
            c: 0.0,
            d: 1.0,
        }
    }

    /// A thin lens of focal length `f` (positive converging).
    pub fn thin_lens(f: f64) -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: -1.0 / f,
            d: 1.0,
        }
    }

    /// A curved mirror of radius `r` (concave `r > 0`), focal length `r/2`.
    pub fn curved_mirror(r: f64) -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: -2.0 / r,
            d: 1.0,
        }
    }

    /// Refraction at a flat interface from index `n1` into `n2`.
    pub fn flat_interface(n1: f64, n2: f64) -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: n1 / n2,
        }
    }

    /// The matrix product `self ∘ next` describing "pass through `self`, then
    /// `next`" — i.e. the combined matrix `next · self`.
    pub fn then(self, next: RayMatrix) -> RayMatrix {
        RayMatrix {
            a: next.a * self.a + next.b * self.c,
            b: next.a * self.b + next.b * self.d,
            c: next.c * self.a + next.d * self.c,
            d: next.c * self.b + next.d * self.d,
        }
    }

    /// The determinant `A·D − B·C` — equal to `n_in / n_out` for a lossless
    /// system, hence `1` when the input and output media match.
    pub fn determinant(&self) -> f64 {
        self.a * self.d - self.b * self.c
    }

    /// Transform a ray `(y, θ)` through this matrix, returning `(y', θ')`.
    pub fn apply(&self, y: f64, theta: f64) -> (f64, f64) {
        (self.a * y + self.b * theta, self.c * y + self.d * theta)
    }
}

/// The **Rayleigh range** `z_R = π·w0²/λ`: the distance from the waist over which
/// the beam area doubles.
pub fn rayleigh_range(w0: f64, lambda: f64) -> f64 {
    std::f64::consts::PI * w0 * w0 / lambda
}

/// The `1/e²` **beam radius** at distance `z` from the waist:
/// `w(z) = w0·√(1 + (z/z_R)²)`.
pub fn beam_radius(w0: f64, z: f64, lambda: f64) -> f64 {
    let zr = rayleigh_range(w0, lambda);
    w0 * (1.0 + (z / zr).powi(2)).sqrt()
}

/// The **wavefront radius of curvature** at distance `z` from the waist:
/// `R(z) = z·(1 + (z_R/z)²)`; infinite (flat) at the waist.
pub fn radius_of_curvature(w0: f64, z: f64, lambda: f64) -> f64 {
    if z == 0.0
    {
        return f64::INFINITY;
    }
    let zr = rayleigh_range(w0, lambda);
    z * (1.0 + (zr / z).powi(2))
}

/// The far-field **half-angle divergence** `θ = λ/(π·w0)`.
pub fn divergence(w0: f64, lambda: f64) -> f64 {
    lambda / (std::f64::consts::PI * w0)
}

/// The **Gouy phase** `ζ(z) = atan(z/z_R)` accumulated relative to a plane wave.
pub fn gouy_phase(w0: f64, z: f64, lambda: f64) -> f64 {
    (z / rayleigh_range(w0, lambda)).atan()
}

/// The complex beam parameter `q` at the waist: `q0 = i·z_R` (flat wavefront,
/// minimum spot).
pub fn q_at_waist(w0: f64, lambda: f64) -> Complex {
    Complex::new(0.0, rayleigh_range(w0, lambda))
}

/// Propagate a Gaussian beam parameter `q` through an ABCD `matrix`:
/// `q' = (A·q + B)/(C·q + D)`.
pub fn propagate_q(q: Complex, matrix: &RayMatrix) -> Complex {
    let num = matrix.a * q + Complex::new(matrix.b, 0.0);
    let den = matrix.c * q + Complex::new(matrix.d, 0.0);
    num / den
}

/// The `1/e²` beam radius implied by a beam parameter `q`, from
/// `Im(1/q) = −λ/(π·w²)`.
pub fn beam_radius_from_q(q: Complex, lambda: f64) -> f64 {
    let inv = Complex::new(1.0, 0.0) / q;
    (lambda / (std::f64::consts::PI * (-inv.im))).sqrt()
}

/// The wavefront radius of curvature implied by `q`, from `Re(1/q) = 1/R`.
pub fn radius_from_q(q: Complex) -> f64 {
    let inv = Complex::new(1.0, 0.0) / q;
    1.0 / inv.re
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lossless_element_matrices_have_unit_determinant() {
        assert!((RayMatrix::free_space(0.3).determinant() - 1.0).abs() < 1e-12);
        assert!((RayMatrix::thin_lens(0.1).determinant() - 1.0).abs() < 1e-12);
        assert!((RayMatrix::curved_mirror(0.5).determinant() - 1.0).abs() < 1e-12);
        assert!((RayMatrix::identity().determinant() - 1.0).abs() < 1e-12);
        // A flat interface changes the marginal-ray index ratio: det = n1/n2.
        assert!((RayMatrix::flat_interface(1.5, 1.0).determinant() - 1.5).abs() < 1e-12);
    }

    #[test]
    fn a_collimated_ray_focuses_at_the_focal_length() {
        // A ray parallel to the axis (θ = 0) crosses the axis one focal length
        // behind a thin lens.
        let f = 0.2;
        let system = RayMatrix::thin_lens(f).then(RayMatrix::free_space(f));
        let (y, theta) = system.apply(3.0, 0.0);
        assert!(y.abs() < 1e-12, "height at focus {y}");
        assert!((theta - (-3.0 / f)).abs() < 1e-12);
    }

    #[test]
    fn imaging_condition_zeroes_the_b_element() {
        // free(so) → lens(f) → free(si) images when 1/so + 1/si = 1/f, at which
        // point B = 0 and A = −si/so is the (inverted) magnification.
        let (f, so) = (10.0, 15.0);
        let si = 1.0 / (1.0 / f - 1.0 / so); // = 30
        let system = RayMatrix::free_space(so)
            .then(RayMatrix::thin_lens(f))
            .then(RayMatrix::free_space(si));
        assert!(system.b.abs() < 1e-9, "B (imaging) = {}", system.b);
        assert!(
            (system.a - (-si / so)).abs() < 1e-9,
            "magnification {}",
            system.a
        );
    }

    #[test]
    fn rayleigh_range_and_beam_geometry_match_closed_forms() {
        let (w0, lambda) = (1.0e-3, 1.0e-6);
        let zr = rayleigh_range(w0, lambda);
        assert!((zr - std::f64::consts::PI * w0 * w0 / lambda).abs() < 1e-15);
        // At the waist the radius is w0 and the wavefront is flat.
        assert!((beam_radius(w0, 0.0, lambda) - w0).abs() < 1e-15);
        assert!(radius_of_curvature(w0, 0.0, lambda).is_infinite());
        // One Rayleigh range out, the spot has grown by √2.
        assert!((beam_radius(w0, zr, lambda) - w0 * 2.0_f64.sqrt()).abs() < 1e-12);
        // Far-field divergence.
        assert!((divergence(w0, lambda) - lambda / (std::f64::consts::PI * w0)).abs() < 1e-15);
        // Gouy phase reaches π/4 at one Rayleigh range.
        assert!((gouy_phase(w0, zr, lambda) - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
    }

    #[test]
    fn q_parameter_free_space_matches_the_beam_radius_formula() {
        // Propagating the waist q through free space must reproduce w(z) and R(z).
        let (w0, lambda, z) = (0.5e-3, 0.633e-6, 0.4);
        let q0 = q_at_waist(w0, lambda);
        let q = propagate_q(q0, &RayMatrix::free_space(z));
        assert!((beam_radius_from_q(q, lambda) - beam_radius(w0, z, lambda)).abs() < 1e-9);
        assert!((radius_from_q(q) - radius_of_curvature(w0, z, lambda)).abs() < 1e-6);
        // The waist itself: q0 gives exactly w0.
        assert!((beam_radius_from_q(q0, lambda) - w0).abs() < 1e-12);
    }

    #[test]
    fn a_lens_forms_a_new_gaussian_waist_at_the_predicted_plane() {
        // A beam at its waist (collimated) hitting a thin lens of focal length f
        // forms a new waist at s' = f·z_R²/(f² + z_R²) after the lens — the
        // Gaussian result, which tends to f only as z_R ≫ f. There the wavefront
        // is flat again (Re(1/q) = 0).
        let (w0, lambda, f) = (1.0e-3, 1.0e-6, 0.15);
        let zr = rayleigh_range(w0, lambda);
        let s_prime = f * zr * zr / (f * f + zr * zr);
        let q0 = q_at_waist(w0, lambda);
        let system = RayMatrix::thin_lens(f).then(RayMatrix::free_space(s_prime));
        let q = propagate_q(q0, &system);
        let inv_re = (Complex::new(1.0, 0.0) / q).re;
        assert!(
            inv_re.abs() < 1e-9,
            "new waist not flat: Re(1/q) = {inv_re}"
        );
    }
}
