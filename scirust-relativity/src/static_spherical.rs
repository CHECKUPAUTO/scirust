//! Shared machinery for static, spherically symmetric spacetimes described by
//! a single lapse function `f(r)`, with metric
//! `diag(-f, 1/f, r^2, r^2 sin^2 theta)` in coordinates `(t, r, theta, phi)`
//! and signature `(-,+,+,+)`.
//!
//! The Christoffel symbols of such a metric depend on the background only
//! through `f(r)` and its radial derivative `f'(r)`, via the standard
//! formulas below. `Schwarzschild` (`f = 1 - 2M/r`) and `ReissnerNordstrom`
//! (`f = 1 - 2M/r + Q^2/r^2`) predate this helper and keep their own hand-
//! written symbols; `DeSitter`/`AntiDeSitter` (`f = 1 - Lambda r^2 / 3`) use
//! it, so no new duplication is introduced.

/// Covariant metric `diag(-f, 1/f, r^2, r^2 sin^2 theta)` of a static,
/// spherically symmetric spacetime with lapse `lapse = f(r)`.
#[must_use]
pub(crate) fn lapse_metric(lapse: f64, radius: f64, polar_angle: f64) -> [[f64; 4]; 4] {
    let radius_squared = radius * radius;
    let sine = polar_angle.sin();

    [
        [-lapse, 0.0, 0.0, 0.0],
        [0.0, 1.0 / lapse, 0.0, 0.0],
        [0.0, 0.0, radius_squared, 0.0],
        [0.0, 0.0, 0.0, radius_squared * sine * sine],
    ]
}

/// Levi-Civita Christoffel symbols `Gamma^rho_(mu nu)` (indexed
/// `[rho][mu][nu]`) of the lapse metric, from `lapse = f(r)` and
/// `lapse_derivative = f'(r)`.
///
/// The non-zero symbols are
///
/// ```text
/// Gamma^t_(t r)       =  f' / (2 f)
/// Gamma^r_(t t)       =  f f' / 2
/// Gamma^r_(r r)       = -f' / (2 f)
/// Gamma^r_(theta theta) = -r f
/// Gamma^r_(phi phi)   = -r f sin^2 theta
/// Gamma^theta_(r theta) = 1 / r
/// Gamma^theta_(phi phi) = -sin theta cos theta
/// Gamma^phi_(r phi)   = 1 / r
/// Gamma^phi_(theta phi) = cot theta
/// ```
///
/// (each symmetric symbol filled in both `mu <-> nu` orders).
#[must_use]
pub(crate) fn lapse_christoffel(
    lapse: f64,
    lapse_derivative: f64,
    radius: f64,
    polar_angle: f64,
) -> [[[f64; 4]; 4]; 4] {
    let sine = polar_angle.sin();
    let cosine = polar_angle.cos();
    let sine_squared = sine * sine;
    let inverse_radius = 1.0 / radius;
    let radial_common = lapse_derivative / (2.0 * lapse);

    let mut symbols = [[[0.0_f64; 4]; 4]; 4];

    symbols[0][0][1] = radial_common;
    symbols[0][1][0] = radial_common;

    symbols[1][0][0] = 0.5 * lapse * lapse_derivative;
    symbols[1][1][1] = -radial_common;
    symbols[1][2][2] = -radius * lapse;
    symbols[1][3][3] = -radius * lapse * sine_squared;

    symbols[2][1][2] = inverse_radius;
    symbols[2][2][1] = inverse_radius;
    symbols[2][3][3] = -sine * cosine;

    symbols[3][1][3] = inverse_radius;
    symbols[3][3][1] = inverse_radius;
    symbols[3][2][3] = cosine / sine;
    symbols[3][3][2] = cosine / sine;

    symbols
}
