//! Spatially flat Friedmann–Lemaître–Robertson–Walker (FLRW) spacetime.
//!
//! In comoving Cartesian coordinates `(t, x, y, z)` with signature `(-,+,+,+)`
//! the spatially flat (`k = 0`) FLRW line element is
//!
//! ```text
//! ds^2 = -dt^2 + a(t)^2 (dx^2 + dy^2 + dz^2),
//! ```
//!
//! where `a(t)` is the scale factor. The metric and Levi-Civita connection
//! depend on the cosmology only through `a` and its first derivative
//! `a_dot = da/dt`; the non-zero Christoffel symbols are
//!
//! ```text
//! Gamma^t_(i i)   = a a_dot          (i in {x, y, z})
//! Gamma^i_(t i)   = a_dot / a        (i in {x, y, z})
//! ```
//!
//! The curvature is a fixed function of `a`, `a_dot`, and `a_ddot`
//! (the Friedmann relations):
//!
//! ```text
//! R           = 6 (a_ddot/a + (a_dot/a)^2)
//! R_(t t)     = -3 a_ddot/a,   R_(i i) = a a_ddot + 2 a_dot^2
//! G_(t t)     = 3 (a_dot/a)^2,  G_(i i) = -(2 a a_ddot + a_dot^2)
//! K           = 12 ((a_ddot/a)^2 + (a_dot/a)^4)
//! ```
//!
//! which serve as exact analytic oracles for the numerical curvature engine.
//! The [`ScaleFactor`] trait supplies `a`, `a_dot`, and `a_ddot`; the metric
//! and connection use the first two, and the tests use all three to form the
//! oracle. [`ExponentialScaleFactor`] (`a = exp(H t)`) reproduces de Sitter
//! space — a maximally symmetric geometry, giving a coordinate-independence
//! cross-check against [`crate::DeSitter`].

use crate::{Connection, Metric};

/// A cosmological scale factor `a(t)` with its first two time derivatives.
///
/// The metric and connection of [`Flrw`] use [`ScaleFactor::value`] and
/// [`ScaleFactor::first_derivative`]; [`ScaleFactor::second_derivative`]
/// supplies the acceleration used to form the exact curvature oracle.
pub trait ScaleFactor {
    /// The scale factor `a(t)`.
    fn value(&self, cosmic_time: f64) -> f64;

    /// The first derivative `a_dot(t) = da/dt`.
    fn first_derivative(&self, cosmic_time: f64) -> f64;

    /// The second derivative `a_ddot(t) = d^2 a / dt^2`.
    fn second_derivative(&self, cosmic_time: f64) -> f64;
}

/// Exponential (de Sitter) scale factor `a(t) = exp(H t)` with constant Hubble
/// rate `H`.
///
/// The resulting FLRW spacetime is de Sitter space in flat (spatially flat)
/// slicing, a maximally symmetric geometry with cosmological constant
/// `Lambda = 3 H^2`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExponentialScaleFactor {
    hubble: f64,
}

impl ExponentialScaleFactor {
    /// Construct from a finite, strictly positive Hubble rate `H`.
    #[must_use]
    pub fn try_new(hubble: f64) -> Option<Self> {
        if hubble.is_finite() && hubble > 0.0
        {
            Some(Self { hubble })
        }
        else
        {
            None
        }
    }

    /// Return the Hubble rate `H`.
    #[must_use]
    pub const fn hubble(&self) -> f64 {
        self.hubble
    }

    /// Return the equivalent cosmological constant `Lambda = 3 H^2`.
    #[must_use]
    pub fn cosmological_constant(&self) -> f64 {
        3.0 * self.hubble * self.hubble
    }
}

impl ScaleFactor for ExponentialScaleFactor {
    fn value(&self, cosmic_time: f64) -> f64 {
        (self.hubble * cosmic_time).exp()
    }

    fn first_derivative(&self, cosmic_time: f64) -> f64 {
        self.hubble * (self.hubble * cosmic_time).exp()
    }

    fn second_derivative(&self, cosmic_time: f64) -> f64 {
        self.hubble * self.hubble * (self.hubble * cosmic_time).exp()
    }
}

/// Power-law scale factor `a(t) = (t / t_ref)^p` for `t > 0`, with exponent `p`
/// (for example `p = 1/2` radiation-dominated, `p = 2/3` matter-dominated).
///
/// Unlike [`ExponentialScaleFactor`], this is *not* maximally symmetric: the
/// curvature varies with cosmic time, exercising the engine on a genuinely
/// time-dependent geometry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PowerLawScaleFactor {
    exponent: f64,
    reference_time: f64,
}

impl PowerLawScaleFactor {
    /// Construct from a finite exponent `p` and a finite, strictly positive
    /// reference time `t_ref`.
    #[must_use]
    pub fn try_new(exponent: f64, reference_time: f64) -> Option<Self> {
        if exponent.is_finite() && reference_time.is_finite() && reference_time > 0.0
        {
            Some(Self {
                exponent,
                reference_time,
            })
        }
        else
        {
            None
        }
    }

    /// Return the power-law exponent `p`.
    #[must_use]
    pub const fn exponent(&self) -> f64 {
        self.exponent
    }
}

impl ScaleFactor for PowerLawScaleFactor {
    fn value(&self, cosmic_time: f64) -> f64 {
        (cosmic_time / self.reference_time).powf(self.exponent)
    }

    fn first_derivative(&self, cosmic_time: f64) -> f64 {
        self.exponent / cosmic_time * (cosmic_time / self.reference_time).powf(self.exponent)
    }

    fn second_derivative(&self, cosmic_time: f64) -> f64 {
        self.exponent * (self.exponent - 1.0) / (cosmic_time * cosmic_time)
            * (cosmic_time / self.reference_time).powf(self.exponent)
    }
}

/// Spatially flat FLRW spacetime with a given [`ScaleFactor`], in comoving
/// Cartesian coordinates `(t, x, y, z)`.
///
/// # Example
///
/// Exponential expansion is de Sitter space, whose Ricci scalar is exactly
/// `4 Lambda = 12 H^2`, recovered by the curvature engine.
///
/// ```
/// use scirust_relativity::{CurvatureTensors, ExponentialScaleFactor, Flrw};
///
/// let hubble = 0.5;
/// let background = Flrw::new(ExponentialScaleFactor::try_new(hubble).unwrap());
/// let curvature = CurvatureTensors::compute(&background, &[1.0, 0.0, 0.0, 0.0], 1.0e-4)
///     .expect("finite FLRW curvature");
///
/// assert!((curvature.ricci_scalar() - 12.0 * hubble * hubble).abs() < 1.0e-6);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Flrw<S> {
    scale_factor: S,
}

impl<S: ScaleFactor> Flrw<S> {
    /// Construct an FLRW spacetime from a scale factor.
    #[must_use]
    pub const fn new(scale_factor: S) -> Self {
        Self { scale_factor }
    }

    /// Borrow the scale factor.
    #[must_use]
    pub const fn scale_factor(&self) -> &S {
        &self.scale_factor
    }

    /// Return the Hubble parameter `H(t) = a_dot / a` at cosmic time `t`.
    #[must_use]
    pub fn hubble_parameter(&self, cosmic_time: f64) -> f64 {
        self.scale_factor.first_derivative(cosmic_time) / self.scale_factor.value(cosmic_time)
    }
}

impl<S: ScaleFactor> Metric<4> for Flrw<S> {
    fn components(&self, coordinates: &[f64; 4]) -> [[f64; 4]; 4] {
        let scale = self.scale_factor.value(coordinates[0]);
        let spatial = scale * scale;

        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, spatial, 0.0, 0.0],
            [0.0, 0.0, spatial, 0.0],
            [0.0, 0.0, 0.0, spatial],
        ]
    }
}

impl<S: ScaleFactor> Connection<4> for Flrw<S> {
    fn christoffel(&self, coordinates: &[f64; 4]) -> [[[f64; 4]; 4]; 4] {
        let cosmic_time = coordinates[0];
        let scale = self.scale_factor.value(cosmic_time);
        let rate = self.scale_factor.first_derivative(cosmic_time);
        let scale_times_rate = scale * rate;
        let hubble = rate / scale;

        let mut symbols = [[[0.0_f64; 4]; 4]; 4];

        // Spatial axes x, y, z are indices 1, 2, 3.
        for spatial in [1, 2, 3]
        {
            // Gamma^t_(i i) = a a_dot.
            symbols[0][spatial][spatial] = scale_times_rate;
            // Gamma^i_(t i) = Gamma^i_(i t) = a_dot / a.
            symbols[spatial][0][spatial] = hubble;
            symbols[spatial][spatial][0] = hubble;
        }

        symbols
    }
}
