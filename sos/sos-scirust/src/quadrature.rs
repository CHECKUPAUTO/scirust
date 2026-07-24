//! A real [`Simulate`] backend wrapping `scirust-solvers`' adaptive Simpson
//! quadrature — gap #3 of the `sos-scirust` scoping plan, third entry, and
//! the third demonstration of the `L2`-plus-certificate pattern
//! [`crate::ode::Dopri5OdeSimulator`] first established.
//!
//! [`QuadratureSimulator`] estimates `∫ₐᵇ f(x) dx` via
//! `scirust_solvers::quadrature::simpson_adaptive_strict` — the *strict*
//! variant, deliberately, not the plain `simpson_adaptive`: strict returns an
//! explicit error when recursion depth is exhausted before the declared
//! tolerance is met, rather than silently returning whatever estimate it last
//! computed. A certificate that could be wrong under a bad case is not a
//! certificate; using the variant that refuses to lie is what makes the `L2`
//! label honest here.
//!
//! ## Determinism, honestly
//!
//! [`DeterminismLevel::L2`], not `L3`, for the same reason as
//! [`crate::ode::Dopri5OdeSimulator`]: adaptive Simpson's subdivide-or-accept
//! decision at each interval branches on a computed error estimate against
//! the caller's `tol`, the textbook "iterative solver to a tolerance" case.
//! Unlike DOPRI5's [`crate::ode::CertifiedTrajectory`], a *successful*
//! `simpson_adaptive_strict` call is by construction guaranteed to have met
//! `tol` (or it would have returned `Err` instead) — so [`CertifiedIntegral`]
//! doesn't need separate accepted/rejected bookkeeping as evidence; recording
//! the declared `tol` and bounds alongside the value *is* the certificate.
//!
//! `seed` is threaded through and stamped on the [`Observation`] — the
//! trait's contract is uniform across backends — but this backend does not
//! consume it in the computation.
//!
//! ## Canonical config, honestly
//!
//! [`QuadratureConfig`] carries `f64` fields, encoded exactly the same way
//! [`crate::ode`]'s configs are — see [`crate::solver::encode_f64`]'s docs.

use scirust_solvers::quadrature::simpson::simpson_adaptive_strict;
use sos_core::DeterminismLevel;
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_simulation::{Observation, Result, SimDescriptor, Simulate};

use crate::solver::{encode_f64, map_solver_error};

/// The configuration for one adaptive-quadrature estimate: the integration
/// bounds `[a, b]`, the target absolute tolerance `tol`, and the maximum
/// recursion depth.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuadratureConfig {
    /// The lower integration bound.
    pub a: f64,
    /// The upper integration bound.
    pub b: f64,
    /// The target absolute error tolerance (must be finite and `> 0`).
    pub tol: f64,
    /// The maximum recursive subdivision depth (must be `> 0`).
    pub max_depth: usize,
}

impl QuadratureConfig {
    /// Construct a quadrature configuration. Validity is checked by
    /// [`QuadratureSimulator::run`], not here — a config is just data.
    #[must_use]
    pub fn new(a: f64, b: f64, tol: f64, max_depth: usize) -> Self {
        Self {
            a,
            b,
            tol,
            max_depth,
        }
    }
}

impl Canonical for QuadratureConfig {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        encode_f64(enc, self.a);
        encode_f64(enc, self.b);
        encode_f64(enc, self.tol);
        enc.u64(self.max_depth as u64);
    }
}

/// An integral estimate bundled with the tolerance certificate that bounds
/// its accuracy. Because [`QuadratureSimulator::run`] only ever returns this
/// on success, `value` is *guaranteed* accurate to within `tol` on `[a, b]` —
/// [`scirust_solvers::quadrature::simpson::simpson_adaptive_strict`] errors
/// instead of returning a non-compliant estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CertifiedIntegral {
    /// The estimated value of `∫ₐᵇ f(x) dx`.
    pub value: f64,
    /// The lower integration bound this result is certified over.
    pub a: f64,
    /// The upper integration bound this result is certified over.
    pub b: f64,
    /// The absolute tolerance this result is certified to.
    pub tol: f64,
}

/// A real [`Simulate`] backend: estimates `∫ₐᵇ f(x) dx` via
/// `scirust-solvers`' adaptive Simpson quadrature.
///
/// See [`crate::ode::Rk4OdeSimulator`] for why `descriptor` is
/// caller-supplied rather than hardcoded — the same reasoning applies here.
pub struct QuadratureSimulator<F> {
    descriptor: SimDescriptor,
    f: F,
}

impl<F> QuadratureSimulator<F>
where
    F: Fn(f64) -> f64,
{
    /// Wrap `f` (the integrand) as a named, versioned quadrature backend.
    #[must_use]
    pub fn new(descriptor: SimDescriptor, f: F) -> Self {
        Self { descriptor, f }
    }
}

impl<F> Simulate for QuadratureSimulator<F>
where
    F: Fn(f64) -> f64,
{
    type Config = QuadratureConfig;
    type Output = CertifiedIntegral;

    fn descriptor(&self) -> SimDescriptor {
        self.descriptor.clone()
    }

    fn level(&self) -> DeterminismLevel {
        DeterminismLevel::L2
    }

    fn run(&self, config: &QuadratureConfig, seed: u64) -> Result<Observation<CertifiedIntegral>> {
        let value =
            simpson_adaptive_strict(&self.f, config.a, config.b, config.tol, config.max_depth)
                .map_err(map_solver_error)?;
        let certified = CertifiedIntegral {
            value,
            a: config.a,
            b: config.b,
            tol: config.tol,
        };
        Ok(Observation::new(certified, self.level(), seed))
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use sos_core::SemVer;
    use sos_simulation::{SimError, Vcr};

    use super::*;

    fn descriptor(name: &str) -> SimDescriptor {
        SimDescriptor::new(name, SemVer::new(1, 0, 0))
    }

    #[test]
    fn integrates_sin_zero_to_pi_and_is_l2() {
        // ∫₀^π sin(x) dx = 2, exactly.
        let sim = QuadratureSimulator::new(descriptor("test/sin"), |x: f64| x.sin());
        let config = QuadratureConfig::new(0.0, PI, 1e-10, 30);
        let obs = sim.run(&config, 0).unwrap();

        assert_eq!(obs.level(), DeterminismLevel::L2);
        assert!((obs.output.value - 2.0).abs() < 1e-8);
    }

    #[test]
    fn certificate_carries_the_requested_bounds_and_tolerance() {
        let sim = QuadratureSimulator::new(descriptor("test/certificate"), |x: f64| x * x * x);
        let config = QuadratureConfig::new(0.0, 1.0, 1e-12, 10);
        let obs = sim.run(&config, 0).unwrap();
        assert_eq!(obs.output.a, 0.0);
        assert_eq!(obs.output.b, 1.0);
        assert_eq!(obs.output.tol, 1e-12);
        // ∫₀¹ x³ dx = 0.25 exactly (Simpson is exact on cubics).
        assert!((obs.output.value - 0.25).abs() < 1e-13);
    }

    #[test]
    fn is_bit_reproducible_given_the_same_config() {
        let sim = QuadratureSimulator::new(descriptor("test/repro"), |x: f64| 1.0 / (1.0 + x * x));
        let config = QuadratureConfig::new(-5.0, 5.0, 1e-10, 30);
        let a = sim.run(&config, 3).unwrap();
        let b = sim.run(&config, 3).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn zero_tolerance_is_a_clean_invalid_config_error() {
        let sim = QuadratureSimulator::new(descriptor("test/bad-tol"), |x: f64| x);
        let config = QuadratureConfig::new(0.0, 1.0, 0.0, 10);
        assert!(matches!(
            sim.run(&config, 0),
            Err(SimError::InvalidConfig(_))
        ));
    }

    #[test]
    fn depth_exhaustion_is_a_backend_error_not_a_wrong_answer() {
        // An aggressively oscillating integrand with an unreasonably tight
        // tolerance and almost no recursion budget: strict must error, not
        // silently return an inaccurate estimate.
        let sim = QuadratureSimulator::new(descriptor("test/depth-exhausted"), |x: f64| {
            (1000.0 * x).sin()
        });
        let config = QuadratureConfig::new(0.0, 1.0, 1e-30, 1);
        assert!(matches!(sim.run(&config, 0), Err(SimError::Backend(_))));
    }

    #[test]
    fn canonical_encoding_reflects_every_field() {
        let base = QuadratureConfig::new(0.0, 1.0, 1e-6, 20);
        assert_eq!(base.canonical_bytes(), base.canonical_bytes());
        assert_ne!(
            base.canonical_bytes(),
            QuadratureConfig::new(0.5, 1.0, 1e-6, 20).canonical_bytes()
        );
        assert_ne!(
            base.canonical_bytes(),
            QuadratureConfig::new(0.0, 1.0, 1e-7, 20).canonical_bytes()
        );
        assert_ne!(
            base.canonical_bytes(),
            QuadratureConfig::new(0.0, 1.0, 1e-6, 25).canonical_bytes()
        );
    }

    #[test]
    fn sub_nanoscale_tolerances_do_not_collide() {
        // Regression: nanoscale fixed-point quantization used to collapse
        // 1e-10 and 1e-12 to the same encoded value — this crate's own VCR
        // test below caught it.
        let a = QuadratureConfig::new(0.0, 1.0, 1e-10, 30);
        let b = QuadratureConfig::new(0.0, 1.0, 1e-12, 30);
        assert_ne!(a.canonical_bytes(), b.canonical_bytes());
    }

    #[test]
    fn the_vcr_records_then_replays_a_real_solver_run() {
        let sim = QuadratureSimulator::new(descriptor("test/vcr"), |x: f64| x.sin());
        let config = QuadratureConfig::new(0.0, PI, 1e-10, 30);
        let mut vcr = Vcr::new();

        let first = vcr.observe(&sim, &config, 0).unwrap();
        assert!(!first.replayed);
        let replay = vcr.observe(&sim, &config, 0).unwrap();
        assert!(replay.replayed);
        assert_eq!(replay.observation, first.observation);
        assert_eq!(vcr.len(), 1);

        // A different tolerance is a fresh run, not a replay.
        let tighter = QuadratureConfig::new(0.0, PI, 1e-12, 30);
        let fresh = vcr.observe(&sim, &tighter, 0).unwrap();
        assert!(!fresh.replayed);
        assert_eq!(vcr.len(), 2);
    }
}
