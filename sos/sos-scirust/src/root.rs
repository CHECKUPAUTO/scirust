//! [`BroydenRootSimulator`] — a real [`Simulate`] backend for nonlinear
//! systems `F(x) = 0`, wrapping `scirust-solvers`' Broyden quasi-Newton
//! solver. `L2`, with a certificate.
//!
//! ## The "own look" this backend was waiting for
//!
//! Gap #3's remaining item was flagged as *"a different fit than `Simulate`'s
//! 'observe an experiment' framing, so it needs its own look rather than a
//! mechanical repeat of this pattern"*. Having looked: it **does** fit, but
//! not for the reason the ODE and quadrature backends do, and the difference
//! has to be built into the types rather than mentioned in passing.
//!
//! For an integration or a quadrature, the configuration determines the
//! answer: `∫ₐᵇ f` is a property of `f`, `a` and `b`, and any correct solver
//! reaching it agrees. **Root-finding is not like that.** A system can have
//! many roots, and which one an iteration reaches depends on where it
//! started. `F(x) = x² − 4` has roots at `+2` and `−2`; starting at `1`
//! reaches one and starting at `−1` reaches the other, and neither is more
//! "the" root than the other.
//!
//! So the observation this backend produces is *"the root reached from this
//! starting point"* — never *"the root of F"*. [`CertifiedRoot`] therefore
//! carries [`started_from`](CertifiedRoot::started_from) alongside the root
//! itself, in the **output**, not only in the config that produced it. A
//! stored result travels without its config; if the starting point lived only
//! there, a reader would be free to mistake a basin-dependent answer for a
//! unique one, which is exactly the kind of quiet over-claim the determinism
//! levels exist to prevent. Two runs of the same system from different starts
//! are two different observations, and their content addresses say so.
//!
//! ## Why `L2`, and why the certificate is trustworthy
//!
//! Broyden iterates until the residual norm meets a declared tolerance, so
//! every step branches on a computed quantity — the textbook `L2`
//! "iterative solver to a tolerance" case, the same classification
//! [`Dopri5OdeSimulator`](crate::ode::Dopri5OdeSimulator) carries and for the
//! same reason. It is not `L3`.
//!
//! The certificate is meaningful because `scirust-solvers` returns
//! `Err(NoConvergence)` when the iteration limit is reached rather than an
//! `Ok` carrying `converged: false`. A [`CertifiedRoot`] therefore only ever
//! describes a converged solve — a non-converged attempt is a
//! [`SimError::Backend`](sos_simulation::SimError::Backend), never an
//! observation. Same reasoning as the quadrature backend's use of the
//! *strict* Simpson variant: a certificate that could be wrong is not one.
//!
//! ## Broyden rather than Newton, and what that costs
//!
//! `scirust-solvers` also offers `newton_system`, which builds an exact
//! Jacobian by dual-number autodiff and converges in fewer iterations. Its
//! model signature is `Fn(&[Dual], &mut [Dual])` — so choosing it would make
//! this the only backend in the crate whose models cannot be written in
//! ordinary Rust, and would push `scirust-autodiff` into the definition of
//! every system a caller wants to solve. Broyden takes the same plain
//! `Fn(&[f64], &mut [f64])` shape the ODE backends use, approximating the
//! Jacobian by finite differences.
//!
//! The cost is real — more iterations to the same tolerance — and it is
//! *recorded* rather than hidden: [`CertifiedRoot::iterations`] reports what
//! the solve actually spent, so a caller comparing backends can see the
//! difference instead of being told about it here.

use scirust_solvers::Tolerance;
use scirust_solvers::nonlinear::broyden::broyden;
use sos_core::DeterminismLevel;
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_simulation::{Observation, Result, SimDescriptor, Simulate};

use crate::solver::{ExactF64Seq, encode_f64, map_solver_error};

/// The configuration for one root solve: where to start, and how close is
/// close enough.
#[derive(Debug, Clone, PartialEq)]
pub struct RootConfig {
    /// The starting point. Which root the iteration reaches depends on this —
    /// see the module docs.
    pub x0: Vec<f64>,
    /// Absolute residual tolerance: the solve stops when `‖F(x)‖∞` is below
    /// it.
    pub abs_tol: f64,
    /// Relative tolerance, applied against the initial residual.
    pub rel_tol: f64,
    /// The iteration budget. Exhausting it is a backend error, not a
    /// low-quality answer.
    pub max_iter: usize,
}

impl RootConfig {
    /// Construct a root-solve configuration. Validity (finite tolerances, a
    /// non-empty `x0`) is checked by [`BroydenRootSimulator::run`], not here
    /// — a config is just data.
    #[must_use]
    pub fn new(x0: Vec<f64>, abs_tol: f64, rel_tol: f64, max_iter: usize) -> Self {
        Self {
            x0,
            abs_tol,
            rel_tol,
            max_iter,
        }
    }
}

impl Canonical for RootConfig {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.value(&ExactF64Seq(&self.x0));
        encode_f64(enc, self.abs_tol);
        encode_f64(enc, self.rel_tol);
        enc.u64(self.max_iter as u64);
    }
}

/// A converged root, with the evidence that makes it interpretable.
#[derive(Debug, Clone, PartialEq)]
pub struct CertifiedRoot {
    /// The root reached.
    pub root: Vec<f64>,
    /// The starting point it was reached **from**. Carried in the output
    /// deliberately: a root is basin-dependent, and a stored result that
    /// omitted this could be misread as the system's unique solution.
    pub started_from: Vec<f64>,
    /// The residual norm `‖F(root)‖∞` actually achieved.
    pub residual: f64,
    /// How many iterations the solve spent.
    pub iterations: usize,
    /// The absolute tolerance this result is certified to.
    pub abs_tol: f64,
    /// The relative tolerance this result is certified to.
    pub rel_tol: f64,
}

impl Canonical for CertifiedRoot {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.value(&ExactF64Seq(&self.root));
        enc.value(&ExactF64Seq(&self.started_from));
        encode_f64(enc, self.residual);
        enc.u64(self.iterations as u64);
        encode_f64(enc, self.abs_tol);
        encode_f64(enc, self.rel_tol);
    }
}

/// A real [`Simulate`] backend: solves `F(x) = 0` via `scirust-solvers`'
/// Broyden quasi-Newton iteration.
///
/// `f` is arbitrary Rust closed over at construction — the specific system
/// being solved — which is why `descriptor` is caller-supplied rather than
/// hardcoded, exactly as in [`crate::ode::Rk4OdeSimulator`]: two different
/// systems sharing this same solver need distinct descriptors to cache
/// distinctly.
pub struct BroydenRootSimulator<F> {
    descriptor: SimDescriptor,
    f: F,
}

impl<F> BroydenRootSimulator<F>
where
    F: Fn(&[f64], &mut [f64]),
{
    /// Wrap `f` (the residual function `F`) as a named, versioned solver.
    #[must_use]
    pub fn new(descriptor: SimDescriptor, f: F) -> Self {
        Self { descriptor, f }
    }
}

impl<F> Simulate for BroydenRootSimulator<F>
where
    F: Fn(&[f64], &mut [f64]),
{
    type Config = RootConfig;
    type Output = CertifiedRoot;

    fn descriptor(&self) -> SimDescriptor {
        self.descriptor.clone()
    }

    fn level(&self) -> DeterminismLevel {
        DeterminismLevel::L2
    }

    fn run(&self, config: &RootConfig, seed: u64) -> Result<Observation<CertifiedRoot>> {
        let tol = Tolerance {
            abs: config.abs_tol,
            rel: config.rel_tol,
            max_iter: config.max_iter,
        };
        let solution = broyden(&self.f, config.x0.clone(), tol).map_err(map_solver_error)?;
        let certified = CertifiedRoot {
            root: solution.value,
            started_from: config.x0.clone(),
            residual: solution.info.residual,
            iterations: solution.info.iterations,
            abs_tol: config.abs_tol,
            rel_tol: config.rel_tol,
        };
        Ok(Observation::new(certified, self.level(), seed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sos_core::SemVer;
    use sos_simulation::{SimError, Vcr};

    fn descriptor(name: &str) -> SimDescriptor {
        SimDescriptor::new(name, SemVer::new(1, 0, 0))
    }

    /// `F(x) = x² − 4`, whose roots are exactly `±2`.
    fn quadratic(x: &[f64], out: &mut [f64]) {
        out[0] = x[0] * x[0] - 4.0;
    }

    /// A 2×2 system with a known solution at `(1, 2)`:
    /// `x + y − 3 = 0`, `x² + y − 3 = 0`.
    fn system_2x2(x: &[f64], out: &mut [f64]) {
        out[0] = x[0] + x[1] - 3.0;
        out[1] = x[0] * x[0] + x[1] - 3.0;
    }

    fn solve<F: Fn(&[f64], &mut [f64])>(f: F, x0: Vec<f64>) -> CertifiedRoot {
        BroydenRootSimulator::new(descriptor("test/root"), f)
            .run(&RootConfig::new(x0, 1e-12, 1e-10, 200), 0)
            .expect("the solve must converge")
            .output
    }

    #[test]
    fn it_finds_a_real_root_of_a_scalar_system() {
        let solved = solve(quadratic, vec![1.0]);
        assert!(
            (solved.root[0] - 2.0).abs() < 1e-8,
            "expected 2, got {}",
            solved.root[0]
        );
        assert!(solved.residual <= 1e-12, "residual {}", solved.residual);
        assert!(solved.iterations > 0, "a real iteration happened");
    }

    #[test]
    fn the_root_reached_depends_on_where_it_started() {
        // The property that made this backend "need its own look": the same
        // system, solved to the same tolerance, yields a different — equally
        // correct — answer from a different start.
        let from_positive = solve(quadratic, vec![1.0]);
        let from_negative = solve(quadratic, vec![-1.0]);
        assert!((from_positive.root[0] - 2.0).abs() < 1e-8);
        assert!((from_negative.root[0] + 2.0).abs() < 1e-8);
        assert_ne!(from_positive.root, from_negative.root);
    }

    #[test]
    fn the_output_records_where_it_started_so_a_root_cannot_be_over_read() {
        // A stored result travels without its config. If the starting point
        // lived only there, this answer could be mistaken for *the* root.
        let solved = solve(quadratic, vec![-1.0]);
        assert_eq!(solved.started_from, vec![-1.0]);
        assert_ne!(
            solved.started_from, solved.root,
            "the certificate distinguishes the question from the answer"
        );
    }

    #[test]
    fn two_starts_reaching_different_roots_are_different_observations() {
        // ...and content addressing says so, so a study cannot conflate them.
        let a = solve(quadratic, vec![1.0]);
        let b = solve(quadratic, vec![-1.0]);
        assert_ne!(a.canonical_bytes(), b.canonical_bytes());
    }

    #[test]
    fn it_solves_a_two_dimensional_system() {
        let solved = solve(system_2x2, vec![0.5, 2.5]);
        assert!(
            (solved.root[0] - 1.0).abs() < 1e-6,
            "x = {}",
            solved.root[0]
        );
        assert!(
            (solved.root[1] - 2.0).abs() < 1e-6,
            "y = {}",
            solved.root[1]
        );
        assert_eq!(solved.started_from.len(), 2);
    }

    #[test]
    fn a_solve_is_l2_not_l3() {
        // Every iteration branches on a computed residual, so this is
        // tolerance-bounded, not bit-exact-by-construction.
        let sim = BroydenRootSimulator::new(descriptor("test/level"), quadratic);
        assert_eq!(sim.level(), DeterminismLevel::L2);
        let obs = sim
            .run(&RootConfig::new(vec![1.0], 1e-12, 1e-10, 200), 7)
            .unwrap();
        assert_eq!(obs.level(), DeterminismLevel::L2);
        assert_eq!(obs.seed, 7);
    }

    #[test]
    fn a_non_converging_solve_is_an_error_not_a_low_quality_answer() {
        // An iteration budget too small to reach the tolerance must not
        // produce a `CertifiedRoot` — a certificate that could be wrong is
        // not a certificate.
        let sim = BroydenRootSimulator::new(descriptor("test/stall"), quadratic);
        let err = sim
            .run(&RootConfig::new(vec![1.0], 1e-15, 1e-15, 1), 0)
            .unwrap_err();
        assert!(matches!(err, SimError::Backend(_)), "{err:?}");
    }

    #[test]
    fn a_system_with_no_root_fails_rather_than_returning_a_nearest_point() {
        // `x² + 1 = 0` has no real root. The solve must report failure, not
        // hand back whatever point it stalled at.
        fn no_real_root(x: &[f64], out: &mut [f64]) {
            out[0] = x[0] * x[0] + 1.0;
        }
        let sim = BroydenRootSimulator::new(descriptor("test/none"), no_real_root);
        assert!(
            sim.run(&RootConfig::new(vec![1.0], 1e-12, 1e-10, 50), 0)
                .is_err()
        );
    }

    #[test]
    fn is_bit_reproducible_given_the_same_config() {
        let a = solve(quadratic, vec![1.0]);
        let b = solve(quadratic, vec![1.0]);
        assert_eq!(a.canonical_bytes(), b.canonical_bytes());
    }

    #[test]
    fn canonical_encoding_reflects_every_field() {
        let base = RootConfig::new(vec![1.0], 1e-12, 1e-10, 200);
        assert_eq!(base.canonical_bytes(), base.clone().canonical_bytes());
        for other in [
            RootConfig::new(vec![-1.0], 1e-12, 1e-10, 200),
            RootConfig::new(vec![1.0], 1e-13, 1e-10, 200),
            RootConfig::new(vec![1.0], 1e-12, 1e-11, 200),
            RootConfig::new(vec![1.0], 1e-12, 1e-10, 201),
            RootConfig::new(vec![1.0, 1.0], 1e-12, 1e-10, 200),
        ]
        {
            assert_ne!(base.canonical_bytes(), other.canonical_bytes(), "{other:?}");
        }
    }

    #[test]
    fn sub_nanoscale_tolerances_do_not_collide() {
        // The cache-key collision this crate already fixed once, re-checked
        // for the field this backend adds.
        let a = RootConfig::new(vec![1.0], 1e-10, 1e-10, 200);
        let b = RootConfig::new(vec![1.0], 1e-12, 1e-10, 200);
        assert_ne!(a.canonical_bytes(), b.canonical_bytes());
    }

    #[test]
    fn the_vcr_records_then_replays_a_real_solver_run() {
        let sim = BroydenRootSimulator::new(descriptor("test/vcr"), quadratic);
        let config = RootConfig::new(vec![1.0], 1e-12, 1e-10, 200);
        let mut vcr = Vcr::new();

        let first = vcr.observe(&sim, &config, 0).unwrap();
        assert!(!first.replayed);
        let replay = vcr.observe(&sim, &config, 0).unwrap();
        assert!(replay.replayed);
        assert_eq!(replay.observation, first.observation);
        assert_eq!(vcr.len(), 1);

        // A different starting point is a fresh run, not a replay — the
        // basin matters, so the cache must not conflate them.
        let elsewhere = RootConfig::new(vec![-1.0], 1e-12, 1e-10, 200);
        let fresh = vcr.observe(&sim, &elsewhere, 0).unwrap();
        assert!(!fresh.replayed);
        assert_eq!(vcr.len(), 2);
        assert_ne!(fresh.observation.output.root, first.observation.output.root);
    }
}
