//! A real [`Simulate`] backend wrapping `scirust-solvers`' fixed-step RK4 ODE
//! integrator — gap #3 of the `sos-scirust` scoping plan (`sos-simulation`
//! backends), first entry.
//!
//! `sos-simulation` ships the backend-independent [`Simulate`] syscall,
//! [`Observation`]'s honest determinism stamping, and the [`Vcr`] record/replay
//! memo — but implements no solver itself, by the same Invariant VIII boundary
//! gap #1 respected for `sos-planner`. [`Rk4OdeSimulator`] is the real
//! computation: it runs `dy/dt = f(t, y)` via `scirust_solvers::ode::rk4_fixed`,
//! the "simple and robust" fixed-step method.
//!
//! ## Determinism, honestly
//!
//! Unlike gap #1's tiers 2/3, this backend is [`DeterminismLevel::L3`], not
//! `L1`: RFC-0002 §08 §1 classifies `scirust-solvers` itself as
//! **seedless-deterministic**, and the fixed-step RK4 loop bears that out
//! concretely — a fixed sequence of scalar `f64` operations with no adaptive
//! step-size branching, no iteration-count-dependent convergence check, and no
//! randomness, so identical `(f, config)` gives bit-identical output on any
//! conforming machine (the same property that makes `scirust-gp`'s Cholesky
//! path `L3` in gap #1 tier 1). `seed` is still threaded through and stamped
//! on the [`Observation`] — the trait's contract is uniform across backends —
//! but this backend does not consume it in the computation.
//!
//! An *adaptive*-step method (`scirust_solvers::ode::dopri5`) or an
//! iterative-to-a-tolerance one (Newton/Broyden root-finding) is the natural
//! `L2`-plus-certificate case SDE §08 §2 also names — that is a separate,
//! deliberately deferred backend, not this one.
//!
//! ## Canonical config, honestly
//!
//! [`OdeConfig`] carries `f64` fields, but the kernel's [`CanonicalEncoder`] is
//! deliberately float-free (`sos_core::canonical` module docs): its `encode`
//! quantizes every field to a fixed-point `i64` at a declared
//! [`FIXED_POINT_SCALE`] before hashing, exactly as those docs prescribe. This
//! affects only content-addressing (the `Vcr`/workflow cache key); the
//! integration itself always runs at full `f64` precision.
//!
//! ## Whose job cache-key disambiguation is
//!
//! [`SimDescriptor`] identifies the backend the caller names it — this module
//! does not hardcode one. Two different physical systems (a harmonic
//! oscillator vs. a predator-prey model) integrated with the same RK4 code
//! must get distinct [`SimDescriptor`]s from their caller if they are meant to
//! cache separately; the right-hand side `f` is arbitrary Rust and is not
//! itself hashed (it cannot be), so descriptor uniqueness across distinct
//! models is the caller's responsibility, the same way it already is for any
//! [`Simulate`] implementor.

use scirust_solvers::SolverError;
use scirust_solvers::ode::rk4::rk4_fixed;
use sos_core::DeterminismLevel;
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_simulation::{Observation, Result, SimDescriptor, SimError, Simulate};

/// Fixed-point scale for quantizing `f64` configuration fields into the
/// kernel's float-free canonical encoding: values are rounded to the nearest
/// `1 / FIXED_POINT_SCALE`Th before hashing (nanoscale precision — ample for
/// any physically meaningful integration bound, initial condition, or step
/// size). This affects only the content hash, never the computation.
const FIXED_POINT_SCALE: f64 = 1e9;

/// Quantize an `f64` to a fixed-point `i64` at [`FIXED_POINT_SCALE`].
fn quantize(v: f64) -> i64 {
    (v * FIXED_POINT_SCALE).round() as i64
}

/// The configuration for one RK4 integration: the interval `[t0, t_end]`, the
/// initial state `y0`, and the fixed step size `step`.
#[derive(Debug, Clone, PartialEq)]
pub struct OdeConfig {
    /// The integration start time.
    pub t0: f64,
    /// The integration end time (must be `>= t0`).
    pub t_end: f64,
    /// The initial state vector.
    pub y0: Vec<f64>,
    /// The fixed step size (must be finite and `> 0`).
    pub step: f64,
}

impl OdeConfig {
    /// Construct an ODE configuration. Validity (`t_end >= t0`, `step` finite
    /// and positive, `y0` all finite) is checked by
    /// [`Rk4OdeSimulator::run`], not here — a config is just data.
    #[must_use]
    pub fn new(t0: f64, t_end: f64, y0: Vec<f64>, step: f64) -> Self {
        Self {
            t0,
            t_end,
            y0,
            step,
        }
    }
}

impl Canonical for OdeConfig {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.i64(quantize(self.t0));
        enc.i64(quantize(self.t_end));
        enc.i64(quantize(self.step));
        let y0_quantized: Vec<i64> = self.y0.iter().map(|&v| quantize(v)).collect();
        enc.seq(&y0_quantized);
    }
}

/// The integration trajectory: `(t, y(t))` at every discretization point,
/// including the initial condition.
pub type Trajectory = Vec<(f64, Vec<f64>)>;

/// A real [`Simulate`] backend: integrates `dy/dt = f(t, y)` via
/// `scirust-solvers`' fixed-step RK4.
///
/// `f` is arbitrary Rust closed over at construction — it is the specific
/// physical model (a harmonic oscillator, a chemical rate law, ...), which is
/// why `descriptor` is supplied by the caller rather than hardcoded: two
/// different models sharing this same RK4 code need distinct descriptors to
/// cache distinctly.
pub struct Rk4OdeSimulator<F> {
    descriptor: SimDescriptor,
    f: F,
}

impl<F> Rk4OdeSimulator<F>
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    /// Wrap `f` (the ODE right-hand side) as a named, versioned RK4 backend.
    #[must_use]
    pub fn new(descriptor: SimDescriptor, f: F) -> Self {
        Self { descriptor, f }
    }
}

impl<F> Simulate for Rk4OdeSimulator<F>
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    type Config = OdeConfig;
    type Output = Trajectory;

    fn descriptor(&self) -> SimDescriptor {
        self.descriptor.clone()
    }

    fn level(&self) -> DeterminismLevel {
        DeterminismLevel::L3
    }

    fn run(&self, config: &OdeConfig, seed: u64) -> Result<Observation<Trajectory>> {
        let trajectory = rk4_fixed(
            &self.f,
            config.t0,
            config.t_end,
            config.y0.clone(),
            config.step,
        )
        .map_err(map_solver_error)?;
        Ok(Observation::new(trajectory, self.level(), seed))
    }
}

/// Map a `scirust-solvers` error to the two-variant `SimError` contract:
/// input rejected before compute began vs. a failure while running.
fn map_solver_error(e: SolverError) -> SimError {
    match e
    {
        SolverError::InvalidInput(msg) => SimError::InvalidConfig(msg),
        other => SimError::Backend(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use sos_core::SemVer;
    use sos_simulation::Vcr;

    use super::*;

    fn descriptor(name: &str) -> SimDescriptor {
        SimDescriptor::new(name, SemVer::new(1, 0, 0))
    }

    #[test]
    fn exponential_decay_matches_the_closed_form() {
        let sim = Rk4OdeSimulator::new(descriptor("test/exp-decay"), |_t, y, dy| dy[0] = -y[0]);
        let config = OdeConfig::new(0.0, 1.0, vec![1.0], 0.01);
        let obs = sim.run(&config, 0).unwrap();
        let (t_final, y_final) = obs.output.last().unwrap();
        assert!((t_final - 1.0).abs() < 1e-9);
        assert!((y_final[0] - (-1.0_f64).exp()).abs() < 1e-6);
        assert_eq!(obs.level(), DeterminismLevel::L3);
    }

    #[test]
    fn harmonic_oscillator_matches_the_closed_form() {
        let sim = Rk4OdeSimulator::new(descriptor("test/harmonic"), |_t, y, dy| {
            dy[0] = y[1];
            dy[1] = -y[0];
        });
        let config = OdeConfig::new(0.0, std::f64::consts::PI, vec![1.0, 0.0], 0.001);
        let obs = sim.run(&config, 0).unwrap();
        let (_, y_final) = obs.output.last().unwrap();
        assert!((y_final[0] - (-1.0)).abs() < 1e-6);
        assert!(y_final[1].abs() < 1e-5);
    }

    #[test]
    fn is_bit_reproducible_given_the_same_config() {
        let sim = Rk4OdeSimulator::new(descriptor("test/repro"), |_t, y, dy| dy[0] = -y[0]);
        let config = OdeConfig::new(0.0, 2.0, vec![1.0], 0.05);
        let a = sim.run(&config, 3).unwrap();
        let b = sim.run(&config, 3).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn invalid_step_is_a_clean_invalid_config_error() {
        let sim = Rk4OdeSimulator::new(descriptor("test/bad-step"), |_t, _y, dy| dy[0] = 0.0);
        let config = OdeConfig::new(0.0, 1.0, vec![1.0], 0.0);
        assert!(matches!(
            sim.run(&config, 0),
            Err(SimError::InvalidConfig(_))
        ));
    }

    #[test]
    fn nan_producing_rhs_is_a_backend_error() {
        let sim = Rk4OdeSimulator::new(descriptor("test/nan"), |_t, _y, dy| {
            dy[0] = f64::MAX / 2.0;
        });
        let config = OdeConfig::new(0.0, 1.0, vec![0.0], 1e-6);
        assert!(matches!(sim.run(&config, 0), Err(SimError::Backend(_))));
    }

    #[test]
    fn canonical_encoding_reflects_every_field() {
        let base = OdeConfig::new(0.0, 1.0, vec![1.0, 2.0], 0.1);
        assert_eq!(base.canonical_bytes(), base.clone().canonical_bytes());
        assert_ne!(
            base.canonical_bytes(),
            OdeConfig::new(0.5, 1.0, vec![1.0, 2.0], 0.1).canonical_bytes()
        );
        assert_ne!(
            base.canonical_bytes(),
            OdeConfig::new(0.0, 1.0, vec![1.0, 2.5], 0.1).canonical_bytes()
        );
        assert_ne!(
            base.canonical_bytes(),
            OdeConfig::new(0.0, 1.0, vec![1.0, 2.0], 0.2).canonical_bytes()
        );
    }

    #[test]
    fn the_vcr_records_then_replays_a_real_solver_run() {
        let sim = Rk4OdeSimulator::new(descriptor("test/vcr"), |_t, y, dy| dy[0] = -y[0]);
        let config = OdeConfig::new(0.0, 1.0, vec![1.0], 0.01);
        let mut vcr = Vcr::new();

        let first = vcr.observe(&sim, &config, 0).unwrap();
        assert!(!first.replayed);
        let replay = vcr.observe(&sim, &config, 0).unwrap();
        assert!(replay.replayed);
        assert_eq!(replay.observation, first.observation);
        assert_eq!(vcr.len(), 1);

        // A different step size is a fresh run, not a replay.
        let finer = OdeConfig::new(0.0, 1.0, vec![1.0], 0.001);
        let fresh = vcr.observe(&sim, &finer, 0).unwrap();
        assert!(!fresh.replayed);
        assert_eq!(vcr.len(), 2);
    }
}
