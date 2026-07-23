//! [`SimDescriptor`] and the [`Simulate`] syscall — the backend-independent
//! simulation interface.

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{DeterminismLevel, SemVer};

use crate::error::SimError;
use crate::observation::Observation;

/// The stable identity of a simulation backend: its name and version. Part of a
/// run's record/replay key, so two backends — or two versions of one — never
/// share a cached observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SimDescriptor {
    /// The simulator name, e.g. `"scirust-solvers/ode-rk4"`.
    pub name: String,
    /// The simulator version.
    pub version: SemVer,
}

impl SimDescriptor {
    /// Construct a simulation descriptor.
    #[must_use]
    pub fn new(name: impl Into<String>, version: SemVer) -> Self {
        Self {
            name: name.into(),
            version,
        }
    }
}

impl Canonical for SimDescriptor {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.str(&self.name);
        enc.value(&self.version);
    }
}

/// A simulation backend — *"an experiment whose executor is a solver"* (RFC-0002
/// §08.5). This is the **syscall** the Discovery loop names instead of a concrete
/// backend, so the loop is identical whether evidence comes from a PDE solve, a
/// signal-processing pipeline, or a wet-lab instrument.
///
/// Backend-independence is real, not nominal: this trait is the interface,
/// `sos-scirust` provides the default implementations (ODE/PDE/FFT/…), and any
/// other backend — an external HPC solver, a WASM-sandboxed model — implements
/// the same trait and **declares its own** [`level`](Simulate::level). The core
/// here defines the contract and the [`Observation`] that carries the result; it
/// implements no solver (Invariant VIII) — no stub.
///
/// The contract every implementation must honor:
/// * **Determinism honesty** — [`run`](Simulate::run) must realize the level
///   [`level`](Simulate::level) declares (or weaker, recorded on the
///   `Observation`), never stronger.
/// * **Seed-reproducibility** — given the same `config` and `seed`, an `L3`
///   backend returns a bit-identical `Observation`, an `L1` backend one identical
///   in distribution.
pub trait Simulate {
    /// The configuration type — [`Canonical`] so a run is content-addressable and
    /// memoizable.
    type Config: Canonical;
    /// The output type the simulation produces.
    type Output;

    /// This backend's stable identity.
    fn descriptor(&self) -> SimDescriptor;

    /// The determinism level this backend realizes (`L3` bit-exact … `L1`
    /// seeded-stochastic). Declared by the backend, stamped on every
    /// [`Observation`], and propagated by the scheduler.
    fn level(&self) -> DeterminismLevel;

    /// Run the simulation for `config` under `seed`, returning an
    /// [`Observation`] stamped with the realized determinism level.
    ///
    /// # Errors
    /// [`SimError::InvalidConfig`] if the configuration is rejected, or
    /// [`SimError::Backend`] if the solver fails.
    fn run(&self, config: &Self::Config, seed: u64) -> Result<Observation<Self::Output>, SimError>;
}
