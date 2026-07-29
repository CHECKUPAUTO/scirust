//! [`OdeStageHandler`] — a real [`Dispatch`](sos_workflow::Dispatch) handler,
//! and with it the first end-to-end workflow execution path in SOS.
//!
//! [`Dispatch`](sos_workflow::Dispatch) (gap #2) resolves a stage's
//! [`StageDescriptor`](sos_workflow::StageDescriptor) to *some* registered
//! implementation, enforcing content-hash pinning and capability
//! authorization on the way — but it deliberately supplies no
//! implementations itself. Handlers belong to the engine crates and backend
//! adapters (Invariant VIII), and this crate is the computational one. So
//! this module is where the two halves meet: a [`Stage`] in a
//! [`Plan`](sos_workflow::Plan) now runs [`Rk4OdeSimulator`], which runs
//! `scirust-solvers`' `rk4_fixed`, and the resulting [`Observation`] becomes
//! a content-addressed object whose [`ObjectId`] the scheduler records in its
//! ledger.
//!
//! ## Two handlers, and the difference between them
//!
//! [`OdeStageHandler`] runs [`Rk4OdeSimulator`], whose model is a Rust closure
//! the handler closes over. [`CatalogStageHandler`] runs
//! [`CatalogSimulator`], whose model is
//! *data*. The mechanics are nearly identical; the guarantee is not.
//!
//! A [`Stage`]'s `config_hash` is all a [`Plan`](sos_workflow::Plan) records
//! about what a stage computes. For the ODE handler that address does not
//! determine the physics — it pins the span, the step and the initial state,
//! while the right-hand side comes from whichever closure this handler was
//! constructed with, so the same plan run against two differently-built
//! handlers computes two different things. For the catalogue handler the model
//! is inside the address, so it cannot. That is also why
//! [`CatalogStageHandler`] can answer
//! [`model_at`](CatalogStageHandler::model_at) — *what science is this plan
//! about to do?* — before running anything, and [`OdeStageHandler`] has no
//! such method to offer.
//!
//! ## Only what these demonstrate
//!
//! Two backends of the five this crate ships are wired
//! ([`crate::ode::Dopri5OdeSimulator`], [`crate::quadrature`],
//! [`crate::root`] and [`crate::spectrum`] are not), and no other engine's
//! stages have handlers. `sos-cli` is connected to neither, so `sos run` is
//! still not real. What can be claimed is precise: a plan — hand-built, or
//! resolved from a TOML study — whose stages name one of these two plugins
//! runs, memoizes, and records real results.
//!
//! ## The seams this module closes
//!
//! **A stage names its config by content address, not by value.** A [`Stage`]
//! carries `config_hash: Digest` — not the configuration itself, because the
//! scheduler is generic over every stage kind and cannot hold their types.
//! Something must map that address back to a typed [`OdeConfig`], and it must
//! not be able to lie about which config an address denotes.
//! [`OdeStageHandler::offer`] takes that responsibility: it *computes* the
//! address from the config's canonical encoding and returns it, so a config
//! is only ever registered under its own true content address. A stage whose
//! `config_hash` matches nothing on offer fails
//! ([`WorkflowError::StageFailed`]) rather than running with a substituted
//! configuration.
//!
//! **A stage returns object ids, not values.** [`StageExecutor::run`] yields
//! `Vec<ObjectId>`, so the [`Observation`] has to be sealed into the object
//! store to have an id at all. That seals in the honest metadata too: the
//! stored object is stamped with the [`DeterminismLevel`] the backend
//! *realized*, its provenance parents are the stage's own inputs, and its
//! [`ReproMeta`] records the stage's seed alongside `scirust-solvers` as the
//! backend that produced it.
//!
//! ## Why the trajectory is serialized as decimal strings
//!
//! A content-addressed object must survive a store round-trip with its id
//! intact — that is the whole basis of `put`-then-`get` integrity checking,
//! and of memoization. Serializing the trajectory's `f64`s as JSON numbers
//! broke exactly that: over a 513-step run of this module's own oscillator
//! test, **259 of 1539 floats came back from `serde_json` with different bits
//! than they went in with** (e.g. `0.009203884727313847` returning as
//! `0.009203884727313849`), so the reloaded object hashed to a different id
//! and [`TypedStore::get_object`] rejected it as corrupt. The store was
//! right; the body was wrong.
//!
//! [`exact_trajectory`] fixes it at the representation: every `f64` is stored
//! as its shortest round-trip decimal string and read back with `str::parse`,
//! which *is* exactly round-tripping (0 of the same 1539 floats lose a bit).
//! That also makes the serialized form and the canonical (hashed) form the
//! same decimal text, so there is no second encoding that could drift from
//! the first — the same reasoning [`OdeConfig`]'s canonical encoding already
//! applies to configuration floats.
//!
//! ## Why the environment record is deliberately unspecified
//!
//! A stage's output id must not depend on the host, or the same plan would
//! memoize differently on two machines and the cache would be worthless
//! across a team. Environment binding already happens one layer up:
//! [`run_plan`](sos_workflow::run_plan) takes an environment [`Digest`] and
//! folds it into every [`CacheKey`](sos_workflow::CacheKey). So the
//! [`EnvRecord`] here names the backend it really used and leaves
//! toolchain/hardware/OS explicitly `"unspecified"` — an accurate statement
//! that this object's identity is host-independent, not a placeholder that
//! someone forgot to fill in.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{
    BackendVersion, Body, DeterminismLevel, Digest, EnvRecord, HashAlgo, Object, ObjectId,
    ProducerRef, ReproMeta, RngId, SemVer,
};
use sos_simulation::{Observation, Simulate};
use sos_store::{ObjectStore, TypedStore};
use sos_workflow::{Stage, StageExecutor, WorkflowError};

use crate::model::{CatalogSimulator, ModelRun, ModelSpec, ModeledTrajectoryBody};
use crate::ode::{OdeConfig, Rk4OdeSimulator, Trajectory};
use crate::solver::{ExactF64Seq, encode_f64};

/// Domain-separation prefix for an ODE stage configuration's content address.
/// Distinct from every object kind's prefix, so a config address can never
/// collide with an [`ObjectId`].
const CONFIG_DOMAIN: &[u8] = b"sos-scirust:ode-stage-config:v1";

/// The content address of `config` — the value a [`Stage`] must carry in its
/// `config_hash` for [`OdeStageHandler`] to run it.
///
/// Derived from the config's canonical encoding, so two configs that differ
/// in any field an integration depends on address differently (and two that
/// are equal address identically, which is what makes memoization correct).
#[must_use]
pub fn config_address(config: &OdeConfig) -> Digest {
    HashAlgo::Sha256.hash(CONFIG_DOMAIN, &config.canonical_bytes())
}

/// `serde` support storing a [`Trajectory`]'s `f64`s as exact shortest
/// round-trip decimal strings rather than JSON numbers.
///
/// See the module docs: JSON-number `f64` round-trips are *not* bit-exact
/// here, which breaks the content address of any object holding them.
/// `f64::to_string` / `str::parse` is exact, and is the same text the
/// canonical encoding hashes.
pub mod exact_trajectory {
    use serde::de::Error as _;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use crate::ode::Trajectory;

    /// Serialize as `[(t, [y...]), ...]` with every float a decimal string.
    ///
    /// # Errors
    /// Propagates the serializer's own errors.
    pub fn serialize<S: Serializer>(traj: &Trajectory, s: S) -> Result<S::Ok, S::Error> {
        let text: Vec<(String, Vec<String>)> = traj
            .iter()
            .map(|(t, y)| {
                (
                    t.to_string(),
                    y.iter().map(std::string::ToString::to_string).collect(),
                )
            })
            .collect();
        text.serialize(s)
    }

    /// Read back the decimal-string form.
    ///
    /// # Errors
    /// Fails if any element is not a parsable `f64` — a corrupted record,
    /// surfaced rather than silently coerced.
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Trajectory, D::Error> {
        let text: Vec<(String, Vec<String>)> = Vec::deserialize(d)?;
        text.into_iter()
            .map(|(t, y)| {
                let t = t.parse::<f64>().map_err(D::Error::custom)?;
                let y = y
                    .into_iter()
                    .map(|v| v.parse::<f64>().map_err(D::Error::custom))
                    .collect::<Result<Vec<f64>, _>>()?;
                Ok((t, y))
            })
            .collect()
    }
}

/// The body of a stored ODE result: the trajectory, the determinism level the
/// backend realized, and the seed it ran under — an [`Observation`] flattened
/// into a first-class SOS object kind, so a trajectory produced by a workflow
/// stage is verifiable, referenceable and garbage-collectable on the same
/// terms as every other scientific object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryBody {
    /// The `(t, y(t))` samples, stored as exact decimal text (see
    /// [`exact_trajectory`]).
    #[serde(with = "exact_trajectory")]
    pub trajectory: Trajectory,
    /// The determinism level the backend realized for this run.
    pub level: DeterminismLevel,
    /// The seed the run used.
    pub seed: u64,
}

impl TrajectoryBody {
    /// Flatten an observation into a storable body.
    #[must_use]
    pub fn from_observation(observation: Observation<Trajectory>) -> Self {
        Self {
            trajectory: observation.output,
            level: observation.level,
            seed: observation.seed,
        }
    }

    /// Rebuild the [`Observation`] this body was stored from.
    #[must_use]
    pub fn observation(&self) -> Observation<Trajectory> {
        Observation::new(self.trajectory.clone(), self.level, self.seed)
    }
}

/// One `(t, y(t))` sample, canonically encoded with the same exact
/// shortest-round-trip float encoding [`OdeConfig`] uses — the kernel's
/// canonical encoding is deliberately float-free, so [`Trajectory`]
/// (`Vec<(f64, Vec<f64>)>`) has no [`Canonical`] impl of its own and must be
/// encoded element-wise here.
struct TrajectorySample<'a> {
    t: f64,
    y: &'a [f64],
}

impl Canonical for TrajectorySample<'_> {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        encode_f64(enc, self.t);
        enc.value(&ExactF64Seq(self.y));
    }
}

impl Canonical for TrajectoryBody {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        let samples: Vec<TrajectorySample<'_>> = self
            .trajectory
            .iter()
            .map(|(t, y)| TrajectorySample { t: *t, y })
            .collect();
        enc.seq(&samples);
        enc.value(&self.level);
        enc.u64(self.seed);
    }
}

impl Body for TrajectoryBody {
    const KIND: &'static str = "OdeTrajectory";
    const SCHEMA_VERSION: u32 = 1;
}

/// A [`StageExecutor`] that runs workflow stages on [`Rk4OdeSimulator`].
///
/// Register it with a [`Dispatch`](sos_workflow::Dispatch) under the plugin
/// name its stages' descriptors use; [`offer`](Self::offer) every
/// configuration those stages may select, and build each [`Stage`] with the
/// address `offer` returned.
///
/// The handler shares its object store with the caller through
/// `Rc<RefCell<_>>`: [`StageExecutor::run`] returns only ids, so the caller
/// needs the same store to read the results back, and the scheduler drives
/// stages sequentially, so no cross-thread sharing is involved.
pub struct OdeStageHandler<F, S> {
    simulator: Rk4OdeSimulator<F>,
    store: Rc<RefCell<S>>,
    configs: BTreeMap<Digest, OdeConfig>,
    backend: BackendVersion,
}

impl<F, S> OdeStageHandler<F, S>
where
    F: Fn(f64, &[f64], &mut [f64]),
    S: ObjectStore,
{
    /// A handler running `simulator`, sealing its results into `store`.
    ///
    /// `backend` pins the exact solver build the results came from; it is
    /// recorded in every stored object's [`ReproMeta`], which is the whole
    /// point of [`BackendVersion`] — a result that cannot name the code that
    /// produced it is not reproducible.
    #[must_use]
    pub fn new(
        simulator: Rk4OdeSimulator<F>,
        store: Rc<RefCell<S>>,
        backend: BackendVersion,
    ) -> Self {
        Self {
            simulator,
            store,
            configs: BTreeMap::new(),
            backend,
        }
    }

    /// Make `config` available to stages, returning the content address that
    /// selects it.
    ///
    /// The address is *computed* from the configuration, never supplied, so a
    /// config cannot be registered under an address it does not have. Build
    /// the stage with the returned [`Digest`] as its `config_hash`.
    pub fn offer(&mut self, config: OdeConfig) -> Digest {
        let address = config_address(&config);
        self.configs.insert(address, config);
        address
    }

    /// The content addresses currently on offer, sorted.
    #[must_use]
    pub fn offered(&self) -> Vec<Digest> {
        self.configs.keys().copied().collect()
    }

    /// The environment record stamped on this handler's outputs. See the
    /// module docs on why toolchain/hardware/OS are `"unspecified"`.
    fn env(&self) -> EnvRecord {
        EnvRecord::new(
            "unspecified",
            vec![self.backend.clone()],
            "unspecified",
            "unspecified",
        )
    }

    /// The producer reference identifying the simulator that ran the stage.
    fn producer(&self) -> ProducerRef {
        let descriptor = self.simulator.descriptor();
        ProducerRef::new(
            descriptor.name.clone(),
            descriptor.version,
            HashAlgo::Sha256.hash(b"sos-producer", &descriptor.canonical_bytes()),
        )
    }
}

impl<F, S> StageExecutor for OdeStageHandler<F, S>
where
    F: Fn(f64, &[f64], &mut [f64]),
    S: ObjectStore,
{
    fn run(&mut self, stage: &Stage) -> Result<Vec<ObjectId>, WorkflowError> {
        // Resolve the configuration by content address. A miss is a failed
        // stage, never a default or a nearest match — running a different
        // configuration than the one the stage pinned is exactly the silent
        // substitution `Dispatch` refuses at the plugin level.
        let config =
            self.configs
                .get(&stage.config_hash)
                .ok_or_else(|| WorkflowError::StageFailed {
                    stage: stage.id.clone(),
                    reason: format!(
                        "no ODE configuration is on offer at content address {}",
                        stage.config_hash
                    ),
                })?;

        let observation =
            self.simulator
                .run(config, stage.seed)
                .map_err(|e| WorkflowError::StageFailed {
                    stage: stage.id.clone(),
                    reason: e.to_string(),
                })?;

        // Stamp the level the backend *realized*, not one assumed here.
        let level = observation.level();
        let repro = ReproMeta::new(
            stage.seed,
            RngId::new("none"),
            self.env().digest(HashAlgo::Sha256),
        )
        .with_inputs(stage.inputs.clone());
        let object = Object::builder(TrajectoryBody::from_observation(observation))
            .level(level)
            .producer(self.producer())
            .parents(stage.inputs.clone())
            .repro(repro)
            .seal();

        let id = self.store.borrow_mut().put_object(&object).map_err(|e| {
            WorkflowError::StageFailed {
                stage: stage.id.clone(),
                reason: e.to_string(),
            }
        })?;
        Ok(vec![id])
    }
}

/// The determinism level every [`OdeStageHandler`] output carries, restated
/// here only so a caller can assert it without running a stage:
/// [`Rk4OdeSimulator`] is `L3`, seedless-deterministic.
pub const RK4_STAGE_LEVEL: DeterminismLevel = DeterminismLevel::L3;

/// A [`BackendVersion`] naming `scirust-solvers` at `version`, with
/// `content_hash` pinning the build. Convenience for the common case; any
/// [`BackendVersion`] works.
#[must_use]
pub fn solvers_backend(version: SemVer, content_hash: Digest) -> BackendVersion {
    BackendVersion::new("scirust-solvers", version, content_hash)
}

/// The same, naming `scirust-sim` — the backend behind
/// [`CatalogStageHandler`].
#[must_use]
pub fn sim_backend(version: SemVer, content_hash: Digest) -> BackendVersion {
    BackendVersion::new("scirust-sim", version, content_hash)
}

/// Domain-separation prefix for a catalogue run's content address. Distinct
/// from [`CONFIG_DOMAIN`] so an ODE config and a model run can never collide,
/// even if their encodings ever coincided.
const MODEL_CONFIG_DOMAIN: &[u8] = b"sos-scirust:model-stage-config:v1";

/// The content address of `run` — the value a [`Stage`] must carry in its
/// `config_hash` for [`CatalogStageHandler`] to run it.
///
/// Unlike [`config_address`], this address covers the **model** as well as the
/// integration parameters, which is what makes a plan referencing it
/// well-determined.
#[must_use]
pub fn model_config_address(run: &ModelRun) -> Digest {
    HashAlgo::Sha256.hash(MODEL_CONFIG_DOMAIN, &run.canonical_bytes())
}

/// The determinism level every [`CatalogStageHandler`] output carries:
/// [`CatalogSimulator`] is `L3`, seedless-deterministic.
pub const CATALOG_STAGE_LEVEL: DeterminismLevel = DeterminismLevel::L3;

/// A [`StageExecutor`] that runs workflow stages on
/// [`CatalogSimulator`] — `scirust-sim`'s catalogued models.
///
/// Note what is *absent* compared to [`OdeStageHandler`]: no type parameter
/// for a right-hand side, because there is no closure to carry. Every
/// difference between two runs lives in their [`ModelRun`]s, and therefore in
/// their content addresses.
pub struct CatalogStageHandler<S> {
    simulator: CatalogSimulator,
    store: Rc<RefCell<S>>,
    runs: BTreeMap<Digest, ModelRun>,
    backend: BackendVersion,
}

impl<S: ObjectStore> CatalogStageHandler<S> {
    /// A handler sealing its results into `store`, recording `backend` in
    /// every stored object's [`ReproMeta`].
    #[must_use]
    pub fn new(store: Rc<RefCell<S>>, backend: BackendVersion) -> Self {
        Self {
            simulator: CatalogSimulator::new(),
            store,
            runs: BTreeMap::new(),
            backend,
        }
    }

    /// Make `run` available to stages, returning the content address that
    /// selects it. The address is computed, never supplied.
    pub fn offer(&mut self, run: ModelRun) -> Digest {
        let address = model_config_address(&run);
        self.runs.insert(address, run);
        address
    }

    /// The content addresses currently on offer, sorted.
    #[must_use]
    pub fn offered(&self) -> Vec<Digest> {
        self.runs.keys().copied().collect()
    }

    /// Which model the run at `address` will integrate — answerable *before*
    /// the stage runs.
    ///
    /// This is the audit that model identity buys and that
    /// [`OdeStageHandler`] cannot offer: given a [`Plan`](sos_workflow::Plan),
    /// a reviewer can ask what science it is about to do rather than reading
    /// the source of whatever closure was compiled in.
    #[must_use]
    pub fn model_at(&self, address: &Digest) -> Option<&ModelSpec> {
        self.runs.get(address).map(|run| &run.model)
    }

    /// See [`OdeStageHandler::env`] — same reasoning, different backend.
    fn env(&self) -> EnvRecord {
        EnvRecord::new(
            "unspecified",
            vec![self.backend.clone()],
            "unspecified",
            "unspecified",
        )
    }

    fn producer(&self) -> ProducerRef {
        let descriptor = self.simulator.descriptor();
        ProducerRef::new(
            descriptor.name.clone(),
            descriptor.version,
            HashAlgo::Sha256.hash(b"sos-producer", &descriptor.canonical_bytes()),
        )
    }
}

impl<S: ObjectStore> StageExecutor for CatalogStageHandler<S> {
    fn run(&mut self, stage: &Stage) -> Result<Vec<ObjectId>, WorkflowError> {
        let run = self
            .runs
            .get(&stage.config_hash)
            .ok_or_else(|| WorkflowError::StageFailed {
                stage: stage.id.clone(),
                reason: format!(
                    "no model run is on offer at content address {}",
                    stage.config_hash
                ),
            })?;

        let observation =
            self.simulator
                .run(run, stage.seed)
                .map_err(|e| WorkflowError::StageFailed {
                    stage: stage.id.clone(),
                    reason: e.to_string(),
                })?;

        let level = observation.level();
        let repro = ReproMeta::new(
            stage.seed,
            RngId::new("none"),
            self.env().digest(HashAlgo::Sha256),
        )
        .with_inputs(stage.inputs.clone());
        let object = Object::builder(ModeledTrajectoryBody::from_observation(observation))
            .level(level)
            .producer(self.producer())
            .parents(stage.inputs.clone())
            .repro(repro)
            .seal();

        let id = self.store.borrow_mut().put_object(&object).map_err(|e| {
            WorkflowError::StageFailed {
                stage: stage.id.clone(),
                reason: e.to_string(),
            }
        })?;
        Ok(vec![id])
    }
}

#[cfg(test)]
mod tests {
    use sos_simulation::SimDescriptor;
    use sos_store::MemoryStore;
    use sos_workflow::{StageDescriptor, StageId};

    use super::*;

    /// `dy/dt = [v, -x]` — the harmonic oscillator, whose exact solution
    /// `x(t) = cos t`, `v(t) = -sin t` lets a test check that a *real*
    /// integration happened.
    fn oscillator(_t: f64, y: &[f64], dy: &mut [f64]) {
        dy[0] = y[1];
        dy[1] = -y[0];
    }

    /// The right-hand side as a plain `fn` pointer, so the handler's type is
    /// nameable in the test helpers below.
    type Rhs = fn(f64, &[f64], &mut [f64]);

    fn handler() -> OdeStageHandler<Rhs, MemoryStore> {
        let simulator = Rk4OdeSimulator::new(
            SimDescriptor::new("scirust-solvers/ode-rk4", SemVer::new(1, 0, 0)),
            oscillator as fn(f64, &[f64], &mut [f64]),
        );
        let backend = solvers_backend(
            SemVer::new(0, 1, 0),
            HashAlgo::Sha256.hash(b"test-backend", b"scirust-solvers"),
        );
        OdeStageHandler::new(
            simulator,
            Rc::new(RefCell::new(MemoryStore::new())),
            backend,
        )
    }

    fn stage_at(config_hash: Digest) -> Stage {
        let d = HashAlgo::Sha256.hash(b"plugin", b"ode");
        Stage::new(
            StageId::new("integrate"),
            StageDescriptor::new("ode", SemVer::new(1, 0, 0), d),
            vec![],
            config_hash,
            0,
            vec![],
        )
    }

    /// A quarter period of the oscillator, finely stepped — 513 samples of
    /// thoroughly irrational floats, which is what makes it a real test of
    /// the storage round-trip.
    fn quarter_period_config() -> OdeConfig {
        let quarter = std::f64::consts::FRAC_PI_2;
        OdeConfig::new(0.0, quarter, vec![1.0, 0.0], quarter / 512.0)
    }

    #[test]
    fn offer_returns_the_configs_own_content_address() {
        let mut h = handler();
        let config = OdeConfig::new(0.0, 1.0, vec![1.0, 0.0], 0.01);
        let address = h.offer(config.clone());
        assert_eq!(address, config_address(&config));
        assert_eq!(h.offered(), vec![address]);
    }

    #[test]
    fn equal_configs_share_an_address_and_differing_ones_do_not() {
        let a = OdeConfig::new(0.0, 1.0, vec![1.0, 0.0], 0.01);
        let same = OdeConfig::new(0.0, 1.0, vec![1.0, 0.0], 0.01);
        let finer = OdeConfig::new(0.0, 1.0, vec![1.0, 0.0], 0.005);
        assert_eq!(config_address(&a), config_address(&same));
        assert_ne!(config_address(&a), config_address(&finer));
    }

    #[test]
    fn running_a_stage_integrates_and_stores_the_trajectory() {
        let mut h = handler();
        let store = Rc::clone(&h.store);
        let quarter = std::f64::consts::FRAC_PI_2;
        let address = h.offer(quarter_period_config());

        let ids = h.run(&stage_at(address)).unwrap();
        assert_eq!(ids.len(), 1);

        let stored: Object<TrajectoryBody> = store
            .borrow()
            .get_object(ids[0])
            .unwrap()
            .expect("the stage's output must be in the store");
        let (t_end, y_end) = stored.body.trajectory.last().unwrap().clone();
        assert!((t_end - quarter).abs() < 1e-9);
        // x(pi/2) = 0, v(pi/2) = -1. RK4 at this step size is far more
        // accurate than 1e-6; a stub or a wrong right-hand side could not
        // land here.
        assert!(
            y_end[0].abs() < 1e-6,
            "x(pi/2) should be 0, got {}",
            y_end[0]
        );
        assert!(
            (y_end[1] + 1.0).abs() < 1e-6,
            "v(pi/2) should be -1, got {}",
            y_end[1]
        );
    }

    #[test]
    fn a_stored_trajectory_reloads_bit_for_bit() {
        // The regression behind `exact_trajectory`: serializing these floats
        // as JSON *numbers* returned different bits for 259 of the 1539
        // values below, so the reloaded object hashed to a different id and
        // the store rejected it as corrupt. Every sample must return
        // bit-identical, not merely close.
        let mut h = handler();
        let store = Rc::clone(&h.store);
        let address = h.offer(quarter_period_config());
        let ids = h.run(&stage_at(address)).unwrap();

        let before = h.simulator.run(&quarter_period_config(), 0).unwrap().output;
        let stored: Object<TrajectoryBody> = store.borrow().get_object(ids[0]).unwrap().unwrap();
        let after = &stored.body.trajectory;

        assert_eq!(before.len(), after.len());
        assert!(before.len() > 500, "needs enough samples to be meaningful");
        for (i, ((t0, y0), (t1, y1))) in before.iter().zip(after).enumerate()
        {
            assert_eq!(t0.to_bits(), t1.to_bits(), "t differs at sample {i}");
            for (k, (a, b)) in y0.iter().zip(y1).enumerate()
            {
                assert_eq!(a.to_bits(), b.to_bits(), "y[{k}] differs at sample {i}");
            }
        }
    }

    #[test]
    fn the_stored_object_carries_the_realized_level_and_the_stages_seed() {
        let mut h = handler();
        let store = Rc::clone(&h.store);
        let address = h.offer(OdeConfig::new(0.0, 0.5, vec![1.0, 0.0], 0.01));
        let mut stage = stage_at(address);
        stage.seed = 4242;

        let ids = h.run(&stage).unwrap();
        let stored: Object<TrajectoryBody> = store.borrow().get_object(ids[0]).unwrap().unwrap();
        assert_eq!(stored.level, RK4_STAGE_LEVEL);
        assert_eq!(stored.body.level, RK4_STAGE_LEVEL);
        assert_eq!(stored.body.seed, 4242);
        assert_eq!(stored.body.observation().seed, 4242);
        assert_eq!(stored.repro.seed, 4242);
        // The backend that really produced it is named, not left blank.
        assert_eq!(stored.producer.name, "scirust-solvers/ode-rk4");
        assert_eq!(stored.repro.rng, RngId::new("none"));
    }

    #[test]
    fn an_unoffered_config_address_fails_the_stage() {
        let mut h = handler();
        let never_offered = config_address(&OdeConfig::new(0.0, 1.0, vec![1.0], 0.1));
        let err = h.run(&stage_at(never_offered)).unwrap_err();
        assert!(matches!(err, WorkflowError::StageFailed { .. }), "{err:?}");
        assert!(err.to_string().contains("no ODE configuration is on offer"));
    }

    #[test]
    fn an_invalid_config_fails_the_stage_rather_than_producing_an_object() {
        let mut h = handler();
        let store = Rc::clone(&h.store);
        // A non-positive step is rejected by the backend, not by `offer` — a
        // config is just data until something integrates with it.
        let address = h.offer(OdeConfig::new(0.0, 1.0, vec![1.0, 0.0], 0.0));
        assert!(matches!(
            h.run(&stage_at(address)),
            Err(WorkflowError::StageFailed { .. })
        ));
        assert!(
            store.borrow().object_ids().is_empty(),
            "a failed stage must store nothing"
        );
    }

    #[test]
    fn the_same_stage_run_twice_produces_the_same_object_id() {
        let mut h = handler();
        let address = h.offer(quarter_period_config());
        let first = h.run(&stage_at(address)).unwrap();
        let second = h.run(&stage_at(address)).unwrap();
        assert_eq!(first, second, "an L3 backend must be bit-reproducible");
    }

    #[test]
    fn a_different_seed_gives_a_different_object_even_at_l3() {
        // RK4 ignores the seed numerically, but the observation records it,
        // so the objects must still be distinct — a result must not claim to
        // have been produced under a seed it was not.
        let mut h = handler();
        let address = h.offer(OdeConfig::new(0.0, 0.5, vec![1.0, 0.0], 0.01));
        let a = h.run(&stage_at(address)).unwrap();
        let mut other = stage_at(address);
        other.seed = 99;
        let b = h.run(&other).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn a_stages_inputs_become_the_outputs_provenance_parents() {
        let mut h = handler();
        let store = Rc::clone(&h.store);
        let input = ObjectId::compute(HashAlgo::Sha256, b"test", b"upstream");
        let address = h.offer(OdeConfig::new(0.0, 0.5, vec![1.0, 0.0], 0.01));
        let mut stage = stage_at(address);
        stage.inputs = vec![input];

        let ids = h.run(&stage).unwrap();
        let stored: Object<TrajectoryBody> = store.borrow().get_object(ids[0]).unwrap().unwrap();
        assert_eq!(stored.parents, vec![input]);
        assert_eq!(stored.repro.inputs, vec![input]);
    }

    #[test]
    fn floats_are_serialized_as_exact_decimal_text_not_json_numbers() {
        // Pins the representation the round-trip guarantee rests on. If this
        // ever reverts to JSON numbers, the bit-for-bit test above fails too
        // — this test says *why*.
        let body = TrajectoryBody {
            trajectory: vec![(0.009_203_884_727_313_847, vec![0.5, -0.25])],
            level: DeterminismLevel::L3,
            seed: 0,
        };
        let json = serde_json::to_string(&body).unwrap();
        assert!(
            json.contains("\"0.009203884727313847\""),
            "floats must be quoted decimal text, got {json}"
        );
        let back: TrajectoryBody = serde_json::from_str(&json).unwrap();
        assert_eq!(
            back.trajectory[0].0.to_bits(),
            body.trajectory[0].0.to_bits()
        );
    }

    #[test]
    fn a_corrupted_float_field_is_an_error_not_a_silent_zero() {
        let json = r#"{"trajectory":[["not-a-number",["1.0"]]],"level":"L3","seed":0}"#;
        assert!(serde_json::from_str::<TrajectoryBody>(json).is_err());
    }

    // ---- the catalogue handler -------------------------------------------

    mod catalogue {
        use super::super::*;
        use super::stage_at;
        use crate::model::ModelKind;
        use sos_store::MemoryStore;

        fn handler() -> CatalogStageHandler<MemoryStore> {
            CatalogStageHandler::new(
                Rc::new(RefCell::new(MemoryStore::new())),
                sim_backend(
                    SemVer::new(0, 1, 0),
                    HashAlgo::Sha256.hash(b"test-backend", b"scirust-sim"),
                ),
            )
        }

        /// Logistic growth toward carrying capacity — a model with a closed
        /// form, so the stage's output can be checked against real physics.
        fn logistic() -> ModelRun {
            ModelRun::new(
                ModelSpec::new(ModelKind::LogisticGrowth, [0.8, 100.0]),
                [5.0],
                0.0,
                6.0,
                0.001,
            )
        }

        #[test]
        fn running_a_stage_integrates_the_named_model_and_stores_it() {
            let mut h = handler();
            let store = Rc::clone(&h.store);
            let address = h.offer(logistic());

            let ids = h.run(&stage_at(address)).unwrap();
            assert_eq!(ids.len(), 1);

            let stored: Object<ModeledTrajectoryBody> =
                store.borrow().get_object(ids[0]).unwrap().unwrap();
            // The physics: x(6) for r = 0.8, K = 100, x0 = 5 is ~86.478.
            let x_end = stored.body.trajectory.last().unwrap().1[0];
            let exact = 100.0 / (1.0 + 19.0 * (-0.8 * 6.0_f64).exp());
            assert!((x_end - exact).abs() < 1e-6, "{x_end} vs {exact}");
            // And the object knows what produced it.
            assert_eq!(stored.body.model, logistic().model);
            assert_eq!(stored.level, CATALOG_STAGE_LEVEL);
            assert_eq!(stored.producer.name, "scirust-sim/catalog-rk4");
        }

        #[test]
        fn a_plan_can_be_audited_before_it_runs() {
            // The capability model identity buys, and the one `OdeStageHandler`
            // cannot offer: given only a stage's `config_hash`, say which
            // science the stage will do.
            let mut h = handler();
            let address = h.offer(logistic());
            assert_eq!(
                h.model_at(&address).map(|m| m.kind),
                Some(ModelKind::LogisticGrowth)
            );
            assert_eq!(h.model_at(&address).unwrap().params, vec![0.8, 100.0]);
            assert_eq!(h.offered(), vec![address]);

            let never_offered = model_config_address(&ModelRun::new(
                ModelSpec::new(ModelKind::Sir, [0.6, 0.2]),
                [0.99, 0.01, 0.0],
                0.0,
                1.0,
                0.01,
            ));
            assert!(h.model_at(&never_offered).is_none());
        }

        #[test]
        fn two_models_are_two_addresses_even_with_identical_run_parameters() {
            // The property the ODE handler cannot have: the plan itself
            // distinguishes the experiments.
            let growth = logistic();
            let mut cooling = growth.clone();
            cooling.model.kind = ModelKind::NewtonCooling;
            assert_ne!(
                model_config_address(&growth),
                model_config_address(&cooling)
            );
            // And neither collides with an ODE config address, which uses a
            // different domain prefix.
            let ode = config_address(&OdeConfig::new(0.0, 6.0, vec![5.0], 0.001));
            assert_ne!(model_config_address(&growth), ode);
        }

        #[test]
        fn an_unoffered_address_fails_the_stage() {
            let mut h = handler();
            let never = model_config_address(&logistic());
            let err = h.run(&stage_at(never)).unwrap_err();
            assert!(matches!(err, WorkflowError::StageFailed { .. }), "{err:?}");
            assert!(err.to_string().contains("no model run is on offer"));
        }

        #[test]
        fn an_invalid_run_fails_the_stage_rather_than_storing_anything() {
            let mut h = handler();
            let store = Rc::clone(&h.store);
            // A state vector of the wrong length for this model — rejected by
            // the backend, not by `offer`, since a config is just data.
            let address = h.offer(ModelRun::new(
                ModelSpec::new(ModelKind::Sir, [0.6, 0.2]),
                [0.99, 0.01],
                0.0,
                1.0,
                0.01,
            ));
            assert!(matches!(
                h.run(&stage_at(address)),
                Err(WorkflowError::StageFailed { .. })
            ));
            assert!(
                store.borrow().object_ids().is_empty(),
                "a failed stage must store nothing"
            );
        }

        #[test]
        fn the_same_stage_run_twice_produces_the_same_object_id() {
            let mut h = handler();
            let address = h.offer(logistic());
            let first = h.run(&stage_at(address)).unwrap();
            let second = h.run(&stage_at(address)).unwrap();
            assert_eq!(first, second, "an L3 backend must be bit-reproducible");
        }

        #[test]
        fn the_stage_seed_and_inputs_reach_the_stored_object() {
            let mut h = handler();
            let store = Rc::clone(&h.store);
            let input = ObjectId::compute(HashAlgo::Sha256, b"test", b"upstream");
            let address = h.offer(logistic());
            let mut stage = stage_at(address);
            stage.seed = 4242;
            stage.inputs = vec![input];

            let ids = h.run(&stage).unwrap();
            let stored: Object<ModeledTrajectoryBody> =
                store.borrow().get_object(ids[0]).unwrap().unwrap();
            assert_eq!(stored.body.seed, 4242);
            assert_eq!(stored.repro.seed, 4242);
            assert_eq!(stored.parents, vec![input]);
            assert_eq!(stored.repro.inputs, vec![input]);
        }
    }
}
