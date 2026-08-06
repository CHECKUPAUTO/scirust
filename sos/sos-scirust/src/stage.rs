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
//! ## Handlers, and the difference between them
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
//! ## The third handler, and the line it falls on
//!
//! [`SpectrumStageHandler`] and [`WelchStageHandler`] join
//! [`CatalogStageHandler`] on the data side: a [`SpectrumConfig`] is samples,
//! a rate and a named window, and a [`WelchConfig`] adds a segment length and
//! an overlap, so both can be written down, audited before they run
//! ([`measurement_at`](SpectrumStageHandler::measurement_at),
//! [`segmentation_at`](WelchStageHandler::segmentation_at)) and driven from
//! `sos run`.
//!
//! [`TrajectorySpectrumStageHandler`] is the one that breaks the pattern, and
//! deliberately: its configuration names a component and a segmentation but
//! carries **no signal**, because the signal is whatever trajectory the stage
//! `consumes`. That makes it the first handler for which the store is a
//! genuine *input* rather than only a sink — and the first stage that can
//! analyse something an earlier stage computed.
//!
//! That is the whole partition, and it is worth stating because it is not a
//! matter of effort. Of this crate's seven `Simulate` backends, exactly
//! **four** take configurations a file can express.
//! [`crate::ode::Dopri5OdeSimulator`], [`crate::quadrature`] and
//! [`crate::root`] each take a *function* — a right-hand side, an integrand,
//! a residual — and no file can name a function. They can gain handlers; they
//! can never gain a CLI binding without a transport that ships code
//! (RFC-0002 §10's WASM or MCP transports), which is a different thing from
//! more plumbing here.
//!
//! ## Only what these demonstrate
//!
//! Five of the seven backends have handlers, and no other engine's stages have
//! any. What can be claimed is precise: a plan — hand-built, or resolved from
//! a TOML study, or run from `sos run` for the four data-configured
//! backends — whose stages name one of these plugins runs, memoizes, and
//! records real results.
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
use sos_simulation::{Observation, SimDescriptor, Simulate};
use sos_store::{ObjectStore, TypedStore};
use sos_workflow::{Stage, StageExecutor, WorkflowError};

use crate::model::{
    AdaptiveCatalogSimulator, AdaptiveModelRun, CatalogSimulator, CertifiedModeledTrajectoryBody,
    ModelRun, ModelSpec, ModeledTrajectoryBody,
};
use crate::ode::{OdeConfig, Rk4OdeSimulator, Trajectory};
use crate::pipeline::{
    TrajectorySpectrogramBody, TrajectorySpectrogramConfig, TrajectorySpectrogramSimulator,
    TrajectorySpectrumBody, TrajectorySpectrumConfig, TrajectorySpectrumSimulator,
};
use crate::solver::{ExactF64Seq, encode_f64};
use crate::spectrum::{
    AveragedSpectrumBody, PeriodogramSimulator, SpectrumBody, SpectrumConfig, WelchConfig,
    WelchSimulator, WindowKind,
};

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

/// Domain-separation prefix for a spectral measurement's content address.
const SPECTRUM_CONFIG_DOMAIN: &[u8] = b"sos-scirust:spectrum-stage-config:v1";

/// The content address of `config` — what a [`Stage`] must carry in its
/// `config_hash` for [`SpectrumStageHandler`] to run it.
#[must_use]
pub fn spectrum_config_address(config: &SpectrumConfig) -> Digest {
    HashAlgo::Sha256.hash(SPECTRUM_CONFIG_DOMAIN, &config.canonical_bytes())
}

/// A [`BackendVersion`] naming `scirust-signal`.
#[must_use]
pub fn signal_backend(version: SemVer, content_hash: Digest) -> BackendVersion {
    BackendVersion::new("scirust-signal", version, content_hash)
}

/// The determinism level every [`SpectrumStageHandler`] output carries.
///
/// `L3` — and, as [`crate::spectrum`]'s own docs insist at length, that is a
/// reproducibility claim and not an accuracy one.
pub const SPECTRUM_STAGE_LEVEL: DeterminismLevel = DeterminismLevel::L3;

/// A [`StageExecutor`] running spectral measurements on
/// [`PeriodogramSimulator`].
///
/// Like [`CatalogStageHandler`] and unlike [`OdeStageHandler`], it carries no
/// type parameter for a closure: a [`SpectrumConfig`] is data — samples, a
/// rate, a named window — so it can be written down. That is what makes this
/// the *second* backend a study author can reach without writing Rust, and
/// the asymmetry is worth naming: of this crate's five `Simulate` backends,
/// exactly two take configurations that a file can express. The other three
/// take a function (`dy/dt = f(t, y)`, an integrand, a residual), and no file
/// can name a function.
pub struct SpectrumStageHandler<S> {
    simulator: PeriodogramSimulator,
    store: Rc<RefCell<S>>,
    configs: BTreeMap<Digest, SpectrumConfig>,
    backend: BackendVersion,
}

impl<S: ObjectStore> SpectrumStageHandler<S> {
    /// A handler sealing its results into `store`.
    #[must_use]
    pub fn new(store: Rc<RefCell<S>>, backend: BackendVersion) -> Self {
        Self {
            simulator: PeriodogramSimulator::new(SimDescriptor::new(
                "scirust-signal/periodogram",
                SemVer::new(1, 0, 0),
            )),
            store,
            configs: BTreeMap::new(),
            backend,
        }
    }

    /// Make `config` available to stages, returning its content address.
    pub fn offer(&mut self, config: SpectrumConfig) -> Digest {
        let address = spectrum_config_address(&config);
        self.configs.insert(address, config);
        address
    }

    /// The content addresses currently on offer, sorted.
    #[must_use]
    pub fn offered(&self) -> Vec<Digest> {
        self.configs.keys().copied().collect()
    }

    /// Which window the measurement at `address` will apply, and over how many
    /// samples — the audit [`CatalogStageHandler::model_at`] offers, for the
    /// choice that shapes a spectrum.
    #[must_use]
    pub fn measurement_at(&self, address: &Digest) -> Option<(WindowKind, usize)> {
        self.configs
            .get(address)
            .map(|c| (c.window, c.signal.len()))
    }

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

impl<S: ObjectStore> StageExecutor for SpectrumStageHandler<S> {
    fn run(&mut self, stage: &Stage) -> Result<Vec<ObjectId>, WorkflowError> {
        let config =
            self.configs
                .get(&stage.config_hash)
                .ok_or_else(|| WorkflowError::StageFailed {
                    stage: stage.id.clone(),
                    reason: format!(
                        "no spectral measurement is on offer at content address {}",
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

        let level = observation.level();
        let repro = ReproMeta::new(
            stage.seed,
            RngId::new("none"),
            self.env().digest(HashAlgo::Sha256),
        )
        .with_inputs(stage.inputs.clone());
        let object = Object::builder(SpectrumBody::from_observation(observation))
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

/// Domain-separation prefix for a Welch measurement's content address.
const WELCH_CONFIG_DOMAIN: &[u8] = b"sos-scirust:welch-stage-config:v1";

/// The content address of `config` — what a [`Stage`] must carry in its
/// `config_hash` for [`WelchStageHandler`] to run it.
#[must_use]
pub fn welch_config_address(config: &WelchConfig) -> Digest {
    HashAlgo::Sha256.hash(WELCH_CONFIG_DOMAIN, &config.canonical_bytes())
}

/// The determinism level every [`WelchStageHandler`] output carries: `L3`.
///
/// Identical to [`SPECTRUM_STAGE_LEVEL`], and the equality is the point — the
/// two backends differ in *estimator quality*, which a determinism tag does
/// not and should not express. What says how good a Welch estimate is lives in
/// its output, as [`crate::spectrum::AveragedSpectrum::segments`].
pub const WELCH_STAGE_LEVEL: DeterminismLevel = DeterminismLevel::L3;

/// A [`StageExecutor`] running Welch averaged-periodogram measurements.
pub struct WelchStageHandler<S> {
    simulator: WelchSimulator,
    store: Rc<RefCell<S>>,
    configs: BTreeMap<Digest, WelchConfig>,
    backend: BackendVersion,
}

impl<S: ObjectStore> WelchStageHandler<S> {
    /// A handler sealing its results into `store`.
    #[must_use]
    pub fn new(store: Rc<RefCell<S>>, backend: BackendVersion) -> Self {
        Self {
            simulator: WelchSimulator::new(SimDescriptor::new(
                "scirust-signal/welch",
                SemVer::new(1, 0, 0),
            )),
            store,
            configs: BTreeMap::new(),
            backend,
        }
    }

    /// Make `config` available to stages, returning its content address.
    pub fn offer(&mut self, config: WelchConfig) -> Digest {
        let address = welch_config_address(&config);
        self.configs.insert(address, config);
        address
    }

    /// The content addresses currently on offer, sorted.
    #[must_use]
    pub fn offered(&self) -> Vec<Digest> {
        self.configs.keys().copied().collect()
    }

    /// How the measurement at `address` will be segmented — window, segment
    /// length and overlap — answerable before the stage runs.
    ///
    /// The pre-run audit again, and here it answers the question that decides
    /// whether the result will be worth anything: how many segments a reader
    /// should expect to be averaged.
    #[must_use]
    pub fn segmentation_at(&self, address: &Digest) -> Option<(WindowKind, usize, usize)> {
        self.configs
            .get(address)
            .map(|c| (c.window, c.segment_len, c.overlap))
    }

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

impl<S: ObjectStore> StageExecutor for WelchStageHandler<S> {
    fn run(&mut self, stage: &Stage) -> Result<Vec<ObjectId>, WorkflowError> {
        let config =
            self.configs
                .get(&stage.config_hash)
                .ok_or_else(|| WorkflowError::StageFailed {
                    stage: stage.id.clone(),
                    reason: format!(
                        "no Welch measurement is on offer at content address {}",
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

        let level = observation.level();
        let repro = ReproMeta::new(
            stage.seed,
            RngId::new("none"),
            self.env().digest(HashAlgo::Sha256),
        )
        .with_inputs(stage.inputs.clone());
        let object = Object::builder(AveragedSpectrumBody::from_observation(observation))
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

/// Load the one trajectory a stage consumed.
///
/// Reads both trajectory kinds this crate produces — a bare
/// [`TrajectoryBody`] from the ODE backends and a [`ModeledTrajectoryBody`]
/// from the catalogue — by dispatching on the stored record's kind, the
/// same body-type-erased trick provenance reading uses. A stage should not
/// have to care which backend made the signal it is measuring.
fn consumed_trajectory<S: ObjectStore>(
    store: &Rc<RefCell<S>>,
    stage: &Stage,
) -> core::result::Result<Trajectory, String> {
    let [input] = stage.inputs.as_slice()
    else
    {
        return Err(match stage.inputs.len()
        {
            0 => "this stage measures a trajectory but consumed none — a \
                  `consumes = [\"<stage>\"]` edge is missing from the study"
                .to_owned(),
            n => format!(
                "this stage measures *a* trajectory but consumed {n}; which one is meant \
                 cannot be guessed"
            ),
        });
    };
    let store = store.borrow();
    let record = store
        .get_raw(*input)
        .ok_or_else(|| format!("the consumed object {input} is not in the store"))?;
    let name = record.kind.name.as_str();
    if name == <TrajectoryBody as sos_core::Body>::KIND
    {
        store
            .get_object::<TrajectoryBody>(*input)
            .map_err(|e| e.to_string())?
            .map(|o| o.body.trajectory)
            .ok_or_else(|| format!("{input} vanished mid-read"))
    }
    else if name == <ModeledTrajectoryBody as sos_core::Body>::KIND
    {
        store
            .get_object::<ModeledTrajectoryBody>(*input)
            .map_err(|e| e.to_string())?
            .map(|o| o.body.trajectory)
            .ok_or_else(|| format!("{input} vanished mid-read"))
    }
    else
    {
        Err(format!(
            "this stage measures a trajectory, but the object it consumed is a `{name}`"
        ))
    }
}

/// Domain-separation prefix for a trajectory spectrogram's content address.
const SPECTROGRAM_CONFIG_DOMAIN: &[u8] = b"sos-scirust:spectrogram-stage-config:v1";

/// The content address of `config` — what a [`Stage`] must carry in its
/// `config_hash` for [`TrajectorySpectrogramStageHandler`] to run it.
///
/// Distinct from [`trajectory_config_address`] even though the two
/// configurations have the same fields: they are different measurements, and a
/// shared address would let a stage route to the wrong backend.
#[must_use]
pub fn spectrogram_config_address(config: &TrajectorySpectrogramConfig) -> Digest {
    HashAlgo::Sha256.hash(SPECTROGRAM_CONFIG_DOMAIN, &config.canonical_bytes())
}

/// Binds [`TrajectorySpectrogramSimulator`] as a workflow stage: how a
/// simulated system's spectrum changes, rather than what it averages to.
///
/// Reads its input the same way [`TrajectorySpectrumStageHandler`] does — one
/// consumed trajectory, either kind, dispatched on the stored record's kind —
/// and differs only in what it computes from it.
pub struct TrajectorySpectrogramStageHandler<S> {
    simulator: TrajectorySpectrogramSimulator,
    store: Rc<RefCell<S>>,
    configs: BTreeMap<Digest, TrajectorySpectrogramConfig>,
    backend: BackendVersion,
}

impl<S: ObjectStore> TrajectorySpectrogramStageHandler<S> {
    /// A handler reading trajectories from, and sealing results into, `store`.
    #[must_use]
    pub fn new(store: Rc<RefCell<S>>, backend: BackendVersion) -> Self {
        Self {
            simulator: TrajectorySpectrogramSimulator::new(SimDescriptor::new(
                "sos-scirust/trajectory-spectrogram",
                SemVer::new(1, 0, 0),
            )),
            store,
            configs: BTreeMap::new(),
            backend,
        }
    }

    /// Offer a configuration, returning its content address.
    pub fn offer(&mut self, config: TrajectorySpectrogramConfig) -> Digest {
        let address = spectrogram_config_address(&config);
        self.configs.insert(address, config);
        address
    }

    /// The content addresses currently on offer, sorted.
    #[must_use]
    pub fn offered(&self) -> Vec<Digest> {
        self.configs.keys().copied().collect()
    }

    /// Which component the spectrogram at `address` reads, and how it is
    /// framed — the choices that set its time and frequency resolution.
    #[must_use]
    pub fn framing_at(&self, address: &Digest) -> Option<(usize, WindowKind, usize, usize)> {
        self.configs
            .get(address)
            .map(|c| (c.component, c.window, c.segment_len, c.overlap))
    }

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

impl<S: ObjectStore> StageExecutor for TrajectorySpectrogramStageHandler<S> {
    fn run(&mut self, stage: &Stage) -> core::result::Result<Vec<ObjectId>, WorkflowError> {
        let config = self
            .configs
            .get(&stage.config_hash)
            .ok_or_else(|| WorkflowError::StageFailed {
                stage: stage.id.clone(),
                reason: format!(
                    "no trajectory spectrogram is on offer at content address {}",
                    stage.config_hash
                ),
            })?
            .clone();

        let trajectory = consumed_trajectory(&self.store, stage).map_err(|reason| {
            WorkflowError::StageFailed {
                stage: stage.id.clone(),
                reason,
            }
        })?;

        let observation = self
            .simulator
            .measure(&config, &trajectory, stage.seed)
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
        let object = Object::builder(TrajectorySpectrogramBody::from_observation(observation))
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

/// Domain-separation prefix for an adaptive catalogue run's content address.
const ADAPTIVE_MODEL_CONFIG_DOMAIN: &[u8] = b"sos-scirust:adaptive-model-stage-config:v1";

/// The content address of `run` — what a [`Stage`] must carry in its
/// `config_hash` for [`AdaptiveCatalogStageHandler`] to run it.
#[must_use]
pub fn adaptive_model_config_address(run: &AdaptiveModelRun) -> Digest {
    HashAlgo::Sha256.hash(ADAPTIVE_MODEL_CONFIG_DOMAIN, &run.canonical_bytes())
}

/// Binds [`AdaptiveCatalogSimulator`] as a workflow stage — **the first `L2`
/// result a study file can produce**.
///
/// Every other CLI-reachable handler is `L3`: bit-reproducible, and saying
/// nothing about accuracy. This one integrates a catalogued model adaptively
/// and stores a [`CertifiedTrajectory`](crate::ode::CertifiedTrajectory), so the
/// object carries the tolerances
/// it is certified to and the accepted/rejected step counts it took to get
/// there.
///
/// That matters beyond the numerics. `sos-repro`'s reproduction contract
/// treats `L2` differently from `L3` — an `L2` node cannot be verified by
/// comparing object ids, because two runs meeting the same tolerance need not
/// be bit-identical, and must instead be judged by a `Certifier` that
/// understands the quantity. Until this handler existed, no `L2` node was
/// reachable from `sos run`, so that whole branch of the contract was
/// unexercised from the command line.
pub struct AdaptiveCatalogStageHandler<S> {
    simulator: AdaptiveCatalogSimulator,
    store: Rc<RefCell<S>>,
    configs: BTreeMap<Digest, AdaptiveModelRun>,
    backend: BackendVersion,
}

impl<S: ObjectStore> AdaptiveCatalogStageHandler<S> {
    /// A handler sealing its results into `store`.
    #[must_use]
    pub fn new(store: Rc<RefCell<S>>, backend: BackendVersion) -> Self {
        Self {
            simulator: AdaptiveCatalogSimulator::new(),
            store,
            configs: BTreeMap::new(),
            backend,
        }
    }

    /// Offer a run, returning its content address.
    pub fn offer(&mut self, run: AdaptiveModelRun) -> Digest {
        let address = adaptive_model_config_address(&run);
        self.configs.insert(address, run);
        address
    }

    /// The content addresses currently on offer, sorted.
    #[must_use]
    pub fn offered(&self) -> Vec<Digest> {
        self.configs.keys().copied().collect()
    }

    /// Which model the run at `address` integrates, and to what tolerances —
    /// the audit [`CatalogStageHandler::model_at`] offers, plus the numbers
    /// that make this an `L2` claim rather than an `L3` one.
    #[must_use]
    pub fn run_at(&self, address: &Digest) -> Option<(ModelSpec, f64, f64)> {
        self.configs
            .get(address)
            .map(|r| (r.model.clone(), r.rtol, r.atol))
    }

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

impl<S: ObjectStore> StageExecutor for AdaptiveCatalogStageHandler<S> {
    fn run(&mut self, stage: &Stage) -> core::result::Result<Vec<ObjectId>, WorkflowError> {
        let config = self
            .configs
            .get(&stage.config_hash)
            .ok_or_else(|| WorkflowError::StageFailed {
                stage: stage.id.clone(),
                reason: format!(
                    "no adaptive model run is on offer at content address {}",
                    stage.config_hash
                ),
            })?
            .clone();

        let observation =
            self.simulator
                .run(&config, stage.seed)
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
        let object = Object::builder(CertifiedModeledTrajectoryBody::from_observation(
            observation,
        ))
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

/// Domain-separation prefix for a trajectory measurement's content address.
const TRAJECTORY_CONFIG_DOMAIN: &[u8] = b"sos-scirust:trajectory-stage-config:v1";

/// The content address of `config` — what a [`Stage`] must carry in its
/// `config_hash` for [`TrajectorySpectrumStageHandler`] to run it.
#[must_use]
pub fn trajectory_config_address(config: &TrajectorySpectrumConfig) -> Digest {
    HashAlgo::Sha256.hash(TRAJECTORY_CONFIG_DOMAIN, &config.canonical_bytes())
}

/// Binds [`TrajectorySpectrumSimulator`] as a workflow stage — the first
/// handler whose input is another stage's output.
///
/// Every other handler here is a pure function of its configuration: offer a
/// [`ModelRun`] or a [`SpectrumConfig`], and running the stage needs nothing
/// else. This one is different by design. A [`TrajectorySpectrumConfig`]
/// carries no signal, so the handler must *read* the trajectory the stage
/// consumed out of the store before it can compute anything.
///
/// That makes the store a genuine input rather than only a sink, and it is
/// why the stage's `inputs` must be exactly one object: "the trajectory this
/// measures" is singular, and picking one of several silently would be the
/// kind of guess the rest of this system refuses. Zero inputs means the study
/// forgot the `consumes` edge — the error says so, because that mistake is
/// otherwise puzzling.
pub struct TrajectorySpectrumStageHandler<S> {
    simulator: TrajectorySpectrumSimulator,
    store: Rc<RefCell<S>>,
    configs: BTreeMap<Digest, TrajectorySpectrumConfig>,
    backend: BackendVersion,
}

impl<S: ObjectStore> TrajectorySpectrumStageHandler<S> {
    /// A handler reading trajectories from, and sealing results into, `store`.
    #[must_use]
    pub fn new(store: Rc<RefCell<S>>, backend: BackendVersion) -> Self {
        Self {
            simulator: TrajectorySpectrumSimulator::new(SimDescriptor::new(
                "sos-scirust/trajectory-spectrum",
                SemVer::new(1, 0, 0),
            )),
            store,
            configs: BTreeMap::new(),
            backend,
        }
    }

    /// Offer a configuration, returning its content address.
    pub fn offer(&mut self, config: TrajectorySpectrumConfig) -> Digest {
        let address = trajectory_config_address(&config);
        self.configs.insert(address, config);
        address
    }

    /// The content addresses currently on offer, sorted.
    #[must_use]
    pub fn offered(&self) -> Vec<Digest> {
        self.configs.keys().copied().collect()
    }

    /// Which component the measurement at `address` reads, and how it is
    /// segmented — the audit the other handlers offer, for the choices that
    /// shape this result.
    #[must_use]
    pub fn measurement_at(&self, address: &Digest) -> Option<(usize, WindowKind, usize)> {
        self.configs
            .get(address)
            .map(|c| (c.component, c.window, c.segment_len))
    }

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

impl<S: ObjectStore> StageExecutor for TrajectorySpectrumStageHandler<S> {
    fn run(&mut self, stage: &Stage) -> core::result::Result<Vec<ObjectId>, WorkflowError> {
        let config = self
            .configs
            .get(&stage.config_hash)
            .ok_or_else(|| WorkflowError::StageFailed {
                stage: stage.id.clone(),
                reason: format!(
                    "no trajectory measurement is on offer at content address {}",
                    stage.config_hash
                ),
            })?
            .clone();

        let trajectory = consumed_trajectory(&self.store, stage).map_err(|reason| {
            WorkflowError::StageFailed {
                stage: stage.id.clone(),
                reason,
            }
        })?;

        let observation = self
            .simulator
            .measure(&config, &trajectory, stage.seed)
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
        let object = Object::builder(TrajectorySpectrumBody::from_observation(observation))
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

    mod spectral {
        use super::super::*;
        use super::stage_at;
        use sos_store::MemoryStore;
        use std::f64::consts::TAU;

        fn handler() -> SpectrumStageHandler<MemoryStore> {
            SpectrumStageHandler::new(
                Rc::new(RefCell::new(MemoryStore::new())),
                signal_backend(
                    SemVer::new(0, 1, 0),
                    HashAlgo::Sha256.hash(b"test-backend", b"scirust-signal"),
                ),
            )
        }

        /// A 64 Hz tone sampled at 1024 Hz over 1024 samples — exactly bin 64.
        fn tone() -> SpectrumConfig {
            let signal = (0..1024)
                .map(|i| (TAU * 64.0 * f64::from(i) / 1024.0).sin())
                .collect();
            SpectrumConfig::new(signal, 1024.0, WindowKind::Hann)
        }

        #[test]
        fn running_a_stage_measures_the_spectrum_and_stores_it() {
            let mut h = handler();
            let store = Rc::clone(&h.store);
            let address = h.offer(tone());

            let ids = h.run(&stage_at(address)).unwrap();
            let stored: Object<SpectrumBody> = store.borrow().get_object(ids[0]).unwrap().unwrap();

            // Real physics through the stage: the tone lands in its own bin.
            let peak = stored
                .body
                .spectrum
                .psd
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .expect("a measured spectrum has bins")
                .0;
            assert_eq!(peak, 64, "the tone must land in bin 64");
            assert_eq!(stored.body.spectrum.window, WindowKind::Hann);
            assert_eq!(stored.level, SPECTRUM_STAGE_LEVEL);
            assert_eq!(stored.producer.name, "scirust-signal/periodogram");
        }

        #[test]
        fn a_measurement_can_be_audited_before_it_runs() {
            let mut h = handler();
            let address = h.offer(tone());
            assert_eq!(h.measurement_at(&address), Some((WindowKind::Hann, 1024)));
            assert_eq!(h.offered(), vec![address]);
            let elsewhere = spectrum_config_address(&SpectrumConfig::new(
                vec![0.0; 8],
                8.0,
                WindowKind::FlatTop,
            ));
            assert!(h.measurement_at(&elsewhere).is_none());
        }

        #[test]
        fn the_window_is_part_of_the_address_and_never_collides_with_another_kind() {
            let mut hann = tone();
            hann.window = WindowKind::Hann;
            let mut flat = tone();
            flat.window = WindowKind::FlatTop;
            assert_ne!(
                spectrum_config_address(&hann),
                spectrum_config_address(&flat)
            );
            // Domain separation: never the same as an ODE or model address.
            let ode = config_address(&OdeConfig::new(0.0, 1.0, vec![1.0], 0.1));
            assert_ne!(spectrum_config_address(&hann), ode);
        }

        #[test]
        fn an_unoffered_address_fails_the_stage() {
            let mut h = handler();
            let err = h
                .run(&stage_at(spectrum_config_address(&tone())))
                .unwrap_err();
            assert!(
                err.to_string()
                    .contains("no spectral measurement is on offer"),
                "{err}"
            );
        }

        #[test]
        fn a_record_the_fft_would_assert_on_fails_the_stage_and_stores_nothing() {
            // `fft_real` panics on a non-power-of-two record. A stage must not.
            let mut h = handler();
            let store = Rc::clone(&h.store);
            let address = h.offer(SpectrumConfig::new(
                vec![0.5; 100],
                1024.0,
                WindowKind::Hann,
            ));
            assert!(matches!(
                h.run(&stage_at(address)),
                Err(WorkflowError::StageFailed { .. })
            ));
            assert!(store.borrow().object_ids().is_empty());
        }

        #[test]
        fn the_same_stage_run_twice_produces_the_same_object_id() {
            let mut h = handler();
            let address = h.offer(tone());
            assert_eq!(
                h.run(&stage_at(address)).unwrap(),
                h.run(&stage_at(address)).unwrap()
            );
        }

        #[test]
        fn a_stored_spectrum_reloads_bit_for_bit() {
            let mut h = handler();
            let store = Rc::clone(&h.store);
            let address = h.offer(tone());
            let ids = h.run(&stage_at(address)).unwrap();

            let fresh = h.simulator.run(&tone(), 0).unwrap().output;
            let stored: Object<SpectrumBody> = store.borrow().get_object(ids[0]).unwrap().unwrap();
            assert_eq!(fresh.psd.len(), stored.body.spectrum.psd.len());
            for (i, (a, b)) in fresh.psd.iter().zip(&stored.body.spectrum.psd).enumerate()
            {
                assert_eq!(a.to_bits(), b.to_bits(), "psd differs at bin {i}");
            }
            assert_eq!(
                fresh.centroid_hz.to_bits(),
                stored.body.spectrum.centroid_hz.to_bits()
            );
        }

        #[test]
        fn a_config_survives_a_file_at_the_same_address() {
            // What makes a spectral measurement authorable outside Rust.
            let config = tone();
            let json = serde_json::to_string(&config).unwrap();
            assert!(json.contains("\"hann\""), "window travels as its code");
            let back: SpectrumConfig = serde_json::from_str(&json).unwrap();
            assert_eq!(back, config);
            assert_eq!(
                spectrum_config_address(&back),
                spectrum_config_address(&config)
            );
        }

        #[test]
        fn a_window_this_build_does_not_have_fails_to_load() {
            let json = r#"{"signal":["1","0"],"sample_rate_hz":"8","window":"kaiser"}"#;
            let err = serde_json::from_str::<SpectrumConfig>(json).unwrap_err();
            assert!(err.to_string().contains("kaiser"), "{err}");
        }
    }

    // ---- the simulate-then-analyse pipeline --------------------------------

    fn pipeline_backend() -> BackendVersion {
        signal_backend(
            SemVer::new(0, 1, 0),
            HashAlgo::Sha256.hash(b"test-backend", b"scirust-signal"),
        )
    }

    fn trajectory_handler(
        store: &Rc<RefCell<MemoryStore>>,
    ) -> TrajectorySpectrumStageHandler<MemoryStore> {
        TrajectorySpectrumStageHandler::new(Rc::clone(store), pipeline_backend())
    }

    /// A stage consuming `inputs`, at `address`.
    fn consuming_stage(id: &str, address: Digest, inputs: Vec<ObjectId>) -> Stage {
        Stage::new(
            StageId::new(id),
            StageDescriptor::new(
                "trajectory-spectrum",
                SemVer::new(1, 0, 0),
                HashAlgo::default().hash(b"pin", b"trajectory-spectrum"),
            ),
            inputs,
            address,
            0,
            vec![],
        )
    }

    /// Store a uniformly sampled trajectory and return its id.
    fn stored_trajectory(store: &Rc<RefCell<MemoryStore>>, n: usize) -> ObjectId {
        let trajectory: Trajectory = (0..n)
            .map(|i| {
                let t = f64::from(u32::try_from(i).unwrap()) / 1024.0;
                (t, vec![(std::f64::consts::TAU * 64.0 * t).sin(), 0.25 * t])
            })
            .collect();
        let object = Object::builder(TrajectoryBody {
            trajectory,
            level: DeterminismLevel::L3,
            seed: 0,
        })
        .level(DeterminismLevel::L3)
        .seal();
        store.borrow_mut().put_object(&object).unwrap()
    }

    #[test]
    fn a_stage_measures_the_trajectory_it_consumed() {
        // The whole point: a signal that was *computed* by an earlier stage,
        // not written into a file by hand.
        let store = Rc::new(RefCell::new(MemoryStore::new()));
        let upstream = stored_trajectory(&store, 4096);
        let mut handler = trajectory_handler(&store);
        let address = handler.offer(TrajectorySpectrumConfig::new(
            0,
            WindowKind::Hann,
            1024,
            512,
        ));

        let out = handler
            .run(&consuming_stage("measure", address, vec![upstream]))
            .unwrap();
        assert_eq!(out.len(), 1);

        let stored: Object<TrajectorySpectrumBody> =
            store.borrow().get_object(out[0]).unwrap().unwrap();
        // The rate was read off the trajectory, not declared anywhere.
        assert!((stored.body.measured.sample_rate_hz - 1024.0).abs() < 1e-9);
        assert_eq!(stored.body.measured.component, 0);
        assert_eq!(stored.body.measured.spectrum.segments, 7);
        // And the provenance names what it measured.
        assert_eq!(stored.parents, vec![upstream]);
    }

    #[test]
    fn a_stage_that_consumed_nothing_is_told_what_is_missing() {
        // The likeliest authoring mistake, and an otherwise puzzling one.
        let store = Rc::new(RefCell::new(MemoryStore::new()));
        let mut handler = trajectory_handler(&store);
        let address = handler.offer(TrajectorySpectrumConfig::new(0, WindowKind::Hann, 512, 0));

        let err = handler
            .run(&consuming_stage("measure", address, vec![]))
            .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("consumes"), "{message}");
    }

    #[test]
    fn consuming_two_trajectories_is_refused_rather_than_guessed() {
        let store = Rc::new(RefCell::new(MemoryStore::new()));
        let a = stored_trajectory(&store, 2048);
        let b = stored_trajectory(&store, 4096);
        let mut handler = trajectory_handler(&store);
        let address = handler.offer(TrajectorySpectrumConfig::new(0, WindowKind::Hann, 512, 0));

        let err = handler
            .run(&consuming_stage("measure", address, vec![a, b]))
            .unwrap_err();
        assert!(err.to_string().contains("cannot be guessed"), "{err}");
    }

    #[test]
    fn consuming_something_that_is_not_a_trajectory_is_refused() {
        // A `consumes` edge can point at any object; this one says what went
        // wrong rather than failing to decode.
        let store = Rc::new(RefCell::new(MemoryStore::new()));
        let mut spectra = SpectrumStageHandler::new(Rc::clone(&store), pipeline_backend());
        let signal: Vec<f64> = (0..1024)
            .map(|i| (std::f64::consts::TAU * 64.0 * f64::from(i) / 1024.0).sin())
            .collect();
        let spectrum_address = spectra.offer(SpectrumConfig::new(signal, 1024.0, WindowKind::Hann));
        let not_a_trajectory = spectra.run(&stage_at(spectrum_address)).unwrap()[0];

        let mut handler = trajectory_handler(&store);
        let address = handler.offer(TrajectorySpectrumConfig::new(0, WindowKind::Hann, 512, 0));
        let err = handler
            .run(&consuming_stage("measure", address, vec![not_a_trajectory]))
            .unwrap_err();
        assert!(err.to_string().contains("Spectrum"), "{err}");
    }

    #[test]
    fn a_catalogue_trajectory_can_be_measured_too() {
        // Both trajectory kinds this crate produces are readable — a stage
        // should not care which backend made the signal.
        let store = Rc::new(RefCell::new(MemoryStore::new()));
        let mut models = CatalogStageHandler::new(Rc::clone(&store), pipeline_backend());
        let model_address = models.offer(ModelRun::new(
            ModelSpec::new(crate::model::ModelKind::LotkaVolterra, [1.1, 0.4, 0.4, 0.1]),
            [10.0, 5.0],
            0.0,
            40.96,
            0.01,
        ));
        let trajectory = models.run(&stage_at(model_address)).unwrap()[0];

        let mut handler = trajectory_handler(&store);
        let address = handler.offer(TrajectorySpectrumConfig::new(1, WindowKind::Hann, 512, 256));
        let out = handler
            .run(&consuming_stage("measure", address, vec![trajectory]))
            .unwrap();

        let stored: Object<TrajectorySpectrumBody> =
            store.borrow().get_object(out[0]).unwrap().unwrap();
        // 0.01 s between samples.
        assert!((stored.body.measured.sample_rate_hz - 100.0).abs() < 1e-6);
        assert_eq!(
            stored.body.measured.component, 1,
            "the predator, not the prey"
        );
        assert_eq!(stored.parents, vec![trajectory]);
    }
}
