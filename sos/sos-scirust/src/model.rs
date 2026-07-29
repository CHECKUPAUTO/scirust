//! [`ModelSpec`] — scientific models with **content addresses**, and
//! [`CatalogSimulator`], the [`Simulate`] backend that runs them. `L3`.
//!
//! ## The problem this closes
//!
//! [`crate::ode::Rk4OdeSimulator`] integrates real numerics, but the *model*
//! it integrates is an anonymous Rust closure. Nothing can address a closure,
//! so the model's identity has to be asserted out of band by a caller-supplied
//! [`SimDescriptor`] — and an assertion is exactly what a content-addressed
//! store is built to avoid needing. Two entirely different physical systems
//! can be handed the same descriptor, cache under the same key, and produce a
//! stored trajectory whose provenance is a claim rather than a fact.
//!
//! That is also why `sos run` could not be real. A study manifest is text; it
//! cannot carry a differential equation's right-hand side, so a manifest could
//! name a *plugin* but never a *model*. Inventing toy models inside SOS to
//! paper over that would be the placeholder this workspace refuses to ship.
//!
//! The gap turns out not to need one. [`scirust_sim`] already carries a
//! catalogue of real, oracle-tested scientific models — SIR/SEIR epidemics,
//! Lotka-Volterra, RC circuits, Van der Pol, damped oscillators — and each is
//! a **named struct with scalar parameters**, not a closure. A name plus a
//! parameter vector *is* canonically encodable. So a model can have a content
//! address after all, without SOS inventing any science of its own.
//!
//! ## What changes as a result
//!
//! * A [`ModelSpec`] is data. It hashes, so two runs of the same model
//!   memoize together and two different models cannot collide.
//! * A [`ModelRun`]'s content address **determines what will be computed**.
//!   An [`crate::ode::OdeConfig`]'s does not: it fixes the span, the step and
//!   the initial state, but the physics lives in whichever closure the handler
//!   happened to be built with, so the same digest can denote two different
//!   experiments. That difference is what a study manifest needs, because
//!   `sos-workflow`'s [`StageSpec`](sos_workflow::StageSpec) identifies a
//!   stage's configuration *by digest* — a manifest cannot inline a
//!   right-hand side, but it can reference a config that names its own model.
//! * [`ModelKind::code`] is a stable string, so a model can also be named in
//!   plain text and [`ModelKind::from_code`] resolves it back.
//! * The [`SimDescriptor`] here is **fixed**, not caller-supplied, and that is
//!   the point: it names the integrator (`scirust-sim`'s RK4) and nothing
//!   else, because the model is no longer hiding inside it.
//! * The output is a [`ModeledTrajectory`], carrying the spec beside the
//!   numbers — the same reasoning [`crate::root::CertifiedRoot`] carries its
//!   starting point and [`crate::spectrum::Spectrum`] carries its window: a
//!   stored result travels without its config.
//!
//! ## What this is not
//!
//! It is not `sos run`. This module supplies the primitive that was missing —
//! a model with an identity — and nothing downstream is wired to it yet:
//! there is no [`StageExecutor`](sos_workflow::StageExecutor) for the
//! catalogue the way [`crate::stage`] provides one for the closure-based ODE
//! backend, and the CLI is not connected to either. Stated precisely, what
//! can be claimed after this module is that a catalogued model can be
//! specified, addressed, stored, integrated and reproduced — not that a study
//! written as text runs one end to end.
//!
//! ## The catalogue is closed, deliberately
//!
//! [`ModelKind`] is a fixed enum, so SOS can run exactly the models
//! `scirust-sim` ships and no others. That is a real limit and worth stating
//! plainly rather than implying a generality this does not have: a
//! third-party model still needs an out-of-process plugin transport
//! (RFC-0002 §10), which this module does not provide and does not pretend
//! to. What it does provide is the thing that was actually missing — models
//! that *have identity at all*. Every entry is a model `scirust-sim` already
//! validates against an analytic solution or a conservation law in its own
//! test suite, which is why the tests below can check real physics instead of
//! re-asserting this adapter's own arithmetic.

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use scirust_sim::ecology::{LogisticGrowth, LotkaVolterra};
use scirust_sim::electrical::{RcCircuit, VanDerPol};
use scirust_sim::epidemiology::{Seir, Sir};
use scirust_sim::mechanics::{Pendulum, SpringMassDamper};
use scirust_sim::thermal::NewtonCooling;
use scirust_sim::{System, simulate};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{Body, DeterminismLevel};
use sos_simulation::{Observation, Result, SimDescriptor, SimError, Simulate};

use crate::ode::Trajectory;
use crate::solver::{ExactF64Seq, encode_f64};
use crate::stage::exact_trajectory;

/// Which model from `scirust-sim`'s catalogue.
///
/// Each variant's documentation names its parameters **in order**; that order
/// is part of the wire format, since [`ModelSpec`] stores parameters
/// positionally and hashes them that way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum ModelKind {
    /// Logistic population growth, `x' = r·x·(1 − x/K)`. `rate`, `capacity`.
    LogisticGrowth,
    /// Lotka-Volterra predator-prey. `alpha`, `beta`, `delta`, `gamma`.
    LotkaVolterra,
    /// SIR compartmental epidemic, over population **fractions** summing to
    /// `1` — not head counts. `beta`, `gamma`.
    Sir,
    /// SEIR compartmental epidemic with an exposed class, over population
    /// **fractions**. `beta`, `sigma`, `gamma`.
    Seir,
    /// Newton cooling toward an ambient temperature. `rate`, `ambient`.
    NewtonCooling,
    /// An RC circuit charging toward a source voltage. `resistance`,
    /// `capacitance`, `v_source`.
    RcCircuit,
    /// The Van der Pol limit-cycle oscillator. `mu`.
    VanDerPol,
    /// A damped spring-mass oscillator. `mass`, `damping`, `stiffness`.
    SpringMassDamper,
    /// A (optionally damped) simple pendulum. `length`, `gravity`, `damping`.
    Pendulum,
}

impl ModelKind {
    /// Every model in the catalogue, in a stable order.
    pub const ALL: &'static [ModelKind] = &[
        ModelKind::LogisticGrowth,
        ModelKind::LotkaVolterra,
        ModelKind::Sir,
        ModelKind::Seir,
        ModelKind::NewtonCooling,
        ModelKind::RcCircuit,
        ModelKind::VanDerPol,
        ModelKind::SpringMassDamper,
        ModelKind::Pendulum,
    ];

    /// The stable text token that names this model — what a study manifest
    /// writes and [`from_code`](Self::from_code) resolves.
    ///
    /// Stability matters more than prettiness here: these strings are part of
    /// every content address a run produces, so renaming one silently
    /// invalidates stored results.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self
        {
            Self::LogisticGrowth => "ecology/logistic-growth",
            Self::LotkaVolterra => "ecology/lotka-volterra",
            Self::Sir => "epidemiology/sir",
            Self::Seir => "epidemiology/seir",
            Self::NewtonCooling => "thermal/newton-cooling",
            Self::RcCircuit => "electrical/rc-circuit",
            Self::VanDerPol => "electrical/van-der-pol",
            Self::SpringMassDamper => "mechanics/spring-mass-damper",
            Self::Pendulum => "mechanics/pendulum",
        }
    }

    /// The model named by `code`, or `None` if the catalogue has no such
    /// entry.
    ///
    /// An unknown name is deliberately not resolved to a default: a manifest
    /// naming a model this build cannot run must fail, not quietly run a
    /// different one.
    #[must_use]
    pub fn from_code(code: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|k| k.code() == code)
    }

    /// This model's parameter names, in the order [`ModelSpec`] stores them.
    #[must_use]
    pub const fn parameters(self) -> &'static [&'static str] {
        match self
        {
            Self::LogisticGrowth => &["rate", "capacity"],
            Self::LotkaVolterra => &["alpha", "beta", "delta", "gamma"],
            Self::Sir => &["beta", "gamma"],
            Self::Seir => &["beta", "sigma", "gamma"],
            Self::NewtonCooling => &["rate", "ambient"],
            Self::RcCircuit => &["resistance", "capacitance", "v_source"],
            Self::VanDerPol => &["mu"],
            Self::SpringMassDamper => &["mass", "damping", "stiffness"],
            Self::Pendulum => &["length", "gravity", "damping"],
        }
    }

    /// This model's state components, in order — what `y0` means.
    ///
    /// Present because the state's *units and convention* are not guessable
    /// from the model's name, and getting them wrong produces a plausible run
    /// rather than an error. The SIR family is the sharp case: it is written
    /// over population **fractions**, so an initial state of `[990, 10, 0]`
    /// head-counts does not model a town of 1000 people — it diverges, since
    /// the `β·s·i` term is quadratic. (This module's own test suite made
    /// exactly that mistake first, which is why this method exists.)
    #[must_use]
    pub const fn state(self) -> &'static [&'static str] {
        match self
        {
            Self::LogisticGrowth => &["population"],
            Self::LotkaVolterra => &["prey", "predator"],
            Self::Sir => &[
                "susceptible-fraction",
                "infected-fraction",
                "recovered-fraction",
            ],
            Self::Seir => &[
                "susceptible-fraction",
                "exposed-fraction",
                "infected-fraction",
                "recovered-fraction",
            ],
            Self::NewtonCooling => &["temperature"],
            Self::RcCircuit => &["capacitor-voltage"],
            Self::VanDerPol => &["position", "velocity"],
            Self::SpringMassDamper => &["position", "velocity"],
            Self::Pendulum => &["angle", "angular-velocity"],
        }
    }

    /// The dimension of this model's state vector — the length `y0` must
    /// have. Always `self.state().len()`.
    #[must_use]
    pub const fn state_dim(self) -> usize {
        self.state().len()
    }
}

impl std::fmt::Display for ModelKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.code())
    }
}

impl std::str::FromStr for ModelKind {
    type Err = UnknownModel;

    fn from_str(s: &str) -> std::result::Result<Self, UnknownModel> {
        Self::from_code(s).ok_or_else(|| UnknownModel(s.to_owned()))
    }
}

/// A name that is not in the catalogue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownModel(pub String);

impl std::fmt::Display for UnknownModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "no model named {:?} in the catalogue", self.0)
    }
}

impl std::error::Error for UnknownModel {}

impl Canonical for ModelKind {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.str(self.code());
    }
}

/// A named model with its parameters bound — a scientific model that has an
/// identity, which is the whole point of this module.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSpec {
    /// Which model.
    pub kind: ModelKind,
    /// Its parameters, positionally, in [`ModelKind::parameters`] order.
    pub params: Vec<f64>,
}

impl ModelSpec {
    /// Bind `params` to `kind`. Arity and parameter validity are checked by
    /// [`instantiate`](Self::instantiate), not here — a spec is just data,
    /// the same contract every other config in this crate follows.
    #[must_use]
    pub fn new(kind: ModelKind, params: impl Into<Vec<f64>>) -> Self {
        Self {
            kind,
            params: params.into(),
        }
    }

    /// Build the runnable system this spec denotes.
    ///
    /// # Errors
    /// [`SimError::InvalidConfig`] if the parameter count does not match the
    /// model's arity, or if `scirust-sim` rejects the parameters themselves
    /// (a negative rate, a zero capacitance, ...). The model's own validation
    /// is reused rather than re-implemented — this adapter must not
    /// second-guess what the domain crate considers a valid rate constant.
    pub fn instantiate(&self) -> Result<Box<dyn System>> {
        let expected = self.kind.parameters();
        if self.params.len() != expected.len()
        {
            return Err(SimError::InvalidConfig(format!(
                "{} takes {} parameters ({}), got {}",
                self.kind,
                expected.len(),
                expected.join(", "),
                self.params.len()
            )));
        }
        let p = &self.params;
        let built: std::result::Result<Box<dyn System>, scirust_sim::SimError> = match self.kind
        {
            ModelKind::LogisticGrowth => LogisticGrowth::new(p[0], p[1]).map(boxed),
            ModelKind::LotkaVolterra => LotkaVolterra::new(p[0], p[1], p[2], p[3]).map(boxed),
            ModelKind::Sir => Sir::new(p[0], p[1]).map(boxed),
            ModelKind::Seir => Seir::new(p[0], p[1], p[2]).map(boxed),
            ModelKind::NewtonCooling => NewtonCooling::new(p[0], p[1]).map(boxed),
            ModelKind::RcCircuit => RcCircuit::new(p[0], p[1], p[2]).map(boxed),
            ModelKind::VanDerPol => VanDerPol::new(p[0]).map(boxed),
            ModelKind::SpringMassDamper => SpringMassDamper::new(p[0], p[1], p[2]).map(boxed),
            ModelKind::Pendulum => Pendulum::new(p[0], p[1], p[2]).map(boxed),
        };
        built.map_err(|e| SimError::InvalidConfig(format!("{}: {e}", self.kind)))
    }
}

/// `Box` a concrete system as a trait object, so every catalogue arm can share
/// one `Result` type.
fn boxed<S: System + 'static>(s: S) -> Box<dyn System> {
    Box::new(s)
}

impl Canonical for ModelSpec {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.value(&self.kind);
        enc.value(&ExactF64Seq(&self.params));
    }
}

/// The serialized shape of a [`ModelSpec`]: the model's stable code, and its
/// parameters as exact shortest round-trip decimal strings.
///
/// Strings rather than JSON numbers for the same reason
/// [`exact_trajectory`] uses them — `serde_json`'s `f64` round-trip is not
/// bit-exact, and a parameter that came back a bit different would change the
/// spec's content address and make the stored object fail its own integrity
/// check. It also makes the serialized text and the hashed canonical text the
/// same, so the two cannot drift.
#[derive(Serialize, Deserialize)]
struct SpecRepr {
    model: String,
    params: Vec<String>,
}

impl Serialize for ModelSpec {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        SpecRepr {
            model: self.kind.code().to_owned(),
            params: self
                .params
                .iter()
                .map(std::string::ToString::to_string)
                .collect(),
        }
        .serialize(s)
    }
}

impl<'de> Deserialize<'de> for ModelSpec {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let repr = SpecRepr::deserialize(d)?;
        // A name this build does not know is an error, never a fallback: a
        // stored result naming a model that cannot be resolved must not load
        // as some other model.
        let kind = ModelKind::from_code(&repr.model)
            .ok_or_else(|| D::Error::custom(UnknownModel(repr.model.clone()).to_string()))?;
        let params = repr
            .params
            .iter()
            .map(|p| p.parse::<f64>().map_err(D::Error::custom))
            .collect::<std::result::Result<Vec<f64>, _>>()?;
        Ok(ModelSpec { kind, params })
    }
}

/// One run of one catalogued model: which model, from what state, over what
/// span, at what step.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelRun {
    /// The model to integrate.
    pub model: ModelSpec,
    /// The initial state. Its length must be [`ModelKind::state_dim`].
    pub y0: Vec<f64>,
    /// Integration start time.
    pub t0: f64,
    /// Integration end time; must be strictly greater than `t0`.
    pub t_end: f64,
    /// The fixed step size; must be finite and positive.
    pub step: f64,
}

impl ModelRun {
    /// A run specification. Validity is checked by
    /// [`CatalogSimulator::run`], not here.
    #[must_use]
    pub fn new(model: ModelSpec, y0: impl Into<Vec<f64>>, t0: f64, t_end: f64, step: f64) -> Self {
        Self {
            model,
            y0: y0.into(),
            t0,
            t_end,
            step,
        }
    }
}

impl Canonical for ModelRun {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.value(&self.model);
        enc.value(&ExactF64Seq(&self.y0));
        encode_f64(enc, self.t0);
        encode_f64(enc, self.t_end);
        encode_f64(enc, self.step);
    }
}

/// A trajectory that names the model it came from.
///
/// [`crate::ode::Rk4OdeSimulator`] cannot offer this — its model is a closure
/// with no identity to record. Here the model is data, so the result can carry
/// it, and a reader of a stored trajectory can check which model produced it
/// instead of trusting a descriptor.
#[derive(Debug, Clone, PartialEq)]
pub struct ModeledTrajectory {
    /// The model, with its parameters, that produced `trajectory`.
    pub model: ModelSpec,
    /// The `(t, y(t))` samples, including the initial condition.
    pub trajectory: Trajectory,
}

impl Canonical for ModeledTrajectory {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.value(&self.model);
        let samples: Vec<Sample<'_>> = self
            .trajectory
            .iter()
            .map(|(t, y)| Sample { t: *t, y })
            .collect();
        enc.seq(&samples);
    }
}

/// The body of a stored catalogue run: the model, its trajectory, the
/// determinism level the backend realized, and the seed it ran under.
///
/// This is what makes a modeled result a **first-class SOS object** rather
/// than a value living only in memory — referenceable, verifiable and
/// garbage-collectable on the same terms as every other scientific object,
/// and, unlike [`crate::stage::TrajectoryBody`], able to answer *which model
/// produced me* from the object alone.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModeledTrajectoryBody {
    /// The model, with its parameters, that produced `trajectory`.
    pub model: ModelSpec,
    /// The `(t, y(t))` samples, stored as exact decimal text.
    #[serde(with = "exact_trajectory")]
    pub trajectory: Trajectory,
    /// The determinism level the backend realized for this run.
    pub level: DeterminismLevel,
    /// The seed the run used.
    pub seed: u64,
}

impl ModeledTrajectoryBody {
    /// Flatten an observation into a storable body.
    #[must_use]
    pub fn from_observation(observation: Observation<ModeledTrajectory>) -> Self {
        let (level, seed) = (observation.level(), observation.seed);
        Self {
            model: observation.output.model,
            trajectory: observation.output.trajectory,
            level,
            seed,
        }
    }

    /// Rebuild the [`Observation`] this body was stored from.
    #[must_use]
    pub fn observation(&self) -> Observation<ModeledTrajectory> {
        Observation::new(
            ModeledTrajectory {
                model: self.model.clone(),
                trajectory: self.trajectory.clone(),
            },
            self.level,
            self.seed,
        )
    }
}

impl Canonical for ModeledTrajectoryBody {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.value(&ModeledTrajectory {
            model: self.model.clone(),
            trajectory: self.trajectory.clone(),
        });
        enc.value(&self.level);
        enc.u64(self.seed);
    }
}

impl Body for ModeledTrajectoryBody {
    const KIND: &'static str = "ModeledTrajectory";
    const SCHEMA_VERSION: u32 = 1;
}

/// One `(t, y)` row, encoded with the crate's exact float encoding — the
/// kernel's canonical encoder is deliberately float-free, so a
/// `Vec<(f64, Vec<f64>)>` has no impl of its own.
struct Sample<'a> {
    t: f64,
    y: &'a [f64],
}

impl Canonical for Sample<'_> {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        encode_f64(enc, self.t);
        enc.value(&ExactF64Seq(self.y));
    }
}

/// The descriptor every [`CatalogSimulator`] reports.
///
/// Fixed rather than caller-supplied, unlike [`crate::ode::Rk4OdeSimulator`]'s:
/// it names the integrator and only the integrator, because the model it
/// integrates is now addressed data rather than something hidden in this
/// string.
const CATALOG_BACKEND: &str = "scirust-sim/catalog-rk4";

/// A real [`Simulate`] backend running any catalogued model on `scirust-sim`'s
/// fixed-step RK4.
///
/// `L3`, for the same reason [`crate::ode::Rk4OdeSimulator`] is: a fixed
/// sequence of scalar `f64` operations with no adaptive branching and no
/// sampling.
#[derive(Debug, Clone, Copy, Default)]
pub struct CatalogSimulator;

impl CatalogSimulator {
    /// The catalogue backend. It takes no configuration: everything that
    /// distinguishes one run from another lives in the [`ModelRun`], which is
    /// the whole point of this module.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Simulate for CatalogSimulator {
    type Config = ModelRun;
    type Output = ModeledTrajectory;

    fn descriptor(&self) -> SimDescriptor {
        SimDescriptor::new(CATALOG_BACKEND, sos_core::SemVer::new(1, 0, 0))
    }

    fn level(&self) -> DeterminismLevel {
        DeterminismLevel::L3
    }

    fn run(&self, config: &ModelRun, seed: u64) -> Result<Observation<ModeledTrajectory>> {
        let system = config.model.instantiate()?;
        // `scirust-sim`'s own `validate_run` already rejects a mismatched
        // state length, non-finite entries, a degenerate time span and a
        // non-positive step — so those checks are *not* duplicated here. The
        // one thing worth adding is naming the model in the message, since a
        // bare "expected 3, got 2" is much less useful once a manifest can
        // choose which model is running.
        let run = Dynamic(system);
        let traj = simulate(&run, &config.y0, config.t0, config.t_end, config.step)
            .map_err(|e| map_sim_error(&config.model, e))?;

        let trajectory: Trajectory = traj.t.into_iter().zip(traj.y).collect();
        Ok(Observation::new(
            ModeledTrajectory {
                model: config.model.clone(),
                trajectory,
            },
            self.level(),
            seed,
        ))
    }
}

/// A `Box<dyn System>` seen as a sized [`System`], because `scirust-sim`'s
/// `simulate` is generic over `S: System` and so cannot take an unsized
/// trait object directly.
struct Dynamic(Box<dyn System>);

impl System for Dynamic {
    fn dim(&self) -> usize {
        self.0.dim()
    }

    fn derivatives(&self, t: f64, y: &[f64], dydt: &mut [f64]) {
        self.0.derivatives(t, y, dydt);
    }
}

/// Map `scirust-sim`'s error onto [`SimError`]'s two-variant contract —
/// rejected before compute began, versus failed while running — naming the
/// model in either case.
fn map_sim_error(model: &ModelSpec, e: scirust_sim::SimError) -> SimError {
    use scirust_sim::SimError as E;
    match e
    {
        E::BadInput(_) | E::DimMismatch { .. } =>
        {
            SimError::InvalidConfig(format!("{}: {e}", model.kind))
        },
        E::NonFinite { .. } | E::StepUnderflow { .. } =>
        {
            SimError::Backend(format!("{}: {e}", model.kind))
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_sim::ecology::{LogisticGrowth as Lg, LotkaVolterra as Lv};
    use scirust_sim::electrical::RcCircuit as Rc;
    use scirust_sim::mechanics::SpringMassDamper as Smd;
    use scirust_sim::thermal::NewtonCooling as Nc;
    use sos_simulation::Vcr;

    fn run(config: &ModelRun) -> ModeledTrajectory {
        CatalogSimulator::new()
            .run(config, 0)
            .expect("a valid run must integrate")
            .output
    }

    fn final_state(config: &ModelRun) -> Vec<f64> {
        run(config).trajectory.last().unwrap().1.clone()
    }

    // ---- real physics, checked against scirust-sim's own oracles ----------
    //
    // Every model below is one `scirust-sim` validates against an analytic
    // solution or a conservation law. Those oracles are reused here, so these
    // tests check that the catalogue really integrates the model it names —
    // not that this adapter agrees with itself.

    #[test]
    fn logistic_growth_matches_its_closed_form() {
        let (rate, capacity, x0, t_end) = (0.8, 100.0, 5.0, 6.0);
        let config = ModelRun::new(
            ModelSpec::new(ModelKind::LogisticGrowth, [rate, capacity]),
            [x0],
            0.0,
            t_end,
            0.001,
        );
        let exact = Lg::new(rate, capacity).unwrap().exact(x0, t_end).unwrap();
        let got = final_state(&config)[0];
        assert!((got - exact).abs() < 1e-6, "{got} vs exact {exact}");
        // And it is genuinely on the way to carrying capacity, not stuck:
        // the closed form puts x(6) at ~86.5, well past x0 = 5.
        assert!(got > 80.0 && got < capacity, "{got}");
    }

    #[test]
    fn newton_cooling_matches_its_closed_form() {
        let (rate, ambient, t0_temp, t_end) = (0.4, 20.0, 90.0, 5.0);
        let config = ModelRun::new(
            ModelSpec::new(ModelKind::NewtonCooling, [rate, ambient]),
            [t0_temp],
            0.0,
            t_end,
            0.001,
        );
        let exact = Nc::new(rate, ambient)
            .unwrap()
            .exact(t0_temp, t_end)
            .unwrap();
        let got = final_state(&config)[0];
        assert!((got - exact).abs() < 1e-6, "{got} vs exact {exact}");
        assert!(got > ambient, "cooling approaches ambient from above");
    }

    #[test]
    fn an_rc_circuit_matches_its_closed_form() {
        let (r, c, vs, v0, t_end) = (1000.0, 1e-5, 5.0, 0.0, 0.05);
        let config = ModelRun::new(
            ModelSpec::new(ModelKind::RcCircuit, [r, c, vs]),
            [v0],
            0.0,
            t_end,
            1e-6,
        );
        let exact = Rc::new(r, c, vs).unwrap().exact(v0, t_end).unwrap();
        let got = final_state(&config)[0];
        assert!((got - exact).abs() < 1e-6, "{got} vs exact {exact}");
    }

    #[test]
    fn lotka_volterra_conserves_its_first_integral() {
        // The oracle for a model with no closed-form solution: the orbit is a
        // level set of V(x, y), so a real integration keeps V nearly fixed.
        let params = [1.1, 0.4, 0.1, 0.4];
        let y0 = [10.0, 5.0];
        let model = Lv::new(params[0], params[1], params[2], params[3]).unwrap();
        let v0 = model.first_integral(&y0).unwrap();

        let out = run(&ModelRun::new(
            ModelSpec::new(ModelKind::LotkaVolterra, params),
            y0,
            0.0,
            20.0,
            0.001,
        ));
        for (t, y) in &out.trajectory
        {
            let v = model.first_integral(y).unwrap();
            assert!((v - v0).abs() < 1e-4, "V drifted to {v} from {v0} at t={t}");
        }
        // And the populations actually moved — a frozen state would also
        // conserve V, so the invariant alone is not enough. Measured over the
        // whole orbit, not endpoint-to-endpoint: the orbit is periodic, so a
        // real run can return close to where it started.
        let prey: Vec<f64> = out.trajectory.iter().map(|(_, y)| y[0]).collect();
        let swing = prey.iter().copied().fold(f64::MIN, f64::max)
            - prey.iter().copied().fold(f64::MAX, f64::min);
        assert!(
            swing > 5.0,
            "the prey population barely moved: swing {swing}"
        );
    }

    #[test]
    fn an_undamped_oscillator_conserves_energy() {
        let params = [1.0, 0.0, 4.0]; // mass, damping = 0, stiffness
        let y0 = [1.0, 0.0];
        let model = Smd::new(params[0], params[1], params[2]).unwrap();
        let e0 = model.energy(&y0).unwrap();

        let out = run(&ModelRun::new(
            ModelSpec::new(ModelKind::SpringMassDamper, params),
            y0,
            0.0,
            25.0,
            0.001,
        ));
        for (t, y) in &out.trajectory
        {
            let e = model.energy(y).unwrap();
            assert!(
                (e - e0).abs() < 1e-6,
                "energy drifted to {e} from {e0} at t={t}"
            );
        }
    }

    #[test]
    fn a_damped_oscillator_loses_energy_monotonically() {
        // The companion check: with damping the same model must *not*
        // conserve energy, so the test above is measuring the physics rather
        // than a quantity this code happens to hold fixed.
        let params = [1.0, 0.5, 4.0];
        let y0 = [1.0, 0.0];
        let model = Smd::new(params[0], params[1], params[2]).unwrap();
        let out = run(&ModelRun::new(
            ModelSpec::new(ModelKind::SpringMassDamper, params),
            y0,
            0.0,
            10.0,
            0.001,
        ));
        let e0 = model.energy(&y0).unwrap();
        let e_end = model.energy(&out.trajectory.last().unwrap().1).unwrap();
        assert!(e_end < e0 * 0.01, "{e_end} should be far below {e0}");
    }

    #[test]
    fn an_epidemic_conserves_its_population() {
        // RK4 preserves linear invariants exactly, so the total is a sharp
        // check. Note the state is *fractions* — see `ModelKind::state`.
        for (kind, params, y0) in [
            (ModelKind::Sir, vec![0.6, 0.2], vec![0.99, 0.01, 0.0]),
            (
                ModelKind::Seir,
                vec![0.6, 0.3, 0.2],
                vec![0.99, 0.0, 0.01, 0.0],
            ),
        ]
        {
            let total: f64 = y0.iter().sum();
            let out = run(&ModelRun::new(
                ModelSpec::new(kind, params),
                y0,
                0.0,
                120.0,
                0.01,
            ));
            for (t, y) in &out.trajectory
            {
                let n: f64 = y.iter().sum();
                assert!((n - total).abs() < 1e-12, "{kind}: N = {n} at t = {t}");
            }
            // The epidemic actually ran: R0 = beta/gamma = 3, so most of the
            // susceptible fraction should be gone by the end.
            let last = out.trajectory.last().unwrap().1.clone();
            assert!(last[0] < 0.5, "{kind}: no epidemic happened: {last:?}");
            assert!(
                *last.last().unwrap() > 0.4,
                "{kind}: nobody recovered: {last:?}"
            );
        }
    }

    #[test]
    fn van_der_pol_reaches_its_limit_cycle_from_either_side() {
        // The defining property: trajectories starting inside and outside the
        // cycle both converge to the same amplitude (~2 for small mu).
        let amplitude_from = |x0: f64| {
            let out = run(&ModelRun::new(
                ModelSpec::new(ModelKind::VanDerPol, [1.0]),
                [x0, 0.0],
                0.0,
                60.0,
                0.001,
            ));
            out.trajectory
                .iter()
                .skip(out.trajectory.len() * 3 / 4)
                .map(|(_, y)| y[0].abs())
                .fold(0.0_f64, f64::max)
        };
        let (small, large) = (amplitude_from(0.1), amplitude_from(4.0));
        assert!((small - large).abs() < 0.05, "{small} vs {large}");
        assert!((1.8..2.3).contains(&small), "limit cycle amplitude {small}");
    }

    // ---- identity: the thing this module exists for ----------------------

    #[test]
    fn the_model_is_part_of_the_content_address() {
        // The failure Rk4OdeSimulator cannot prevent: two different physical
        // systems, same integrator, same everything else. Here they cannot
        // collide, because the model is data that hashes.
        let a = ModelRun::new(
            ModelSpec::new(ModelKind::LogisticGrowth, [0.5, 10.0]),
            [1.0],
            0.0,
            1.0,
            0.01,
        );
        let mut b = a.clone();
        b.model.kind = ModelKind::NewtonCooling;
        assert_ne!(a.canonical_bytes(), b.canonical_bytes());
    }

    #[test]
    fn parameters_are_part_of_the_content_address() {
        let base = ModelSpec::new(ModelKind::Sir, [0.6, 0.2]);
        assert_eq!(base.canonical_bytes(), base.clone().canonical_bytes());
        for other in [
            ModelSpec::new(ModelKind::Sir, [0.6, 0.200_000_1]),
            ModelSpec::new(ModelKind::Sir, [0.2, 0.6]),
            ModelSpec::new(ModelKind::Sir, [0.6]),
        ]
        {
            assert_ne!(base.canonical_bytes(), other.canonical_bytes(), "{other:?}");
        }
    }

    #[test]
    fn every_catalogue_entry_has_a_distinct_stable_address() {
        for (i, a) in ModelKind::ALL.iter().enumerate()
        {
            for b in &ModelKind::ALL[i + 1..]
            {
                assert_ne!(a.canonical_bytes(), b.canonical_bytes(), "{a} vs {b}");
            }
        }
    }

    #[test]
    fn a_model_name_round_trips_through_text() {
        // The half of `sos run` that was missing: a manifest can write the
        // name, and it resolves back to exactly one model.
        for kind in ModelKind::ALL
        {
            assert_eq!(ModelKind::from_code(kind.code()), Some(*kind));
            assert_eq!(kind.code().parse::<ModelKind>().unwrap(), *kind);
            assert_eq!(kind.to_string(), kind.code());
        }
        assert_eq!(ModelKind::ALL.len(), 9, "the catalogue is a fixed size");
    }

    #[test]
    fn an_unknown_model_name_is_an_error_not_a_default() {
        assert_eq!(ModelKind::from_code("epidemiology/sirs"), None);
        let err = "not-a-model".parse::<ModelKind>().unwrap_err();
        assert_eq!(err, UnknownModel("not-a-model".to_owned()));
        assert!(err.to_string().contains("not-a-model"));
    }

    #[test]
    fn the_output_carries_the_model_that_produced_it() {
        let spec = ModelSpec::new(ModelKind::LogisticGrowth, [0.5, 10.0]);
        let out = run(&ModelRun::new(spec.clone(), [1.0], 0.0, 1.0, 0.01));
        assert_eq!(out.model, spec);
    }

    #[test]
    fn the_descriptor_names_the_integrator_and_not_the_model() {
        // Deliberate: the model no longer hides in this string.
        let d = CatalogSimulator::new().descriptor();
        assert_eq!(d.name, CATALOG_BACKEND);
        assert!(
            !ModelKind::ALL.iter().any(|k| d.name.contains(k.code())),
            "the descriptor must not encode a model"
        );
    }

    // ---- validation -------------------------------------------------------

    #[test]
    fn a_wrong_parameter_count_is_an_error_that_names_what_was_expected() {
        // `Box<dyn System>` is not `Debug`, so no `unwrap_err` here.
        let Err(err) = ModelSpec::new(ModelKind::LotkaVolterra, [1.0, 2.0]).instantiate()
        else
        {
            panic!("a wrong parameter count must be rejected");
        };
        assert!(matches!(err, SimError::InvalidConfig(_)));
        let msg = err.to_string();
        assert!(msg.contains("takes 4 parameters"), "{msg}");
        assert!(msg.contains("alpha, beta, delta, gamma"), "{msg}");
        assert!(msg.contains("ecology/lotka-volterra"), "{msg}");
    }

    #[test]
    fn parameters_the_domain_crate_rejects_are_refused_here_too() {
        // The adapter reuses scirust-sim's validation rather than inventing
        // its own notion of a valid rate constant.
        for spec in [
            ModelSpec::new(ModelKind::LogisticGrowth, [0.5, -1.0]),
            ModelSpec::new(ModelKind::Sir, [-0.6, 0.2]),
            ModelSpec::new(ModelKind::RcCircuit, [0.0, 1e-5, 5.0]),
            ModelSpec::new(ModelKind::SpringMassDamper, [0.0, 0.1, 1.0]),
        ]
        {
            assert!(
                matches!(spec.instantiate(), Err(SimError::InvalidConfig(_))),
                "{spec:?} must be rejected"
            );
        }
    }

    #[test]
    fn a_state_vector_of_the_wrong_length_is_an_error() {
        let sim = CatalogSimulator::new();
        for kind in ModelKind::ALL
        {
            let params = vec![0.5; kind.parameters().len()];
            // One component too few for this model, whatever its dimension.
            let y0 = vec![1.0; kind.state_dim() - 1];
            let config = ModelRun::new(ModelSpec::new(*kind, params), y0, 0.0, 1.0, 0.01);
            let err = sim.run(&config, 0).unwrap_err();
            assert!(matches!(err, SimError::InvalidConfig(_)), "{kind}: {err:?}");
        }
    }

    #[test]
    fn state_dim_agrees_with_what_the_model_reports() {
        // The declared dimension is documentation until something checks it
        // against the model itself.
        for kind in ModelKind::ALL
        {
            let params = vec![0.5; kind.parameters().len()];
            let system = ModelSpec::new(*kind, params).instantiate().unwrap();
            assert_eq!(system.dim(), kind.state_dim(), "{kind}");
        }
    }

    #[test]
    fn a_degenerate_run_is_an_error_not_an_empty_trajectory() {
        let sim = CatalogSimulator::new();
        let spec = ModelSpec::new(ModelKind::LogisticGrowth, [0.5, 10.0]);
        for (t0, t_end, step) in [(0.0, 0.0, 0.01), (1.0, 0.0, 0.01), (0.0, 1.0, 0.0)]
        {
            let config = ModelRun::new(spec.clone(), [1.0], t0, t_end, step);
            assert!(
                matches!(sim.run(&config, 0), Err(SimError::InvalidConfig(_))),
                "[{t0}, {t_end}] @ {step} must be rejected"
            );
        }
    }

    // ---- the stored form -------------------------------------------------

    #[test]
    fn a_stored_body_keeps_its_model_and_its_floats_exactly() {
        let config = ModelRun::new(
            ModelSpec::new(ModelKind::Pendulum, [1.0, 9.807, 0.05]),
            [0.4, 0.0],
            0.0,
            2.0,
            0.01,
        );
        let body = ModeledTrajectoryBody::from_observation(
            CatalogSimulator::new().run(&config, 11).unwrap(),
        );
        let json = serde_json::to_string(&body).unwrap();
        // Parameters and samples alike are quoted decimal text, not JSON
        // numbers — see the type's own docs on why.
        assert!(json.contains("\"9.807\""), "{}", &json[..200]);
        assert!(json.contains("mechanics/pendulum"), "{}", &json[..200]);

        let back: ModeledTrajectoryBody = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model, body.model);
        assert_eq!(back.seed, 11);
        assert_eq!(back.level, DeterminismLevel::L3);
        assert_eq!(back.canonical_bytes(), body.canonical_bytes());
        for (i, ((t0, y0), (t1, y1))) in body.trajectory.iter().zip(&back.trajectory).enumerate()
        {
            assert_eq!(t0.to_bits(), t1.to_bits(), "t at {i}");
            for (a, b) in y0.iter().zip(y1)
            {
                assert_eq!(a.to_bits(), b.to_bits(), "y at {i}");
            }
        }
    }

    #[test]
    fn a_stored_model_this_build_cannot_resolve_fails_to_load() {
        // Never a fallback: an object naming a model this build does not have
        // must not deserialize as some other model.
        let json = r#"{"model":{"model":"epidemiology/sirs","params":["0.5"]},
                       "trajectory":[["0",["1"]]],"level":"L3","seed":0}"#;
        let err = serde_json::from_str::<ModeledTrajectoryBody>(json).unwrap_err();
        assert!(err.to_string().contains("epidemiology/sirs"), "{err}");
    }

    #[test]
    fn the_body_round_trips_through_an_observation() {
        let config = ModelRun::new(
            ModelSpec::new(ModelKind::VanDerPol, [1.0]),
            [1.0, 0.0],
            0.0,
            1.0,
            0.01,
        );
        let observed = CatalogSimulator::new().run(&config, 5).unwrap();
        let body = ModeledTrajectoryBody::from_observation(observed.clone());
        let rebuilt = body.observation();
        assert_eq!(rebuilt.output, observed.output);
        assert_eq!(rebuilt.level(), observed.level());
        assert_eq!(rebuilt.seed, observed.seed);
    }

    // ---- determinism and replay ------------------------------------------

    #[test]
    fn the_run_is_l3_and_bit_reproducible() {
        let sim = CatalogSimulator::new();
        let config = ModelRun::new(
            ModelSpec::new(ModelKind::LotkaVolterra, [1.1, 0.4, 0.1, 0.4]),
            [10.0, 5.0],
            0.0,
            5.0,
            0.01,
        );
        assert_eq!(sim.level(), DeterminismLevel::L3);
        let a = sim.run(&config, 7).unwrap();
        let b = sim.run(&config, 7).unwrap();
        assert_eq!(a.output.canonical_bytes(), b.output.canonical_bytes());
        assert_eq!(a.level(), DeterminismLevel::L3);
        assert_eq!(a.seed, 7);
    }

    #[test]
    fn the_vcr_treats_two_models_as_two_measurements() {
        let sim = CatalogSimulator::new();
        let mut vcr = Vcr::new();
        let logistic = ModelRun::new(
            ModelSpec::new(ModelKind::LogisticGrowth, [0.5, 10.0]),
            [1.0],
            0.0,
            1.0,
            0.01,
        );
        let cooling = ModelRun::new(
            ModelSpec::new(ModelKind::NewtonCooling, [0.5, 10.0]),
            [1.0],
            0.0,
            1.0,
            0.01,
        );

        assert!(!vcr.observe(&sim, &logistic, 0).unwrap().replayed);
        assert!(vcr.observe(&sim, &logistic, 0).unwrap().replayed);
        // Same parameters, same state, same span — a different model.
        assert!(!vcr.observe(&sim, &cooling, 0).unwrap().replayed);
        assert_eq!(vcr.len(), 2);
    }
}
