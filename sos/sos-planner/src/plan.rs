//! [`Candidate`], [`RankedDesign`], [`StopVerdict`], and the [`Plan`] object.

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{Author, Body, Object, ObjectId};

use crate::estimate::{Cost, Estimate};
use crate::policy::UtilityPolicy;

/// A candidate experiment to plan over: the design's object id, its EIG
/// [`Estimate`], and its [`Cost`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Candidate {
    /// The candidate design / experiment object.
    pub experiment: ObjectId,
    /// The expected information gain it is estimated to yield.
    pub eig: Estimate,
    /// What it costs to run.
    pub cost: Cost,
}

impl Candidate {
    /// Construct a candidate.
    #[must_use]
    pub fn new(experiment: ObjectId, eig: Estimate, cost: Cost) -> Self {
        Self {
            experiment,
            eig,
            cost,
        }
    }
}

/// A candidate scored by the planner: its EIG, cost, and computed utility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RankedDesign {
    /// The design / experiment object.
    pub experiment: ObjectId,
    /// Its EIG estimate.
    pub eig: Estimate,
    /// Its cost.
    pub cost: Cost,
    /// Its fixed-point utility under the plan's policy.
    pub utility: i64,
}

impl Canonical for RankedDesign {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.value(&self.experiment);
        enc.value(&self.eig);
        enc.value(&self.cost);
        enc.i64(self.utility);
    }
}

/// The planner's verdict: run a specific experiment, or stop because no candidate
/// is worth running.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopVerdict {
    /// Run this experiment next — the top-utility design that clears the EIG floor
    /// (`ξ*`).
    Recommend(ObjectId),
    /// **Information is exhausted**: no candidate's EIG clears the floor, so no
    /// experiment can teach more than `ε`. An honest first-class output, not a
    /// silent loop-forever (SDE §05.7).
    InformationExhausted,
}

/// A recommendation the planner produces (SDE §05.6): the candidate designs
/// ranked by utility — each annotated with its EIG (and *that* estimate's
/// uncertainty), cost, and utility — plus the stopping verdict and the policy
/// used. Because every quantity is recorded, the choice is **defensible**: "B
/// buys 0.9 bits vs 0.05, and 0.45 utility vs 0.05."
///
/// A `Plan` is a content-addressed [`Body`], so the research plan is itself a
/// citable object in the graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Plan {
    /// The utility policy used to rank.
    pub policy: UtilityPolicy,
    /// The information-exhaustion floor `ε`, in millibits.
    pub eig_floor_milli: i64,
    /// Candidate designs, best-utility first (ties broken by object id).
    pub ranked: Vec<RankedDesign>,
    /// Whether to run `ξ*` or stop.
    pub verdict: StopVerdict,
}

impl Plan {
    /// The top-ranked design, if any.
    #[must_use]
    pub fn best(&self) -> Option<&RankedDesign> {
        self.ranked.first()
    }

    /// Whether the plan recommends running an experiment (vs stopping).
    #[must_use]
    pub fn recommends(&self) -> bool {
        matches!(self.verdict, StopVerdict::Recommend(_))
    }
}

impl Canonical for StopVerdict {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        match self
        {
            Self::Recommend(id) =>
            {
                enc.str("recommend");
                enc.value(id);
            },
            Self::InformationExhausted => enc.str("information-exhausted"),
        }
    }
}

impl Canonical for Plan {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.value(&self.policy);
        enc.i64(self.eig_floor_milli);
        enc.seq(&self.ranked);
        enc.value(&self.verdict);
    }
}

impl Body for Plan {
    /// `"ExperimentPlan"`, not `"Plan"`.
    ///
    /// `sos-workflow`'s [`Plan`](../../sos_workflow/struct.Plan.html) — a
    /// validated DAG of stages — also declares a [`Body`] kind, and both
    /// originally called themselves `"Plan"`. Two structurally different types
    /// claiming one kind is incoherent in a content-addressed store: the
    /// store's `KindMismatch` guard compares declared kind names, so a
    /// collision defeats exactly the check that exists to stop a record being
    /// decoded as the wrong type. `sos-cli`'s `sos verify` found it by trying
    /// to read a stored workflow plan as this one.
    ///
    /// This is the side that renamed because the workflow `Plan` is the one
    /// RFC-0002 §08 calls *the* plan, while this is a **recommendation** — a
    /// ranking of candidate experiments and a stopping verdict. Naming it for
    /// what it is costs nothing and removes the ambiguity.
    const KIND: &'static str = "ExperimentPlan";
    const SCHEMA_VERSION: u32 = 1;
}

/// Seal a [`Plan`] as a storable `Object<Plan>`.
#[must_use]
pub fn seal_plan(plan: Plan, author: Author) -> Object<Plan> {
    Object::builder(plan).author(author).seal()
}
