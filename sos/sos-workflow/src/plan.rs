//! The plan model: [`StageId`], [`Stage`], and the immutable [`Plan`] DAG with
//! deterministic topological scheduling.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{Body, Digest, HashAlgo, ObjectId};

use crate::cache::CacheKey;
use crate::descriptor::StageDescriptor;
use crate::error::{Result, WorkflowError};

/// Domain-separation prefix for the plan digest.
const PLAN_DOMAIN: &[u8] = b"sos-workflow:plan:v1";

/// A stable identity for a stage within a plan (its manifest name, e.g.
/// `"hypothesis"`). Scheduling ties are broken by `StageId` ordering, so the
/// schedule is fully deterministic.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct StageId(pub String);

impl StageId {
    /// Construct a stage id.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl core::fmt::Display for StageId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Canonical for StageId {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.str(&self.0);
    }
}

/// One node of a workflow: a pure pass that reads `inputs` and appends objects,
/// identified for memoization by its [`CacheKey`] (RFC-0002 §08.1).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stage {
    /// This stage's id within the plan.
    pub id: StageId,
    /// The plugin identity (name + version + content hash).
    pub descriptor: StageDescriptor,
    /// The exact object ids this stage consumes (stored as a sorted set).
    pub inputs: Vec<ObjectId>,
    /// A content hash of the stage configuration.
    pub config_hash: Digest,
    /// The mandatory seed — a stage you cannot reseed is not reproducible.
    pub seed: u64,
    /// The stages this one depends on (must run after; stored as a sorted set).
    pub deps: Vec<StageId>,
    /// The stages whose **outputs this one reads** (stored as a sorted set).
    ///
    /// [`deps`](Self::deps) is ordering only — "must run after" — and says
    /// nothing about data. Until this field existed there was no way to
    /// express dataflow between stages at all: [`inputs`](Self::inputs) takes
    /// literal [`ObjectId`]s, and an upstream stage's output ids do not exist
    /// until it has run, so a plan could sequence its stages but never feed
    /// one into the next. Every stage's output recorded zero provenance
    /// parents as a result — a workflow engine producing a graph with no edges
    /// between its own nodes.
    ///
    /// Consuming a stage implies depending on it, so an id listed here does
    /// not also need to appear in `deps`; [`Stage::consuming`] folds it in.
    ///
    /// Defaulted on deserialization: a plan written before this field existed
    /// consumed nothing, which is exactly what an empty set means. No stored
    /// plan is invalidated by its arrival.
    #[serde(default)]
    pub consumes: Vec<StageId>,
}

impl Stage {
    /// Construct a stage, normalizing `inputs` and `deps` to sorted, deduplicated
    /// sets so a stage's identity and schedule position do not depend on order.
    ///
    /// Consumes nothing; use [`consuming`](Self::consuming) for a stage that
    /// reads an upstream stage's output.
    #[must_use]
    pub fn new(
        id: StageId,
        descriptor: StageDescriptor,
        inputs: Vec<ObjectId>,
        config_hash: Digest,
        seed: u64,
        deps: Vec<StageId>,
    ) -> Self {
        Self::consuming(id, descriptor, inputs, config_hash, seed, deps, Vec::new())
    }

    /// The same, plus the stages whose outputs this one reads.
    ///
    /// `consumes` implies ordering: every id in it is folded into `deps`, so
    /// a caller never has to state the dependency twice and cannot state it
    /// inconsistently.
    #[must_use]
    pub fn consuming(
        id: StageId,
        descriptor: StageDescriptor,
        mut inputs: Vec<ObjectId>,
        config_hash: Digest,
        seed: u64,
        mut deps: Vec<StageId>,
        mut consumes: Vec<StageId>,
    ) -> Self {
        inputs.sort_unstable();
        inputs.dedup();
        consumes.sort();
        consumes.dedup();
        deps.extend(consumes.iter().cloned());
        deps.sort();
        deps.dedup();
        Self {
            id,
            descriptor,
            inputs,
            config_hash,
            seed,
            deps,
            consumes,
        }
    }

    /// This stage with `resolved` as its inputs — what the scheduler hands to
    /// an executor once every consumed stage's outputs are known.
    ///
    /// Returning a new `Stage` rather than mutating is what keeps memoization
    /// correct: [`cache_key`](Self::cache_key) reads `inputs`, so the key is
    /// computed from the *resolved* set and a stage whose upstream changed
    /// necessarily misses the cache.
    #[must_use]
    pub fn with_resolved_inputs(&self, mut resolved: Vec<ObjectId>) -> Self {
        resolved.sort_unstable();
        resolved.dedup();
        Self {
            inputs: resolved,
            ..self.clone()
        }
    }

    /// This stage's [`CacheKey`] under the given environment digest.
    #[must_use]
    pub fn cache_key(&self, env_digest: &Digest) -> CacheKey {
        CacheKey::compute(
            &self.descriptor,
            &self.inputs,
            &self.config_hash,
            self.seed,
            env_digest,
        )
    }
}

impl Canonical for Stage {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.value(&self.id);
        enc.value(&self.descriptor);
        enc.seq(&self.inputs);
        enc.bytes(self.config_hash.as_bytes());
        enc.u64(self.seed);
        enc.seq(&self.deps);
        enc.seq(&self.consumes);
    }
}

/// An immutable workflow: a validated DAG of [`Stage`]s.
///
/// Constructed through [`Plan::new`], which rejects duplicate ids, dangling
/// dependencies, and cycles — so a `Plan` is always a schedulable DAG.
///
/// A `Plan` is also a storable [`Body`], so the plan a study ran can live in
/// the object store beside the [`RunLedger`](crate::RunLedger) that recorded
/// the run — the same "control flow is data too" argument the ledger already
/// makes. Re-executing a recorded run needs the plan back, and a digest alone
/// cannot supply it.
///
/// **Deserialization re-validates.** It routes through [`Plan::new`] rather
/// than filling the field directly, so the type's invariant survives a
/// round-trip through bytes nobody controls: a stored plan whose stages were
/// tampered into a cycle fails to load rather than becoming a `Plan` that is
/// not a DAG.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Plan {
    stages: Vec<Stage>,
}

impl<'de> Deserialize<'de> for Plan {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> core::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Raw {
            stages: Vec<Stage>,
        }
        let raw = Raw::deserialize(d)?;
        Self::new(raw.stages).map_err(serde::de::Error::custom)
    }
}

impl Canonical for Plan {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        // Sorted by id so the encoding — and therefore [`Plan::digest`] — does
        // not depend on the order stages were supplied in.
        let mut ordered: Vec<&Stage> = self.stages.iter().collect();
        ordered.sort_by(|a, b| a.id.cmp(&b.id));
        enc.seq(&ordered);
    }
}

impl Body for Plan {
    const KIND: &'static str = "Plan";
    const SCHEMA_VERSION: u32 = 1;
}

impl Plan {
    /// Validate `stages` and build the plan.
    ///
    /// # Errors
    /// * [`WorkflowError::DuplicateStage`] if two stages share an id.
    /// * [`WorkflowError::MissingDependency`] if a dependency names no stage.
    /// * [`WorkflowError::Cycle`] if the dependency graph is not acyclic.
    pub fn new(stages: Vec<Stage>) -> Result<Self> {
        let mut ids = BTreeSet::new();
        for s in &stages
        {
            if !ids.insert(s.id.clone())
            {
                return Err(WorkflowError::DuplicateStage(s.id.clone()));
            }
        }
        for s in &stages
        {
            for dep in &s.deps
            {
                if !ids.contains(dep)
                {
                    return Err(WorkflowError::MissingDependency {
                        stage: s.id.clone(),
                        dep: dep.clone(),
                    });
                }
            }
        }
        let plan = Self { stages };
        // A full topological order exists iff the graph is acyclic.
        if plan.topo_order().len() != plan.stages.len()
        {
            return Err(WorkflowError::Cycle);
        }
        Ok(plan)
    }

    /// The stages, in construction order.
    #[must_use]
    pub fn stages(&self) -> &[Stage] {
        &self.stages
    }

    /// Look up a stage by id.
    #[must_use]
    pub fn get(&self, id: &StageId) -> Option<&Stage> {
        self.stages.iter().find(|s| &s.id == id)
    }

    /// The deterministic topological schedule: dependencies before dependents,
    /// ties broken by [`StageId`] ordering. Because stages are pure, any order
    /// consistent with this one yields identical results; this canonical order
    /// makes the *schedule itself* reproducible (recorded in the ledger).
    #[must_use]
    pub fn schedule(&self) -> Vec<StageId> {
        self.topo_order()
    }

    /// A content digest of the whole plan (order-independent over stages), used
    /// to bind a [`crate::RunLedger`] to the plan it ran.
    #[must_use]
    pub fn digest(&self) -> Digest {
        HashAlgo::Sha256.hash(PLAN_DOMAIN, &self.canonical_bytes())
    }

    /// Kahn's algorithm with a sorted ready-set — deterministic, and returns a
    /// partial order (shorter than the stage count) exactly when a cycle exists.
    fn topo_order(&self) -> Vec<StageId> {
        // Remaining dependency count per stage, and the reverse edges.
        let mut remaining: BTreeMap<StageId, usize> = BTreeMap::new();
        let mut dependents: BTreeMap<StageId, Vec<StageId>> = BTreeMap::new();
        for s in &self.stages
        {
            remaining.insert(s.id.clone(), s.deps.len());
            for dep in &s.deps
            {
                dependents
                    .entry(dep.clone())
                    .or_default()
                    .push(s.id.clone());
            }
        }

        // Ready = stages with no remaining dependencies, processed in id order.
        let mut ready: BTreeSet<StageId> = remaining
            .iter()
            .filter(|(_, &n)| n == 0)
            .map(|(id, _)| id.clone())
            .collect();
        let mut order = Vec::with_capacity(self.stages.len());

        while let Some(id) = ready.iter().next().cloned()
        {
            ready.remove(&id);
            order.push(id.clone());
            if let Some(children) = dependents.get(&id)
            {
                for child in children
                {
                    let n = remaining.get_mut(child).expect("dependent is a stage");
                    *n -= 1;
                    if *n == 0
                    {
                        ready.insert(child.clone());
                    }
                }
            }
        }
        order
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plan_round_trips_through_its_stored_form() {
        let plan = Plan::new(vec![stage("b", &["a"]), stage("a", &[])]).unwrap();
        let json = serde_json::to_string(&plan).unwrap();
        let back: Plan = serde_json::from_str(&json).unwrap();
        assert_eq!(back, plan);
        assert_eq!(back.digest(), plan.digest());
        assert_eq!(back.schedule(), plan.schedule());
    }

    #[test]
    fn deserializing_re_validates_rather_than_trusting_the_bytes() {
        // A `Plan` is documented as always being a schedulable DAG. Bytes from
        // a store are not under this crate's control, so loading must re-run
        // the same checks `Plan::new` does — otherwise a tampered record would
        // become a `Plan` that is not a DAG, and every downstream guarantee
        // built on that invariant would quietly rest on nothing.
        let cyclic = r#"{"stages":[
            {"id":"a","descriptor":{"name":"p","version":{"major":1,"minor":0,"patch":0},
             "content_hash":"0000000000000000000000000000000000000000000000000000000000000000"},
             "inputs":[],"config_hash":"0000000000000000000000000000000000000000000000000000000000000000",
             "seed":0,"deps":["b"]},
            {"id":"b","descriptor":{"name":"p","version":{"major":1,"minor":0,"patch":0},
             "content_hash":"0000000000000000000000000000000000000000000000000000000000000000"},
             "inputs":[],"config_hash":"0000000000000000000000000000000000000000000000000000000000000000",
             "seed":0,"deps":["a"]}]}"#;
        let err = serde_json::from_str::<Plan>(cyclic).unwrap_err();
        assert!(err.to_string().contains("cycle"), "{err}");

        // Likewise a dangling dependency.
        let dangling = cyclic.replace(r#""deps":["b"]"#, r#""deps":["ghost"]"#);
        assert!(serde_json::from_str::<Plan>(&dangling).is_err());
    }

    #[test]
    fn the_canonical_encoding_and_the_digest_agree_on_ordering() {
        // `digest` is defined in terms of the `Canonical` impl, so a plan
        // built in either order is the same object — which is what lets a
        // stored plan be found by the `plan_digest` a ledger recorded.
        let forward = Plan::new(vec![stage("a", &[]), stage("b", &["a"])]).unwrap();
        let reverse = Plan::new(vec![stage("b", &["a"]), stage("a", &[])]).unwrap();
        assert_eq!(forward.digest(), reverse.digest());
        assert_eq!(forward.canonical_bytes(), reverse.canonical_bytes());
    }
    use sos_core::SemVer;

    fn digest(tag: &[u8]) -> Digest {
        HashAlgo::default().hash(b"test", tag)
    }

    fn stage(id: &str, deps: &[&str]) -> Stage {
        Stage::new(
            StageId::new(id),
            StageDescriptor::new(id, SemVer::new(1, 0, 0), digest(id.as_bytes())),
            vec![],
            digest(b"cfg"),
            0,
            deps.iter().map(|d| StageId::new(*d)).collect(),
        )
    }

    #[test]
    fn schedule_is_topological_and_deterministic() {
        // Diamond: a -> {b, c} -> d.
        let plan = Plan::new(vec![
            stage("d", &["b", "c"]),
            stage("b", &["a"]),
            stage("c", &["a"]),
            stage("a", &[]),
        ])
        .unwrap();
        let order = plan.schedule();
        assert_eq!(
            order,
            vec![
                StageId::new("a"),
                StageId::new("b"),
                StageId::new("c"),
                StageId::new("d"),
            ]
        );
        assert_eq!(plan.schedule(), order); // deterministic
    }

    #[test]
    fn duplicate_dangling_and_cyclic_plans_are_rejected() {
        assert!(matches!(
            Plan::new(vec![stage("a", &[]), stage("a", &[])]),
            Err(WorkflowError::DuplicateStage(_))
        ));
        assert!(matches!(
            Plan::new(vec![stage("a", &["ghost"])]),
            Err(WorkflowError::MissingDependency { .. })
        ));
        assert!(matches!(
            Plan::new(vec![stage("a", &["b"]), stage("b", &["a"])]),
            Err(WorkflowError::Cycle)
        ));
    }
}
