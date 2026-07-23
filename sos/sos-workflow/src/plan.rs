//! The plan model: [`StageId`], [`Stage`], and the immutable [`Plan`] DAG with
//! deterministic topological scheduling.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{Digest, HashAlgo, ObjectId};

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
}

impl Stage {
    /// Construct a stage, normalizing `inputs` and `deps` to sorted, deduplicated
    /// sets so a stage's identity and schedule position do not depend on order.
    #[must_use]
    pub fn new(
        id: StageId,
        descriptor: StageDescriptor,
        mut inputs: Vec<ObjectId>,
        config_hash: Digest,
        seed: u64,
        mut deps: Vec<StageId>,
    ) -> Self {
        inputs.sort_unstable();
        inputs.dedup();
        deps.sort();
        deps.dedup();
        Self {
            id,
            descriptor,
            inputs,
            config_hash,
            seed,
            deps,
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
    }
}

/// An immutable workflow: a validated DAG of [`Stage`]s.
///
/// Constructed through [`Plan::new`], which rejects duplicate ids, dangling
/// dependencies, and cycles — so a `Plan` is always a schedulable DAG.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    stages: Vec<Stage>,
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
        let mut ordered: Vec<&Stage> = self.stages.iter().collect();
        ordered.sort_by(|a, b| a.id.cmp(&b.id));
        let mut enc = CanonicalEncoder::new();
        enc.seq(&ordered);
        HashAlgo::Sha256.hash(PLAN_DOMAIN, &enc.finish())
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
