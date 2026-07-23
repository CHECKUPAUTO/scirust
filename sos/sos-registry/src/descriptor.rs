//! Plugin descriptors: [`PluginDescriptor`], [`Role`], [`PluginName`],
//! [`DomainTag`], and [`VersionReq`].

use core::fmt;

use serde::{Deserialize, Serialize};
use sos_core::hash::Digest;
use sos_core::{DeterminismLevel, SemVer};

use crate::capability::{Capability, CapabilitySet};

/// A plugin's stable name, e.g. `"sos-scirust/symreg"`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PluginName(pub String);

impl fmt::Display for PluginName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A domain a plugin applies to, e.g. `"physics"` or `"chemistry"`. An empty
/// [`PluginDescriptor::domains`] list means "applies to any domain".
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DomainTag(pub String);

impl fmt::Display for DomainTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// The engine syscall or discovery stage a plugin implements — the "slot" it
/// plugs into. Covers the SOS engine syscalls (RFC-0002 §02) and the SDE
/// discovery stages (RFC-0001 §07), with [`Role::Custom`] for extension.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Role {
    /// Knowledge Engine syscall.
    Knowledge,
    /// Reasoning Engine syscall.
    Reasoning,
    /// Curiosity Engine syscall.
    Curiosity,
    /// Simulation Engine syscall (effectful).
    Simulation,
    /// Planning Engine syscall.
    Planning,
    /// Publication Engine syscall.
    Publication,
    /// Cognitive-memory syscall.
    Memory,
    /// Discovery stage: hypothesis generation.
    HypothesisGenerator,
    /// Discovery stage: prediction.
    Predictor,
    /// Discovery stage: experiment design.
    ExperimentDesigner,
    /// Discovery stage: execution (effectful).
    Executor,
    /// Discovery stage: evidence extraction.
    EvidenceExtractor,
    /// Discovery stage: statistical evaluation.
    StatisticalEvaluator,
    /// Discovery stage: hypothesis ranking.
    HypothesisRanker,
    /// Discovery stage: theory revision.
    TheoryReviser,
    /// A plugin-specific role, named by string.
    Custom(String),
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::Custom(s) => write!(f, "custom:{s}"),
            other => write!(f, "{other:?}"),
        }
    }
}

/// A requirement over a plugin's semantic version, used when resolving.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VersionReq {
    /// Any version.
    Any,
    /// Exactly this version.
    Exact(SemVer),
    /// This version or newer.
    AtLeast(SemVer),
    /// Same major version and at least this version (caret semantics).
    Compatible(SemVer),
}

impl VersionReq {
    /// Whether `version` satisfies this requirement.
    #[must_use]
    pub fn matches(&self, version: &SemVer) -> bool {
        match self
        {
            Self::Any => true,
            Self::Exact(r) => version == r,
            Self::AtLeast(r) => version >= r,
            Self::Compatible(r) => version.major == r.major && version >= r,
        }
    }
}

/// A content-pinned description of a plugin: what it is, what it needs, and how
/// reproducible it is.
///
/// Construct via [`PluginDescriptor::new`] (name + version + content hash +
/// [`Role`] are required) and refine with the builder setters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginDescriptor {
    /// Stable plugin name.
    pub name: PluginName,
    /// The plugin's semantic version.
    pub version: SemVer,
    /// Content hash of the plugin artifact — pins the exact code, so a
    /// re-resolution to a different hash is a detected drift.
    pub content_hash: Digest,
    /// The role (engine syscall / discovery stage) this plugin implements.
    pub role: Role,
    /// The determinism level this plugin realizes.
    pub level: DeterminismLevel,
    /// The capabilities this plugin needs to run.
    pub capabilities: CapabilitySet,
    /// The domains this plugin applies to; empty means "any domain".
    pub domains: Vec<DomainTag>,
}

impl PluginDescriptor {
    /// Create a descriptor. `role` is required because a plugin with no role
    /// plugs into nothing; the level defaults to [`DeterminismLevel::L3`] and
    /// capabilities/domains default to empty (needs nothing, applies to all).
    #[must_use]
    pub fn new(name: impl Into<String>, version: SemVer, content_hash: Digest, role: Role) -> Self {
        Self {
            name: PluginName(name.into()),
            version,
            content_hash,
            role,
            level: DeterminismLevel::L3,
            capabilities: CapabilitySet::new(),
            domains: Vec::new(),
        }
    }

    /// Set the determinism level this plugin realizes.
    #[must_use]
    pub fn level(mut self, level: DeterminismLevel) -> Self {
        self.level = level;
        self
    }

    /// Declare a capability this plugin needs.
    #[must_use]
    pub fn needs(mut self, cap: Capability) -> Self {
        self.capabilities.insert(cap);
        self
    }

    /// Declare a domain this plugin applies to (repeatable).
    #[must_use]
    pub fn domain(mut self, domain: impl Into<String>) -> Self {
        self.domains.push(DomainTag(domain.into()));
        self
    }

    /// Whether this plugin applies to `domain` (true if it lists `domain`, or
    /// lists no domains at all — meaning "any").
    #[must_use]
    pub fn applies_to(&self, domain: &DomainTag) -> bool {
        self.domains.is_empty() || self.domains.contains(domain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_req_semantics() {
        let v = SemVer::new(1, 2, 3);
        assert!(VersionReq::Any.matches(&v));
        assert!(VersionReq::Exact(SemVer::new(1, 2, 3)).matches(&v));
        assert!(!VersionReq::Exact(SemVer::new(1, 2, 4)).matches(&v));
        assert!(VersionReq::AtLeast(SemVer::new(1, 0, 0)).matches(&v));
        assert!(!VersionReq::AtLeast(SemVer::new(2, 0, 0)).matches(&v));
        assert!(VersionReq::Compatible(SemVer::new(1, 0, 0)).matches(&v));
        assert!(!VersionReq::Compatible(SemVer::new(2, 0, 0)).matches(&v));
        // Compatible requires the same major AND version >= requirement.
        assert!(!VersionReq::Compatible(SemVer::new(1, 3, 0)).matches(&v));
    }

    #[test]
    fn applies_to_treats_empty_as_any() {
        let hash = sos_core::HashAlgo::default().hash(b"p", b"x");
        let any = PluginDescriptor::new("p", SemVer::new(0, 1, 0), hash, Role::Planning);
        assert!(any.applies_to(&DomainTag("anything".into())));

        let phys = any.clone().domain("physics");
        assert!(phys.applies_to(&DomainTag("physics".into())));
        assert!(!phys.applies_to(&DomainTag("finance".into())));
    }
}
