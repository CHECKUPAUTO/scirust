//! Capabilities and least-privilege authorization.
//!
//! A plugin declares the capabilities it *needs* (network, GPU, filesystem, the
//! right to cause effects); a study *grants* a set of capabilities; and
//! [`authorize`] lets a plugin run only if every capability it needs was
//! granted — refusing by default. This mirrors the signed, least-privilege
//! `ScopeAuthorization` posture of `scirust-discovery`.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::descriptor::PluginDescriptor;
use crate::error::{RegistryError, Result};

/// A capability a plugin may require in order to run.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Capability {
    /// Outbound network access.
    Network,
    /// GPU / accelerator access.
    Gpu,
    /// Filesystem access beyond the object store.
    Filesystem,
    /// The right to cause external effects (run an experiment, touch an
    /// instrument) — the one impure boundary of the system.
    Effectful,
    /// A domain- or plugin-specific capability, named by string.
    Custom(String),
}

impl core::fmt::Display for Capability {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self
        {
            Self::Network => f.write_str("network"),
            Self::Gpu => f.write_str("gpu"),
            Self::Filesystem => f.write_str("filesystem"),
            Self::Effectful => f.write_str("effectful"),
            Self::Custom(s) => write!(f, "custom:{s}"),
        }
    }
}

/// An unordered, deduplicated set of capabilities (sorted iteration for
/// determinism).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilitySet(BTreeSet<Capability>);

impl CapabilitySet {
    /// An empty set.
    #[must_use]
    pub fn new() -> Self {
        Self(BTreeSet::new())
    }

    /// Add a capability, returning `self` for chaining.
    #[must_use]
    pub fn with(mut self, cap: Capability) -> Self {
        self.0.insert(cap);
        self
    }

    /// Add a capability in place.
    pub fn insert(&mut self, cap: Capability) {
        self.0.insert(cap);
    }

    /// Whether the set contains a capability.
    #[must_use]
    pub fn contains(&self, cap: &Capability) -> bool {
        self.0.contains(cap)
    }

    /// Whether the set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The number of capabilities.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// The capabilities in `self` that are **not** in `other`, sorted.
    #[must_use]
    pub fn missing_from(&self, other: &CapabilitySet) -> Vec<Capability> {
        self.0.difference(&other.0).cloned().collect()
    }

    /// Whether every capability in `self` is also in `other`.
    #[must_use]
    pub fn is_subset_of(&self, other: &CapabilitySet) -> bool {
        self.0.is_subset(&other.0)
    }

    /// Iterate the capabilities in sorted order.
    pub fn iter(&self) -> impl Iterator<Item = &Capability> {
        self.0.iter()
    }
}

/// The set of capabilities a study grants to the plugins it runs.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Grant(CapabilitySet);

impl Grant {
    /// A grant of nothing — the least-privilege default. A plugin needing any
    /// capability is denied under this grant.
    #[must_use]
    pub fn new() -> Self {
        Self(CapabilitySet::new())
    }

    /// Grant a capability, returning `self` for chaining.
    #[must_use]
    pub fn allow(mut self, cap: Capability) -> Self {
        self.0.insert(cap);
        self
    }

    /// The granted capability set.
    #[must_use]
    pub fn capabilities(&self) -> &CapabilitySet {
        &self.0
    }
}

/// Authorize a plugin against a grant: `Ok` iff every capability the plugin
/// needs was granted.
///
/// # Errors
/// [`RegistryError::Denied`] listing the missing capabilities if the plugin
/// needs any capability the grant does not include.
pub fn authorize(descriptor: &PluginDescriptor, grant: &Grant) -> Result<()> {
    let missing = descriptor.capabilities.missing_from(grant.capabilities());
    if missing.is_empty()
    {
        Ok(())
    }
    else
    {
        Err(RegistryError::Denied {
            name: descriptor.name.to_string(),
            missing,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_from_computes_the_shortfall() {
        let needs = CapabilitySet::new()
            .with(Capability::Gpu)
            .with(Capability::Network);
        let has = CapabilitySet::new().with(Capability::Gpu);
        assert_eq!(needs.missing_from(&has), vec![Capability::Network]);
        assert!(CapabilitySet::new().missing_from(&has).is_empty());
    }

    #[test]
    fn subset_relation() {
        let small = CapabilitySet::new().with(Capability::Gpu);
        let big = CapabilitySet::new()
            .with(Capability::Gpu)
            .with(Capability::Network);
        assert!(small.is_subset_of(&big));
        assert!(!big.is_subset_of(&small));
    }

    #[test]
    fn display_is_stable() {
        assert_eq!(Capability::Gpu.to_string(), "gpu");
        assert_eq!(Capability::Custom("dsp".into()).to_string(), "custom:dsp");
    }
}
