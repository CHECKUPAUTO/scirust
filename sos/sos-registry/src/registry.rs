//! The [`Registry`] — a content-pinned index of plugin descriptors.

use std::collections::BTreeMap;

use sos_core::hash::Digest;

use crate::descriptor::{DomainTag, PluginDescriptor, PluginName, Role, VersionReq};
use crate::error::{RegistryError, Result};

/// An in-memory index of [`PluginDescriptor`]s, keyed by name and ordered by
/// version. Resolution binds a plugin by name + [`VersionReq`], and
/// [`Registry::resolve_pinned`] detects content-hash drift between runs.
///
/// Iteration order is deterministic ([`BTreeMap`] by name, versions sorted
/// ascending), so listing and discovery never depend on insertion order.
#[derive(Debug, Clone, Default)]
pub struct Registry {
    by_name: BTreeMap<PluginName, Vec<PluginDescriptor>>,
}

impl Registry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a plugin descriptor. Idempotent on the exact `(name, version,
    /// content_hash)` triple; distinct hashes for the same version are both kept
    /// (a conflict a pinned resolve will surface as drift).
    pub fn register(&mut self, descriptor: PluginDescriptor) {
        let entry = self.by_name.entry(descriptor.name.clone()).or_default();
        if !entry
            .iter()
            .any(|d| d.version == descriptor.version && d.content_hash == descriptor.content_hash)
        {
            entry.push(descriptor);
            entry.sort_by_key(|d| d.version);
        }
    }

    /// Resolve the **highest** registered version of `name` that satisfies
    /// `req`.
    ///
    /// # Errors
    /// [`RegistryError::NotFound`] if no plugin has that name;
    /// [`RegistryError::VersionUnsatisfied`] if none of its versions match.
    pub fn resolve(&self, name: &str, req: &VersionReq) -> Result<&PluginDescriptor> {
        let key = PluginName(name.to_string());
        let versions = self
            .by_name
            .get(&key)
            .ok_or_else(|| RegistryError::NotFound(name.to_string()))?;
        versions
            .iter()
            .rev() // highest version first
            .find(|d| req.matches(&d.version))
            .ok_or_else(|| RegistryError::VersionUnsatisfied {
                name: name.to_string(),
            })
    }

    /// Resolve as [`Registry::resolve`], then verify the bound plugin's content
    /// hash equals `expected` — the reproducibility check that catches an
    /// implementation drifting between runs.
    ///
    /// # Errors
    /// [`RegistryError::Drift`] if the resolved content hash differs from
    /// `expected`, plus any error from [`Registry::resolve`].
    pub fn resolve_pinned(
        &self,
        name: &str,
        req: &VersionReq,
        expected: &Digest,
    ) -> Result<&PluginDescriptor> {
        let d = self.resolve(name, req)?;
        if &d.content_hash != expected
        {
            return Err(RegistryError::Drift {
                name: name.to_string(),
                expected: *expected,
                found: d.content_hash,
            });
        }
        Ok(d)
    }

    /// All plugins implementing `role`, optionally filtered to those applying to
    /// `domain`. Sorted by name then version.
    #[must_use]
    pub fn find(&self, role: &Role, domain: Option<&DomainTag>) -> Vec<&PluginDescriptor> {
        let mut out: Vec<&PluginDescriptor> = self
            .by_name
            .values()
            .flatten()
            .filter(|d| &d.role == role)
            .filter(|d| domain.is_none_or(|dom| d.applies_to(dom)))
            .collect();
        out.sort_by(|a, b| a.name.cmp(&b.name).then(a.version.cmp(&b.version)));
        out
    }

    /// Every registered descriptor, sorted by name then version.
    #[must_use]
    pub fn all(&self) -> Vec<&PluginDescriptor> {
        let mut out: Vec<&PluginDescriptor> = self.by_name.values().flatten().collect();
        out.sort_by(|a, b| a.name.cmp(&b.name).then(a.version.cmp(&b.version)));
        out
    }

    /// The registered plugin names, sorted.
    #[must_use]
    pub fn names(&self) -> Vec<&PluginName> {
        self.by_name.keys().collect()
    }

    /// The total number of registered descriptors (across all versions).
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_name.values().map(Vec::len).sum()
    }

    /// Whether the registry holds no plugins.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sos_core::{HashAlgo, SemVer};

    fn desc(name: &str, ver: SemVer, role: Role) -> PluginDescriptor {
        let hash = HashAlgo::default().hash(b"plugin", format!("{name}-{ver}").as_bytes());
        PluginDescriptor::new(name, ver, hash, role)
    }

    #[test]
    fn resolve_picks_highest_matching_version() {
        let mut r = Registry::new();
        r.register(desc("p", SemVer::new(0, 1, 0), Role::Planning));
        r.register(desc("p", SemVer::new(0, 2, 0), Role::Planning));
        r.register(desc("p", SemVer::new(1, 0, 0), Role::Planning));

        assert_eq!(
            r.resolve("p", &VersionReq::Any).unwrap().version,
            SemVer::new(1, 0, 0)
        );
        assert_eq!(
            r.resolve("p", &VersionReq::Compatible(SemVer::new(0, 1, 0)))
                .unwrap()
                .version,
            SemVer::new(0, 2, 0) // highest 0.x
        );
    }

    #[test]
    fn resolve_errors_are_precise() {
        let mut r = Registry::new();
        r.register(desc("p", SemVer::new(1, 0, 0), Role::Planning));
        assert_eq!(
            r.resolve("nope", &VersionReq::Any),
            Err(RegistryError::NotFound("nope".into()))
        );
        assert!(matches!(
            r.resolve("p", &VersionReq::AtLeast(SemVer::new(2, 0, 0))),
            Err(RegistryError::VersionUnsatisfied { .. })
        ));
    }

    #[test]
    fn pinned_resolution_detects_drift() {
        let mut r = Registry::new();
        let d = desc("p", SemVer::new(1, 0, 0), Role::Planning);
        let good = d.content_hash;
        r.register(d);
        assert!(r.resolve_pinned("p", &VersionReq::Any, &good).is_ok());
        let wrong = HashAlgo::default().hash(b"plugin", b"different");
        assert!(matches!(
            r.resolve_pinned("p", &VersionReq::Any, &wrong),
            Err(RegistryError::Drift { .. })
        ));
    }

    #[test]
    fn register_is_idempotent_on_identical_triple() {
        let mut r = Registry::new();
        let d = desc("p", SemVer::new(1, 0, 0), Role::Planning);
        r.register(d.clone());
        r.register(d);
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn find_by_role_and_domain() {
        let mut r = Registry::new();
        r.register(desc("a", SemVer::new(1, 0, 0), Role::HypothesisGenerator).domain("physics"));
        r.register(desc("b", SemVer::new(1, 0, 0), Role::HypothesisGenerator)); // any domain
        r.register(desc("c", SemVer::new(1, 0, 0), Role::Planning));

        let phys = DomainTag("physics".into());
        let found = r.find(&Role::HypothesisGenerator, Some(&phys));
        let names: Vec<String> = found.iter().map(|d| d.name.to_string()).collect();
        assert_eq!(names, vec!["a", "b"]); // both apply to physics; sorted

        let fin = DomainTag("finance".into());
        let found_fin = r.find(&Role::HypothesisGenerator, Some(&fin));
        let names_fin: Vec<String> = found_fin.iter().map(|d| d.name.to_string()).collect();
        assert_eq!(names_fin, vec!["b"]); // only the any-domain plugin

        assert_eq!(r.find(&Role::Planning, None).len(), 1);
    }
}
