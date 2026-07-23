//! [`StageDescriptor`] — the stable identity of a stage plugin.

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{Digest, SemVer};

/// The stable identity of a stage: its plugin **name**, **version**, and the
/// **content hash** of the plugin itself (RFC-0002 §08.2). This is the first
/// ingredient of a stage's [`crate::CacheKey`]: bumping a plugin's version or
/// changing its bytes changes every cache key that used it, so stale results are
/// never silently reused.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageDescriptor {
    /// The plugin name (e.g. `"sde-scirust/symreg"`).
    pub name: String,
    /// The plugin's semantic version.
    pub version: SemVer,
    /// A content hash of the plugin implementation.
    pub content_hash: Digest,
}

impl StageDescriptor {
    /// Construct a stage descriptor.
    #[must_use]
    pub fn new(name: impl Into<String>, version: SemVer, content_hash: Digest) -> Self {
        Self {
            name: name.into(),
            version,
            content_hash,
        }
    }
}

impl Canonical for StageDescriptor {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.str(&self.name);
        enc.value(&self.version);
        enc.bytes(self.content_hash.as_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sos_core::HashAlgo;

    fn digest(tag: &[u8]) -> Digest {
        HashAlgo::default().hash(b"test", tag)
    }

    #[test]
    fn canonical_reflects_every_field() {
        let base = StageDescriptor::new("s", SemVer::new(1, 0, 0), digest(b"a"));
        assert_eq!(base.canonical_bytes(), base.clone().canonical_bytes());
        assert_ne!(
            base.canonical_bytes(),
            StageDescriptor::new("s", SemVer::new(1, 0, 1), digest(b"a")).canonical_bytes()
        );
        assert_ne!(
            base.canonical_bytes(),
            StageDescriptor::new("s", SemVer::new(1, 0, 0), digest(b"b")).canonical_bytes()
        );
    }
}
