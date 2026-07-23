//! [`SemVer`] — a minimal semantic version for object content lineage and
//! producer/backend versions.
//!
//! This is intentionally tiny (no pre-release/build metadata): SOS uses it to
//! *record* versions in provenance, not to resolve dependency ranges. It is
//! `Canonical` (so it can sit inside a hashed object) and parses/prints as
//! `major.minor.patch`.

use core::fmt;
use core::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::canonical::{Canonical, CanonicalEncoder};
use crate::error::SosError;

/// A `major.minor.patch` semantic version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SemVer {
    /// Major version — incremented on incompatible changes.
    pub major: u32,
    /// Minor version — incremented on backward-compatible additions.
    pub minor: u32,
    /// Patch version — incremented on backward-compatible fixes.
    pub patch: u32,
}

impl SemVer {
    /// Construct a version from its three components.
    #[must_use]
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }
}

impl Canonical for SemVer {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.u64(u64::from(self.major));
        enc.u64(u64::from(self.minor));
        enc.u64(u64::from(self.patch));
    }
}

impl fmt::Display for SemVer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl FromStr for SemVer {
    type Err = SosError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut it = s.split('.');
        let mut next = || -> Result<u32, SosError> {
            it.next()
                .ok_or_else(|| SosError::InvalidSemVer(s.to_string()))?
                .parse::<u32>()
                .map_err(|_| SosError::InvalidSemVer(s.to_string()))
        };
        let major = next()?;
        let minor = next()?;
        let patch = next()?;
        if it.next().is_some()
        {
            return Err(SosError::InvalidSemVer(s.to_string()));
        }
        Ok(Self::new(major, minor, patch))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_and_parse_roundtrip() {
        let v = SemVer::new(1, 89, 0);
        assert_eq!(v.to_string(), "1.89.0");
        assert_eq!("1.89.0".parse::<SemVer>().unwrap(), v);
    }

    #[test]
    fn ordering_is_semantic() {
        assert!(SemVer::new(1, 0, 0) < SemVer::new(1, 0, 1));
        assert!(SemVer::new(1, 2, 0) < SemVer::new(1, 10, 0));
        assert!(SemVer::new(0, 9, 9) < SemVer::new(1, 0, 0));
    }

    #[test]
    fn malformed_versions_are_rejected() {
        assert!("1.2".parse::<SemVer>().is_err());
        assert!("1.2.3.4".parse::<SemVer>().is_err());
        assert!("a.b.c".parse::<SemVer>().is_err());
        assert!("".parse::<SemVer>().is_err());
    }

    #[test]
    fn canonical_distinguishes_versions() {
        assert_ne!(
            SemVer::new(1, 0, 0).canonical_bytes(),
            SemVer::new(1, 0, 1).canonical_bytes()
        );
    }
}
