//! [`EnvLock`] — the hermetic environment "lockfile", and [`Drift`] detection.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{Body, Digest, EnvRecord, HashAlgo, SemVer};

/// A hermetic environment **pin** — the reproducibility lockfile (RFC-0002
/// §09.7). Where `sos-provenance` *records* the environment an object was
/// produced in, an `EnvLock` *pins* it: the exact toolchain, the exact backend
/// versions and content hashes, the hardware class, the OS. Re-execution binds
/// the **same** pins or declares the [drift](EnvLock::drift) — no "works on my
/// machine".
///
/// The lock's [`env_digest`](EnvLock::env_digest) is the key every workflow stage
/// is memoized against, so re-running under a matching lock reproduces from cache
/// and re-running under a drifted lock re-executes (see `sos_repro::rerun`).
///
/// It is a content-addressed [`Body`], so the environment a result depends on is
/// itself a citable object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvLock {
    env: EnvRecord,
}

impl EnvLock {
    /// Pin an [`EnvRecord`] as a lock.
    #[must_use]
    pub fn pin(env: EnvRecord) -> Self {
        Self { env }
    }

    /// The pinned environment record.
    #[must_use]
    pub fn env(&self) -> &EnvRecord {
        &self.env
    }

    /// The lock's environment digest — the reproducibility key (SHA-256 over the
    /// canonical [`EnvRecord`]).
    #[must_use]
    pub fn env_digest(&self) -> Digest {
        self.env.digest(HashAlgo::default())
    }

    /// Whether this lock pins the **identical** environment as `other` (no drift).
    #[must_use]
    pub fn binds(&self, other: &EnvLock) -> bool {
        self.drift(other).is_empty()
    }

    /// The itemized differences from this lock (the *expected* pin) to `other`
    /// (the *actual* environment). An empty result means the environments are
    /// byte-identical for reproducibility purposes; a non-empty result **localizes
    /// every difference**, so drift is declared, never silent.
    ///
    /// Drifts are returned in a deterministic order: toolchain, OS, hardware, then
    /// backend differences sorted by backend name.
    #[must_use]
    pub fn drift(&self, other: &EnvLock) -> Vec<Drift> {
        let mut out = Vec::new();
        if self.env.toolchain != other.env.toolchain
        {
            out.push(Drift::Toolchain {
                expected: self.env.toolchain.clone(),
                actual: other.env.toolchain.clone(),
            });
        }
        if self.env.os != other.env.os
        {
            out.push(Drift::Os {
                expected: self.env.os.clone(),
                actual: other.env.os.clone(),
            });
        }
        if self.env.hardware != other.env.hardware
        {
            out.push(Drift::Hardware {
                expected: self.env.hardware.clone(),
                actual: other.env.hardware.clone(),
            });
        }

        let expected: BTreeMap<&str, &_> = self
            .env
            .backends
            .iter()
            .map(|b| (b.name.as_str(), b))
            .collect();
        let actual: BTreeMap<&str, &_> = other
            .env
            .backends
            .iter()
            .map(|b| (b.name.as_str(), b))
            .collect();

        // Union of backend names, sorted (BTreeMap keys are sorted).
        let mut names: Vec<&str> = expected.keys().chain(actual.keys()).copied().collect();
        names.sort_unstable();
        names.dedup();

        for name in names
        {
            match (expected.get(name), actual.get(name))
            {
                (Some(_), None) => out.push(Drift::BackendRemoved(name.to_string())),
                (None, Some(_)) => out.push(Drift::BackendAdded(name.to_string())),
                (Some(e), Some(a)) =>
                {
                    if e.version != a.version
                    {
                        out.push(Drift::BackendVersionChanged {
                            name: name.to_string(),
                            expected: e.version,
                            actual: a.version,
                        });
                    }
                    else if e.content_hash != a.content_hash
                    {
                        out.push(Drift::BackendHashChanged {
                            name: name.to_string(),
                        });
                    }
                },
                (None, None) => unreachable!("name came from the union of both maps"),
            }
        }
        out
    }
}

impl Canonical for EnvLock {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.value(&self.env);
    }
}

impl Body for EnvLock {
    const KIND: &'static str = "EnvLock";
    const SCHEMA_VERSION: u32 = 1;
}

/// A single itemized difference between two [`EnvLock`]s (RFC-0002 §09.7 — "binds
/// the same pins or declares the drift").
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Drift {
    /// The Rust toolchain differs.
    Toolchain {
        /// The expected (pinned) toolchain.
        expected: String,
        /// The actual toolchain.
        actual: String,
    },
    /// The operating system differs.
    Os {
        /// The expected OS.
        expected: String,
        /// The actual OS.
        actual: String,
    },
    /// The hardware class differs.
    Hardware {
        /// The expected hardware class.
        expected: String,
        /// The actual hardware class.
        actual: String,
    },
    /// A backend present in the pin is missing from the actual environment.
    BackendRemoved(String),
    /// A backend present in the actual environment is not in the pin.
    BackendAdded(String),
    /// A backend's version changed.
    BackendVersionChanged {
        /// The backend name.
        name: String,
        /// The expected (pinned) version.
        expected: SemVer,
        /// The actual version.
        actual: SemVer,
    },
    /// A backend kept its name and version but its content hash changed — the same
    /// version was built from different bytes.
    BackendHashChanged {
        /// The backend name.
        name: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use sos_core::BackendVersion;

    fn backend(name: &str, v: (u32, u32, u32), tag: &[u8]) -> BackendVersion {
        BackendVersion::new(
            name,
            SemVer::new(v.0, v.1, v.2),
            HashAlgo::default().hash(b"backend", tag),
        )
    }

    fn env() -> EnvRecord {
        EnvRecord::new(
            "1.89.0-stable",
            vec![backend("scirust-solvers", (0, 1, 0), b"solvers")],
            "x86_64/avx2/openblas",
            "linux",
        )
    }

    #[test]
    fn identical_locks_bind_with_no_drift() {
        let a = EnvLock::pin(env());
        let b = EnvLock::pin(env());
        assert!(a.binds(&b));
        assert!(a.drift(&b).is_empty());
        assert_eq!(a.env_digest(), b.env_digest());
    }

    #[test]
    fn drift_is_itemized_and_localized() {
        let a = EnvLock::pin(env());
        let mut e2 = env();
        e2.hardware = "aarch64/neon".into();
        e2.backends = vec![
            backend("scirust-solvers", (0, 2, 0), b"solvers"), // version bump
            backend("scirust-signal", (0, 1, 0), b"signal"),   // added
        ];
        let b = EnvLock::pin(e2);

        let drift = a.drift(&b);
        assert!(!a.binds(&b));
        assert!(a.env_digest() != b.env_digest());
        // Hardware drift, a version change, and an added backend — deterministic order.
        assert!(drift.contains(&Drift::Hardware {
            expected: "x86_64/avx2/openblas".into(),
            actual: "aarch64/neon".into(),
        }));
        assert!(drift.iter().any(|d| matches!(
            d,
            Drift::BackendVersionChanged { name, .. } if name == "scirust-solvers"
        )));
        assert!(drift.contains(&Drift::BackendAdded("scirust-signal".into())));
    }

    #[test]
    fn hash_change_at_same_version_is_flagged() {
        let a = EnvLock::pin(env());
        let mut e2 = env();
        e2.backends = vec![backend("scirust-solvers", (0, 1, 0), b"rebuilt")]; // same ver, new hash
        let b = EnvLock::pin(e2);
        assert_eq!(
            a.drift(&b),
            vec![Drift::BackendHashChanged {
                name: "scirust-solvers".into()
            }]
        );
    }
}
