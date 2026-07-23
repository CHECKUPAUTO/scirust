//! # `sos-repro` — the SOS Reproducibility Engine (the Nix analogy)
//!
//! Where the Provenance Engine *records* the environment, the Reproducibility
//! Engine **pins and re-realizes** it (RFC-0002 §09.7). This is the subsystem
//! that turns "reproducible" from a hope into a **checkable property of a graph**.
//!
//! This crate provides the deterministic core:
//!
//! * [`EnvLock`] — the hermetic environment lockfile: exact toolchain, backend
//!   versions + content hashes, hardware class, OS. Its
//!   [`env_digest`](EnvLock::env_digest) keys the workflow cache, so a lock that
//!   [`binds`](EnvLock::binds) the original reproduces from cache and a drifted
//!   lock re-executes.
//! * [`Drift`] / [`EnvLock::drift`] — itemized, localized drift detection: "binds
//!   the same pins or **declares** the drift", never a silent "works on my
//!   machine".
//! * The **reproduction contract** ([`verify_reproduction`], [`MatchRule`],
//!   [`NodeVerdict`], [`VerifyReport`]) — level-aware verification: `L3` bit-exact
//!   and `L0` replay are decided here by object-id equality; `L2` within-certificate
//!   and `L1` in-distribution take a backend-supplied verdict. Any deviation is
//!   localized to a specific node and its declared level.
//! * [`rerun`] — re-realize a `sos_workflow::Plan` under a pinned lock.
//!
//! ## What is deliberately *not* here yet
//!
//! The numeric / statistical evaluation behind an `L2`/`L1` node's verdict is the
//! backend's job (`sos-scirust`), supplied to the contract as a
//! [`Reproduced::Certified`] outcome — this crate never fabricates it. A full
//! store-driven `verify(object)` that walks an object's sub-DAG, re-executes it,
//! and auto-diffs composes [`rerun`] with [`verify_reproduction`]; the piece it
//! awaits is the object-graph re-execution driver. No stub crosses that line.
//!
//! ## Example — a lock binds, or declares its drift
//!
//! ```
//! use sos_core::{BackendVersion, EnvRecord, HashAlgo, SemVer};
//! use sos_repro::{EnvLock, Drift};
//!
//! let backend = BackendVersion::new(
//!     "scirust-solvers", SemVer::new(0, 1, 0), HashAlgo::default().hash(b"b", b"v1"),
//! );
//! let env = EnvRecord::new("1.89.0-stable", vec![backend], "x86_64/avx2", "linux");
//! let pinned = EnvLock::pin(env.clone());
//!
//! // The same environment binds — reproduction is possible.
//! assert!(pinned.binds(&EnvLock::pin(env.clone())));
//!
//! // A different CPU class is drift, and it is named, not hidden.
//! let mut moved = env;
//! moved.hardware = "aarch64/neon".into();
//! let other = EnvLock::pin(moved);
//! assert!(!pinned.binds(&other));
//! assert_eq!(pinned.drift(&other), vec![Drift::Hardware {
//!     expected: "x86_64/avx2".into(),
//!     actual: "aarch64/neon".into(),
//! }]);
//! // Different pins ⇒ different cache key ⇒ a re-run recomputes rather than reuses.
//! assert_ne!(pinned.env_digest(), other.env_digest());
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

pub mod contract;
pub mod error;
pub mod lock;
pub mod rerun;

pub use contract::{
    MatchRule, NodeClaim, NodeReport, NodeVerdict, Reproduced, VerifyReport, verify_node,
    verify_reproduction,
};
pub use error::{ReproError, Result};
pub use lock::{Drift, EnvLock};
pub use rerun::rerun;
