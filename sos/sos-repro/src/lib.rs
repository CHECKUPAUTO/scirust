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
//! [`Reproduced::Certified`] outcome — this crate never fabricates it, and now
//! does not have to: `sos-scirust`'s `verdict` module supplies real ones
//! (certificate-sum agreement for `L2`, a two-sample test for `L1`), and
//! [`verify_rerun`] now joins them: it pairs an original run's ledger against a
//! re-run's, reads each node's declared level from the store, decides `L3`/`L0`
//! by id and routes `L2`/`L1` to that backend through a [`Certifier`]. What
//! [`verify_object`] closes the loop from the store side: it finds the run
//! that recorded an object, loads the [`Plan`](sos_workflow::Plan) that run's
//! digest names — a `Plan` is a storable object now, precisely so a recorded
//! run can be re-executed — re-runs it, and verifies. It refuses rather than
//! guesses when nothing produced the object, when several runs did, or when
//! the plan was never stored. No stub crosses that line.
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
pub mod driver;
pub mod error;
pub mod lock;
pub mod rerun;

pub use contract::{
    MatchRule, NodeClaim, NodeReport, NodeVerdict, Reproduced, VerifyReport, verify_node,
    verify_reproduction,
};
pub use driver::{Certifier, NoCertifier, verify_object, verify_rerun};
pub use error::{ReproError, Result};
pub use lock::{Drift, EnvLock};
pub use rerun::rerun;
