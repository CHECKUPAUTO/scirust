//! # `sos-registry` — the SOS Plugin System
//!
//! The registry is SOS's "device-driver table" (RFC-0002 §07.4): it binds the
//! engine syscalls and discovery stages to concrete plugins by **name + semantic
//! version + content hash**, so a workflow's choice of implementation is
//! explicit, discoverable, and reproducible.
//!
//! This crate is the *metadata and resolution* layer — the stable, pure part of
//! the plugin system. It has three jobs:
//!
//! 1. **Describe** a plugin: [`PluginDescriptor`] pins its name, version, content
//!    hash, the [`Role`] it implements, the [`DeterminismLevel`] it realizes, the
//!    [`Capability`]s it needs, and the domains it applies to.
//! 2. **Resolve** a plugin by name and a [`VersionReq`], returning the highest
//!    matching version — and detecting **drift** ([`Registry::resolve_pinned`])
//!    when a re-resolution binds a *different* content hash than a prior run
//!    recorded.
//! 3. **Authorize** it: [`authorize`] enforces least privilege — a plugin may
//!    run only if every capability it needs was granted, refusing by default
//!    (the `scirust-discovery::ScopeAuthorization` posture).
//!
//! Binding a descriptor to an actual trait object (the *factory*) is the job of
//! the engine crates that own those traits; the registry deliberately stays
//! decoupled from them so it depends only on `sos-core`.
//!
//! [`DeterminismLevel`]: sos_core::DeterminismLevel
//!
//! ## Example
//!
//! ```
//! use sos_core::{SemVer, HashAlgo, DeterminismLevel};
//! use sos_registry::{
//!     PluginDescriptor, Registry, Role, VersionReq, Capability, Grant, authorize,
//! };
//!
//! let hash = HashAlgo::default().hash(b"plugin", b"symreg-0.1.0");
//! let desc = PluginDescriptor::new(
//!     "sos-scirust/symreg", SemVer::new(0, 1, 0), hash, Role::HypothesisGenerator,
//! )
//! .level(DeterminismLevel::L1)
//! .needs(Capability::Gpu)
//! .domain("physics");
//!
//! let mut reg = Registry::new();
//! reg.register(desc);
//!
//! // Resolve by name + a compatible-version requirement.
//! let bound = reg
//!     .resolve("sos-scirust/symreg", &VersionReq::Compatible(SemVer::new(0, 1, 0)))
//!     .unwrap();
//! assert_eq!(bound.version, SemVer::new(0, 1, 0));
//!
//! // Least privilege: a study that grants GPU may run it; one that doesn't, can't.
//! let with_gpu = Grant::new().allow(Capability::Gpu);
//! assert!(authorize(bound, &with_gpu).is_ok());
//! assert!(authorize(bound, &Grant::new()).is_err());
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

pub mod capability;
pub mod descriptor;
pub mod error;
pub mod registry;

pub use capability::{Capability, CapabilitySet, Grant, authorize};
pub use descriptor::{DomainTag, PluginDescriptor, PluginName, Role, VersionReq};
pub use error::{RegistryError, Result};
pub use registry::Registry;
