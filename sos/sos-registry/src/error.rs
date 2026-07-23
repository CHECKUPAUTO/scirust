//! Plugin-system error type.

use sos_core::hash::Digest;
use thiserror::Error;

use crate::capability::Capability;

/// Errors produced by the plugin registry.
#[derive(Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum RegistryError {
    /// No plugin is registered under the requested name.
    #[error("no plugin named `{0}`")]
    NotFound(String),

    /// A plugin with that name exists, but no registered version satisfies the
    /// requested [`crate::VersionReq`].
    #[error("no version of `{name}` satisfies the requirement")]
    VersionUnsatisfied {
        /// The plugin name that was resolved.
        name: String,
    },

    /// A pinned resolution bound a **different** content hash than expected — the
    /// implementation drifted between runs, which breaks reproducibility.
    #[error("plugin `{name}` drifted: expected content {expected}, resolved {found}")]
    Drift {
        /// The plugin name.
        name: String,
        /// The content hash a prior run recorded.
        expected: Digest,
        /// The content hash resolved now.
        found: Digest,
    },

    /// A plugin was denied because it needs capabilities the study did not
    /// grant (least privilege). Lists the missing capabilities.
    #[error("plugin `{name}` denied: missing capabilities {missing:?}")]
    Denied {
        /// The plugin name.
        name: String,
        /// The capabilities the plugin needs but was not granted.
        missing: Vec<Capability>,
    },
}

/// Convenience alias for registry results.
pub type Result<T> = core::result::Result<T, RegistryError>;
