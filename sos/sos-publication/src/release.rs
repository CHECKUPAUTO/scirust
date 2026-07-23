//! [`ReleaseManifest`] — an attestation that a specific publication object was
//! reviewed and released against a specific graph state.
//!
//! A release is its **own** content-addressed object whose parent is the sealed
//! publication it attests. Keeping it separate is what makes "has it changed
//! since release?" answerable without circularity: the manifest records the
//! exact [`ObjectId`] of the publication as released, so re-sealing the current
//! publication and comparing ids ([`check_release`](crate::verify::check_release))
//! detects any post-release edit. The engine never silently repairs or
//! reinterprets a released publication — it reports the divergence.

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{Author, Body, Object, ObjectId};

use crate::policy::PolicyId;

/// An attestation binding a released publication to the graph state it was
/// reviewed against and the policy it was judged under.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleaseManifest {
    /// The content address of the sealed `Object<Publication>` this release
    /// attests. Any later edit to the publication changes this id.
    pub publication: ObjectId,
    /// The graph state (e.g. a study root) that was reviewed at release time, if
    /// recorded.
    pub reviewed_state: Option<ObjectId>,
    /// The support policy the publication was judged under at release.
    pub policy: PolicyId,
    /// A human-readable release/attestation statement.
    pub statement: String,
}

impl ReleaseManifest {
    /// A release attesting `publication` under `policy`, with `statement`.
    /// `reviewed_state` is unset (add with [`reviewed`](Self::reviewed)).
    #[must_use]
    pub fn new(publication: ObjectId, policy: PolicyId, statement: impl Into<String>) -> Self {
        Self {
            publication,
            reviewed_state: None,
            policy,
            statement: statement.into(),
        }
    }

    /// Record the reviewed graph state.
    #[must_use]
    pub fn reviewed(mut self, state: ObjectId) -> Self {
        self.reviewed_state = Some(state);
        self
    }
}

impl Canonical for ReleaseManifest {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.value(&self.publication);
        enc.option(&self.reviewed_state);
        enc.value(&self.policy);
        enc.str(&self.statement);
    }
}

impl Body for ReleaseManifest {
    const KIND: &'static str = "ReleaseManifest";
    const SCHEMA_VERSION: u32 = 1;
}

/// Seal a [`ReleaseManifest`] as an `Object<ReleaseManifest>` whose parent is the
/// publication it attests, authored by the releasing `principal`.
///
/// The parent link makes the release a child of the publication in the Merkle
/// DAG. A cryptographic signature (if the deployment has a real signing
/// abstraction in `sos-provenance`) attaches to this object; this crate does not
/// fabricate one.
#[must_use]
pub fn seal_release(manifest: ReleaseManifest, principal: Author) -> Object<ReleaseManifest> {
    let publication = manifest.publication;
    Object::builder(manifest)
        .parents(vec![publication])
        .author(principal)
        .seal()
}
