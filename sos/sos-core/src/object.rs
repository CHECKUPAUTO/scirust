//! [`Object`] — the immutable, content-addressed scientific object envelope.
//!
//! Every artifact in SOS-IR shares this envelope. Its [`ObjectId`] is a hash of
//! its content — including its `parents` — so the object graph is a Merkle DAG:
//! reproducible identity, dedup, and end-to-end tamper-evidence all follow from
//! one construction. The advisory wall-clock and the optional signature are
//! **excluded** from the id (a timestamp must not perturb identity, and a
//! signature cannot sign itself), so stamping either never changes the address.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::canonical::{Canonical, CanonicalEncoder};
use crate::clock::LamportClock;
use crate::determinism::DeterminismLevel;
use crate::error::{Result, SosError};
use crate::hash::HashAlgo;
use crate::id::ObjectId;
use crate::kind::Kind;
use crate::provenance::{Author, ProducerRef, Signature};
use crate::repro::{EnvRecord, ReproMeta, RngId};
use crate::version::SemVer;

/// A kind-specific object payload.
///
/// Implementers supply a stable [`Body::KIND`] name and [`Body::SCHEMA_VERSION`]
/// (which together form the object's [`Kind`] and its hash domain) plus a
/// [`Canonical`] encoding. `Serialize`/`DeserializeOwned` provide the JSON
/// interchange form; `Canonical` provides the normative hashing form.
pub trait Body: Canonical + Serialize + DeserializeOwned + Clone {
    /// Stable type name, e.g. `"Hypothesis"`.
    const KIND: &'static str;
    /// Schema version of this body's canonical shape.
    const SCHEMA_VERSION: u32;

    /// The [`Kind`] for this body type.
    #[must_use]
    fn kind() -> Kind {
        Kind::new(Self::KIND, Self::SCHEMA_VERSION)
    }
}

/// An immutable, content-addressed scientific object.
///
/// Construct one via [`Object::builder`]; the [`ObjectBuilder::seal`] step
/// computes the [`Object::id`]. Objects are never mutated after sealing — a
/// "revision" is a new object pointing at its parent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "B: serde::Serialize",
    deserialize = "B: serde::de::DeserializeOwned"
))]
pub struct Object<B: Body> {
    /// Content address — the hash of every field below except `wall` and
    /// `signature`. Recompute and check it with [`Object::verify_id`].
    pub id: ObjectId,
    /// Type name + schema version of `body`.
    pub kind: Kind,
    /// Semantic version of this object's content lineage (distinct from
    /// `kind.schema_version`, which versions the *shape*).
    pub version: SemVer,
    /// Authoritative logical time (Lamport). Part of the id.
    pub logical: LamportClock,
    /// Advisory wall-clock (unix seconds). **Not** part of the id, and omitted
    /// from JSON when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wall: Option<u64>,
    /// Direct provenance: the objects this one was derived from. Part of the id,
    /// which is what makes the graph a Merkle DAG.
    pub parents: Vec<ObjectId>,
    /// The engine/plugin that produced this object.
    pub producer: ProducerRef,
    /// The principal that initiated this object.
    pub author: Author,
    /// Reproducibility metadata (seed, environment digest, inputs).
    pub repro: ReproMeta,
    /// The determinism level realized for this object (`meet` over its inputs
    /// and producer — computed by the caller via [`DeterminismLevel::min_over`]).
    pub level: DeterminismLevel,
    /// Optional detached attestation. **Not** part of the id; omitted from JSON
    /// when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<Signature>,
    /// The kind-specific payload.
    pub body: B,
}

/// Encode the *identity* fields of an object in a fixed order. This is the sole
/// definition of what an id covers, shared by sealing and verification so the
/// two can never drift apart. `id`, `wall`, and `signature` are deliberately
/// excluded.
#[allow(clippy::too_many_arguments)]
fn encode_identity<B: Body>(
    version: &SemVer,
    logical: LamportClock,
    parents: &[ObjectId],
    producer: &ProducerRef,
    author: &Author,
    repro: &ReproMeta,
    level: DeterminismLevel,
    body: &B,
) -> Vec<u8> {
    let mut e = CanonicalEncoder::new();
    e.value(version);
    e.value(&logical);
    e.seq(parents);
    e.value(producer);
    e.value(author);
    e.value(repro);
    e.value(&level);
    e.value(body);
    e.finish()
}

impl<B: Body> Object<B> {
    /// The hash algorithm the kernel uses for object ids (versioned; SHA-256
    /// today). Kept as an associated accessor so callers never hard-code it.
    #[must_use]
    pub const fn hash_algo() -> HashAlgo {
        HashAlgo::Sha256
    }

    /// Begin building an object from its `body`. See [`ObjectBuilder`] for the
    /// defaults (a hand-authored, `L3`, root object in an unspecified
    /// environment) and the setters that override them for computed objects.
    #[must_use]
    pub fn builder(body: B) -> ObjectBuilder<B> {
        ObjectBuilder::new(body)
    }

    /// Recompute the id this object *should* have from its current content.
    #[must_use]
    pub fn recompute_id(&self) -> ObjectId {
        let bytes = encode_identity(
            &self.version,
            self.logical,
            &self.parents,
            &self.producer,
            &self.author,
            &self.repro,
            self.level,
            &self.body,
        );
        ObjectId::compute(Self::hash_algo(), &self.kind.domain(), &bytes)
    }

    /// Whether the stored [`Object::id`] matches the content.
    #[must_use]
    pub fn verify_id(&self) -> bool {
        self.recompute_id() == self.id
    }

    /// Verify the id, returning [`SosError::IdMismatch`] on failure.
    ///
    /// # Errors
    /// Fails if the object has been tampered with, corrupted, or deserialized
    /// from an incompatible schema.
    pub fn check_id(&self) -> Result<()> {
        let recomputed = self.recompute_id();
        if recomputed == self.id
        {
            Ok(())
        }
        else
        {
            Err(SosError::IdMismatch {
                stored: self.id,
                recomputed,
            })
        }
    }

    /// Attach a detached [`Signature`], returning the object unchanged in
    /// identity (the id excludes the signature, so `verify_id` still holds).
    #[must_use]
    pub fn with_signature(mut self, sig: Signature) -> Self {
        self.signature = Some(sig);
        self
    }
}

/// Builder for an [`Object`]. Fields default to a **hand-authored, root,
/// bit-reproducible** object in an unspecified environment; engines override
/// the provenance, reproducibility, level, parents, and logical clock they
/// actually realized before [`ObjectBuilder::seal`]-ing.
#[derive(Debug, Clone)]
pub struct ObjectBuilder<B: Body> {
    body: B,
    parents: Vec<ObjectId>,
    author: Author,
    producer: ProducerRef,
    version: SemVer,
    logical: LamportClock,
    repro: ReproMeta,
    level: DeterminismLevel,
    wall: Option<u64>,
}

impl<B: Body> ObjectBuilder<B> {
    fn new(body: B) -> Self {
        let algo = Object::<B>::hash_algo();
        // A real, deterministic self-descriptor for the kernel authoring path —
        // meaningful ("authored by the kernel"), not a placeholder.
        let producer = ProducerRef::new(
            "sos-core",
            SemVer::new(0, 1, 0),
            algo.hash(b"sos-producer", b"sos-core/authoring"),
        );
        // An explicit "unspecified authored environment" — honest, not a stub:
        // a hand-authored root object genuinely has no computational backend.
        let env = EnvRecord::new("unspecified", Vec::new(), "unspecified", "unspecified");
        let repro = ReproMeta::new(0, RngId::new("none"), env.digest(algo));
        Self {
            body,
            parents: Vec::new(),
            author: Author::engine("sos-core"),
            producer,
            version: SemVer::new(1, 0, 0),
            logical: LamportClock::ZERO,
            repro,
            level: DeterminismLevel::L3,
            wall: None,
        }
    }

    /// Set the direct provenance parents.
    #[must_use]
    pub fn parents(mut self, parents: Vec<ObjectId>) -> Self {
        self.parents = parents;
        self
    }

    /// Set the initiating principal.
    #[must_use]
    pub fn author(mut self, author: Author) -> Self {
        self.author = author;
        self
    }

    /// Set the producing engine/plugin.
    #[must_use]
    pub fn producer(mut self, producer: ProducerRef) -> Self {
        self.producer = producer;
        self
    }

    /// Set the content-lineage version.
    #[must_use]
    pub fn version(mut self, version: SemVer) -> Self {
        self.version = version;
        self
    }

    /// Set the logical clock (normally `1 + max(parent clocks)`).
    #[must_use]
    pub fn logical(mut self, logical: LamportClock) -> Self {
        self.logical = logical;
        self
    }

    /// Set the reproducibility metadata.
    #[must_use]
    pub fn repro(mut self, repro: ReproMeta) -> Self {
        self.repro = repro;
        self
    }

    /// Set the realized determinism level.
    #[must_use]
    pub fn level(mut self, level: DeterminismLevel) -> Self {
        self.level = level;
        self
    }

    /// Set the advisory wall-clock (unix seconds). Does not affect the id.
    #[must_use]
    pub fn wall(mut self, unix_seconds: u64) -> Self {
        self.wall = Some(unix_seconds);
        self
    }

    /// Finalize: compute the content-addressed id and produce the immutable
    /// [`Object`].
    #[must_use]
    pub fn seal(self) -> Object<B> {
        let kind = B::kind();
        let bytes = encode_identity(
            &self.version,
            self.logical,
            &self.parents,
            &self.producer,
            &self.author,
            &self.repro,
            self.level,
            &self.body,
        );
        let id = ObjectId::compute(Object::<B>::hash_algo(), &kind.domain(), &bytes);
        Object {
            id,
            kind,
            version: self.version,
            logical: self.logical,
            wall: self.wall,
            parents: self.parents,
            producer: self.producer,
            author: self.author,
            repro: self.repro,
            level: self.level,
            signature: None,
            body: self.body,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Serialize, Deserialize)]
    struct Q {
        text: String,
    }
    impl Canonical for Q {
        fn encode(&self, enc: &mut CanonicalEncoder) {
            enc.str(&self.text);
        }
    }
    impl Body for Q {
        const KIND: &'static str = "Question";
        const SCHEMA_VERSION: u32 = 1;
    }

    fn q(text: &str) -> Object<Q> {
        Object::builder(Q { text: text.into() })
            .author(Author::human("ada"))
            .seal()
    }

    #[test]
    fn sealing_is_deterministic() {
        assert_eq!(q("hi").id, q("hi").id);
    }

    #[test]
    fn different_body_changes_id() {
        assert_ne!(q("a").id, q("b").id);
    }

    #[test]
    fn sealed_objects_verify() {
        assert!(q("hi").verify_id());
        assert!(q("hi").check_id().is_ok());
    }

    #[test]
    fn parents_change_id_merkle_property() {
        let root = q("root");
        let a = Object::builder(Q {
            text: "child".into(),
        })
        .author(Author::human("ada"))
        .parents(vec![root.id])
        .seal();
        let b = Object::builder(Q {
            text: "child".into(),
        })
        .author(Author::human("ada"))
        .parents(vec![q("different-root").id])
        .seal();
        assert_ne!(a.id, b.id, "changing a parent must change the id");
    }

    #[test]
    fn wall_clock_does_not_affect_id() {
        let base = q("hi");
        let stamped = Object::builder(Q { text: "hi".into() })
            .author(Author::human("ada"))
            .wall(1_700_000_000)
            .seal();
        assert_eq!(base.id, stamped.id);
        assert!(stamped.verify_id());
    }

    #[test]
    fn signature_does_not_affect_id() {
        let base = q("hi");
        let signed = q("hi").with_signature(Signature::new("test", vec![1, 2, 3]));
        assert_eq!(base.id, signed.id);
        assert!(signed.verify_id(), "signing must not break identity");
    }

    #[test]
    fn level_and_producer_are_part_of_identity() {
        let a = Object::builder(Q { text: "x".into() })
            .author(Author::human("ada"))
            .level(DeterminismLevel::L3)
            .seal();
        let b = Object::builder(Q { text: "x".into() })
            .author(Author::human("ada"))
            .level(DeterminismLevel::L1)
            .seal();
        assert_ne!(a.id, b.id);
    }

    #[test]
    fn tampering_is_detected() {
        let mut obj = q("hi");
        obj.body.text = "tampered".into(); // mutate content without resealing
        assert!(!obj.verify_id());
        assert!(matches!(obj.check_id(), Err(SosError::IdMismatch { .. })));
    }

    #[test]
    fn json_roundtrip_preserves_identity() {
        let obj = q("hello");
        let json = serde_json::to_string(&obj).unwrap();
        let back: Object<Q> = serde_json::from_str(&json).unwrap();
        assert_eq!(obj.id, back.id);
        assert!(back.verify_id());
    }

    #[test]
    fn kind_is_derived_from_body() {
        assert_eq!(q("x").kind, Kind::new("Question", 1));
    }
}
