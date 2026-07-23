//! [`PublicationObjectSource`] — the read-only window the verifier looks at the
//! object graph through — plus [`ObjectFacts`], a [`StoreSource`] adapter over
//! any [`ObjectStore`], an in-memory [`MapSource`], and [`dependency_closure`].
//!
//! Verification must never do implicit filesystem reads or trust external files;
//! everything it learns about an object comes through this one trait, which
//! yields only the *facts* needed to judge support and scope — an object's kind,
//! its parents, and the determinism level it realized. Bodies are never
//! deserialized (the verifier is body-type agnostic), and the source is
//! read-only (verification cannot mutate the graph).

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sos_core::{Body, DeterminismLevel, Kind, Object, ObjectId};
use sos_store::ObjectStore;

use crate::error::SourceError;

/// The facts about a graph object the verifier needs: its identity, kind,
/// provenance parents, and realized determinism level. Deliberately *not* the
/// body — support and scope are decided from these alone.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObjectFacts {
    /// The object's content address.
    pub id: ObjectId,
    /// The object's kind (type name + schema version).
    pub kind: Kind,
    /// The object's direct provenance parents.
    pub parents: Vec<ObjectId>,
    /// The determinism level the object realized.
    pub level: DeterminismLevel,
}

impl ObjectFacts {
    /// Construct facts directly.
    #[must_use]
    pub fn new(id: ObjectId, kind: Kind, parents: Vec<ObjectId>, level: DeterminismLevel) -> Self {
        Self {
            id,
            kind,
            parents,
            level,
        }
    }

    /// The facts of a sealed [`Object`] — a convenience for callers that already
    /// hold the typed object (e.g. right after sealing an engine result).
    #[must_use]
    pub fn of<B: Body>(object: &Object<B>) -> Self {
        Self {
            id: object.id,
            kind: object.kind.clone(),
            parents: object.parents.clone(),
            level: object.level,
        }
    }
}

/// A read-only window onto the object graph.
///
/// The sole primitive is [`facts`](Self::facts): given an id, return its
/// [`ObjectFacts`], or `Ok(None)` if no such object exists. Absence is a
/// finding, not an error; only a genuine backend or decode fault is a
/// [`SourceError`].
pub trait PublicationObjectSource {
    /// The facts of the object at `id`, or `Ok(None)` if it is not present.
    ///
    /// # Errors
    /// [`SourceError`] if the backend fails or a stored object's header cannot be
    /// decoded. **Not** returned for a simple absence.
    fn facts(&self, id: ObjectId) -> Result<Option<ObjectFacts>, SourceError>;
}

/// Just the header fields [`StoreSource`] needs from a stored object's JSON — the
/// rest of the envelope is ignored by serde. The `kind` comes from the store's
/// own record, so it is not re-read here.
#[derive(Deserialize)]
struct FactsHeader {
    parents: Vec<ObjectId>,
    level: DeterminismLevel,
}

/// A [`PublicationObjectSource`] over any [`ObjectStore`].
///
/// It reads the type-erased record at an id and projects the header fields the
/// verifier needs — without knowing the body type. This is the production path:
/// verify a publication against the same content-addressed store the objects
/// live in.
#[derive(Debug, Clone, Copy)]
pub struct StoreSource<'s, S: ObjectStore + ?Sized> {
    store: &'s S,
}

impl<'s, S: ObjectStore + ?Sized> StoreSource<'s, S> {
    /// Wrap a store as an object source.
    #[must_use]
    pub fn new(store: &'s S) -> Self {
        Self { store }
    }
}

impl<S: ObjectStore + ?Sized> PublicationObjectSource for StoreSource<'_, S> {
    fn facts(&self, id: ObjectId) -> Result<Option<ObjectFacts>, SourceError> {
        let Some(record) = self.store.get_raw(id)
        else
        {
            return Ok(None);
        };
        let header: FactsHeader = serde_json::from_slice(&record.bytes)
            .map_err(|e| SourceError::Decode(format!("{id}: {e}")))?;
        Ok(Some(ObjectFacts {
            id,
            kind: record.kind,
            parents: header.parents,
            level: header.level,
        }))
    }
}

/// An in-memory [`PublicationObjectSource`] built from explicit [`ObjectFacts`].
///
/// Useful when a caller already holds the facts (or is composing a graph to
/// check) and does not want to stand up a full store. Deterministic: lookups are
/// exact and iteration order never matters.
#[derive(Debug, Clone, Default)]
pub struct MapSource {
    facts: BTreeMap<ObjectId, ObjectFacts>,
}

impl MapSource {
    /// An empty source.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert (or replace) the facts for an object.
    pub fn insert(&mut self, facts: ObjectFacts) {
        self.facts.insert(facts.id, facts);
    }

    /// Builder-style [`insert`](Self::insert).
    #[must_use]
    pub fn with(mut self, facts: ObjectFacts) -> Self {
        self.insert(facts);
        self
    }

    /// How many objects the source knows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.facts.len()
    }

    /// Whether the source is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }
}

impl PublicationObjectSource for MapSource {
    fn facts(&self, id: ObjectId) -> Result<Option<ObjectFacts>, SourceError> {
        Ok(self.facts.get(&id).cloned())
    }
}

/// The transitive dependency closure of `roots`: the set containing each root
/// and every ancestor reachable by following provenance parents through
/// `source`.
///
/// The result is a [`BTreeSet`], so it is deterministic regardless of traversal
/// order. A root (or parent) that the source does not know is simply a leaf of
/// the walk — its *absence* is reported elsewhere; here it just contributes no
/// further ancestors. This is the mechanism behind the "declared scope" check:
/// an object a claim binds to is *in scope* iff it lies in this closure.
///
/// # Errors
/// [`SourceError`] if the source faults while resolving any object.
pub fn dependency_closure<S: PublicationObjectSource + ?Sized>(
    source: &S,
    roots: &[ObjectId],
) -> Result<BTreeSet<ObjectId>, SourceError> {
    let mut seen: BTreeSet<ObjectId> = BTreeSet::new();
    let mut stack: Vec<ObjectId> = roots.to_vec();
    while let Some(id) = stack.pop()
    {
        if !seen.insert(id)
        {
            continue;
        }
        if let Some(facts) = source.facts(id)?
        {
            for parent in facts.parents
            {
                if !seen.contains(&parent)
                {
                    stack.push(parent);
                }
            }
        }
    }
    Ok(seen)
}
