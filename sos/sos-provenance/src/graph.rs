//! [`ProvenanceGraph`] — a queryable snapshot of the object graph's edges.
//!
//! Built once from any [`ObjectStore`], it answers ancestor/descendant/root/tip
//! queries in memory. Reachability follows `parents`, so *ancestors* are what an
//! object was derived from ("why do we believe X") and *descendants* are what
//! was derived from it ("what breaks if X is retracted").

use std::collections::{BTreeMap, BTreeSet};

use sos_core::ObjectId;
use sos_store::{ObjectHeader, ObjectStore};

use crate::error::Result;

/// An in-memory, deterministic snapshot of the provenance DAG.
///
/// All query results are **sorted** (by [`ObjectId`]) so nothing downstream
/// depends on iteration order.
#[derive(Debug, Clone, Default)]
pub struct ProvenanceGraph {
    /// id → its direct parents (as authored).
    parents: BTreeMap<ObjectId, Vec<ObjectId>>,
    /// id → its direct children (reverse edges), sorted & deduped.
    children: BTreeMap<ObjectId, Vec<ObjectId>>,
    /// every stored object id.
    nodes: BTreeSet<ObjectId>,
}

impl ProvenanceGraph {
    /// Build the graph by scanning every object in `store`.
    ///
    /// # Errors
    /// [`crate::ProvError::MalformedHeader`] if a stored object's bytes are not a
    /// valid object header (were not written via
    /// [`sos_store::TypedStore::put_object`]).
    pub fn build<S: ObjectStore + ?Sized>(store: &S) -> Result<Self> {
        let mut parents = BTreeMap::new();
        let mut children: BTreeMap<ObjectId, Vec<ObjectId>> = BTreeMap::new();
        let mut nodes = BTreeSet::new();
        for oid in store.object_ids()
        {
            nodes.insert(oid);
            if let Some(rec) = store.get_raw(oid)
            {
                let header: ObjectHeader = serde_json::from_slice(&rec.bytes)?;
                for &p in &header.parents
                {
                    children.entry(p).or_default().push(oid);
                }
                parents.insert(oid, header.parents);
            }
        }
        for kids in children.values_mut()
        {
            kids.sort();
            kids.dedup();
        }
        Ok(Self {
            parents,
            children,
            nodes,
        })
    }

    /// The number of objects in the graph.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Every object id, sorted.
    #[must_use]
    pub fn ids(&self) -> Vec<ObjectId> {
        self.nodes.iter().copied().collect()
    }

    /// The direct provenance parents of `id` (empty if none / unknown).
    #[must_use]
    pub fn direct_parents(&self, id: ObjectId) -> &[ObjectId] {
        self.parents.get(&id).map_or(&[], Vec::as_slice)
    }

    /// The direct children of `id` (objects that cite it as a parent), sorted.
    #[must_use]
    pub fn direct_children(&self, id: ObjectId) -> Vec<ObjectId> {
        self.children.get(&id).cloned().unwrap_or_default()
    }

    /// All transitive ancestors of `id` — everything it was derived from.
    /// Excludes `id`; sorted.
    #[must_use]
    pub fn ancestors(&self, id: ObjectId) -> Vec<ObjectId> {
        Self::closure(id, &self.parents)
    }

    /// All transitive descendants of `id` — everything derived from it.
    /// Excludes `id`; sorted.
    #[must_use]
    pub fn descendants(&self, id: ObjectId) -> Vec<ObjectId> {
        Self::closure(id, &self.children)
    }

    /// The roots: objects with no parents (a study's questions and axioms),
    /// sorted.
    #[must_use]
    pub fn roots(&self) -> Vec<ObjectId> {
        self.nodes
            .iter()
            .copied()
            .filter(|id| self.parents.get(id).is_none_or(Vec::is_empty))
            .collect()
    }

    /// The tips: objects nothing was derived from (the open frontier / current
    /// conclusions), sorted.
    #[must_use]
    pub fn tips(&self) -> Vec<ObjectId> {
        self.nodes
            .iter()
            .copied()
            .filter(|id| self.children.get(id).is_none_or(Vec::is_empty))
            .collect()
    }

    /// Transitive closure over an adjacency map, starting from `start`'s direct
    /// neighbours (so `start` itself is excluded). Result is sorted.
    fn closure(start: ObjectId, adj: &BTreeMap<ObjectId, Vec<ObjectId>>) -> Vec<ObjectId> {
        let mut seen = BTreeSet::new();
        let mut stack: Vec<ObjectId> = adj.get(&start).cloned().unwrap_or_default();
        while let Some(cur) = stack.pop()
        {
            if !seen.insert(cur)
            {
                continue;
            }
            if let Some(next) = adj.get(&cur)
            {
                stack.extend(next.iter().copied());
            }
        }
        // A DAG never cycles back to `start`, but exclude it defensively.
        seen.remove(&start);
        seen.into_iter().collect()
    }
}

/// Convenience: the transitive ancestors of `id` in `store`.
///
/// # Errors
/// Propagates [`ProvenanceGraph::build`]'s errors.
pub fn ancestors<S: ObjectStore + ?Sized>(store: &S, id: ObjectId) -> Result<Vec<ObjectId>> {
    Ok(ProvenanceGraph::build(store)?.ancestors(id))
}

/// Convenience: the transitive descendants of `id` in `store`.
///
/// # Errors
/// Propagates [`ProvenanceGraph::build`]'s errors.
pub fn descendants<S: ObjectStore + ?Sized>(store: &S, id: ObjectId) -> Result<Vec<ObjectId>> {
    Ok(ProvenanceGraph::build(store)?.descendants(id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sos_core::{HashAlgo, Kind};
    use sos_store::{MemoryStore, ObjectStore, StoredRecord};

    fn oid(tag: &[u8]) -> ObjectId {
        ObjectId::compute(HashAlgo::default(), b"sos-obj:T:v1", tag)
    }

    /// Store a synthetic object header (id + parents) without needing a full
    /// `sos-core` object — exercises the graph logic in isolation.
    fn put_node(store: &mut MemoryStore, id: ObjectId, parents: &[ObjectId]) {
        let parents_hex: Vec<String> = parents.iter().map(ObjectId::to_prefixed_hex).collect();
        let v = serde_json::json!({ "id": id.to_prefixed_hex(), "parents": parents_hex });
        store.put_raw(
            id,
            StoredRecord::new(Kind::new("T", 1), serde_json::to_vec(&v).unwrap()),
        );
    }

    #[test]
    fn diamond_dag_queries() {
        // a is root; b and c derive from a; d derives from b and c.
        //     a
        //    / \
        //   b   c
        //    \ /
        //     d
        let (a, b, c, d) = (oid(b"a"), oid(b"b"), oid(b"c"), oid(b"d"));
        let mut s = MemoryStore::new();
        put_node(&mut s, a, &[]);
        put_node(&mut s, b, &[a]);
        put_node(&mut s, c, &[a]);
        put_node(&mut s, d, &[b, c]);

        let g = ProvenanceGraph::build(&s).unwrap();
        assert_eq!(g.len(), 4);

        // ancestors of d = {a,b,c}; of a = {}.
        let mut anc_d = g.ancestors(d);
        anc_d.sort();
        let mut want = vec![a, b, c];
        want.sort();
        assert_eq!(anc_d, want);
        assert!(g.ancestors(a).is_empty());

        // descendants of a = {b,c,d}; of d = {}.
        let mut desc_a = g.descendants(a);
        desc_a.sort();
        let mut want2 = vec![b, c, d];
        want2.sort();
        assert_eq!(desc_a, want2);
        assert!(g.descendants(d).is_empty());

        assert_eq!(g.roots(), vec![a]);
        assert_eq!(g.tips(), vec![d]);
    }

    #[test]
    fn results_are_sorted_and_deterministic() {
        let (a, b, c) = (oid(b"1"), oid(b"2"), oid(b"3"));
        let mut s = MemoryStore::new();
        put_node(&mut s, a, &[]);
        put_node(&mut s, b, &[a]);
        put_node(&mut s, c, &[a]);
        let g = ProvenanceGraph::build(&s).unwrap();
        // descendants(a) returns sorted, and equals a re-query.
        let d1 = g.descendants(a);
        let mut manual = d1.clone();
        manual.sort();
        assert_eq!(d1, manual);
        assert_eq!(d1, ProvenanceGraph::build(&s).unwrap().descendants(a));
    }
}
