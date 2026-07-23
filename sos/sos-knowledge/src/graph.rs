//! [`KnowledgeGraph`] — a deterministic structural view over stored [`Edge`]s,
//! and the [`Knowledge`] query trait.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use sos_core::{Body, Object, ObjectId};
use sos_store::ObjectStore;

use crate::edge::Edge;
use crate::error::Result;
use crate::relation::Relation;

/// A resolved edge in the graph: the edge object's id plus its endpoints and
/// relation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeRef {
    /// The `Object<Edge>` id (the edge is itself a content-addressed object).
    pub id: ObjectId,
    /// Source node.
    pub from: ObjectId,
    /// Target node.
    pub to: ObjectId,
    /// The typed relation.
    pub relation: Relation,
}

/// An in-memory, deterministic snapshot of the knowledge graph: the typed edges
/// stored in an [`ObjectStore`], indexed for structural queries.
///
/// All query results are **sorted**, so nothing downstream depends on iteration
/// order.
#[derive(Debug, Clone, Default)]
pub struct KnowledgeGraph {
    out: BTreeMap<ObjectId, Vec<(Relation, ObjectId)>>,
    inn: BTreeMap<ObjectId, Vec<(Relation, ObjectId)>>,
    edges: Vec<EdgeRef>,
    nodes: BTreeSet<ObjectId>,
}

impl KnowledgeGraph {
    /// Build the graph by reading every stored [`Edge`] object in `store`.
    /// Non-edge objects are ignored (they are nodes only if an edge references
    /// them).
    ///
    /// # Errors
    /// [`crate::KnowledgeError::MalformedEdge`] if an object tagged `Edge`
    /// cannot be deserialized as an `Object<Edge>`.
    pub fn build<S: ObjectStore + ?Sized>(store: &S) -> Result<Self> {
        let edge_kind = Edge::kind();
        let mut out: BTreeMap<ObjectId, Vec<(Relation, ObjectId)>> = BTreeMap::new();
        let mut inn: BTreeMap<ObjectId, Vec<(Relation, ObjectId)>> = BTreeMap::new();
        let mut edges = Vec::new();
        let mut nodes = BTreeSet::new();

        for oid in store.object_ids()
        {
            let Some(rec) = store.get_raw(oid)
            else
            {
                continue;
            };
            if rec.kind != edge_kind
            {
                continue;
            }
            let obj: Object<Edge> = serde_json::from_slice(&rec.bytes)?;
            let from = obj.body.from;
            let to = obj.body.to;
            let relation = obj.body.relation;
            nodes.insert(from);
            nodes.insert(to);
            out.entry(from).or_default().push((relation.clone(), to));
            inn.entry(to).or_default().push((relation.clone(), from));
            edges.push(EdgeRef {
                id: oid,
                from,
                to,
                relation,
            });
        }

        for adj in out.values_mut()
        {
            adj.sort();
            adj.dedup();
        }
        for adj in inn.values_mut()
        {
            adj.sort();
            adj.dedup();
        }
        edges.sort_by(|a, b| {
            a.from
                .cmp(&b.from)
                .then(a.to.cmp(&b.to))
                .then(a.relation.cmp(&b.relation))
        });

        Ok(Self {
            out,
            inn,
            edges,
            nodes,
        })
    }

    /// The number of nodes (distinct edge endpoints).
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph has no edges.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// The number of edges.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// All nodes, sorted.
    #[must_use]
    pub fn nodes(&self) -> Vec<ObjectId> {
        self.nodes.iter().copied().collect()
    }

    /// All edges, sorted by `(from, to, relation)`.
    #[must_use]
    pub fn edges(&self) -> &[EdgeRef] {
        &self.edges
    }

    /// The outgoing `(relation, target)` pairs from `id`, sorted.
    #[must_use]
    pub fn out_edges(&self, id: ObjectId) -> Vec<(Relation, ObjectId)> {
        self.out.get(&id).cloned().unwrap_or_default()
    }

    /// The incoming `(relation, source)` pairs to `id`, sorted.
    #[must_use]
    pub fn in_edges(&self, id: ObjectId) -> Vec<(Relation, ObjectId)> {
        self.inn.get(&id).cloned().unwrap_or_default()
    }
}

/// The read-side Knowledge Engine syscall: structural queries over the graph.
///
/// Asserting knowledge (adding nodes and edges) is done by storing objects via
/// [`sos_store::TypedStore::put_object`]; this trait is the query surface that
/// engines depend on.
pub trait Knowledge {
    /// Out-neighbours of `id` along `relation`, sorted.
    fn neighbors(&self, id: ObjectId, relation: &Relation) -> Vec<ObjectId>;
    /// In-neighbours of `id` along `relation` (objects pointing at `id`), sorted.
    fn in_neighbors(&self, id: ObjectId, relation: &Relation) -> Vec<ObjectId>;
    /// The relations directly connecting `from` to `to`, sorted.
    fn related(&self, from: ObjectId, to: ObjectId) -> Vec<Relation>;
    /// A shortest directed path `from → … → to` following edges (optionally
    /// restricted to a single `relation`), or `None` if unreachable. The path
    /// includes both endpoints; among equal-length paths the choice is
    /// deterministic (sorted neighbour exploration).
    fn path(
        &self,
        from: ObjectId,
        to: ObjectId,
        relation: Option<&Relation>,
    ) -> Option<Vec<ObjectId>>;
}

impl Knowledge for KnowledgeGraph {
    fn neighbors(&self, id: ObjectId, relation: &Relation) -> Vec<ObjectId> {
        let mut v: Vec<ObjectId> = self
            .out
            .get(&id)
            .into_iter()
            .flatten()
            .filter(|(r, _)| r == relation)
            .map(|(_, t)| *t)
            .collect();
        v.sort();
        v.dedup();
        v
    }

    fn in_neighbors(&self, id: ObjectId, relation: &Relation) -> Vec<ObjectId> {
        let mut v: Vec<ObjectId> = self
            .inn
            .get(&id)
            .into_iter()
            .flatten()
            .filter(|(r, _)| r == relation)
            .map(|(_, s)| *s)
            .collect();
        v.sort();
        v.dedup();
        v
    }

    fn related(&self, from: ObjectId, to: ObjectId) -> Vec<Relation> {
        let mut v: Vec<Relation> = self
            .out
            .get(&from)
            .into_iter()
            .flatten()
            .filter(|(_, t)| *t == to)
            .map(|(r, _)| r.clone())
            .collect();
        v.sort();
        v.dedup();
        v
    }

    fn path(
        &self,
        from: ObjectId,
        to: ObjectId,
        relation: Option<&Relation>,
    ) -> Option<Vec<ObjectId>> {
        if from == to
        {
            return Some(vec![from]);
        }
        let mut prev: BTreeMap<ObjectId, ObjectId> = BTreeMap::new();
        let mut visited: BTreeSet<ObjectId> = BTreeSet::new();
        let mut queue: VecDeque<ObjectId> = VecDeque::new();
        visited.insert(from);
        queue.push_back(from);

        while let Some(cur) = queue.pop_front()
        {
            // Deterministic exploration: sorted, relation-filtered neighbours.
            let mut nbrs: Vec<ObjectId> = self
                .out
                .get(&cur)
                .into_iter()
                .flatten()
                .filter(|(r, _)| relation.is_none_or(|want| r == want))
                .map(|(_, t)| *t)
                .collect();
            nbrs.sort();
            nbrs.dedup();
            for n in nbrs
            {
                if visited.insert(n)
                {
                    prev.insert(n, cur);
                    if n == to
                    {
                        return Some(reconstruct(&prev, from, to));
                    }
                    queue.push_back(n);
                }
            }
        }
        None
    }
}

/// Rebuild a path from the BFS predecessor map.
fn reconstruct(prev: &BTreeMap<ObjectId, ObjectId>, from: ObjectId, to: ObjectId) -> Vec<ObjectId> {
    let mut path = vec![to];
    let mut cur = to;
    while cur != from
    {
        // `prev` is total on the discovered path, so this cannot loop forever.
        let p = prev[&cur];
        path.push(p);
        cur = p;
    }
    path.reverse();
    path
}

#[cfg(test)]
mod tests {
    use super::*;
    use sos_core::{Author, HashAlgo};
    use sos_store::{MemoryStore, TypedStore};

    use crate::edge::seal_edge;

    fn node(tag: &[u8]) -> ObjectId {
        ObjectId::compute(HashAlgo::default(), b"sos-obj:N:v1", tag)
    }

    fn store_edges(pairs: &[(ObjectId, Relation, ObjectId)]) -> MemoryStore {
        let mut s = MemoryStore::new();
        for (f, r, t) in pairs
        {
            s.put_object(&seal_edge(*f, *t, r.clone(), Author::engine("test")))
                .unwrap();
        }
        s
    }

    #[test]
    fn neighbors_and_relations() {
        let (a, b, c) = (node(b"a"), node(b"b"), node(b"c"));
        let s = store_edges(&[
            (a, Relation::Specializes, b),
            (a, Relation::Cites, c),
            (a, Relation::Specializes, c),
        ]);
        let kg = KnowledgeGraph::build(&s).unwrap();

        assert_eq!(kg.neighbors(a, &Relation::Specializes), {
            let mut v = vec![b, c];
            v.sort();
            v
        });
        assert_eq!(kg.neighbors(a, &Relation::Cites), vec![c]);
        assert_eq!(kg.in_neighbors(b, &Relation::Specializes), vec![a]);
        assert_eq!(kg.related(a, c), {
            let mut v = vec![Relation::Cites, Relation::Specializes];
            v.sort();
            v
        });
        assert_eq!(kg.edge_count(), 3);
    }

    #[test]
    fn shortest_path_is_found_and_deterministic() {
        // a -> b -> d and a -> c -> d ; both length 2. BFS + sorted exploration
        // yields a single deterministic shortest path.
        let (a, b, c, d) = (node(b"a"), node(b"b"), node(b"c"), node(b"d"));
        let s = store_edges(&[
            (a, Relation::DerivesFrom, b),
            (a, Relation::DerivesFrom, c),
            (b, Relation::DerivesFrom, d),
            (c, Relation::DerivesFrom, d),
        ]);
        let kg = KnowledgeGraph::build(&s).unwrap();
        let p = kg.path(a, d, None).unwrap();
        assert_eq!(p.first(), Some(&a));
        assert_eq!(p.last(), Some(&d));
        assert_eq!(p.len(), 3); // a, {b or c}, d
        assert_eq!(kg.path(a, d, None), kg.path(a, d, None)); // deterministic
    }

    #[test]
    fn path_respects_relation_filter_and_reachability() {
        let (a, b, c) = (node(b"a"), node(b"b"), node(b"c"));
        let s = store_edges(&[(a, Relation::Cites, b), (b, Relation::Contradicts, c)]);
        let kg = KnowledgeGraph::build(&s).unwrap();
        // Following only `cites`, c is unreachable from a.
        assert_eq!(kg.path(a, c, Some(&Relation::Cites)), None);
        // Following any relation, a -> b -> c.
        assert_eq!(kg.path(a, c, None), Some(vec![a, b, c]));
        // Self-path is trivially the single node.
        assert_eq!(kg.path(a, a, None), Some(vec![a]));
    }

    #[test]
    fn empty_graph() {
        let s = MemoryStore::new();
        let kg = KnowledgeGraph::build(&s).unwrap();
        assert!(kg.is_empty());
        assert_eq!(kg.edge_count(), 0);
        assert!(kg.neighbors(node(b"x"), &Relation::IsA).is_empty());
    }
}
