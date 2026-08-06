//! # Graph inspection & query primitives
//!
//! Read-only causal analyses over a [`MemoryGraph`] that power three
//! inspection subcommands:
//!
//! - `ccos top`    — the hottest nodes by causal score (the working set).
//! - `ccos blame`  — a node's upstream *causes* and downstream *blast radius*.
//! - `ccos export` — the causal graph as GraphML for interactive graph tools.
//!
//! Everything here is **deterministic**: neighbour lists are sorted before
//! traversal and results are totally ordered, so the same graph always yields
//! byte-identical output (a core CCOS invariant — see the crate docs).
//!
//! ## Edge direction
//!
//! CCOS edges point `source → target` (container/dependent → contained/
//! dependency), the same direction failures propagate in
//! [`MemoryGraph::propagate_failure`]. A [`Direction::Downstream`] walk
//! therefore yields the **impact set** (what breaks if the origin breaks) and a
//! [`Direction::Upstream`] walk yields the **causes** (what the origin rests on).

use crate::memory::{MemoryGraph, NodeId, NodeType};
use std::collections::{BTreeMap, HashSet, VecDeque};

/// A node reached during a causal walk, annotated with its distance (in edges)
/// from the origin and the causal score CCOS currently assigns it.
#[derive(Debug, Clone, PartialEq)]
pub struct Reached {
    /// Identifier of the reached node.
    pub id: NodeId,
    /// Number of edges between the origin and this node (origin itself is 0 and
    /// is never included in results).
    pub distance: u32,
    /// The node's causal score at walk time (see [`MemoryGraph::compute_node_score`]).
    pub score: f64,
}

/// Direction of a causal walk over the graph's directed edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Follow edges `source → target` — the same way failures propagate. Yields
    /// the origin's **blast radius**: everything affected if the origin fails.
    Downstream,
    /// Follow edges `target → source` — the inverse. Yields the origin's
    /// **causes**: everything it transitively contains or depends upon.
    Upstream,
}

/// Breadth-first causal walk from `origin` up to `max_depth` edges away, in the
/// requested [`Direction`]. The origin is excluded; every other node appears at
/// most once (at its shortest distance). Results are ordered by `(distance,
/// score desc, id)` for deterministic output.
///
/// Returns an empty vector if `origin` is not present in the graph.
pub fn walk(
    graph: &MemoryGraph,
    origin: &NodeId,
    max_depth: u32,
    direction: Direction,
) -> Vec<Reached> {
    if !graph.nodes.contains_key(origin) {
        return Vec::new();
    }

    // Build a sorted adjacency map in the requested direction so traversal is
    // independent of edge insertion order.
    let mut adj: BTreeMap<&NodeId, Vec<&NodeId>> = BTreeMap::new();
    for e in &graph.edges {
        let (from, to) = match direction {
            Direction::Downstream => (&e.source, &e.target),
            Direction::Upstream => (&e.target, &e.source),
        };
        adj.entry(from).or_default().push(to);
    }
    for neighbours in adj.values_mut() {
        neighbours.sort();
        neighbours.dedup();
    }

    let mut visited: HashSet<&NodeId> = HashSet::new();
    visited.insert(origin);
    let mut queue: VecDeque<(&NodeId, u32)> = VecDeque::new();
    queue.push_back((origin, 0));

    let mut out: Vec<Reached> = Vec::new();
    while let Some((node, dist)) = queue.pop_front() {
        if dist >= max_depth {
            continue;
        }
        if let Some(neighbours) = adj.get(node) {
            for &next in neighbours {
                if visited.insert(next) {
                    let score = graph
                        .nodes
                        .get(next)
                        .map(|n| graph.compute_node_score(n))
                        .unwrap_or(0.0);
                    out.push(Reached {
                        id: next.clone(),
                        distance: dist + 1,
                        score,
                    });
                    queue.push_back((next, dist + 1));
                }
            }
        }
    }

    out.sort_by(|a, b| {
        a.distance
            .cmp(&b.distance)
            .then(
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
            .then_with(|| a.id.cmp(&b.id))
    });
    out
}

/// Downstream blast radius of `origin`: every node affected if it fails, up to
/// `max_depth` edges away. Convenience wrapper over [`walk`] with
/// [`Direction::Downstream`].
pub fn impact_set(graph: &MemoryGraph, origin: &NodeId, max_depth: u32) -> Vec<Reached> {
    walk(graph, origin, max_depth, Direction::Downstream)
}

/// Upstream causes of `origin`: everything it transitively contains or depends
/// upon, up to `max_depth` edges away. Convenience wrapper over [`walk`] with
/// [`Direction::Upstream`].
pub fn source_set(graph: &MemoryGraph, origin: &NodeId, max_depth: u32) -> Vec<Reached> {
    walk(graph, origin, max_depth, Direction::Upstream)
}

/// The `limit` hottest nodes by causal score (descending, ties broken by id).
/// This is the "working set" the kernel would page into a context window first.
pub fn hot_set(graph: &MemoryGraph, limit: usize) -> Vec<(NodeId, f64)> {
    graph.get_node_scores().into_iter().take(limit).collect()
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn node_type_name(t: &NodeType) -> &'static str {
    match t {
        NodeType::Module => "Module",
        NodeType::Symbol => "Symbol",
        NodeType::ContextBlock => "ContextBlock",
        NodeType::AnalysisResult => "AnalysisResult",
        NodeType::CodeRegion => "CodeRegion",
        NodeType::Unknown => "Unknown",
    }
}

/// Render the causal graph as [GraphML](http://graphml.graphdrawing.org/) — an
/// XML interchange format consumed by Gephi, yEd, Cytoscape, networkx, etc.
///
/// Node data carries `label`, `type` and the causal `score`; edge data carries
/// `weight` and `type`. Nodes and edges are emitted in deterministic
/// (id-sorted) order so the output is reproducible and diff-friendly.
pub fn to_graphml(graph: &MemoryGraph) -> String {
    let mut out = String::new();
    out.push_str(r#"<?xml version="1.0" encoding="UTF-8"?>"#);
    out.push('\n');
    out.push_str(r#"<graphml xmlns="http://graphml.graphdrawing.org/xmlns">"#);
    out.push('\n');
    // Attribute declarations.
    out.push_str(r#"  <key id="label" for="node" attr.name="label" attr.type="string"/>"#);
    out.push('\n');
    out.push_str(r#"  <key id="type" for="node" attr.name="type" attr.type="string"/>"#);
    out.push('\n');
    out.push_str(r#"  <key id="score" for="node" attr.name="score" attr.type="double"/>"#);
    out.push('\n');
    out.push_str(r#"  <key id="weight" for="edge" attr.name="weight" attr.type="double"/>"#);
    out.push('\n');
    out.push_str(r#"  <key id="etype" for="edge" attr.name="type" attr.type="string"/>"#);
    out.push('\n');
    out.push_str(r#"  <graph id="ccos" edgedefault="directed">"#);
    out.push('\n');

    let mut ids: Vec<&NodeId> = graph.nodes.keys().collect();
    ids.sort();
    for id in &ids {
        let n = &graph.nodes[*id];
        let score = graph.compute_node_score(n);
        out.push_str(&format!("    <node id=\"{}\">\n", xml_escape(&id.0)));
        out.push_str(&format!(
            "      <data key=\"label\">{}</data>\n",
            xml_escape(&n.label)
        ));
        out.push_str(&format!(
            "      <data key=\"type\">{}</data>\n",
            node_type_name(&n.node_type)
        ));
        out.push_str(&format!("      <data key=\"score\">{score:.6}</data>\n"));
        out.push_str("    </node>\n");
    }

    let mut edges: Vec<&crate::memory::GraphEdge> = graph.edges.iter().collect();
    edges.sort_by(|a, b| {
        a.source
            .cmp(&b.source)
            .then_with(|| a.target.cmp(&b.target))
            .then_with(|| format!("{:?}", a.edge_type).cmp(&format!("{:?}", b.edge_type)))
    });
    for (i, e) in edges.iter().enumerate() {
        out.push_str(&format!(
            "    <edge id=\"e{}\" source=\"{}\" target=\"{}\">\n",
            i,
            xml_escape(&e.source.0),
            xml_escape(&e.target.0)
        ));
        out.push_str(&format!(
            "      <data key=\"weight\">{:.6}</data>\n",
            e.weight
        ));
        out.push_str(&format!(
            "      <data key=\"etype\">{:?}</data>\n",
            e.edge_type
        ));
        out.push_str("    </edge>\n");
    }

    out.push_str("  </graph>\n");
    out.push_str("</graphml>\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::EdgeType;

    /// a → b → c → d (a chain), plus a → e (a branch).
    fn chain_graph() -> MemoryGraph {
        let mut g = MemoryGraph::new(0.2, 1000);
        for id in ["a", "b", "c", "d", "e"] {
            g.upsert_node(id.into(), id.into(), "".into(), NodeType::Module);
        }
        g.add_edge("a".into(), "b".into(), 1.0, EdgeType::DependsOn);
        g.add_edge("b".into(), "c".into(), 1.0, EdgeType::DependsOn);
        g.add_edge("c".into(), "d".into(), 1.0, EdgeType::DependsOn);
        g.add_edge("a".into(), "e".into(), 1.0, EdgeType::Contains);
        g
    }

    #[test]
    fn impact_set_is_downstream_reachability() {
        let g = chain_graph();
        let impact = impact_set(&g, &"a".into(), 10);
        let ids: Vec<&str> = impact.iter().map(|r| r.id.0.as_str()).collect();
        // a reaches b, c, d (chain) and e (branch); not itself.
        assert!(
            ids.contains(&"b") && ids.contains(&"c") && ids.contains(&"d") && ids.contains(&"e")
        );
        assert!(!ids.contains(&"a"));
        // distances: b/e at 1, c at 2, d at 3.
        let dist = |name: &str| impact.iter().find(|r| r.id.0 == name).unwrap().distance;
        assert_eq!(dist("b"), 1);
        assert_eq!(dist("e"), 1);
        assert_eq!(dist("c"), 2);
        assert_eq!(dist("d"), 3);
    }

    #[test]
    fn source_set_is_upstream_reachability() {
        let g = chain_graph();
        let causes = source_set(&g, &"d".into(), 10);
        let ids: Vec<&str> = causes.iter().map(|r| r.id.0.as_str()).collect();
        // d is caused by c, b, a (upstream); e is unrelated.
        assert!(ids.contains(&"a") && ids.contains(&"b") && ids.contains(&"c"));
        assert!(!ids.contains(&"e"));
        assert!(!ids.contains(&"d"));
    }

    #[test]
    fn depth_limits_the_walk() {
        let g = chain_graph();
        let shallow = impact_set(&g, &"a".into(), 1);
        let ids: Vec<&str> = shallow.iter().map(|r| r.id.0.as_str()).collect();
        // Only direct neighbours at depth 1.
        assert!(ids.contains(&"b") && ids.contains(&"e"));
        assert!(!ids.contains(&"c") && !ids.contains(&"d"));
    }

    #[test]
    fn missing_origin_yields_empty() {
        let g = chain_graph();
        assert!(impact_set(&g, &"ghost".into(), 5).is_empty());
        assert!(source_set(&g, &"ghost".into(), 5).is_empty());
    }

    #[test]
    fn hot_set_is_sorted_and_capped() {
        let g = chain_graph();
        let hot = hot_set(&g, 3);
        assert_eq!(hot.len(), 3);
        // Descending by score.
        for w in hot.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    #[test]
    fn graphml_is_wellformed_and_deterministic() {
        let g = chain_graph();
        let a = to_graphml(&g);
        let b = to_graphml(&g);
        assert_eq!(a, b, "GraphML export must be deterministic");
        assert!(a.starts_with("<?xml"));
        assert!(a.contains("<graphml"));
        assert!(a.contains("</graphml>"));
        assert!(a.contains(r#"<node id="a">"#));
        assert!(a.contains(r#"source="a" target="b""#));
    }

    #[test]
    fn graphml_escapes_special_characters() {
        let mut g = MemoryGraph::new(0.2, 10);
        g.upsert_node("x".into(), "a<b>&\"c\"".into(), "".into(), NodeType::Symbol);
        let xml = to_graphml(&g);
        assert!(xml.contains("a&lt;b&gt;&amp;&quot;c&quot;"));
        assert!(!xml.contains("a<b>"));
    }
}
