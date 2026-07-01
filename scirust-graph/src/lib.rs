//! Graph pattern matching: subgraph isomorphism, motif discovery, community detection.

pub mod dag;

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

// ─── Graph Representation ───────────────────────────────────────────────────

/// An undirected graph.
#[derive(Debug, Clone)]
pub struct Graph {
    pub n_nodes: usize,
    pub adjacency: Vec<Vec<usize>>,
    pub node_labels: Vec<String>,
    pub edge_labels: HashMap<(usize, usize), String>,
}

impl Graph {
    pub fn new(n_nodes: usize) -> Self {
        Self {
            n_nodes,
            adjacency: vec![Vec::new(); n_nodes],
            node_labels: vec![String::new(); n_nodes],
            edge_labels: HashMap::new(),
        }
    }

    pub fn add_edge(&mut self, u: usize, v: usize) {
        if !self.adjacency[u].contains(&v)
        {
            self.adjacency[u].push(v);
            self.adjacency[v].push(u);
        }
    }

    pub fn add_edge_labeled(&mut self, u: usize, v: usize, label: &str) {
        self.add_edge(u, v);
        self.edge_labels
            .insert((u.min(v), u.max(v)), label.to_string());
    }

    pub fn set_node_label(&mut self, node: usize, label: &str) {
        self.node_labels[node] = label.to_string();
    }

    pub fn degree(&self, node: usize) -> usize {
        self.adjacency[node].len()
    }

    pub fn n_edges(&self) -> usize {
        self.adjacency.iter().map(|a| a.len()).sum::<usize>() / 2
    }

    pub fn neighbors(&self, node: usize) -> &[usize] {
        &self.adjacency[node]
    }

    /// Check if two nodes are adjacent.
    pub fn are_adjacent(&self, u: usize, v: usize) -> bool {
        self.adjacency[u].contains(&v)
    }

    /// Get edge label.
    pub fn edge_label(&self, u: usize, v: usize) -> Option<&str> {
        self.edge_labels
            .get(&(u.min(v), u.max(v)))
            .map(|s| s.as_str())
    }

    /// BFS from a node.
    pub fn bfs(&self, start: usize) -> Vec<usize> {
        let mut visited = vec![false; self.n_nodes];
        let mut queue = VecDeque::new();
        let mut order = Vec::new();

        visited[start] = true;
        queue.push_back(start);

        while let Some(node) = queue.pop_front()
        {
            order.push(node);
            for &neighbor in &self.adjacency[node]
            {
                if !visited[neighbor]
                {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
        order
    }

    /// DFS from a node.
    pub fn dfs(&self, start: usize) -> Vec<usize> {
        let mut visited = vec![false; self.n_nodes];
        let mut order = Vec::new();
        self.dfs_recursive(start, &mut visited, &mut order);
        order
    }

    fn dfs_recursive(&self, node: usize, visited: &mut Vec<bool>, order: &mut Vec<usize>) {
        visited[node] = true;
        order.push(node);
        for &neighbor in &self.adjacency[node]
        {
            if !visited[neighbor]
            {
                self.dfs_recursive(neighbor, visited, order);
            }
        }
    }

    /// Shortest path between two nodes (BFS).
    pub fn shortest_path(&self, start: usize, end: usize) -> Option<Vec<usize>> {
        if start == end
        {
            return Some(vec![start]);
        }

        let mut visited = vec![false; self.n_nodes];
        let mut parent = vec![usize::MAX; self.n_nodes];
        let mut queue = VecDeque::new();

        visited[start] = true;
        queue.push_back(start);

        while let Some(node) = queue.pop_front()
        {
            for &neighbor in &self.adjacency[node]
            {
                if !visited[neighbor]
                {
                    visited[neighbor] = true;
                    parent[neighbor] = node;
                    if neighbor == end
                    {
                        let mut path = vec![end];
                        let mut current = end;
                        while parent[current] != usize::MAX
                        {
                            current = parent[current];
                            path.push(current);
                        }
                        path.reverse();
                        return Some(path);
                    }
                    queue.push_back(neighbor);
                }
            }
        }
        None
    }

    /// Get all nodes within a given distance.
    pub fn nodes_within_distance(&self, start: usize, max_dist: usize) -> Vec<(usize, usize)> {
        let mut visited = vec![false; self.n_nodes];
        let mut result = Vec::new();
        let mut queue = VecDeque::new();

        visited[start] = true;
        queue.push_back((start, 0));

        while let Some((node, dist)) = queue.pop_front()
        {
            result.push((node, dist));
            if dist < max_dist
            {
                for &neighbor in &self.adjacency[node]
                {
                    if !visited[neighbor]
                    {
                        visited[neighbor] = true;
                        queue.push_back((neighbor, dist + 1));
                    }
                }
            }
        }
        result
    }
}

// ─── Subgraph Isomorphism ───────────────────────────────────────────────────

/// Check if pattern is a subgraph of the target graph (VF2-like algorithm).
pub fn subgraph_isomorphism(pattern: &Graph, target: &Graph) -> Vec<HashMap<usize, usize>> {
    let mut mappings = Vec::new();
    let mut state = IsoState {
        pattern,
        target,
        mapping: HashMap::new(),
        target_matched: vec![false; target.n_nodes],
    };

    iso_recursive(&mut state, 0, &mut mappings);
    mappings
}

struct IsoState<'a> {
    pattern: &'a Graph,
    target: &'a Graph,
    // Pattern nodes are matched in increasing `depth` order, so the recursion
    // depth already records which ones are bound; only the target side needs an
    // explicit "already used" marker.
    mapping: HashMap<usize, usize>,
    target_matched: Vec<bool>,
}

fn iso_recursive(state: &mut IsoState, depth: usize, mappings: &mut Vec<HashMap<usize, usize>>) {
    if depth == state.pattern.n_nodes
    {
        mappings.push(state.mapping.clone());
        return;
    }

    // Try to match pattern node `depth` with each unmatched target node
    for t in 0..state.target.n_nodes
    {
        if state.target_matched[t]
        {
            continue;
        }

        // Check if node labels match
        if !state.pattern.node_labels[depth].is_empty()
            && state.pattern.node_labels[depth] != state.target.node_labels[t]
        {
            continue;
        }

        // Check degree constraint
        if state.pattern.degree(depth) > state.target.degree(t)
        {
            continue;
        }

        // Check adjacency consistency with already matched nodes
        let mut consistent = true;
        for (&p, &m) in &state.mapping
        {
            let p_adj = state.pattern.are_adjacent(p, depth);
            let t_adj = state.target.are_adjacent(m, t);
            if p_adj != t_adj
            {
                consistent = false;
                break;
            }
            // Check edge labels
            if p_adj
            {
                if let Some(label) = state.pattern.edge_label(p, depth)
                {
                    if state.target.edge_label(m, t) != Some(label)
                    {
                        consistent = false;
                        break;
                    }
                }
            }
        }

        if !consistent
        {
            continue;
        }

        // Make assignment
        state.mapping.insert(depth, t);
        state.target_matched[t] = true;

        iso_recursive(state, depth + 1, mappings);

        // Backtrack
        state.mapping.remove(&depth);
        state.target_matched[t] = false;
    }
}

/// Check if pattern is isomorphic to target (exact match).
pub fn graph_isomorphism(pattern: &Graph, target: &Graph) -> bool {
    if pattern.n_nodes != target.n_nodes
    {
        return false;
    }
    if pattern.n_edges() != target.n_edges()
    {
        return false;
    }
    !subgraph_isomorphism(pattern, target).is_empty()
}

// ─── Motif Discovery ────────────────────────────────────────────────────────

/// Find all motifs (common subgraphs) of a given size.
pub fn find_motifs(graph: &Graph, motif_size: usize) -> Vec<Motif> {
    let mut motif_counts: HashMap<Vec<Vec<usize>>, usize> = HashMap::new();

    // Enumerate all subsets of nodes of the given size
    let combinations = combinations(graph.n_nodes, motif_size);

    for combo in &combinations
    {
        // Extract subgraph induced by these nodes
        let subgraph = induced_subgraph(graph, combo);
        let canonical = canonical_form(&subgraph);
        *motif_counts.entry(canonical).or_insert(0) += 1;
    }

    // Convert to Motif structs
    motif_counts
        .into_iter()
        .map(|(adj, count)| Motif {
            nodes: adj.len(),
            edges: adj.iter().map(|a| a.len()).sum::<usize>() / 2,
            adjacency: adj,
            count,
            frequency: count as f64 / combinations.len() as f64,
        })
        .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Motif {
    pub nodes: usize,
    pub edges: usize,
    pub adjacency: Vec<Vec<usize>>,
    pub count: usize,
    pub frequency: f64,
}

fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut current = Vec::new();
    combinations_recursive(n, k, 0, &mut current, &mut result);
    result
}

fn combinations_recursive(
    n: usize,
    k: usize,
    start: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Vec<usize>>,
) {
    if current.len() == k
    {
        result.push(current.clone());
        return;
    }
    for i in start..n
    {
        current.push(i);
        combinations_recursive(n, k, i + 1, current, result);
        current.pop();
    }
}

fn induced_subgraph(graph: &Graph, nodes: &[usize]) -> Graph {
    let node_map: HashMap<usize, usize> = nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();
    let mut sub = Graph::new(nodes.len());

    for (i, &n) in nodes.iter().enumerate()
    {
        sub.node_labels[i] = graph.node_labels[n].clone();
    }

    for (i, &n) in nodes.iter().enumerate()
    {
        for &neighbor in &graph.adjacency[n]
        {
            if let Some(&mapped) = node_map.get(&neighbor)
            {
                if i < mapped
                {
                    match graph.edge_label(n, neighbor)
                    {
                        Some(label) => sub.add_edge_labeled(i, mapped, label),
                        None => sub.add_edge(i, mapped),
                    }
                }
            }
        }
    }
    sub
}

/// Canonical structural form of a (small) graph, invariant under node
/// relabeling. Two graphs are isomorphic (ignoring labels) iff their canonical
/// forms are equal.
///
/// For each permutation of the node ids we build the relabeled, sorted
/// adjacency lists and keep the lexicographically smallest representative. This
/// is exponential in the node count but is only ever applied to motifs (size 3
/// or 4), where it is exact and cheap.
fn canonical_form(graph: &Graph) -> Vec<Vec<usize>> {
    let n = graph.n_nodes;
    let mut best: Option<Vec<Vec<usize>>> = None;

    for perm in permutations(n)
    {
        // perm[old] = new id for that node.
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (old, neighbors) in graph.adjacency.iter().enumerate()
        {
            for &nb in neighbors
            {
                adj[perm[old]].push(perm[nb]);
            }
        }
        for a in &mut adj
        {
            a.sort_unstable();
        }
        let smaller = match &best
        {
            Some(current) => adj < *current,
            None => true,
        };
        if smaller
        {
            best = Some(adj);
        }
    }

    best.unwrap_or_default()
}

/// All permutations of `0..n` as `Vec<usize>` (Heap-free, recursive).
fn permutations(n: usize) -> Vec<Vec<usize>> {
    let mut items: Vec<usize> = (0..n).collect();
    let mut result = Vec::new();
    permute_recursive(&mut items, 0, &mut result);
    result
}

fn permute_recursive(items: &mut Vec<usize>, start: usize, result: &mut Vec<Vec<usize>>) {
    if start == items.len()
    {
        result.push(items.clone());
        return;
    }
    for i in start..items.len()
    {
        items.swap(start, i);
        permute_recursive(items, start + 1, result);
        items.swap(start, i);
    }
}

// ─── Community Detection ────────────────────────────────────────────────────

/// Label Propagation Algorithm for community detection.
pub fn label_propagation(graph: &Graph, max_iterations: usize) -> Vec<usize> {
    let mut labels: Vec<usize> = (0..graph.n_nodes).collect();

    for _ in 0..max_iterations
    {
        let mut changed = false;
        let mut order: Vec<usize> = (0..graph.n_nodes).collect();
        // Shuffle order (simplified: use node index)
        order.sort_by_key(|&x| (x * 7 + 13) % graph.n_nodes); // pseudo-random

        for &node in &order
        {
            if graph.adjacency[node].is_empty()
            {
                continue;
            }

            // Count labels of neighbors
            let mut label_counts: HashMap<usize, usize> = HashMap::new();
            for &neighbor in &graph.adjacency[node]
            {
                *label_counts.entry(labels[neighbor]).or_insert(0) += 1;
            }

            // Find most common label. `label_counts` is a HashMap, whose
            // iteration order is randomized, so we break count ties by picking
            // the smallest label to keep the result deterministic.
            let max_count = label_counts.values().copied().max().unwrap_or(0);
            let new_label = label_counts
                .iter()
                .filter(|(_, &c)| c == max_count)
                .map(|(&l, _)| l)
                .min()
                .unwrap_or(labels[node]);
            if labels[node] != new_label
            {
                labels[node] = new_label;
                changed = true;
            }
        }

        if !changed
        {
            break;
        }
    }

    // Relabel to 0..k
    let mut label_map = HashMap::new();
    let mut next_label = 0;
    for &label in &labels
    {
        label_map.entry(label).or_insert_with(|| {
            let l = next_label;
            next_label += 1;
            l
        });
    }
    labels
        .iter()
        .map(|&l| *label_map.get(&l).unwrap())
        .collect()
}

/// Newman–Girvan modularity `Q` of a community assignment — how much more
/// intra-community edge density the partition has than a degree-preserving
/// random graph: `Q = (1/2m) Σ_ij [A_ij − k_i·k_j/(2m)] δ(c_i, c_j)`. Higher `Q`
/// means stronger community structure. This *scores* a given partition; it does
/// not optimise or detect communities (see `label_propagation` / `girvan_newman`).
pub fn modularity(graph: &Graph, communities: &[usize]) -> f64 {
    let m = graph.n_edges() as f64;
    if m < 1.0
    {
        return 0.0;
    }

    let mut q = 0.0;
    for u in 0..graph.n_nodes
    {
        for v in 0..graph.n_nodes
        {
            if communities[u] == communities[v]
            {
                let a_uv = if graph.are_adjacent(u, v) { 1.0 } else { 0.0 };
                let k_u = graph.degree(u) as f64;
                let k_v = graph.degree(v) as f64;
                q += a_uv - (k_u * k_v) / (2.0 * m);
            }
        }
    }
    q / (2.0 * m)
}

/// Girvan-Newman community detection.
///
/// Repeatedly removes the edge with the highest betweenness until the graph
/// splits into at least `n_communities` connected components, then returns the
/// connected-component label of every node. If the graph cannot be split that
/// far (i.e. all edges are removed first), the components of the resulting
/// edgeless graph are returned.
pub fn girvan_newman(graph: &Graph, n_communities: usize) -> Vec<usize> {
    let mut g = graph.clone();

    // A graph's communities under Girvan-Newman are its connected components.
    // We remove the highest-betweenness edge until enough components appear.
    loop
    {
        let components = connected_components(&g);
        if count_communities(&components) >= n_communities || g.n_edges() == 0
        {
            return components;
        }

        // Compute edge betweenness and drop the single highest-scoring edge.
        // `betweenness` is a HashMap with randomized iteration order, so ties on
        // the betweenness score are broken by the (smaller) edge key to keep the
        // choice — and hence the whole result — deterministic.
        let betweenness = edge_betweenness(&g);
        let Some((&(u, v), _)) = betweenness.iter().max_by(|&(ea, sa), &(eb, sb)| {
            sa.partial_cmp(sb).unwrap().then_with(|| eb.cmp(ea))
        })
        else
        {
            return components;
        };

        g.adjacency[u].retain(|&x| x != v);
        g.adjacency[v].retain(|&x| x != u);
    }
}

fn count_communities(communities: &[usize]) -> usize {
    let mut seen = HashSet::new();
    for &c in communities
    {
        seen.insert(c);
    }
    seen.len()
}

/// Label every node with the id of its connected component.
///
/// Component ids are assigned in increasing order of the smallest node they
/// contain, so a connected graph always yields all-zeros and the labelling is
/// deterministic.
pub fn connected_components(graph: &Graph) -> Vec<usize> {
    let mut component = vec![usize::MAX; graph.n_nodes];
    let mut next = 0;

    for start in 0..graph.n_nodes
    {
        if component[start] != usize::MAX
        {
            continue;
        }
        // BFS the component reachable from `start`.
        component[start] = next;
        let mut queue = VecDeque::new();
        queue.push_back(start);
        while let Some(node) = queue.pop_front()
        {
            for &neighbor in &graph.adjacency[node]
            {
                if component[neighbor] == usize::MAX
                {
                    component[neighbor] = next;
                    queue.push_back(neighbor);
                }
            }
        }
        next += 1;
    }

    component
}

/// Compute edge betweenness centrality.
pub fn edge_betweenness(graph: &Graph) -> HashMap<(usize, usize), f64> {
    let mut betweenness = HashMap::new();

    for s in 0..graph.n_nodes
    {
        // BFS from s
        let mut visited = vec![false; graph.n_nodes];
        let mut distance = vec![-1i32; graph.n_nodes];
        let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); graph.n_nodes];
        let mut sigma = vec![0.0f64; graph.n_nodes];
        let mut order = Vec::new();

        visited[s] = true;
        distance[s] = 0;
        sigma[s] = 1.0;
        let mut queue = VecDeque::new();
        queue.push_back(s);

        while let Some(v) = queue.pop_front()
        {
            order.push(v);
            for &w in &graph.adjacency[v]
            {
                if !visited[w]
                {
                    visited[w] = true;
                    distance[w] = distance[v] + 1;
                    queue.push_back(w);
                }
                if distance[w] == distance[v] + 1
                {
                    sigma[w] += sigma[v];
                    predecessors[w].push(v);
                }
            }
        }

        // Back-propagation
        let mut delta = vec![0.0f64; graph.n_nodes];
        for &w in order.iter().rev()
        {
            for &v in &predecessors[w]
            {
                let edge = (v.min(w), v.max(w));
                let contribution = (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                *betweenness.entry(edge).or_insert(0.0) += contribution;
                delta[v] += contribution;
            }
        }
    }

    // Normalize
    let n = graph.n_nodes as f64;
    let norm = if n > 2.0 { (n - 1.0) * (n - 2.0) } else { 1.0 };
    for v in betweenness.values_mut()
    {
        *v /= norm;
    }

    betweenness
}

// ─── Graph Statistics ───────────────────────────────────────────────────────

/// Degree distribution.
pub fn degree_distribution(graph: &Graph) -> HashMap<usize, usize> {
    let mut dist = HashMap::new();
    for node in 0..graph.n_nodes
    {
        let deg = graph.degree(node);
        *dist.entry(deg).or_insert(0) += 1;
    }
    dist
}

/// Clustering coefficient for a node.
pub fn clustering_coefficient(graph: &Graph, node: usize) -> f64 {
    let neighbors = graph.neighbors(node);
    let k = neighbors.len();
    if k < 2
    {
        return 0.0;
    }

    let mut triangles = 0;
    for i in 0..k
    {
        for j in (i + 1)..k
        {
            if graph.are_adjacent(neighbors[i], neighbors[j])
            {
                triangles += 1;
            }
        }
    }

    triangles as f64 / (k * (k - 1) / 2) as f64
}

/// Average clustering coefficient.
pub fn average_clustering(graph: &Graph) -> f64 {
    let sum: f64 = (0..graph.n_nodes)
        .map(|n| clustering_coefficient(graph, n))
        .sum();
    sum / graph.n_nodes as f64
}

/// Graph density.
pub fn density(graph: &Graph) -> f64 {
    let n = graph.n_nodes as f64;
    let m = graph.n_edges() as f64;
    let max_edges = n * (n - 1.0) / 2.0;
    if max_edges < 1.0 { 0.0 } else { m / max_edges }
}

/// Diameter of the graph: the longest shortest-path distance between any pair
/// of nodes that are actually connected.
///
/// Unreachable pairs (in a disconnected graph) are ignored rather than counted
/// as infinite, so the result is the largest finite eccentricity. An empty or
/// fully isolated graph has diameter `0`.
pub fn diameter(graph: &Graph) -> usize {
    let mut max_dist = 0;
    for start in 0..graph.n_nodes
    {
        let distances = bfs_distances(graph, start);
        for &d in &distances
        {
            if d != usize::MAX && d > max_dist
            {
                max_dist = d;
            }
        }
    }
    max_dist
}

fn bfs_distances(graph: &Graph, start: usize) -> Vec<usize> {
    let mut distances = vec![usize::MAX; graph.n_nodes];
    let mut queue = VecDeque::new();

    distances[start] = 0;
    queue.push_back(start);

    while let Some(node) = queue.pop_front()
    {
        for &neighbor in &graph.adjacency[node]
        {
            if distances[neighbor] == usize::MAX
            {
                distances[neighbor] = distances[node] + 1;
                queue.push_back(neighbor);
            }
        }
    }
    distances
}

/// Betweenness centrality for all nodes.
pub fn betweenness_centrality(graph: &Graph) -> Vec<f64> {
    let mut centrality = vec![0.0f64; graph.n_nodes];

    for s in 0..graph.n_nodes
    {
        let mut visited = vec![false; graph.n_nodes];
        let mut distance = vec![-1i32; graph.n_nodes];
        let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); graph.n_nodes];
        let mut sigma = vec![0.0f64; graph.n_nodes];
        let mut order = Vec::new();

        visited[s] = true;
        distance[s] = 0;
        sigma[s] = 1.0;
        let mut queue = VecDeque::new();
        queue.push_back(s);

        while let Some(v) = queue.pop_front()
        {
            order.push(v);
            for &w in &graph.adjacency[v]
            {
                if !visited[w]
                {
                    visited[w] = true;
                    distance[w] = distance[v] + 1;
                    queue.push_back(w);
                }
                if distance[w] == distance[v] + 1
                {
                    sigma[w] += sigma[v];
                    predecessors[w].push(v);
                }
            }
        }

        let mut delta = vec![0.0f64; graph.n_nodes];
        for &w in order.iter().rev()
        {
            for &v in &predecessors[w]
            {
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
            }
            if w != s
            {
                centrality[w] += delta[w];
            }
        }
    }

    // Normalize
    let n = graph.n_nodes as f64;
    let norm = if n > 2.0 { (n - 1.0) * (n - 2.0) } else { 1.0 };
    for c in &mut centrality
    {
        *c /= norm;
    }
    centrality
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle_graph() -> Graph {
        let mut g = Graph::new(3);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 0);
        g
    }

    fn path_graph() -> Graph {
        let mut g = Graph::new(4);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        g
    }

    fn complete_graph() -> Graph {
        let mut g = Graph::new(4);
        for i in 0..4
        {
            for j in (i + 1)..4
            {
                g.add_edge(i, j);
            }
        }
        g
    }

    /// Star S4: center node 0 connected to leaves 1, 2, 3.
    fn star_graph() -> Graph {
        let mut g = Graph::new(4);
        g.add_edge(0, 1);
        g.add_edge(0, 2);
        g.add_edge(0, 3);
        g
    }

    #[test]
    fn test_graph_basic() {
        let g = triangle_graph();
        assert_eq!(g.n_nodes, 3);
        assert_eq!(g.n_edges(), 3);
        assert_eq!(g.degree(0), 2);
    }

    #[test]
    fn test_bfs() {
        let g = path_graph();
        let order = g.bfs(0);
        assert_eq!(order[0], 0);
        assert_eq!(order[1], 1);
    }

    #[test]
    fn test_dfs() {
        let g = path_graph();
        let order = g.dfs(0);
        assert_eq!(order[0], 0);
        assert!(order.contains(&3));
    }

    #[test]
    fn test_shortest_path() {
        let g = path_graph();
        let path = g.shortest_path(0, 3);
        assert!(path.is_some());
        assert_eq!(path.unwrap(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_subgraph_isomorphism_triangle_in_k4() {
        // K4 contains exactly C(4,3) = 4 triangles, and each triangle has 3! = 6
        // automorphisms, so an (induced) subgraph search for a triangle pattern
        // must return 4 * 6 = 24 distinct vertex mappings.
        let pattern = triangle_graph();
        let target = complete_graph();
        let mappings = subgraph_isomorphism(&pattern, &target);
        assert_eq!(mappings.len(), 24, "triangle-in-K4 should have 24 mappings");

        // Every mapping must be a genuine triangle: 3 distinct, mutually adjacent
        // target nodes.
        for m in &mappings
        {
            let img: Vec<usize> = (0..3).map(|p| m[&p]).collect();
            assert_eq!(img.iter().collect::<HashSet<_>>().len(), 3);
            assert!(target.are_adjacent(img[0], img[1]));
            assert!(target.are_adjacent(img[1], img[2]));
            assert!(target.are_adjacent(img[0], img[2]));
        }
    }

    #[test]
    fn test_subgraph_isomorphism_is_induced() {
        // A "path" pattern 0-1-2 (no edge 0-2) must NOT match the triangle's
        // node triple, because induced subgraph isomorphism requires the
        // non-edge 0-2 to map to a non-edge — and the triangle has none.
        let mut path = Graph::new(3);
        path.add_edge(0, 1);
        path.add_edge(1, 2);
        let target = triangle_graph();
        assert!(
            subgraph_isomorphism(&path, &target).is_empty(),
            "open path must not match a triangle under induced isomorphism"
        );

        // But the same path pattern DOES appear inside a 4-path 0-1-2-3.
        let host = path_graph();
        assert!(!subgraph_isomorphism(&path, &host).is_empty());
    }

    #[test]
    fn test_graph_isomorphism_negatives() {
        // Path P4 and star S4 both have 4 nodes and 3 edges but are NOT isomorphic
        // (different degree sequences: [1,2,2,1] vs [3,1,1,1]).
        let p4 = path_graph();
        let star = star_graph();
        assert!(!graph_isomorphism(&p4, &star));

        // Triangle (3 edges) vs open path on 3 nodes (2 edges): different edge
        // counts, so not isomorphic.
        let tri = triangle_graph();
        let mut path3 = Graph::new(3);
        path3.add_edge(0, 1);
        path3.add_edge(1, 2);
        assert!(!graph_isomorphism(&tri, &path3));

        // Two relabelings of P4 must be isomorphic.
        let mut p4_relabeled = Graph::new(4);
        p4_relabeled.add_edge(3, 2);
        p4_relabeled.add_edge(2, 1);
        p4_relabeled.add_edge(1, 0);
        assert!(graph_isomorphism(&p4, &p4_relabeled));
    }

    #[test]
    fn test_graph_isomorphism() {
        let g1 = triangle_graph();
        let mut g2 = Graph::new(3);
        g2.add_edge(2, 0);
        g2.add_edge(0, 1);
        g2.add_edge(1, 2);
        assert!(graph_isomorphism(&g1, &g2));
    }

    #[test]
    fn test_find_motifs_complete_graph() {
        // Every size-3 node subset of K4 induces a triangle, so there is exactly
        // ONE motif class, with count C(4,3) = 4 and frequency 1.0.
        let g = complete_graph();
        let motifs = find_motifs(&g, 3);
        assert_eq!(motifs.len(), 1, "K4 has a single size-3 motif class");
        assert_eq!(motifs[0].nodes, 3);
        assert_eq!(motifs[0].edges, 3);
        assert_eq!(motifs[0].count, 4);
        assert!((motifs[0].frequency - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_find_motifs_canonical_merges_isomorphic() {
        // Path P4 = 0-1-2-3. Its four size-3 subsets are:
        //   {0,1,2} -> path (2 edges)   {1,2,3} -> path (2 edges)
        //   {0,1,3} -> single edge      {0,2,3} -> single edge
        // A correct canonical form must merge the two single-edge subgraphs
        // (which differ only by which node is isolated) into ONE motif class.
        let g = path_graph();
        let motifs = find_motifs(&g, 3);
        assert_eq!(
            motifs.len(),
            2,
            "P4 size-3 motifs should be exactly 2 classes"
        );

        let mut by_edges: HashMap<usize, usize> = HashMap::new();
        for m in &motifs
        {
            by_edges.insert(m.edges, m.count);
        }
        assert_eq!(
            by_edges.get(&1),
            Some(&2),
            "two single-edge induced subgraphs"
        );
        assert_eq!(
            by_edges.get(&2),
            Some(&2),
            "two open-path induced subgraphs"
        );

        // Frequencies must sum to 1 across all classes.
        let total: f64 = motifs.iter().map(|m| m.frequency).sum();
        assert!((total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_label_propagation() {
        let g = complete_graph();
        let communities = label_propagation(&g, 100);
        // Complete graph should have 1 community
        assert_eq!(count_communities(&communities), 1);
    }

    #[test]
    fn test_modularity_single_community_complete_graph() {
        // For K4 with every node in one community, modularity is exactly 0.
        // Q = (1/2m) * sum_{i,j same comm} [A_ij - k_i*k_j/(2m)], summing over
        // ALL ordered pairs including i==j (where A_ii = 0).
        // K4: m=6, k_i=3 for all i, 2m=12.
        //   12 off-diagonal pairs (i!=j): each A_ij - 9/12 = 1 - 0.75 = 0.25 -> +3.0
        //   4 diagonal pairs   (i==j): each 0 - 9/12             = -0.75 -> -3.0
        //   sum = 0 -> Q = 0 / 12 = 0.
        let g = complete_graph();
        let q = modularity(&g, &[0, 0, 0, 0]);
        assert!(
            (q - 0.0).abs() < 1e-9,
            "K4 single-community modularity should be 0, got {q}"
        );
    }

    #[test]
    fn test_modularity_two_communities_beats_trivial() {
        // Two triangles 0-1-2 and 3-4-5 joined by a single bridge edge 2-3.
        // m = 7 edges. Degrees: nodes 2 and 3 have degree 3, the rest degree 2.
        // Splitting into the two obvious communities {0,1,2} and {3,4,5} gives
        // (computed by the modularity definition above) Q = 25/70 = 0.357142857...
        let mut g = Graph::new(6);
        for &(a, b) in &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)]
        {
            g.add_edge(a, b);
        }
        let q_split = modularity(&g, &[0, 0, 0, 1, 1, 1]);
        assert!(
            (q_split - 25.0 / 70.0).abs() < 1e-9,
            "two-community Q wrong: {q_split}"
        );

        // Putting everything in one community yields Q = 0 (up to rounding).
        let q_all = modularity(&g, &[0, 0, 0, 0, 0, 0]);
        assert!(
            q_all.abs() < 1e-9,
            "single-community Q should be 0, got {q_all}"
        );

        // The natural split must score strictly higher than the trivial grouping.
        assert!(
            q_split > q_all,
            "split {q_split} should beat trivial {q_all}"
        );
    }

    #[test]
    fn test_clustering_coefficient() {
        let g = triangle_graph();
        let cc = clustering_coefficient(&g, 0);
        assert!((cc - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_average_clustering() {
        let g = complete_graph();
        let acc = average_clustering(&g);
        assert!((acc - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_density() {
        let g = complete_graph();
        let d = density(&g);
        assert!((d - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_diameter() {
        let g = path_graph();
        assert_eq!(diameter(&g), 3);
    }

    #[test]
    fn test_betweenness_centrality_path() {
        // P4 = 0-1-2-3. Unordered shortest-path pairs through each node:
        //   node 1: {0,2},{0,3}            -> 2 pairs
        //   node 2: {0,3},{1,3}            -> 2 pairs
        //   endpoints 0,3: 0 pairs.
        // networkx-style normalization for undirected n=4 is x * 2/((n-1)(n-2))
        // = x * 2/6 = x/3, so node 1 and 2 score 2/3 and endpoints 0.
        let g = path_graph();
        let bc = betweenness_centrality(&g);
        assert!((bc[0] - 0.0).abs() < 1e-9);
        assert!((bc[1] - 2.0 / 3.0).abs() < 1e-9, "got {}", bc[1]);
        assert!((bc[2] - 2.0 / 3.0).abs() < 1e-9, "got {}", bc[2]);
        assert!((bc[3] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_betweenness_centrality_star_and_complete() {
        // Star S4: only the center (node 0) lies on shortest paths between the
        // three leaf pairs {1,2},{1,3},{2,3} -> 3 pairs, normalized 3 * 2/6 = 1.0.
        let star = star_graph();
        let bc = betweenness_centrality(&star);
        assert!(
            (bc[0] - 1.0).abs() < 1e-9,
            "star center should be 1.0, got {}",
            bc[0]
        );
        for &leaf in &bc[1..]
        {
            assert!(leaf.abs() < 1e-9);
        }

        // K4: every pair is directly connected, so NO node is an intermediary;
        // all betweenness scores are 0.
        let k4 = complete_graph();
        for c in betweenness_centrality(&k4)
        {
            assert!(c.abs() < 1e-9);
        }
    }

    #[test]
    fn test_edge_betweenness_path() {
        // P4 = 0-1-2-3. Raw (unordered) edge betweenness:
        //   (0,1): pairs using it {0,1},{0,2},{0,3} -> 3
        //   (1,2): {0,2},{0,3},{1,2},{1,3}          -> 4
        //   (2,3): {0,3},{1,3},{2,3}                -> 3
        // The implementation accumulates the directed value (twice the unordered
        // one) and divides by (n-1)(n-2) = 6, i.e. unordered/3.
        let g = path_graph();
        let eb = edge_betweenness(&g);
        assert_eq!(eb.len(), 3);
        assert!((eb[&(0, 1)] - 1.0).abs() < 1e-9, "got {}", eb[&(0, 1)]);
        assert!(
            (eb[&(1, 2)] - 4.0 / 3.0).abs() < 1e-9,
            "got {}",
            eb[&(1, 2)]
        );
        assert!((eb[&(2, 3)] - 1.0).abs() < 1e-9, "got {}", eb[&(2, 3)]);
        // The central edge must carry the most shortest-path traffic.
        assert!(eb[&(1, 2)] > eb[&(0, 1)]);
    }

    #[test]
    fn test_edge_betweenness_bridge_is_maximal() {
        // Two triangles joined by a single bridge 2-3. The bridge must have the
        // strictly highest edge betweenness, since every path between the two
        // triangles must cross it.
        let mut g = Graph::new(6);
        for &(a, b) in &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)]
        {
            g.add_edge(a, b);
        }
        let eb = edge_betweenness(&g);
        let bridge = eb[&(2, 3)];
        for (&edge, &score) in &eb
        {
            if edge != (2, 3)
            {
                assert!(
                    bridge > score,
                    "bridge {bridge} must exceed {edge:?} = {score}"
                );
            }
        }
    }

    #[test]
    fn test_degree_distribution() {
        let g = triangle_graph();
        let dist = degree_distribution(&g);
        assert_eq!(dist.get(&2), Some(&3));
    }

    #[test]
    fn test_nodes_within_distance() {
        // P4 = 0-1-2-3 from node 0 within distance 2 reaches exactly
        // 0@0, 1@1, 2@2 (node 3 is at distance 3, excluded).
        let g = path_graph();
        let mut nodes = g.nodes_within_distance(0, 2);
        nodes.sort();
        assert_eq!(nodes, vec![(0, 0), (1, 1), (2, 2)]);
    }

    #[test]
    fn test_shortest_path_disconnected_returns_none() {
        // Nodes 2 and 3 are isolated from the 0-1 edge.
        let mut g = Graph::new(4);
        g.add_edge(0, 1);
        assert_eq!(g.shortest_path(0, 1), Some(vec![0, 1]));
        assert_eq!(g.shortest_path(0, 3), None);
        // Trivial self path.
        assert_eq!(g.shortest_path(2, 2), Some(vec![2]));
    }

    #[test]
    fn test_clustering_coefficient_path_middle_is_zero() {
        // In P4, node 1's two neighbors (0 and 2) are not adjacent, so its local
        // clustering coefficient is 0.
        let g = path_graph();
        assert!((clustering_coefficient(&g, 1) - 0.0).abs() < 1e-9);
        // A degree-1 node has fewer than two neighbors -> coefficient 0 by definition.
        assert!((clustering_coefficient(&g, 0) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_average_clustering_partial() {
        // Triangle 0-1-2 plus a pendant node 3 attached to node 0.
        // Node 0: neighbors {1,2,3}; only the pair (1,2) is connected out of 3
        //         possible pairs -> cc = 1/3.
        // Nodes 1,2: neighbors are each other and node 0, both connected -> cc = 1.
        // Node 3: degree 1 -> cc = 0.
        // Average = (1/3 + 1 + 1 + 0) / 4 = (7/3) / 4 = 7/12.
        let mut g = Graph::new(4);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 0);
        g.add_edge(0, 3);
        assert!((clustering_coefficient(&g, 0) - 1.0 / 3.0).abs() < 1e-9);
        assert!((clustering_coefficient(&g, 1) - 1.0).abs() < 1e-9);
        assert!((average_clustering(&g) - 7.0 / 12.0).abs() < 1e-9);
    }

    #[test]
    fn test_diameter_cycle_and_disconnected() {
        // C5: farthest pair is 2 hops apart in either direction -> diameter 2.
        let mut c5 = Graph::new(5);
        for &(a, b) in &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        {
            c5.add_edge(a, b);
        }
        assert_eq!(diameter(&c5), 2);

        // Disconnected: path 0-1-2 (component diameter 2) plus an isolated edge
        // 3-4. Unreachable pairs must be ignored, so the diameter is the largest
        // FINITE distance = 2, not usize::MAX.
        let mut g = Graph::new(5);
        for &(a, b) in &[(0, 1), (1, 2), (3, 4)]
        {
            g.add_edge(a, b);
        }
        assert_eq!(diameter(&g), 2);

        // A fully isolated graph (no edges) has diameter 0.
        assert_eq!(diameter(&Graph::new(3)), 0);
    }

    #[test]
    fn test_connected_components() {
        // Connected graph -> all nodes share component 0.
        assert_eq!(connected_components(&triangle_graph()), vec![0, 0, 0]);

        // Two triangles, no bridge -> components {0,1,2}=0 and {3,4,5}=1.
        let mut g = Graph::new(6);
        for &(a, b) in &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]
        {
            g.add_edge(a, b);
        }
        assert_eq!(connected_components(&g), vec![0, 0, 0, 1, 1, 1]);

        // Isolated nodes each form their own component.
        let mut h = Graph::new(4);
        h.add_edge(0, 1);
        assert_eq!(connected_components(&h), vec![0, 0, 1, 2]);
    }

    #[test]
    fn test_label_propagation_two_disjoint_triangles() {
        // Two disconnected triangles must collapse into exactly two communities,
        // with the three nodes of each triangle sharing one label.
        let mut g = Graph::new(6);
        for &(a, b) in &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]
        {
            g.add_edge(a, b);
        }
        let comm = label_propagation(&g, 100);
        assert_eq!(count_communities(&comm), 2);
        assert_eq!(comm[0], comm[1]);
        assert_eq!(comm[1], comm[2]);
        assert_eq!(comm[3], comm[4]);
        assert_eq!(comm[4], comm[5]);
        assert_ne!(comm[0], comm[3]);
    }

    #[test]
    fn test_girvan_newman_splits_bridge() {
        // Two triangles joined by a single bridge edge 2-3. Removing the bridge
        // (highest edge betweenness) splits the graph into the two triangles, so
        // requesting 2 communities must recover {0,1,2} and {3,4,5}.
        let mut g = Graph::new(6);
        for &(a, b) in &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)]
        {
            g.add_edge(a, b);
        }
        let comm = girvan_newman(&g, 2);
        assert_eq!(count_communities(&comm), 2);
        // Nodes within a triangle share a community; the two triangles differ.
        assert_eq!(comm[0], comm[1]);
        assert_eq!(comm[1], comm[2]);
        assert_eq!(comm[3], comm[4]);
        assert_eq!(comm[4], comm[5]);
        assert_ne!(comm[0], comm[3]);

        // Requesting a single community removes no edges: one component.
        assert_eq!(count_communities(&girvan_newman(&g, 1)), 1);
    }

    #[test]
    fn test_girvan_newman_path_peels_endpoints() {
        // On P4, the most-central edge is (1,2). Removing it first yields two
        // components {0,1} and {2,3}, satisfying a request for 2 communities.
        let g = path_graph();
        let comm = girvan_newman(&g, 2);
        assert_eq!(count_communities(&comm), 2);
        assert_eq!(comm[0], comm[1]);
        assert_eq!(comm[2], comm[3]);
        assert_ne!(comm[1], comm[2]);
    }

    #[test]
    fn test_dfs_path_order_is_exact() {
        // DFS from node 0 on a path must walk straight down it.
        let g = path_graph();
        assert_eq!(g.dfs(0), vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_bfs_is_breadth_first_on_star() {
        // BFS from the center of a star visits the center, then all leaves.
        let g = star_graph();
        let order = g.bfs(0);
        assert_eq!(order[0], 0);
        let mut leaves = order[1..].to_vec();
        leaves.sort();
        assert_eq!(leaves, vec![1, 2, 3]);
    }

    #[test]
    fn test_degree_distribution_path() {
        // P4 degree sequence is [1,2,2,1]: two nodes of degree 1, two of degree 2.
        let g = path_graph();
        let dist = degree_distribution(&g);
        assert_eq!(dist.get(&1), Some(&2));
        assert_eq!(dist.get(&2), Some(&2));
        assert_eq!(dist.get(&3), None);
    }

    #[test]
    fn test_density_path_and_empty() {
        // P4: 3 edges out of max 4*3/2 = 6 -> density 0.5.
        assert!((density(&path_graph()) - 0.5).abs() < 1e-9);
        // No-edge graph has density 0; single node has density 0 (no possible edges).
        assert!((density(&Graph::new(5)) - 0.0).abs() < 1e-9);
        assert!((density(&Graph::new(1)) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_labeled_subgraph_isomorphism_respects_labels() {
        // Pattern: a single A--B edge. Target: a 3-path with labels A-B-A so that
        // node 0=A,1=B,2=A and edges 0-1,1-2. The A--B pattern should match the
        // (0,1) and (1,2) edges but a C--D pattern should not match at all.
        let mut pattern = Graph::new(2);
        pattern.add_edge(0, 1);
        pattern.set_node_label(0, "A");
        pattern.set_node_label(1, "B");

        let mut target = Graph::new(3);
        target.add_edge(0, 1);
        target.add_edge(1, 2);
        target.set_node_label(0, "A");
        target.set_node_label(1, "B");
        target.set_node_label(2, "A");

        let maps = subgraph_isomorphism(&pattern, &target);
        // Mappings: {0->0,1->1} and {0->2,1->1}. (Pattern node 0 = "A", 1 = "B".)
        assert_eq!(
            maps.len(),
            2,
            "A-B edge should map onto both A-B target edges"
        );
        for m in &maps
        {
            assert_eq!(m[&1], 1, "pattern node B must map to the unique target B");
            assert!(target.are_adjacent(m[&0], m[&1]));
        }

        // A pattern whose label is absent from the target yields no mapping.
        let mut nope = Graph::new(2);
        nope.add_edge(0, 1);
        nope.set_node_label(0, "C");
        nope.set_node_label(1, "D");
        assert!(subgraph_isomorphism(&nope, &target).is_empty());
    }

    #[test]
    fn test_label_propagation_is_deterministic_on_ties() {
        // Regression: among neighbor-label count ties, the algorithm must break
        // ties deterministically (smallest label). Previously `candidates[0]`
        // was taken from a HashMap whose iteration order is randomized, so the
        // result varied run-to-run.
        //
        // A 4-cycle 0-1-2-3-0 starts with every node in its own community and is
        // maximally tie-prone: on the first sweep each node sees two neighbors
        // carrying two different labels, one each — a genuine count tie.
        let mut g = Graph::new(4);
        for &(a, b) in &[(0, 1), (1, 2), (2, 3), (3, 0)]
        {
            g.add_edge(a, b);
        }

        // The exact result is fixed by the smallest-label tie-break; recompute it
        // many times and require byte-for-byte identical output every run. Under
        // the old code the randomized HashMap order made this flaky.
        let reference = label_propagation(&g, 100);
        for _ in 0..200
        {
            assert_eq!(
                label_propagation(&g, 100),
                reference,
                "label_propagation must be deterministic across runs"
            );
        }
    }

    #[test]
    fn test_girvan_newman_is_deterministic_on_ties() {
        // Regression: when several edges share the maximal betweenness, the edge
        // removed must be chosen deterministically. Previously `max_by` scanned a
        // HashMap with randomized order, so the peeled edge — and the resulting
        // partition — varied run-to-run.
        //
        // A 6-cycle is fully symmetric: all six edges have identical betweenness,
        // so the first removal is a pure tie. Whatever partition the tie-break
        // produces, it must be the SAME every single run.
        let mut g = Graph::new(6);
        for &(a, b) in &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
        {
            g.add_edge(a, b);
        }

        let reference = girvan_newman(&g, 2);
        assert_eq!(count_communities(&reference), 2);
        for _ in 0..200
        {
            assert_eq!(
                girvan_newman(&g, 2),
                reference,
                "girvan_newman must pick the same edge on betweenness ties"
            );
        }
    }

    #[test]
    fn test_edge_label_roundtrip() {
        let mut g = Graph::new(3);
        g.add_edge_labeled(0, 2, "bond");
        // Labels are stored on the canonical (min,max) key and are direction-free.
        assert_eq!(g.edge_label(0, 2), Some("bond"));
        assert_eq!(g.edge_label(2, 0), Some("bond"));
        assert_eq!(g.edge_label(0, 1), None);
        assert!(g.are_adjacent(0, 2));
        assert!(g.are_adjacent(2, 0));
    }
}
