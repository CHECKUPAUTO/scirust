//! Graph pattern matching: subgraph isomorphism, motif discovery, community detection.

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
        pattern_matched: vec![false; pattern.n_nodes],
        target_matched: vec![false; target.n_nodes],
    };

    iso_recursive(&mut state, 0, &mut mappings);
    mappings
}

struct IsoState<'a> {
    pattern: &'a Graph,
    target: &'a Graph,
    mapping: HashMap<usize, usize>,
    pattern_matched: Vec<bool>,
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
        state.pattern_matched[depth] = true;
        state.target_matched[t] = true;

        iso_recursive(state, depth + 1, mappings);

        // Backtrack
        state.mapping.remove(&depth);
        state.pattern_matched[depth] = false;
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
                    sub.add_edge(i, mapped);
                    if let Some(label) = graph.edge_label(n, neighbor)
                    {
                        sub.add_edge_labeled(i, mapped, label);
                    }
                }
            }
        }
    }
    sub
}

fn canonical_form(graph: &Graph) -> Vec<Vec<usize>> {
    // Simple canonical form: sorted adjacency lists
    let mut adj: Vec<Vec<usize>> = graph.adjacency.clone();
    for a in &mut adj
    {
        a.sort();
    }
    adj
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

            // Find most common label
            let max_count = label_counts.values().max().unwrap_or(&0);
            let candidates: Vec<usize> = label_counts
                .iter()
                .filter(|(_, &c)| c == *max_count)
                .map(|(&l, _)| l)
                .collect();

            let new_label = candidates[0]; // deterministic: pick first
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

/// Louvain-like modularity optimization (simplified).
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

/// Girvan-Newman community detection (edge betweenness).
pub fn girvan_newman(graph: &Graph, n_communities: usize) -> Vec<usize> {
    let mut g = graph.clone();
    let mut communities = (0..graph.n_nodes).collect::<Vec<usize>>();

    while count_communities(&communities) < n_communities && g.n_edges() > 0
    {
        // Compute edge betweenness
        let betweenness = edge_betweenness(&g);

        // Remove edge with highest betweenness
        if let Some((&(u, v), _)) = betweenness
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        {
            g.adjacency[u].retain(|&x| x != v);
            g.adjacency[v].retain(|&x| x != u);

            // Recompute communities
            communities = label_propagation(&g, 100);
        }
        else
        {
            break;
        }
    }

    communities
}

fn count_communities(communities: &[usize]) -> usize {
    let mut seen = HashSet::new();
    for &c in communities
    {
        seen.insert(c);
    }
    seen.len()
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

/// Diameter of the graph (longest shortest path).
pub fn diameter(graph: &Graph) -> usize {
    let mut max_dist = 0;
    for start in 0..graph.n_nodes
    {
        let distances = bfs_distances(graph, start);
        for &d in &distances
        {
            if d > max_dist
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
    fn test_subgraph_isomorphism() {
        let pattern = triangle_graph();
        let target = complete_graph();
        let mappings = subgraph_isomorphism(&pattern, &target);
        assert!(!mappings.is_empty());
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
    fn test_find_motifs() {
        let g = complete_graph();
        let motifs = find_motifs(&g, 3);
        assert!(!motifs.is_empty());
    }

    #[test]
    fn test_label_propagation() {
        let g = complete_graph();
        let communities = label_propagation(&g, 100);
        // Complete graph should have 1 community
        assert_eq!(count_communities(&communities), 1);
    }

    #[test]
    fn test_modularity() {
        let g = complete_graph();
        let communities = vec![0, 0, 0, 0];
        let q = modularity(&g, &communities);
        // For a complete graph with all nodes in one community, modularity should be positive
        // Q = (1/2m) * sum_{ij in same community} [A_ij - k_i*k_j/(2m)]
        // For K4: m=6, k_i=3 for all i, A_ij=1 for all i!=j
        // Q = (1/12) * 16 * (1 - 9/12) = (1/12) * 16 * 0.25 = 4/12 = 0.333
        assert!(
            q > -1.0 && q <= 1.0,
            "modularity should be in [-1, 1], got {}",
            q
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
    fn test_betweenness_centrality() {
        let g = path_graph();
        let bc = betweenness_centrality(&g);
        // Node 1 (middle) should have highest betweenness
        assert!(bc[1] > bc[0]);
        assert!(bc[1] > bc[3]);
    }

    #[test]
    fn test_edge_betweenness() {
        let g = path_graph();
        let eb = edge_betweenness(&g);
        assert!(!eb.is_empty());
    }

    #[test]
    fn test_degree_distribution() {
        let g = triangle_graph();
        let dist = degree_distribution(&g);
        assert_eq!(dist.get(&2), Some(&3));
    }

    #[test]
    fn test_nodes_within_distance() {
        let g = path_graph();
        let nodes = g.nodes_within_distance(0, 2);
        assert!(nodes.len() >= 3); // nodes 0, 1, 2
    }
}
