use crate::core::Reasoner;
use crate::graph::kg::KnowledgeGraph;
use std::collections::{HashMap, HashSet, VecDeque};

/// Reasoning over a [`KnowledgeGraph`]: shortest-path discovery via breadth-first
/// search over directed `subject --relation--> object` edges.
pub struct GraphReasoning {
    pub kg: KnowledgeGraph,
}

impl GraphReasoning {
    pub fn new(kg: KnowledgeGraph) -> Self {
        Self { kg }
    }

    /// Returns the shortest path of entity names from `start` to `end`
    /// (inclusive), or an empty vector if no path exists.
    pub fn find_path(&self, start: &str, end: &str) -> Vec<String> {
        if start == end
        {
            return vec![start.to_string()];
        }
        let mut prev: HashMap<String, String> = HashMap::new();
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<String> = VecDeque::new();
        queue.push_back(start.to_string());
        visited.insert(start.to_string());

        while let Some(current) = queue.pop_front()
        {
            // Outgoing neighbours, visited in a deterministic order.
            let mut neighbours: Vec<&str> = self
                .kg
                .triples
                .iter()
                .filter(|t| t.subject.0 == current)
                .map(|t| t.object.0.as_str())
                .collect();
            neighbours.sort_unstable();
            for nb in neighbours
            {
                if !visited.insert(nb.to_string())
                {
                    continue;
                }
                prev.insert(nb.to_string(), current.clone());
                if nb == end
                {
                    return reconstruct(&prev, start, end);
                }
                queue.push_back(nb.to_string());
            }
        }
        Vec::new()
    }
}

fn reconstruct(prev: &HashMap<String, String>, start: &str, end: &str) -> Vec<String> {
    let mut path = vec![end.to_string()];
    let mut node = end.to_string();
    while node != start
    {
        match prev.get(&node)
        {
            Some(p) =>
            {
                path.push(p.clone());
                node = p.clone();
            },
            None => return Vec::new(),
        }
    }
    path.reverse();
    path
}

impl Reasoner for GraphReasoning {
    fn name(&self) -> &str {
        "GraphReasoning"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::kg::KnowledgeGraph;

    #[test]
    fn finds_multi_hop_path() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple("A", "to", "B");
        kg.add_triple("B", "to", "C");
        kg.add_triple("C", "to", "D");
        let gr = GraphReasoning::new(kg);
        assert_eq!(gr.find_path("A", "D"), vec!["A", "B", "C", "D"]);
    }

    #[test]
    fn returns_empty_when_unreachable() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple("A", "to", "B");
        kg.add_triple("X", "to", "Y");
        let gr = GraphReasoning::new(kg);
        assert!(gr.find_path("A", "Y").is_empty());
    }
}
